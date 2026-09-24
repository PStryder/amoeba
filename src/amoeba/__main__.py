"""Command-line entry point.

    python -m amoeba doctor     --config config.toml
    python -m amoeba supervise  --config config.toml
    python -m amoeba mcp        --config config.toml --transport stdio
    python -m amoeba status     --config config.toml
    python -m amoeba shutdown   --config config.toml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import load_config


def _doctor(cfg: Any) -> int:
    """Report what is actually present and working. No aspirational claims."""
    report: dict[str, Any] = {"checks": [], "ok": True}

    def check(name: str, ok: bool, detail: Any = "", *, fatal: bool = False) -> None:
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail,
                                 "fatal": fatal})
        if fatal and not ok:
            report["ok"] = False

    check("python", sys.version_info[:2] == (3, 11), sys.version.split()[0])
    for label, path in (("state_dir", cfg.state_dir), ("runtime_dir", cfg.runtime_dir),
                        ("models_dir", cfg.models_dir)):
        check(label, Path(path).exists(), str(path), fatal=(label == "state_dir"))

    try:
        import mcp  # noqa: F401
        from mcp.server.fastmcp import FastMCP  # noqa: F401
        check("mcp_sdk", True, "mcp + FastMCP import ok", fatal=True)
    except Exception as exc:  # noqa: BLE001
        check("mcp_sdk", False, repr(exc), fatal=True)

    try:
        import numpy  # noqa: F401
        check("numpy", True, numpy.__version__)
    except Exception as exc:  # noqa: BLE001
        check("numpy", False, repr(exc), fatal=True)

    backend = cfg.backend
    check("backend_kind", True, backend.kind)
    if backend.kind == "llama_cpp":
        lib = Path(backend.lib_path)
        check("llama_dll", lib.exists(), str(lib), fatal=True)
        model = Path(backend.model_path)
        check("model_file", model.exists(),
              f"{model} ({model.stat().st_size/2**30:.2f} GiB)" if model.exists()
              else str(model), fatal=True)
        if lib.exists():
            try:
                from .backends.llama_ffi import GgmlFFI, LlamaFFI

                ggml = GgmlFFI(lib.parent)
                ggml.load_backends()
                devices = ggml.devices()
                gpus = [d for d in devices if d["type"] == 1]
                check("ggml_backends", bool(devices),
                      [f"{d['name']}: {d['description']}" for d in devices])
                check("cuda_device", bool(gpus),
                      [f"{d['description']} free={d['free_bytes']/2**30:.2f}GiB "
                       f"total={d['total_bytes']/2**30:.2f}GiB" for d in gpus])
                ffi = LlamaFFI(lib)
                ffi.silence_logs()
                check("llama_abi", True, ffi.validate_abi(), fatal=True)
                check("gpu_offload_supported",
                      bool(ffi.lib.llama_supports_gpu_offload()),
                      "llama_supports_gpu_offload()")
            except Exception as exc:  # noqa: BLE001
                check("llama_abi", False, repr(exc), fatal=True)
    else:
        check("simulated_backend_warning", True,
              "backend.kind is 'deterministic': output is a hash function, "
              "NOT model inference")

    try:
        from .mind import Mind

        mind = Mind(cfg)
        integrity = mind.verify_integrity(deep=True)
        check("state_store", True, {"state_version": integrity["state_version"],
                                    "counts": integrity["counts"]})
        check("hash_chain", integrity["hash_chain_ok"], integrity["first_bad_event"])
        check("content_addressable_store", integrity["missing_content_count"] == 0,
              integrity["missing_content"][:5])
        mind.close()
    except Exception as exc:  # noqa: BLE001
        check("state_store", False, repr(exc), fatal=True)

    check("supervisor_running", cfg.ready_path.exists(),
          cfg.ready_path.read_text() if cfg.ready_path.exists() else "not running")

    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def _status(cfg: Any) -> int:
    from .rpc import RpcClient, read_or_create_token

    token = read_or_create_token(cfg.token_path)
    client = RpcClient(cfg.supervisor_host, cfg.supervisor_port, token, timeout=30)
    try:
        client.connect(retries=2)
        print(json.dumps(client.call("health"), indent=2, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": repr(exc),
                          "hint": "is the supervisor running? "
                                  "python -m amoeba supervise"}, indent=2))
        return 1
    finally:
        client.close()


def _shutdown(cfg: Any) -> int:
    from .rpc import RpcClient, read_or_create_token

    token = read_or_create_token(cfg.token_path)
    client = RpcClient(cfg.supervisor_host, cfg.supervisor_port, token, timeout=30)
    try:
        client.connect(retries=2)
        print(json.dumps(client.call("shutdown"), indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": repr(exc)}, indent=2))
        return 1
    finally:
        client.close()


def _reset(cfg: Any, *, delete: bool, dry_run: bool,
           rotate_credentials: bool, confirmed: bool) -> int:
    """Start a new organism. The old one is moved aside, not erased.

    Refuses while a supervisor owns the state directory: deleting a database
    under a live writer leaves a half-state nobody can reason about later.
    """
    from . import reset as reset_mod
    from .errors import ResourceExhausted

    live = reset_mod.holder(cfg)
    if live is not None:
        print(f"refusing: a supervisor (pid {live.get('pid')}) owns "
              f"{cfg.state_dir}.\n"
              f"  stop it first: python -m amoeba shutdown --config <config>")
        return 2
    if delete and not (confirmed or dry_run):
        print("refusing: --delete destroys the record of the organism that ran.\n"
              "  archive it instead (omit --delete), or pass --yes to mean it.")
        return 2

    # Held across deciding and doing. The check above is a courtesy that
    # gives a readable message; this is what actually excludes a supervisor.
    try:
        with reset_mod.owned(cfg):
            chosen = reset_mod.plan(cfg, rotate_credentials=rotate_credentials)
            if not chosen["holds_state"]:
                print(f"nothing to reset: {cfg.state_dir} holds no organism state")
                return 0
            out = reset_mod.perform(chosen, delete=delete, dry_run=dry_run)
    except ResourceExhausted as exc:
        print(f"refusing: {exc.message}\n"
              f"  stop it first: python -m amoeba shutdown --config <config>")
        return 2
    verb = ("would move" if dry_run else
            "deleted" if out["deleted"] else "archived")
    print(f"{verb} {len(out['moved'])} item(s) from {cfg.state_dir}:")
    for name in out["moved"]:
        print(f"  {name}")
    if out["archive"]:
        print(f"the old organism is readable at {out['archive']}")
    if out["kept"]:
        print("kept (credentials; --rotate-credentials reissues them):")
        for name in out["kept"]:
            print(f"  {name}")
    if not dry_run:
        print("the next `supervise` starts a new organism at incarnation 1, "
              "with the prompt library rebootstrapped from the shipped files.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amoeba")
    ap.add_argument("command",
                    choices=["doctor", "supervise", "mcp", "status", "shutdown",
                             "reset", "ego", "id", "inference", "neuocyte"])
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    ap.add_argument("--transport", default="stdio")
    ap.add_argument("--neuocyte-id", default=None)
    ap.add_argument("--work-id", default=None)
    # `reset`: what to do with the organism that is there now.
    ap.add_argument("--delete", action="store_true",
                    help="reset: remove the old state instead of archiving it")
    ap.add_argument("--rotate-credentials", action="store_true",
                    help="reset: reissue tokens and API keys as well")
    ap.add_argument("--dry-run", action="store_true",
                    help="reset: say what would move, and move nothing")
    ap.add_argument("--yes", action="store_true",
                    help="reset: confirm an irreversible --delete")
    args, rest = ap.parse_known_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    cfg_args = ["--config", args.config] if args.config else []

    if args.command == "doctor":
        return _doctor(cfg)
    if args.command == "status":
        return _status(cfg)
    if args.command == "shutdown":
        return _shutdown(cfg)
    if args.command == "reset":
        return _reset(cfg, delete=args.delete, dry_run=args.dry_run,
                      rotate_credentials=args.rotate_credentials,
                      confirmed=args.yes)
    if args.command == "supervise":
        from .supervisor import main as sup_main
        return sup_main(cfg_args)
    if args.command == "mcp":
        from .mcp_api import main as mcp_main
        return mcp_main(cfg_args + ["--transport", args.transport])
    if args.command == "inference":
        from .inference_service import main as inf_main
        return inf_main(cfg_args)
    if args.command in ("ego", "id"):
        from .roles import main as roles_main
        return roles_main([args.command, *cfg_args])
    if args.command == "neuocyte":
        from .neuocyte import main as neuocyte_main
        extra = []
        if args.neuocyte_id:
            extra += ["--neuocyte-id", args.neuocyte_id]
        if args.work_id:
            extra += ["--work-id", args.work_id]
        return neuocyte_main(cfg_args + extra)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
