"""Experiment: is a forked Ego prefix physically SHARED or physically COPIED?

VRAM deltas cannot answer this. llama.cpp allocates the whole KV buffer when
the context is created, so a fork never changes VRAM whether or not cells are
duplicated. ``llama_state_seq_get_size`` cannot answer it either: that reports
the *logical* serialized size of a sequence and is identical for a shared and a
copied prefix.

The decisive measurement is capacity. Build a prefix of P tokens, fork it to W
neuocytes, then keep appending tokens until the cache refuses a slot. Count how
many cells the context actually accommodated:

* physically shared -> occupancy is  P + (private tails)
* physically copied -> occupancy is  P*(W+1) + (private tails)

Run both ``kv_unified=True`` (one KV stream) and ``kv_unified=False`` (one
stream per sequence) and compare against the theoretical numbers.

Usage:
    .\\.venv\\Scripts\\python.exe bench\\prefix_sharing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amoeba.backends.llama_engine import LlamaEngine  # noqa: E402
from amoeba.errors import BackendUnavailable, ResourceExhausted  # noqa: E402

def _engine_paths() -> tuple[str, str]:
    """The runtime and model this machine has, from config or the environment.

    These were one machine's absolute paths written into the script, so it ran
    nowhere else. The operator's `config.toml` is asked first because it
    already carries them; AMOEBA_RUNTIME_DIR and AMOEBA_MODEL_PATH override,
    and a missing answer says what to set rather than failing on a path
    nobody here chose.
    """
    import os

    from amoeba.config import load_config

    runtime = os.environ.get("AMOEBA_RUNTIME_DIR", "").strip()
    model = os.environ.get("AMOEBA_MODEL_PATH", "").strip()
    cfg_file = Path(__file__).resolve().parents[1] / "config.toml"
    if (not runtime or not model) and cfg_file.exists():
        cfg = load_config(cfg_file)
        runtime = runtime or str(cfg.runtime_dir)
        model = model or str(cfg.backend.model_path)
    if not runtime or not model:
        raise SystemExit(
            "no runtime or model to load: copy config.example.toml to "
            "config.toml and set backend.model_path, or set "
            "AMOEBA_RUNTIME_DIR and AMOEBA_MODEL_PATH")
    return runtime, model

N_CTX = 2048
N_SEQ_MAX = 4
PREFIX = 900
WORKERS = 3
STEP = 16


def probe(kv_unified: bool) -> dict:
    _r, _m = _engine_paths()
    eng = LlamaEngine(
        runtime_dir=_r, model_path=_m,
        n_ctx=N_CTX, n_seq_max=N_SEQ_MAX, n_gpu_layers=-1,
        n_batch=512, n_ubatch=512, kv_unified=kv_unified,
    )
    report = eng.load()
    out: dict = {
        "kv_unified": kv_unified,
        "n_kv_streams": report["n_kv_streams"],
        "n_ctx_total": report["n_ctx_total"],
        "n_ctx_per_seq": report["n_ctx_per_seq"],
        "vram_consumed_by_load": report["vram_consumed_by_load"],
        "prefix_len": PREFIX,
        "neuocytes": WORKERS,
    }
    try:
        filler = eng.tokenize("The amoeba records evidence before it forms beliefs. ",
                              add_special=False)
        prefix_tokens = (filler * (PREFIX // len(filler) + 2))[:PREFIX]

        ego = eng.open_session(role="ego")
        t0 = time.perf_counter()
        eng.ingest(ego.session_id, prefix_tokens)
        out["prefill_seconds"] = time.perf_counter() - t0
        out["vram_free_after_prefill"] = eng.vram_free()

        t0 = time.perf_counter()
        forks = []
        fork_errors = []
        for i in range(WORKERS):
            try:
                forks.append(eng.fork_prefix(src_session_id=ego.session_id,
                                             prefix_len=PREFIX, role="neuocyte"))
            except Exception as exc:  # noqa: BLE001
                fork_errors.append(repr(exc))
        out["fork_seconds_total"] = time.perf_counter() - t0
        out["forks_created"] = len(forks)
        out["fork_errors"] = fork_errors
        out["vram_free_after_fork"] = eng.vram_free()
        out["vram_delta_at_fork_bytes"] = (
            out["vram_free_after_prefill"] - out["vram_free_after_fork"]
        )
        if forks:
            out["state_seq_size_worker_logical"] = eng.state_seq_size(forks[0].session_id)
            out["state_seq_size_ego_logical"] = eng.state_seq_size(ego.session_id)

        # Fill private tails round-robin until the cache refuses a slot.
        sessions = [ego] + forks
        appended = 0
        exhausted_reason = None
        tok = filler[0]
        try:
            while appended < N_CTX * (WORKERS + 2):
                for s in sessions:
                    eng.ingest(s.session_id, [tok] * STEP, compute_logits=False)
                    appended += STEP
        except ResourceExhausted as exc:
            exhausted_reason = f"ResourceExhausted: {exc.details}"
        except BackendUnavailable as exc:
            exhausted_reason = f"BackendUnavailable: {exc.details}"

        out["tail_tokens_accommodated"] = appended
        out["exhausted_reason"] = exhausted_reason
        out["observed_total_logical_tokens"] = PREFIX * (len(forks) + 1) + appended
        out["cells_if_shared"] = PREFIX + appended
        out["cells_if_copied"] = PREFIX * (len(forks) + 1) + appended
        out["kv_capacity_cells"] = N_CTX if kv_unified else N_CTX * N_SEQ_MAX

        shared_fits = out["cells_if_shared"] <= out["kv_capacity_cells"] + STEP * len(sessions)
        copied_fits = out["cells_if_copied"] <= out["kv_capacity_cells"] + STEP * len(sessions)
        if shared_fits and not copied_fits:
            out["verdict"] = "PHYSICALLY_SHARED"
        elif copied_fits:
            out["verdict"] = "INCONCLUSIVE_capacity_too_large_to_discriminate"
        else:
            out["verdict"] = "PHYSICALLY_COPIED_or_unexpected"
        return out
    finally:
        eng.close()


def main() -> int:
    results = []
    for unified in (True, False):
        print(f"\n{'='*70}\nkv_unified={unified}\n{'='*70}")
        try:
            r = probe(unified)
        except Exception as exc:  # noqa: BLE001
            r = {"kv_unified": unified, "error": repr(exc)}
        results.append(r)
        for k, v in r.items():
            print(f"  {k}: {v}")

    outdir = ROOT / "bench" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "prefix_sharing.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
