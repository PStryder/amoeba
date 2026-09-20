"""Configuration loading.

All absolute paths (runtime binaries, model weights, durable state) live in the
config file so that third-party runtimes, weights, application source and
runtime data stay in separate trees.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HomeostasisSettings:
    elevated: float = 0.55
    high: float = 0.70
    critical: float = 0.85
    role_context_high: float = 0.75
    keep_head_tokens: int = 512
    keep_tail_fraction: float = 0.45
    min_seconds_between_rejuvenations: float = 120.0
    max_rejuvenations_per_hour: int = 12
    auto_rejuvenate: bool = True


@dataclass(slots=True)
class BackendConfig:
    kind: str = "deterministic"          # deterministic | llama_cpp
    lib_path: str = ""                   # absolute path to llama.dll (llama_cpp)
    model_path: str = ""                 # absolute path to .gguf
    n_gpu_layers: int = -1               # -1 = offload all
    n_ctx: int = 16384                   # TOTAL KV cells across all sequences
    n_seq_max: int = 8                   # max concurrent sequences in the context
    n_batch: int = 1024
    n_ubatch: int = 512
    n_threads: int = 8
    flash_attn: bool = True
    type_k: str = "f16"
    type_v: str = "f16"
    seed: int = 1234
    warn_if_fake: bool = True


@dataclass(slots=True)
class ArbiterConfig:
    max_neuocytes: int = 4
    max_outstanding_work: int = 64
    max_prompt_tokens: int = 6144
    max_completion_tokens: int = 512
    neuocyte_wall_seconds: float = 180.0
    neuocyte_token_budget: int = 2048
    neuocyte_max_age_seconds: float = 900.0
    lease_seconds: float = 90.0
    # Tool turns per neuocyte. One of three independent bounds on the tool
    # loop, alongside the token budget and the wall-clock deadline; a model
    # that keeps calling tools is an expected outcome, not a malfunction.
    max_tool_turns: int = 6
    # Weighted-fair split between user-directed work and Id maintenance.
    user_weight: float = 0.7
    maintenance_weight: float = 0.3
    # Neither class may be starved: each is guaranteed this many neuocyte slots.
    user_reserved_slots: int = 1
    maintenance_reserved_slots: int = 1
    max_maintenance_depth: int = 2
    max_maintenance_per_hour: int = 20


@dataclass(slots=True)
class SandboxConfig:
    enabled: bool = True
    wall_seconds: float = 60.0
    cpu_seconds: float = 60.0
    memory_bytes: int = 1024 * 1024 * 1024
    max_processes: int = 8
    max_output_bytes: int = 262144
    max_scratch_bytes: int = 268435456
    max_artifact_bytes: int = 16777216
    max_concurrent: int = 4


@dataclass(slots=True)
class RoleConfig:
    max_context_tokens: int = 6144
    max_turns_resident: int = 40
    system_prompt: str = ""


@dataclass(slots=True)
class FilespaceRoot:
    """One host directory Amoeba is allowed to touch, and how.

    Nothing outside the configured roots is reachable. This is an allowlist at
    the Harness layer rather than a check inside a tool, so there is no handler
    that could be reached with a path the resolver never approved.
    """
    name: str = ""
    path: str = ""
    mode: str = "read_only"          # read_only | read_write
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class FilespaceConfig:
    roots: list[FilespaceRoot] = field(default_factory=list)
    max_read_bytes: int = 4 * 1024 * 1024
    max_write_bytes: int = 16 * 1024 * 1024
    max_listing: int = 500
    # Prior content is content-addressed before any overwrite or delete, so a
    # write is always reversible. Turning this off means a neuocyte-authored
    # write can destroy a file, which is the thing the design exists to stop.
    snapshot_before_overwrite: bool = True
    # An NTFS hard link is not a pointer to a file, it *is* the file: a
    # second directory entry for the same record. Path containment says
    # "inside the root" and is telling the truth about the path while
    # being wrong about the file. Measured: reading through a planted hard
    # link returned content from outside the root. Refusing multiply-linked
    # files is what keeps the allowlist a statement about files.
    allow_multiply_linked: bool = False


@dataclass(slots=True)
class Config:
    state_dir: Path = Path("F:/hexylab/amoeba-state")
    runtime_dir: Path = Path("F:/hexylab/amoeba-runtime")
    models_dir: Path = Path("F:/hexylab/amoeba-models")
    supervisor_host: str = "127.0.0.1"
    supervisor_port: int = 8711
    inference_port: int = 8712
    ego_port: int = 8713
    id_port: int = 8714
    log_level: str = "INFO"
    # Loopback by default. Binding here is not authentication:
    # every request still needs a credential (see http_api).
    api_host: str = "127.0.0.1"
    api_port: int = 8715
    api_enabled: bool = True
    backend: BackendConfig = field(default_factory=BackendConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    filespace: FilespaceConfig = field(default_factory=FilespaceConfig)
    homeostasis: "HomeostasisSettings" = field(default_factory=lambda: HomeostasisSettings())
    ego: RoleConfig = field(default_factory=RoleConfig)
    id: RoleConfig = field(default_factory=RoleConfig)
    source_path: Path | None = None

    # -- derived paths -------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.state_dir / "mind.sqlite3"

    @property
    def blob_dir(self) -> Path:
        return self.state_dir / "blobs"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def sandbox_dir(self) -> Path:
        return self.state_dir / "sandbox"

    @property
    def artifact_dir(self) -> Path:
        """Accepted work product, when no filespace root was named.

        Durable and outside every compute sandbox, which is the point: an
        artifact is authoritative only once it lives here (or in a filespace
        root) and in the blob store. Deliberately not called a "workspace" --
        that word was doing double duty for this and for the ephemeral compute
        sandbox, which have opposite lifetimes.
        """
        return self.state_dir / "artifacts"

    @property
    def ready_path(self) -> Path:
        return self.state_dir / "supervisor.ready"

    @property
    def token_path(self) -> Path:
        """The operator/control credential.

        Held by the supervisor itself and by the MCP facade. Deliberately NOT
        given to Ego, Id or neuocytes: presenting it grants the full method
        table, so handing it to a child would make every capability boundary
        below it decorative.
        """
        return self.state_dir / "control.token"

    @property
    def operator_session_path(self) -> Path:
        """Where the console's session credential is written at startup."""
        return self.state_dir / "operator.session"

    @property
    def api_clients_path(self) -> Path:
        """API keys and the client identity each one carries."""
        return self.state_dir / "api_clients.json"

    def scope_token_path(self, scope: str) -> Path:
        """The credential a child presents to name its own capability scope.

        The scope a caller gets is what its secret *is*, never what it claims,
        so there is no role field to forge. Same-account file permissions are
        the residual limit here, exactly as for the ACL work in security.py:
        this removes accidental and model-driven capability, not a determined
        process running as the Amoeba user.
        """
        safe = "".join(c for c in scope if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError("scope name must not be empty")
        return self.state_dir / f"scope.{safe}.token"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "supervisor.lock"

    def ensure_dirs(self) -> None:
        # One-time rename of the old "workspace" directory. Promoted artifacts
        # are durable, so they are moved rather than orphaned by the renaming.
        legacy = self.state_dir / "workspace"
        if legacy.is_dir() and not self.artifact_dir.exists():
            legacy.rename(self.artifact_dir)
        for p in (self.state_dir, self.blob_dir, self.log_dir,
                  self.sandbox_dir, self.artifact_dir):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("state_dir", "runtime_dir", "models_dir", "source_path"):
            d[k] = str(d[k]) if d[k] is not None else None
        return d


def _apply(obj: Any, data: dict[str, Any], where: str) -> None:
    valid = set(obj.__slots__) if hasattr(obj, "__slots__") else set(vars(obj))
    for key, value in data.items():
        if key not in valid:
            raise ValueError(f"unknown config key [{where}].{key}")
        setattr(obj, key, value)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg = Config()
    if path is None:
        env = os.environ.get("AMOEBA_CONFIG")
        path = env if env else None
    if path is None:
        cfg.ensure_dirs()
        return cfg
    p = Path(path).resolve()
    with p.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg.source_path = p
    top = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    for key, value in top.items():
        if key in ("state_dir", "runtime_dir", "models_dir"):
            setattr(cfg, key, Path(value))
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"unknown config key {key}")
    if "backend" in raw:
        _apply(cfg.backend, raw["backend"], "backend")
    if "arbiter" in raw:
        _apply(cfg.arbiter, raw["arbiter"], "arbiter")
    if "ego" in raw:
        _apply(cfg.ego, raw["ego"], "ego")
    if "id" in raw:
        _apply(cfg.id, raw["id"], "id")
    if "sandbox" in raw:
        _apply(cfg.sandbox, raw["sandbox"], "sandbox")
    if "homeostasis" in raw:
        _apply(cfg.homeostasis, raw["homeostasis"], "homeostasis")
    if "filespace" in raw:
        fs = dict(raw["filespace"])
        roots = fs.pop("roots", [])
        _apply(cfg.filespace, fs, "filespace")
        cfg.filespace.roots = [_root(r, i) for i, r in enumerate(roots)]
    cfg.ensure_dirs()
    return cfg


def _root(raw: dict[str, Any], index: int) -> FilespaceRoot:
    unknown = set(raw) - set(FilespaceRoot.__slots__)
    if unknown:
        raise ValueError(f"unknown filespace.roots[{index}] key(s): {sorted(unknown)}")
    root = FilespaceRoot(**raw)
    if not root.name or not root.path:
        raise ValueError(f"filespace.roots[{index}] needs both name and path")
    if root.mode not in ("read_only", "read_write"):
        raise ValueError(
            f"filespace.roots[{index}].mode must be read_only or read_write, "
            f"got {root.mode!r}")
    return root
