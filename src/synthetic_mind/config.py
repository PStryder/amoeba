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
    max_workers: int = 4
    max_outstanding_work: int = 64
    max_prompt_tokens: int = 6144
    max_completion_tokens: int = 512
    worker_wall_seconds: float = 180.0
    worker_token_budget: int = 2048
    worker_max_age_seconds: float = 900.0
    lease_seconds: float = 90.0
    # Weighted-fair split between user-directed work and Id maintenance.
    user_weight: float = 0.7
    maintenance_weight: float = 0.3
    # Neither class may be starved: each is guaranteed this many worker slots.
    user_reserved_slots: int = 1
    maintenance_reserved_slots: int = 1
    max_maintenance_depth: int = 2
    max_maintenance_per_hour: int = 20


@dataclass(slots=True)
class RoleConfig:
    max_context_tokens: int = 6144
    max_turns_resident: int = 40
    system_prompt: str = ""


@dataclass(slots=True)
class Config:
    state_dir: Path = Path("F:/hexylab/synthetic-mind-state")
    runtime_dir: Path = Path("F:/hexylab/synthetic-mind-runtime")
    models_dir: Path = Path("F:/hexylab/synthetic-mind-models")
    supervisor_host: str = "127.0.0.1"
    supervisor_port: int = 8711
    inference_port: int = 8712
    ego_port: int = 8713
    id_port: int = 8714
    log_level: str = "INFO"
    backend: BackendConfig = field(default_factory=BackendConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
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
    def ready_path(self) -> Path:
        return self.state_dir / "supervisor.ready"

    @property
    def token_path(self) -> Path:
        return self.state_dir / "control.token"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "supervisor.lock"

    def ensure_dirs(self) -> None:
        for p in (self.state_dir, self.blob_dir, self.log_dir):
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
        env = os.environ.get("SYNTHETIC_MIND_CONFIG")
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
    cfg.ensure_dirs()
    return cfg
