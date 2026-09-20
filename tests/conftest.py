from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amoeba.config import Config, load_config  # noqa: E402
from amoeba.mind import Mind  # noqa: E402
from amoeba.rpc import RpcClient, read_or_create_token, wait_for_port  # noqa: E402

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

REAL_CONFIG = ROOT / "config.toml"


_CLAIMED_PORTS: set[int] = set()


def _free_port() -> int:
    """An ephemeral port not already handed to another stack in this session.

    A port is free again the instant the probe socket closes, so two stacks
    started close together can otherwise be given the same number and a new
    supervisor can find a previous run's processes still on it.
    """
    for _ in range(100):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = int(s.getsockname()[1])
        if port not in _CLAIMED_PORTS:
            _CLAIMED_PORTS.add(port)
            return port
    raise RuntimeError("could not find an unclaimed ephemeral port")


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    c = Config()
    c.state_dir = tmp_path / "state"
    c.runtime_dir = tmp_path / "runtime"
    c.models_dir = tmp_path / "models"
    c.backend.kind = "deterministic"
    c.ensure_dirs()
    return c


@pytest.fixture()
def mind(cfg: Config) -> Iterator[Mind]:
    m = Mind(cfg)
    yield m
    m.close()


@pytest.fixture()
def backend() -> Iterator[Any]:
    from amoeba.backends.deterministic import DeterministicBackend

    b = DeterministicBackend(n_seq_max=6, n_ctx=4096)
    b.load()
    yield b
    b.close()


# ---------------------------------------------------------------------------
# A live multi-process stack on the deterministic backend.
# ---------------------------------------------------------------------------
class LiveStack:
    def __init__(self, cfg: Config, proc: subprocess.Popen[bytes]) -> None:
        self.cfg = cfg
        self.proc = proc
        self.token = read_or_create_token(cfg.token_path)
        self.client = RpcClient(cfg.supervisor_host, cfg.supervisor_port, self.token,
                                timeout=180)
        self.client.connect(retries=40, delay=0.5)

    def call(self, method: str, **params: Any) -> Any:
        return self.client.call(method, **params)

    def wait_for_children(self, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            health = self.call("health")
            children = health.get("children", {})
            if all(c.get("reported") for c in children.values()):
                return True
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        try:
            self.client.call("shutdown")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, check=False)
        self.client.close()


def _write_stack_config(tmp_path: Path, *, kind: str = "deterministic",
                        overrides: dict[str, Any] | None = None,
                        extra_toml: str = "") -> Path:
    ports = [_free_port() for _ in range(4)]
    state = (tmp_path / "state").as_posix()
    lines = [
        f'state_dir = "{state}"',
        f'runtime_dir = "{(ROOT / "runtime").as_posix()}"',
        f'models_dir = "{(ROOT / "models").as_posix()}"',
        f"supervisor_port = {ports[0]}",
        f"inference_port = {ports[1]}",
        f"ego_port = {ports[2]}",
        f"id_port = {ports[3]}",
        'log_level = "INFO"',
        "",
        "[backend]",
        f'kind = "{kind}"',
    ]
    if kind == "deterministic":
        lines += ["n_ctx = 65536", "n_seq_max = 8"]
    else:
        real = load_config(REAL_CONFIG)
        lines += [
            f'lib_path = "{Path(real.backend.lib_path).as_posix()}"',
            f'model_path = "{Path(real.backend.model_path).as_posix()}"',
            f"n_gpu_layers = {real.backend.n_gpu_layers}",
            "n_ctx = 8192", "n_seq_max = 6",
        ]
    lines += [
        "",
        "[arbiter]",
        "max_neuocytes = 2",
        "neuocyte_wall_seconds = 45.0",
        "neuocyte_token_budget = 128",
        "lease_seconds = 8.0",
        "max_maintenance_per_hour = 50",
    ]
    for key, value in (overrides or {}).items():
        lines.append(f"{key} = {value!r}")
    if extra_toml:
        lines += ["", extra_toml]
    path = tmp_path / "stack.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def start_stack(tmp_path: Path, *, kind: str = "deterministic",
                timeout: float = 180.0, extra_toml: str = "") -> LiveStack:
    cfg_path = _write_stack_config(tmp_path, kind=kind, extra_toml=extra_toml)
    cfg = load_config(cfg_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env["AMOEBA_STDERR_LOG"] = "0"
    proc = subprocess.Popen(
        [PYTHON, "-m", "amoeba.supervisor", "--config", str(cfg_path)],
        env=env, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not wait_for_port(cfg.supervisor_host, cfg.supervisor_port, timeout=timeout):
        proc.kill()
        raise RuntimeError("supervisor did not start")
    stack = LiveStack(cfg, proc)
    stack.wait_for_children(timeout=timeout)
    return stack


@pytest.fixture()
def stack(tmp_path: Path) -> Iterator[LiveStack]:
    s = start_stack(tmp_path)
    yield s
    s.stop()


# ---------------------------------------------------------------------------
def real_model_available() -> bool:
    if not REAL_CONFIG.exists():
        return False
    try:
        c = load_config(REAL_CONFIG)
    except Exception:  # noqa: BLE001
        return False
    return (Path(c.backend.lib_path).exists() and Path(c.backend.model_path).exists())


requires_gpu = pytest.mark.skipif(
    not real_model_available(),
    reason="real llama.cpp runtime and GGUF model not configured",
)
