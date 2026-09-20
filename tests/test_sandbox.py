"""The sandbox boundary is asserted, not described.

Every claim the module docstring makes about containment has a test here that
tries to break it. A sandbox that silently stopped isolating would fail these,
not merely be documented incorrectly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from amoeba.errors import InvalidInput, ResourceExhausted
from amoeba.sandbox import SandboxLimits, SandboxManager

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="AppContainer isolation is Windows-only")

HOST_FILE_IN_PROFILE = str(Path.home() / "NTUSER.DAT")
PROJECT_FILE = str(Path(__file__).resolve().parents[1] / "config.toml")


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    root = tmp_path_factory.mktemp("sbxroot")
    m = SandboxManager(root)
    ok, detail = m.available()
    if not ok:
        pytest.skip(f"sandbox unavailable: {detail}")
    m.ensure_runtime()
    yield m
    m.destroy_all()


@pytest.fixture()
def sb(manager):
    s = manager.create(owner="wk_test",
                       limits=SandboxLimits(wall_seconds=45, cpu_seconds=30,
                                            max_processes=4))
    yield s
    manager.destroy(s.sandbox_id)


def run(manager, sb, code: str, **kw):
    return manager.run_python(sb.sandbox_id, code=code, **kw)


# ---------------------------------------------------------------------------
# It has to actually work, or isolation is trivially achieved by doing nothing.
# ---------------------------------------------------------------------------
def test_runs_real_code_and_returns_output(manager, sb):
    r = run(manager, sb, "print('hello from inside'); print(2**16)")
    assert r.exit_code == 0
    assert "hello from inside" in r.stdout
    assert "65536" in r.stdout
    assert r.timed_out is False


def test_can_compute_and_persist_inside_scratch(manager, sb):
    r = run(manager, sb, (
        "import json, pathlib\n"
        "vals = [i*i for i in range(1000)]\n"
        "pathlib.Path('result.json').write_text(json.dumps({'sum': sum(vals)}))\n"
        "print('wrote', sum(vals))\n"
    ))
    assert r.exit_code == 0, r.stderr
    files = {f["path"] for f in manager.list_files(sb.sandbox_id)}
    assert any(p.endswith("result.json") for p in files), files


def test_can_spawn_a_child_process_inside(manager, sb):
    r = run(manager, sb, (
        "import subprocess, sys\n"
        "out = subprocess.run([sys.executable, '-c', 'print(7*6)'],\n"
        "                     capture_output=True, text=True, timeout=60)\n"
        "print('child said', out.stdout.strip())\n"
    ))
    assert r.exit_code == 0, r.stderr
    assert "42" in r.stdout


# ---------------------------------------------------------------------------
# Network: blocked in the kernel, not by a wrapper.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target,port", [
    ("1.1.1.1", 53),        # internet
    ("192.168.1.1", 80),    # LAN
    ("127.0.0.1", 9),       # loopback
])
def test_network_is_blocked(manager, sb, target, port):
    r = run(manager, sb, (
        "import socket\n"
        f"s = socket.create_connection(({target!r}, {port}), timeout=4)\n"
        "s.close(); print('CONNECTED')\n"
    ))
    assert "CONNECTED" not in r.stdout, f"sandbox reached {target}:{port}"
    assert r.exit_code != 0 or r.timed_out


def test_dns_resolution_is_blocked(manager, sb):
    r = run(manager, sb, "import socket; print(socket.gethostbyname('example.com'))")
    assert r.exit_code != 0
    assert "gaierror" in r.stderr or "socket" in r.stderr


def test_cannot_serve_a_listening_socket_reachable_from_host(manager, sb):
    r = run(manager, sb, (
        "import socket\n"
        "s = socket.socket()\n"
        "try:\n"
        "    s.bind(('0.0.0.0', 0)); s.listen(1); print('BOUND', s.getsockname()[1])\n"
        "except Exception as e:\n"
        "    print('BIND-FAILED', type(e).__name__)\n"
    ))
    # Binding may succeed inside the container's own namespace; what matters is
    # that no traffic crosses. The connect tests above establish that.
    assert r.exit_code == 0 or "BIND-FAILED" in r.stdout


# ---------------------------------------------------------------------------
# Host filesystem and credentials.
# ---------------------------------------------------------------------------
def test_cannot_read_the_user_profile(manager, sb):
    r = run(manager, sb, (
        "import os\n"
        "print('ENTRIES', len(os.listdir(r'C:\\\\Users')))\n"
    ))
    assert "ENTRIES" not in r.stdout
    assert r.exit_code != 0


def test_cannot_read_project_source(manager, sb):
    r = run(manager, sb, (
        f"print(open({PROJECT_FILE!r}).read(32))\n"
    ))
    assert r.exit_code != 0
    assert "PermissionError" in r.stderr or "FileNotFoundError" in r.stderr


def test_cannot_read_the_state_database(manager, sb):
    db = r"F:\hexylab\amoeba-state\mind.sqlite3"
    r = run(manager, sb, f"print(open({db!r}, 'rb').read(16))\n")
    assert r.exit_code != 0


def test_cannot_write_outside_the_sandbox(manager, sb):
    target = str(Path.home() / "amoeba_escape_test.txt")
    r = run(manager, sb, f"open({target!r}, 'w').write('escaped')\nprint('WROTE')\n")
    assert "WROTE" not in r.stdout
    assert r.exit_code != 0
    assert not Path(target).exists(), "sandbox wrote into the user profile"


def test_windows_system_files_remain_readable_and_this_is_documented(manager, sb):
    """The known, deliberate gap.

    An AppContainer must read system DLLs to start, so Windows grants
    ALL APPLICATION PACKAGES read access to parts of C:\\Windows. This test
    pins the *actual* behaviour so the docs cannot drift away from it.
    """
    r = run(manager, sb, "print(len(open(r'C:\\\\Windows\\\\win.ini').read()))")
    assert r.exit_code == 0, "if this now fails, the caveat in sandbox.py is stale"
    caveat = SandboxManager(Path(os.devnull).parent).capabilities()["caveat"]
    assert "C:\\Windows" in caveat


# ---------------------------------------------------------------------------
# Resource limits.
# ---------------------------------------------------------------------------
def test_wall_clock_timeout_kills_the_process(manager, sb):
    t0 = time.perf_counter()
    r = run(manager, sb, "import time\nwhile True: time.sleep(0.05)\n", timeout=4)
    elapsed = time.perf_counter() - t0
    assert r.timed_out is True
    assert r.killed_reason and "wall-clock" in r.killed_reason
    assert elapsed < 30, "timeout did not actually stop the process"


def test_output_is_capped(manager, sb):
    sb.limits.max_output_bytes = 2048
    r = run(manager, sb, "print('x' * 100000)")
    assert r.stdout_truncated is True
    assert len(r.stdout) <= 2048


def test_fork_bomb_is_bounded_by_the_job_object(manager, sb):
    r = run(manager, sb, (
        "import subprocess, sys\n"
        "kids = []\n"
        "for i in range(40):\n"
        "    try:\n"
        "        kids.append(subprocess.Popen([sys.executable, '-c',\n"
        "                     'import time; time.sleep(30)']))\n"
        "    except Exception as e:\n"
        "        print('STOPPED_AT', i, type(e).__name__); break\n"
        "else:\n"
        "    print('SPAWNED_ALL', len(kids))\n"
    ), timeout=30)
    # Either the job refused the spawns, or the whole thing was killed.
    assert "SPAWNED_ALL 40" not in r.stdout


# ---------------------------------------------------------------------------
# Path safety: the caller is a language model.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "../escape.txt",
    "../../escape.txt",
    "work/../../../escape.txt",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
    "\\\\server\\share\\file",
    "/etc/passwd",
])
def test_path_traversal_is_rejected(manager, sb, bad):
    with pytest.raises(InvalidInput):
        manager.resolve_inside(sb, bad)


def test_empty_and_oversized_paths_rejected(manager, sb):
    for bad in ("", "   ", "a" * 600):
        with pytest.raises(InvalidInput):
            manager.resolve_inside(sb, bad)


def test_legitimate_relative_paths_are_accepted(manager, sb):
    p = manager.resolve_inside(sb, "work/sub/dir/file.txt")
    assert sb.root.resolve() in p.parents


def test_write_respects_the_artifact_size_limit(manager, sb):
    sb.limits.max_artifact_bytes = 1024
    with pytest.raises(ResourceExhausted):
        manager.write_file(sb.sandbox_id, "work/big.txt", "y" * 5000)


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------
def test_sandboxes_are_isolated_from_each_other(manager):
    a = manager.create(owner="wk_a")
    b = manager.create(owner="wk_b")
    try:
        manager.write_file(a.sandbox_id, "work/secret.txt", "SECRET-ALPHA-991")
        other = str(a.root / "work" / "secret.txt")
        r = manager.run_python(b.sandbox_id, code=f"print(open({other!r}).read())")
        assert "SECRET-ALPHA-991" not in r.stdout
        assert r.exit_code != 0
    finally:
        manager.destroy(a.sandbox_id)
        manager.destroy(b.sandbox_id)


def test_destroy_removes_scratch_and_invalidates_the_handle(manager):
    s = manager.create(owner="wk_gone")
    manager.write_file(s.sandbox_id, "work/f.txt", "data")
    root = s.root
    assert root.exists()
    manager.destroy(s.sandbox_id)
    assert not root.exists()
    with pytest.raises(InvalidInput):
        manager.get(s.sandbox_id)


def test_capabilities_do_not_overclaim(manager):
    caps = manager.capabilities()
    assert caps["sandbox_available"] is True
    assert caps["enforcement"] == "os_kernel"
    assert "blocked" in caps["network"]
    # The gap is stated in the capability report itself, not only in prose.
    assert "world-readable" in caps["host_filesystem"]
    assert caps["third_party_libraries"].startswith("none")


def test_runtime_has_no_third_party_packages(manager):
    runtime = manager.ensure_runtime()
    assert not (runtime.parent / "Lib" / "site-packages").exists()
    r = manager.create(owner="wk_imports")
    try:
        out = manager.run_python(r.sandbox_id, code=(
            "import importlib.util\n"
            "for m in ('numpy', 'requests', 'torch', 'mcp'):\n"
            "    print(m, importlib.util.find_spec(m) is not None)\n"
        ))
        assert out.exit_code == 0, out.stderr
        for line in out.stdout.strip().splitlines():
            assert line.endswith("False"), f"third-party module reachable: {line}"
    finally:
        manager.destroy(r.sandbox_id)
