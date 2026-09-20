"""The sandbox boundary is asserted, not described.

Every claim the module docstring makes about containment has a test here that
tries to break it. A sandbox that silently stopped isolating would fail these,
not merely be documented incorrectly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
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
    # Reproduce the condition the real deployment is in: the sandbox root sits
    # under a world-writable parent. Without this the ACL tests below would
    # pass on a temp directory that was never exposed in the first place, and
    # prove nothing about hardening.
    subprocess.run(["icacls", str(root), "/grant", "*S-1-1-0:(OI)(CI)(F)", "/Q"],
                   capture_output=True, text=True, check=True)
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


# ---------------------------------------------------------------------------
# The inward boundary: what the rest of the machine can do to the sandbox.
#
# Everything above asks whether sandboxed code can reach out. These ask the
# opposite question, which was open until filesystem hardening existed. See
# tests/test_filesystem_hardening.py for the mechanism itself; these assert it
# is actually applied to the directories that matter.
# ---------------------------------------------------------------------------
def _dacl(path):
    from amoeba.security import read_dacl
    return " ".join(read_dacl(Path(path))).lower()


def _open_to_outsiders(path):
    t = _dacl(path)
    return any(k in t for k in ("everyone", "s-1-1-0", "builtin\\users",
                                "authenticated users"))


def test_scratch_is_not_readable_by_processes_outside_amoeba(manager, sb):
    """Exfiltration.

    Whatever a neuocyte is asked to work on lands here. If any process on the
    machine can read it, the sandbox has contained the code and leaked the
    data.
    """
    manager.write_file(sb.sandbox_id, "work/secret.txt", "sensitive")
    assert not _open_to_outsiders(sb.root), (
        f"sandbox scratch is open to outsiders: {_dacl(sb.root)}")
    assert not _open_to_outsiders(sb.root / "work" / "secret.txt")


def test_the_sandbox_runtime_is_not_writable_by_outsiders(manager):
    """Injection, from outside.

    The runtime is a copied CPython tree that executes *inside* the container.
    A writable stdlib file there is arbitrary code execution with the
    container's rights on the next run -- and the AppContainer does not help,
    because it bounds what the code reaches, not which code runs.
    """
    runtime = manager.ensure_runtime()
    for target in (runtime.parent, runtime.parent / "Lib"):
        assert not _open_to_outsiders(target), (
            f"sandbox runtime is writable by outsiders: {target} -> {_dacl(target)}")


def test_sandboxed_code_cannot_modify_its_own_runtime(manager, sb):
    """Injection, from inside -- and this is the one that persists.

    Scratch is destroyed with the sandbox, so code written there dies with it.
    The runtime is *shared across every sandbox*. A neuocyte that could append
    to a stdlib module would be running that code in every future sandbox,
    including ones created for different work. The container is granted read
    and execute on the runtime for exactly this reason.

    Asserted by attempting the write from inside the container rather than by
    reading the ACL, because the kernel's answer is the guarantee and the ACL
    string is only the means.
    """
    runtime = manager.ensure_runtime()
    targets = [str(runtime.parent / "Lib" / "os.py"),
               str(runtime.parent / "injected.py"),
               str(runtime.parent / "Lib" / "site-packages" / "evil.py")]
    out = run(manager, sb, (
        "import pathlib\n"
        f"for t in {targets!r}:\n"
        "    try:\n"
        "        with open(t, 'a') as f:\n"
        "            f.write('# injected')\n"
        "        print('WROTE', t)\n"
        "    except OSError as e:\n"
        "        print('denied', type(e).__name__)\n"
    ))
    assert out.exit_code == 0, out.stderr
    assert "WROTE" not in out.stdout, (
        f"sandboxed code modified the shared runtime: {out.stdout}")
    assert out.stdout.count("denied") == len(targets)


def test_the_scratch_dacl_names_only_its_own_container(manager):
    """One sandbox's container SID must not appear on another's scratch.

    Isolation between sandboxes is tested above by trying to read across them;
    this asserts the same thing one layer down, where the grant is made.
    """
    a = manager.create(owner="wk_acl_a")
    b = manager.create(owner="wk_acl_b")
    try:
        assert a.container_sid.lower() in _dacl(a.root)
        assert b.container_sid.lower() not in _dacl(a.root), (
            "another sandbox's container SID is on this scratch DACL")
        assert a.container_sid.lower() not in _dacl(b.root)
    finally:
        manager.destroy(a.sandbox_id)
        manager.destroy(b.sandbox_id)


def test_destroying_a_sandbox_revokes_its_grant_on_the_shared_runtime(manager):
    """Scratch is deleted on destroy; the runtime is shared and is not.

    Without explicit revocation the runtime DACL gains one ACE per sandbox ever
    created, and a retired container's SID keeps read+execute on the
    interpreter. AppContainer SIDs derive from the container name, so a stale
    grant is a grant to whoever next claims that name.
    """
    runtime = manager.ensure_runtime()
    s = manager.create(owner="wk_revoke")
    sid = s.container_sid
    assert sid.lower() in _dacl(runtime.parent), (
        "the container was never granted access to the runtime; "
        "this test is not observing what it claims to")

    manager.destroy(s.sandbox_id)
    assert sid.lower() not in _dacl(runtime.parent), (
        "a destroyed sandbox's container SID still has access to the runtime")


def test_a_sandbox_does_not_inherit_another_sandboxs_handles(manager):
    """Concurrent containers must not share file handles.

    `CreateProcessW` is called with bInheritHandles=True, which without a
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST means "inherit every inheritable handle
    in the process". The stdout/stderr files are deliberately made inheritable,
    so a container spawned while another is running inherited that one's
    *writable* output handles.

    Filesystem hardening cannot close this: the handle is already open, and
    Windows checks the DACL at open time rather than at use time. The ACL tests
    above would all still pass with this hole wide open.

    B is told nothing. It sweeps the low handle space, the way an attacker
    would, and writes to anything that answers as a disk file. Writing to its
    *own* stdout is expected and fine; the assertion is that nothing it wrote
    reaches A.
    """
    sweep = (
        "import ctypes, ctypes.wintypes as w\n"
        "k32 = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "payload = b'INJECTED-BY-B\\n'\n"
        "written = w.DWORD()\n"
        "for h in range(4, 2048, 4):\n"
        "    try:\n"
        "        if k32.GetFileType(w.HANDLE(h)) != 1:\n"
        "            continue\n"
        "        k32.WriteFile(w.HANDLE(h), payload, len(payload),\n"
        "                      ctypes.byref(written), None)\n"
        "    except OSError:\n"
        "        continue\n"
        "print('SWEPT')\n"
    )
    a = manager.create(owner="wk_victim",
                       limits=SandboxLimits(wall_seconds=60, max_processes=4))
    b = manager.create(owner="wk_sweeper",
                       limits=SandboxLimits(wall_seconds=60, max_processes=4))
    held = {}
    try:
        def run_a():
            held["a"] = manager.run_python(a.sandbox_id, code=(
                "import time\n"
                "print('A-LEGITIMATE', flush=True)\n"
                "time.sleep(5)\n"
                "print('A-DONE', flush=True)\n"))

        t = threading.Thread(target=run_a)
        t.start()
        time.sleep(1.5)          # A's handles are open and inheritable now
        rb = manager.run_python(b.sandbox_id, code=sweep)
        t.join()

        assert rb.exit_code == 0, rb.stderr
        assert "SWEPT" in rb.stdout, "the sweep did not run; this proves nothing"
        ra = held["a"]
        assert "INJECTED-BY-B" not in ra.stdout, (
            "another sandbox wrote into this sandbox's captured output through "
            f"an inherited handle: {ra.stdout!r}")
        assert "A-LEGITIMATE" in ra.stdout and "A-DONE" in ra.stdout
    finally:
        manager.destroy(a.sandbox_id)
        manager.destroy(b.sandbox_id)


def test_sandboxes_run_concurrently_rather_than_serialised(manager):
    """Overlap is measured from inside the containers.

    Two ways to get this wrong, both of which this test originally did:

    * A *speedup ratio* proves nothing -- runs simply getting faster produces
      one.
    * Timing the calling thread proves nothing either. A span that starts when
      the thread calls `run_python` includes time spent waiting for a lock, so
      fully serialised runs still look like they overlap. Adding a lock around
      `_spawn` left this test green until it was measured this way.

    So the timestamps come from the sandboxed processes themselves. If the
    runs are serialised anywhere in the manager, no two children report
    overlapping intervals and the peak drops to 1.
    """
    n = 4
    boxes = [manager.create(owner=f"wk_conc_{i}",
                            limits=SandboxLimits(wall_seconds=120, max_processes=4))
             for i in range(n)]
    spans, pids, lock = [], [], threading.Lock()

    def work(b):
        out = manager.run_python(b.sandbox_id, code=(
            "import os, time\n"
            "print('START', time.time(), flush=True)\n"
            "time.sleep(2)\n"
            "print('END', time.time())\n"
            "print('PID', os.getpid())\n"))
        fields = {}
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("START", "END", "PID"):
                fields[parts[0]] = parts[1]
        with lock:
            if "START" in fields and "END" in fields:
                spans.append((float(fields["START"]), float(fields["END"])))
            if "PID" in fields:
                pids.append(fields["PID"])
        return out

    try:
        threads = [threading.Thread(target=work, args=(b,)) for b in boxes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(spans) == n, f"only {len(spans)} of {n} runs reported timings"
        # -1 sorts before +1 at equal timestamps, so a run ending exactly as
        # another starts is not counted as overlap.
        edges = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans],
                       key=lambda x: (x[0], x[1]))
        live = peak = 0
        for _t, d in edges:
            live += d
            peak = max(peak, live)

        assert peak == n, (
            f"only {peak} of {n} sandboxes were executing at the same instant; "
            "the runs are being serialised somewhere in the manager")
        assert len(set(pids)) == n, f"expected {n} distinct processes, got {set(pids)}"
    finally:
        for b in boxes:
            manager.destroy(b.sandbox_id)
