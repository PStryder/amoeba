"""Sandboxed compute workspace: scratch compute with no hands on the host.

A neuocyte can ask the Harness for an isolated workspace, run real code in it,
and propose the artifacts it produces for promotion into durable state. It
cannot reach the host filesystem, the project source, the state database, the
LAN, the internet, or any credential.

**The boundary is a Windows AppContainer**, which is an OS-enforced security
boundary, not a policy check inside Python. Each sandbox gets:

* its own AppContainer profile with **zero capabilities** -- notably without
  ``internetClient``, which is what makes network access fail in the kernel
  rather than in a wrapper the sandboxed code could bypass;
* a scratch directory ACL'd to that container SID and nothing else;
* a stdlib-only Python runtime (no ``site-packages``), so no third-party
  library is reachable by construction;
* a Job Object capping process count, committed memory and CPU time, with
  ``KILL_ON_JOB_CLOSE`` so nothing outlives the sandbox.

Measured on this machine (see ``tests/test_sandbox.py``, which asserts all of
it rather than trusting the description):

| attempt | result |
|---|---|
| connect to loopback / LAN / internet, resolve DNS | blocked |
| read the user profile, the project source, the state DB | blocked |
| write anywhere outside the scratch directory | blocked |
| read world-readable ``C:\\Windows`` system files | **allowed** |
| write inside scratch, spawn a child process, compute | allowed |

That last "allowed" is the honest caveat. An AppContainer must be able to read
system DLLs to start at all, so Windows grants ``ALL APPLICATION PACKAGES``
read access to parts of ``C:\\Windows``. Nothing user-specific or secret is
reachable through it, but "no host filesystem access whatsoever" would be a
false claim and is not made here.

Promotion out of a sandbox is never automatic. A neuocyte *proposes*; the
Harness copies, hashes and receipts. Model-generated strings never become host
paths: every path is resolved inside the scratch root and rejected if it
escapes.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as w
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .errors import CapabilityUnsupported, InvalidInput, ResourceExhausted
from .ids import new_id, sha256_hex
from .logging_setup import get_logger
from .security import audit_path, harden, revoke

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)

EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_SUSPENDED = 0x00000004
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
# Without this, bInheritHandles=True means "inherit every inheritable handle in
# the process", which across concurrent sandboxes means each container
# inheriting the others' open output files. An already-open handle carries the
# access it was granted, and Windows checks the DACL at open time rather than
# at use time, so no amount of ACL hardening closes that. The handle list makes
# inheritance explicit: exactly these handles, nothing else.
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
ERROR_ALREADY_EXISTS_HR = 0x800700B7

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400


# ---------------------------------------------------------------------------
# Win32 structures
# ---------------------------------------------------------------------------
class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
                ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
                ("dwXSize", w.DWORD), ("dwYSize", w.DWORD), ("dwXCountChars", w.DWORD),
                ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD),
                ("dwFlags", w.DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE),
                ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [("AppContainerSid", ctypes.c_void_p), ("Capabilities", ctypes.c_void_p),
                ("CapabilityCount", w.DWORD), ("Reserved", w.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", w.DWORD), ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SandboxLimits:
    wall_seconds: float = 60.0
    cpu_seconds: float = 60.0
    memory_bytes: int = 1024 * 1024 * 1024      # 1 GiB
    max_processes: int = 8
    max_output_bytes: int = 256 * 1024
    max_scratch_bytes: int = 256 * 1024 * 1024  # 256 MiB
    max_artifact_bytes: int = 16 * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class Sandbox:
    sandbox_id: str
    owner: str
    root: Path
    container_name: str
    container_sid: str
    limits: SandboxLimits
    created_at: float
    runs: int = 0
    destroyed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id, "owner": self.owner, "root": str(self.root),
            "container_sid": self.container_sid, "limits": self.limits.to_dict(),
            "created_at": self.created_at, "runs": self.runs, "destroyed": self.destroyed,
        }


@dataclass(slots=True)
class RunResult:
    sandbox_id: str
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool
    killed_reason: str | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    peak_job_memory_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class SandboxUnavailable(CapabilityUnsupported):
    code = "sandbox_unavailable"


# ---------------------------------------------------------------------------
class SandboxManager:
    """Owned by the Harness. Neuocytes never construct one themselves."""

    def __init__(self, root: Path, *, runtime_dir: Path | None = None) -> None:
        self.root = Path(root)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.root / "runtime" / "python"
        self.log = get_logger("sandbox")
        self._lock = threading.RLock()
        self._sandboxes: dict[str, Sandbox] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        # The sandbox root inherits its parent's DACL, which on this machine
        # granted Everyone full control. Harden before anything is written.
        harden(self.root, log_name="sandbox")

    # -- capability -----------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if not IS_WINDOWS:
            return False, "AppContainer isolation is Windows-only"
        if not hasattr(userenv, "CreateAppContainerProfile"):
            return False, "userenv!CreateAppContainerProfile unavailable"
        return True, "ok"

    def capabilities(self) -> dict[str, Any]:
        ok, detail = self.available()
        return {
            "sandbox_available": ok,
            "detail": detail,
            "isolation": "windows_appcontainer" if ok else "none",
            "enforcement": "os_kernel" if ok else "none",
            "network": "blocked (no capabilities granted, incl. internetClient)",
            "host_filesystem": "blocked except world-readable Windows system files",
            "credentials": "unreachable (no user-profile access)",
            "third_party_libraries": "none (stdlib-only runtime, no site-packages)",
            "promotion": "proposal only; the Harness copies, re-hashes and "
                         "refuses content that changed since it was proposed",
            "scratch_isolation": ("inheritance removed; DACL grants only this "
                                  "container, the Amoeba account, SYSTEM and "
                                  "Administrators"),
            "residual_risk": ("a process running as the Amoeba account owns "
                              "these directories and can rewrite the DACL; a "
                              "dedicated service account is the fix"),
            "caveat": ("an AppContainer must read system DLLs to start, so parts of "
                       "C:\\Windows remain readable; nothing user-specific is"),
        }

    # -- runtime --------------------------------------------------------
    def ensure_runtime(self) -> Path:
        """A stdlib-only interpreter in a tree the Harness owns and can ACL.

        The system Python install usually cannot be ACL'd without admin, and
        shipping ``site-packages`` into the sandbox would hand the sandboxed
        code every installed library. Copying just the stdlib solves both.
        """
        exe = self.runtime_dir / "python.exe"
        if exe.exists():
            return exe
        base = Path(sys.base_prefix)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        for f in ("python.exe", "pythonw.exe", "vcruntime140.dll", "vcruntime140_1.dll"):
            if (base / f).exists():
                shutil.copy2(base / f, self.runtime_dir / f)
        for dll in base.glob("python*.dll"):
            shutil.copy2(dll, self.runtime_dir / dll.name)
        shutil.copytree(base / "DLLs", self.runtime_dir / "DLLs", dirs_exist_ok=True)
        shutil.copytree(
            base / "Lib", self.runtime_dir / "Lib", dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("site-packages", "__pycache__", "test", "idlelib",
                                          "tkinter", "turtledemo", "ensurepip"),
        )
        self.log.info("built sandbox runtime at %s", self.runtime_dir)
        return exe

    # -- lifecycle ------------------------------------------------------
    def create(self, *, owner: str, limits: SandboxLimits | None = None) -> Sandbox:
        ok, detail = self.available()
        if not ok:
            raise SandboxUnavailable(detail)
        limits = limits or SandboxLimits()
        sandbox_id = new_id("sbx")
        container_name = f"Amoeba.{sandbox_id}"[:64]
        scratch = self.root / sandbox_id
        scratch.mkdir(parents=True, exist_ok=False)
        (scratch / "work").mkdir()
        (scratch / "out").mkdir()

        sid_str = self._create_container(container_name)
        runtime = self.ensure_runtime()
        # Runtime: read+execute for the container, and nothing for anyone
        # outside Amoeba. A writable runtime is a code-injection path straight
        # into the container -- the AppContainer bounds what the code can
        # reach, not which code runs.
        harden(runtime.parent, container_sid=sid_str,
               container_rights="(OI)(CI)(RX)", log_name="sandbox")
        # Scratch: full control for its own container only.
        harden(scratch, container_sid=sid_str, container_rights="(OI)(CI)(F)",
               log_name="sandbox")

        sb = Sandbox(sandbox_id=sandbox_id, owner=owner, root=scratch,
                     container_name=container_name, container_sid=sid_str,
                     limits=limits, created_at=time.time())
        with self._lock:
            self._sandboxes[sandbox_id] = sb
        self.log.info("sandbox %s created for %s at %s", sandbox_id, owner, scratch)
        return sb

    def _create_container(self, name: str) -> str:
        sid = ctypes.c_void_p()
        userenv.CreateAppContainerProfile.restype = ctypes.c_long
        hr = userenv.CreateAppContainerProfile(
            ctypes.c_wchar_p(name), ctypes.c_wchar_p(name),
            ctypes.c_wchar_p("Amoeba neuocyte scratch compute"), None, 0, ctypes.byref(sid))
        if (hr & 0xFFFFFFFF) == ERROR_ALREADY_EXISTS_HR:
            userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
            userenv.DeriveAppContainerSidFromAppContainerName(
                ctypes.c_wchar_p(name), ctypes.byref(sid))
        if not sid.value:
            raise SandboxUnavailable("could not create an AppContainer profile",
                                     hresult=hex(hr & 0xFFFFFFFF))
        s = ctypes.c_wchar_p()
        advapi.ConvertSidToStringSidW(sid, ctypes.byref(s))
        return s.value

    def get(self, sandbox_id: str) -> Sandbox:
        sb = self._sandboxes.get(sandbox_id)
        if sb is None or sb.destroyed:
            raise InvalidInput("unknown or destroyed sandbox", sandbox_id=sandbox_id)
        return sb

    def list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sandboxes.values() if not s.destroyed]

    def destroy(self, sandbox_id: str, *, keep_scratch: bool = False) -> dict[str, Any]:
        with self._lock:
            sb = self._sandboxes.get(sandbox_id)
            if sb is None:
                return {"sandbox_id": sandbox_id, "already_gone": True}
            # No job handle to close here: a Job Object is created per run
            # and closed in _spawn's finally, where KILL_ON_JOB_CLOSE takes any
            # surviving process with it. Nothing outlives a run.
            try:
                userenv.DeleteAppContainerProfile(ctypes.c_wchar_p(sb.container_name))
            except Exception:  # noqa: BLE001
                self.log.debug("profile delete failed for %s", sb.container_name)
            # The runtime is shared, so its grant to this container is not
            # removed by deleting the scratch. Left in place it would accumulate
            # one dead ACE per sandbox and keep a retired container's SID
            # able to read and execute the interpreter.
            revoke(self.runtime_dir, sb.container_sid, log_name="sandbox")
            if not keep_scratch:
                shutil.rmtree(sb.root, ignore_errors=True)
            sb.destroyed = True
            self.log.info("sandbox %s destroyed", sandbox_id)
            return {"sandbox_id": sandbox_id, "destroyed": True,
                    "scratch_removed": not keep_scratch}

    def destroy_all(self) -> list[str]:
        return [self.destroy(s)["sandbox_id"] for s in list(self._sandboxes)]

    # -- path safety ----------------------------------------------------
    def resolve_inside(self, sb: Sandbox, relpath: str) -> Path:
        """Resolve a caller-supplied path strictly inside the scratch root.

        The caller is ultimately a language model. Absolute paths, ``..``
        traversal and symlinks that leave the root are all rejected rather than
        normalised, because a path that tries to escape is a signal, not a typo.
        """
        if not isinstance(relpath, str) or not relpath.strip():
            raise InvalidInput("path must be a non-empty string")
        if len(relpath) > 512:
            raise InvalidInput("path too long")
        candidate = Path(relpath)
        if candidate.is_absolute() or candidate.drive or relpath.startswith("\\\\"):
            raise InvalidInput("absolute paths are not permitted inside a sandbox",
                               path=relpath)
        root = sb.root.resolve()
        target = (root / candidate).resolve()
        if target != root and root not in target.parents:
            raise InvalidInput("path escapes the sandbox root", path=relpath)
        return target

    def write_file(self, sandbox_id: str, relpath: str, content: str) -> dict[str, Any]:
        """Text convenience wrapper. Binary content must use write_bytes."""
        return self.write_bytes(sandbox_id, relpath, content.encode("utf-8"))

    def write_bytes(self, sandbox_id: str, relpath: str, data: bytes
                    ) -> dict[str, Any]:
        """Put exact bytes into a sandbox.

        The byte-accurate path, and the one anything not known to be UTF-8 text
        has to use. Routing a file through ``str`` replaces every byte that is
        not valid UTF-8 with U+FFFD, which is silent, lossy and -- worse --
        leaves a receipt describing content that is not what landed.
        """
        sb = self.get(sandbox_id)
        target = self.resolve_inside(sb, relpath)
        if len(data) > sb.limits.max_artifact_bytes:
            raise ResourceExhausted("file exceeds the per-artifact limit",
                                    size=len(data), limit=sb.limits.max_artifact_bytes)
        if self.scratch_bytes(sandbox_id) + len(data) > sb.limits.max_scratch_bytes:
            raise ResourceExhausted("sandbox scratch quota exceeded",
                                    limit=sb.limits.max_scratch_bytes)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"path": relpath, "bytes": len(data), "sha256": sha256_hex(data)}

    def read_file(self, sandbox_id: str, relpath: str, *, max_bytes: int = 256 * 1024
                  ) -> dict[str, Any]:
        sb = self.get(sandbox_id)
        target = self.resolve_inside(sb, relpath)
        if not target.is_file():
            raise InvalidInput("no such file in sandbox", path=relpath)
        data = target.read_bytes()
        truncated = len(data) > max_bytes
        head = data[:max_bytes]
        text = head.decode("utf-8", "replace")
        # Say when the text is not the file. A model handed U+FFFD soup with no
        # indication would reason about it as though it were the content.
        lossy = text.encode("utf-8") != head
        return {"path": relpath, "bytes": len(data), "truncated": truncated,
                "sha256": sha256_hex(data), "lossy_decode": lossy,
                "note": ("this file is not UTF-8 text; the content below is a "
                         "lossy rendering and is not what the file contains"
                         if lossy else ""),
                "content": text}

    def list_files(self, sandbox_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """Files in the scratch tree.

        Skips anything that resolves outside the root. Sandboxed code can
        create a junction inside its own scratch, and rglob will happily walk
        through it -- which would leak names and sizes from wherever it points.
        """
        sb = self.get(sandbox_id)
        root = sb.root.resolve()
        out = []
        for p in sorted(root.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            try:
                real = p.resolve()
            except OSError:
                continue
            if real != root and root not in real.parents:
                continue        # a junction pointing out of the sandbox
            if len(out) >= limit:
                break
            out.append({"path": p.relative_to(root).as_posix(),
                        "bytes": p.stat().st_size,
                        "modified": p.stat().st_mtime})
        return out

    def scratch_bytes(self, sandbox_id: str) -> int:
        """Bytes under scratch, not following links out of it."""
        sb = self.get(sandbox_id)
        root = sb.root.resolve()
        total = 0
        for p in root.rglob("*"):
            if p.is_symlink() or not p.is_file():
                continue
            try:
                if p.resolve().parents and root in p.resolve().parents:
                    total += p.stat().st_size
            except OSError:
                continue
        return total

    def audit(self, sandbox_id: str) -> dict[str, Any]:
        """Who can reach this sandbox's scratch, according to the filesystem."""
        return audit_path(self.get(sandbox_id).root)

    # -- execution ------------------------------------------------------
    def run_python(self, sandbox_id: str, *, code: str | None = None,
                   script: str | None = None, argv: Sequence[str] = (),
                   timeout: float | None = None) -> RunResult:
        """Run Python inside the container. Either inline ``code`` or a script
        already written into the sandbox."""
        sb = self.get(sandbox_id)
        runtime = self.ensure_runtime()
        if code is not None and script is not None:
            raise InvalidInput("pass code or script, not both")
        if code is not None:
            name = f"work/_run_{sb.runs:04d}.py"
            self.write_file(sandbox_id, name, code)
            script = name
        if script is None:
            raise InvalidInput("nothing to run")
        target = self.resolve_inside(sb, script)
        if not target.is_file():
            raise InvalidInput("script not found in sandbox", path=script)
        # -I isolates from env vars and user site; -S skips site processing.
        cmd = [str(runtime), "-I", "-S", str(target), *[str(a) for a in argv]]
        return self._spawn(sb, cmd, timeout=timeout)

    def _spawn(self, sb: Sandbox, argv: list[str], *, timeout: float | None) -> RunResult:
        limits = sb.limits
        wall = float(timeout if timeout is not None else limits.wall_seconds)
        wall = min(wall, limits.wall_seconds)

        out_path = sb.root / f"_stdout_{sb.runs}.txt"
        err_path = sb.root / f"_stderr_{sb.runs}.txt"
        sb.runs += 1

        sid = ctypes.c_void_p()
        userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
        userenv.DeriveAppContainerSidFromAppContainerName(
            ctypes.c_wchar_p(sb.container_name), ctypes.byref(sid))
        if not sid.value:
            raise SandboxUnavailable("container SID vanished", sandbox_id=sb.sandbox_id)

        caps = SECURITY_CAPABILITIES()
        caps.AppContainerSid = sid
        caps.Capabilities = None
        caps.CapabilityCount = 0
        caps.Reserved = 0

        job = self._make_job(sb)

        fout = open(out_path, "wb")
        ferr = open(err_path, "wb")
        attrs = None
        try:
            # The attribute list is built *after* the output files exist,
            # because the handle list has to name them.
            h_out = msvcrt_handle(fout)
            h_err = msvcrt_handle(ferr)
            inheritable = (ctypes.c_void_p * 2)(h_out, h_err)

            size = ctypes.c_size_t(0)
            k32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
            attr_buf = (ctypes.c_byte * size.value)()
            attrs = ctypes.cast(attr_buf, ctypes.c_void_p)
            if not k32.InitializeProcThreadAttributeList(attrs, 2, 0, ctypes.byref(size)):
                raise SandboxUnavailable("InitializeProcThreadAttributeList failed",
                                         err=ctypes.get_last_error())
            if not k32.UpdateProcThreadAttribute(
                    attrs, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES),
                    ctypes.byref(caps), ctypes.sizeof(caps), None, None):
                raise SandboxUnavailable("UpdateProcThreadAttribute failed",
                                         err=ctypes.get_last_error())
            # Exactly these two handles are inherited. Without it, a container
            # spawned while another is running inherits that one's writable
            # output handles and can write through them -- measured, not
            # theorised: a sandbox that was told nothing found them by sweeping
            # the handle space and injected a line into another sandbox's
            # recorded stdout.
            if not k32.UpdateProcThreadAttribute(
                    attrs, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_HANDLE_LIST),
                    ctypes.byref(inheritable), ctypes.sizeof(inheritable), None, None):
                raise SandboxUnavailable("could not restrict handle inheritance",
                                         err=ctypes.get_last_error())

            si = STARTUPINFOEXW()
            si.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            si.StartupInfo.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
            si.StartupInfo.hStdOutput = h_out
            si.StartupInfo.hStdError = h_err
            si.StartupInfo.hStdInput = None
            si.lpAttributeList = attrs
            pi = PROCESS_INFORMATION()
            cmdline = subprocess.list2cmdline(argv)
            k32.CreateProcessW.argtypes = [
                w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL, w.DWORD,
                ctypes.c_void_p, w.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p]
            ok = k32.CreateProcessW(
                None, ctypes.create_unicode_buffer(cmdline), None, None, True,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED,
                None, str(sb.root / "work"), ctypes.byref(si), ctypes.byref(pi))
            if not ok:
                raise SandboxUnavailable("CreateProcessW failed",
                                         err=ctypes.get_last_error())
            k32.AssignProcessToJobObject(job, pi.hProcess)
            k32.ResumeThread(pi.hThread)

            t0 = time.perf_counter()
            waited = k32.WaitForSingleObject(pi.hProcess, int(wall * 1000))
            elapsed = time.perf_counter() - t0
            timed_out = waited == 0x00000102  # WAIT_TIMEOUT
            killed_reason = None
            if timed_out:
                k32.TerminateProcess(pi.hProcess, 1)
                killed_reason = f"wall-clock limit {wall:.1f}s exceeded"
                k32.WaitForSingleObject(pi.hProcess, 5000)
            code = w.DWORD()
            k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
            peak = self._job_peak_memory(job)
            k32.CloseHandle(pi.hThread)
            k32.CloseHandle(pi.hProcess)
        finally:
            if attrs is not None:
                k32.DeleteProcThreadAttributeList(attrs)
            fout.close(); ferr.close()
            k32.CloseHandle(job)  # kills any survivors

        def read_capped(p: Path) -> tuple[str, bool]:
            data = p.read_bytes() if p.exists() else b""
            cap = sb.limits.max_output_bytes
            return data[:cap].decode("utf-8", "replace"), len(data) > cap

        stdout, out_trunc = read_capped(out_path)
        stderr, err_trunc = read_capped(err_path)
        out_path.unlink(missing_ok=True)
        err_path.unlink(missing_ok=True)

        return RunResult(
            sandbox_id=sb.sandbox_id, argv=list(argv), seconds=elapsed,
            exit_code=None if timed_out else int(code.value),
            stdout=stdout, stderr=stderr, timed_out=timed_out,
            killed_reason=killed_reason, stdout_truncated=out_trunc,
            stderr_truncated=err_trunc, peak_job_memory_bytes=peak,
        )

    def _make_job(self, sb: Sandbox) -> Any:
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise SandboxUnavailable("CreateJobObject failed", err=ctypes.get_last_error())
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        b = info.BasicLimitInformation
        b.LimitFlags = (JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_JOB_MEMORY
                        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                        | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
                        | JOB_OBJECT_LIMIT_JOB_TIME)
        b.ActiveProcessLimit = sb.limits.max_processes
        # 100-ns units.
        b.PerJobUserTimeLimit = int(sb.limits.cpu_seconds * 10_000_000)
        info.BasicLimitInformation = b
        info.JobMemoryLimit = sb.limits.memory_bytes
        if not k32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                ctypes.sizeof(info)):
            k32.CloseHandle(job)
            raise SandboxUnavailable("SetInformationJobObject failed",
                                     err=ctypes.get_last_error())
        return job

    @staticmethod
    def _job_peak_memory(job: Any) -> int:
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = w.DWORD()
        if k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                         ctypes.byref(info), ctypes.sizeof(info),
                                         ctypes.byref(returned)):
            return int(info.PeakJobMemoryUsed)
        return 0


def msvcrt_handle(fh: Any) -> int:
    import msvcrt

    handle = msvcrt.get_osfhandle(fh.fileno())
    # The child must be able to inherit it.
    k32.SetHandleInformation(handle, 0x00000001, 0x00000001)  # HANDLE_FLAG_INHERIT
    return handle
