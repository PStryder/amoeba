"""Filesystem hardening for the Amoeba state tree.

The sandbox's *outward* boundary — what code inside it can reach — is an
AppContainer, and it holds (see ``sandbox.py``). This module is about the
*inward* boundary: what the rest of the machine can do to Amoeba's files.

Those are different problems and the second one was open. A directory created
under a permissive parent inherits that parent's DACL, and `F:\\hexylab` grants
``Everyone:(OI)(CI)(F)``. Measured before this module existed, every Amoeba
directory was world-writable, which meant:

* **Injection.** The sandbox runtime is a copied CPython tree. Anyone could
  overwrite a stdlib ``.py`` file or a DLL, and the next neuocyte run would
  execute it *inside* the container, with the container's rights. The
  AppContainer bounds what that code can reach; it does nothing about what code
  gets to run.
* **Exfiltration.** Scratch contents — including anything a neuocyte was asked
  to work on — were readable by any process on the machine.
* **Tampering.** The event log, the content-addressed blobs and the promoted
  artifact store were writable by any process. The hash chain detects that,
  detection after the fact is not the same as prevention.

## What hardening does

Each protected directory gets inheritance removed and an explicit DACL:

| Principal | Gets |
|---|---|
| The account Amoeba runs as | full control (it has to work) |
| ``SYSTEM`` | full control (removing it breaks the OS's own housekeeping) |
| ``Administrators`` | full control (excluding them is theatre; see below) |
| An AppContainer SID | only on its own scratch, or read+execute on the runtime |
| **Everyone / Users / Authenticated Users** | **nothing** |

## What it does not do, stated plainly

**A process running as the same account can undo this.** The Amoeba user owns
these directories, and on Windows an object's owner implicitly holds
``WRITE_DAC``: it can rewrite the ACL and restore its own access. Hardening
therefore stops *other* accounts and stops casual or accidental access from
this one — a backup agent, an indexer, a careless script, another tool. It is
not a boundary against a determined process running as you.

The real fix for that threat is a dedicated low-privilege account for Amoeba,
which is a deployment decision rather than something this code can impose.
``audit_paths`` reports the residual exposure so it stays visible instead of
being assumed away.

Administrators and SYSTEM can take ownership regardless, so listing them is
honesty about who is already trusted, not a grant of new power.
"""

from __future__ import annotations

import getpass
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .logging_setup import get_logger

IS_WINDOWS = sys.platform == "win32"

# Principals that must never appear on an Amoeba directory's DACL.
FORBIDDEN_SIDS = {
    "S-1-1-0": "Everyone",
    "S-1-5-32-545": "Users",
    "S-1-5-11": "Authenticated Users",
    "S-1-5-32-546": "Guests",
    "S-1-5-7": "Anonymous",
}
# Principals that are expected and are not a finding.
EXPECTED_SIDS = {
    "S-1-5-18": "SYSTEM",
    "S-1-5-32-544": "Administrators",
}


@dataclass(slots=True)
class HardenResult:
    path: str
    hardened: bool
    detail: str = ""
    removed_inheritance: bool = False
    granted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


def current_account() -> str:
    """The account Amoeba runs as, in a form icacls accepts."""
    domain = os.environ.get("USERDOMAIN", "")
    user = os.environ.get("USERNAME") or getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def _icacls(args: list[str], *, timeout: float = 300.0) -> subprocess.CompletedProcess:
    """Run icacls with a real argv and no shell.

    No shell means no quoting rules to get wrong and no injection surface. It
    also avoids a POSIX-style shell rewriting ``/grant`` into a path, which is
    how this silently failed the first time.
    """
    return subprocess.run(["icacls", *args], capture_output=True, text=True,
                          timeout=timeout)


def _usable(path: Path) -> bool:
    """Can the Amoeba account still read and write here?

    Hardening that locks the owner out is worse than hardening that fails, so
    this is checked immediately afterwards and used to trigger a rollback.
    """
    try:
        probe = Path(path) / ".amoeba_access_probe"
        probe.write_text("probe", encoding="utf-8")
        probe.read_text(encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def harden(
    path: Path,
    *,
    container_sid: str | None = None,
    container_rights: str = "(OI)(CI)(F)",
    log_name: str = "security",
) -> HardenResult:
    """Remove inheritance and apply a minimal explicit DACL.

    **One icacls invocation, on the directory only.** Two details, both learned
    the hard way:

    * Splitting this into ``/inheritance:r`` then ``/grant`` leaves a window
      where a failure between them strips the DACL and locks everyone out,
      including the owner. One call has no such window.
    * ``/T`` is wrong here. ``(OI)(CI)`` are *inheritance* flags and mean
      nothing on a file, so applying this grant to children gives them an ACE
      that conveys no access -- icacls reports success and the files become
      unreadable. Children re-inherit from the hardened directory on their own,
      and the old permissive ACEs disappear with the inheritance that carried
      them.

    Idempotent, and verified afterwards: if the account can no longer use the
    directory, the change is rolled back rather than left in place.
    """
    log = get_logger(log_name)
    path = Path(path)
    if not IS_WINDOWS:
        return HardenResult(str(path), False, "hardening is Windows-only")
    if not path.exists():
        return HardenResult(str(path), False, "path does not exist")

    account = current_account()
    grants = [
        f"{account}:(OI)(CI)(F)",
        "*S-1-5-18:(OI)(CI)(F)",        # SYSTEM
        "*S-1-5-32-544:(OI)(CI)(F)",    # Administrators
    ]
    if container_sid:
        grants.append(f"*{container_sid}:{container_rights}")

    cmd = [str(path), "/inheritance:r"]
    for g in grants:
        cmd += ["/grant", g]
    cmd.append("/Q")

    r = _icacls(cmd)
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = detail[-1][:160] if detail else f"icacls exit {r.returncode}"
        log.warning("could not harden %s: %s", path, msg)
        return HardenResult(str(path), False, msg)

    if not _usable(path):
        # Fail safe, not locked. Restoring inheritance is strictly better than
        # leaving a directory nobody can open.
        _icacls([str(path), "/reset", "/Q"])
        log.error("hardening %s left it unusable; inheritance restored", path)
        return HardenResult(str(path), False,
                            "hardening would have locked out the Amoeba account; "
                            "rolled back")

    log.info("hardened %s (inheritance removed, %d principals)", path, len(grants))
    return HardenResult(str(path), True, "ok", removed_inheritance=True,
                        granted=grants)


def revoke(path: Path, sid: str, *, log_name: str = "security") -> bool:
    """Drop every ACE for one SID.

    Grants on the shared runtime are per-container, so they have to be removed
    when the container goes away. Otherwise the DACL grows by one ACE per
    sandbox ever created, and a destroyed container's SID keeps read+execute on
    the interpreter tree -- an AppContainer SID is derived from its name, so a
    stale grant is a grant to whoever next claims that name.
    """
    if not IS_WINDOWS or not Path(path).exists():
        return False
    r = _icacls([str(path), "/remove", f"*{sid}", "/Q"])
    if r.returncode != 0:
        get_logger(log_name).debug("could not revoke %s on %s", sid, path)
    return r.returncode == 0


def read_dacl(path: Path) -> list[str]:
    """Raw ACE lines for a path, as icacls reports them."""
    if not IS_WINDOWS or not Path(path).exists():
        return []
    r = _icacls([str(path)])
    out = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line or line.startswith("Successfully") or line.startswith("Failed"):
            continue
        # The first line carries the path before the first ACE.
        if line.lower().startswith(str(path).lower()):
            line = line[len(str(path)):].strip()
        if line:
            out.append(line)
    return out


def audit_path(path: Path) -> dict[str, Any]:
    """Is this directory exposed to principals outside Amoeba?

    Reports rather than asserts, so the residual risk stays visible.
    """
    p = Path(path)
    if not IS_WINDOWS:
        return {"path": str(p), "platform": "non-windows", "exposed": None}
    if not p.exists():
        return {"path": str(p), "exists": False, "exposed": None}

    aces = read_dacl(p)
    joined = " ".join(aces)
    inherited = "(I)" in joined
    exposed_to: list[str] = []
    for name in ("Everyone", "BUILTIN\\Users", "NT AUTHORITY\\Authenticated Users",
                 "BUILTIN\\Guests", "ANONYMOUS LOGON"):
        if name.lower() in joined.lower():
            exposed_to.append(name)
    for sid, label in FORBIDDEN_SIDS.items():
        if sid in joined and label not in exposed_to:
            exposed_to.append(label)

    return {
        "path": str(p),
        "exists": True,
        "aces": aces,
        "has_inherited_aces": inherited,
        "exposed_to": exposed_to,
        "exposed": bool(exposed_to),
        "owner_can_restore_access": True,
        "caveat": ("the owning account can rewrite this DACL, so hardening stops "
                   "other accounts and accidental access, not a determined "
                   "process running as the Amoeba user; a dedicated service "
                   "account is the fix for that"),
    }


def audit_paths(paths: Iterable[Path]) -> dict[str, Any]:
    reports = [audit_path(p) for p in paths]
    exposed = [r for r in reports if r.get("exposed")]
    return {
        "checked": len(reports),
        "exposed_count": len(exposed),
        "exposed_paths": [r["path"] for r in exposed],
        "reports": reports,
        "all_protected": not exposed,
    }


def harden_state_tree(cfg: Any, *, log_name: str = "security") -> dict[str, Any]:
    """Harden every directory Amoeba owns.

    Called on supervisor start. The sandbox root is hardened here; individual
    scratch directories are hardened as they are created, because each needs
    its own AppContainer SID on the DACL.
    """
    log = get_logger(log_name)
    targets = [cfg.state_dir, cfg.blob_dir, cfg.log_dir, cfg.sandbox_dir,
               cfg.artifact_dir]
    results = []
    for t in targets:
        Path(t).mkdir(parents=True, exist_ok=True)
    # Every directory is hardened explicitly rather than relying on the root's
    # ACL propagating. Inheritance would cover the common case -- these are
    # normally all under `state_dir` -- but it is configuration, not a
    # guarantee: `blob_dir` and friends can be pointed anywhere, and a child
    # outside the root would silently keep its permissive inherited DACL.
    for t in targets:
        results.append(harden(t, log_name=log_name))
    audit = audit_paths(targets)
    if audit["exposed_count"]:
        log.error("state tree still exposed to %s", audit["exposed_paths"])
    return {"results": [r.to_dict() for r in results], "audit": audit}
