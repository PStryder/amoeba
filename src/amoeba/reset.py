"""Start a new organism, without pretending the old one never existed.

A reset is not a cleanup. Everything in the state directory is the record of
a mind that ran: what it concluded, what it was told, what it did and when.
So the default is to move it aside, intact, under a timestamped name -- a new
organism starts, and the old one remains readable. `--delete` is available
and says what it is.

Three things this refuses to do:

* Reset a running organism. Deleting a database under a live writer produces
  a half-state nobody can reason about afterwards, so the supervisor's own
  lock decides, using the same liveness rule the supervisor uses.
* Touch anything outside the state directory. The configured file roots are
  the operator's own directories; an organism that tidied those would be a
  different kind of program.
* Throw away the plumbing by default. Credentials and API keys are how
  clients reach this machine, not part of the mind: a reset is about the
  organism, and reissuing keys is a separate decision (`--rotate-credentials`).
"""

from __future__ import annotations

import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# What a client or an operator authenticates with. Preserved unless asked
# otherwise: the mind is new, the doors it is reached through need not be.
CREDENTIALS = ("control.token", "operator.session", "api_clients.json")
CREDENTIAL_PREFIXES = ("scope.",)


def is_credential(name: str) -> bool:
    return name in CREDENTIALS or name.startswith(CREDENTIAL_PREFIXES)


# The lock is how ownership is decided, so it is never something a reset
# moves: a reset that archived the lock it was holding would be archiving
# its own claim on the directory.
LOCK_NAME = "supervisor.lock"


def _lock(cfg: Any) -> Any:
    from .supervisor import SingleInstanceLock

    return SingleInstanceLock(Path(cfg.state_dir) / LOCK_NAME)


def holder(cfg: Any) -> dict[str, Any] | None:
    """The live supervisor owning this state directory, if there is one.

    The supervisor's own lock and its own staleness rule, rather than a
    second opinion that could disagree with it. This answers "is it worth
    starting?" and nothing more -- a check is not ownership, and `perform`
    takes the lock rather than trusting this.
    """
    lock = _lock(cfg)
    if not lock.path.exists() or lock._is_stale():
        return None
    return lock._holder() or {"pid": None}


@contextmanager
def owned(cfg: Any) -> Any:
    """Hold the state directory for the whole reset, or do nothing at all.

    Checking for a supervisor and then moving the database is two steps with
    a gap in it, and a supervisor that starts inside that gap has its
    database archived out from under it. The lock a supervisor would have to
    take is taken here instead, for the duration: whoever holds it owns the
    directory, and the loser is refused rather than raced.
    """
    lock = _lock(cfg)
    lock.acquire()                      # raises if a live supervisor owns it
    try:
        yield lock
    finally:
        lock.release()


def plan(cfg: Any, *, rotate_credentials: bool = False) -> dict[str, Any]:
    """What a reset would move, and what it would leave."""
    state = Path(cfg.state_dir)
    archive = state.parent / f"{state.name}.reset-{time.strftime('%Y%m%d-%H%M%S')}"
    move, keep = [], []
    for entry in sorted(state.iterdir()) if state.exists() else []:
        if entry.name == LOCK_NAME:
            continue                    # ours for the duration; see `owned`
        if is_credential(entry.name) and not rotate_credentials:
            keep.append(entry)
        else:
            move.append(entry)
    return {"state_dir": state, "archive": archive, "move": move, "keep": keep,
            "holds_state": any(_holds_state(e) for e in move)}


def _holds_state(entry: Path) -> bool:
    """Is there an organism here, or only the scaffolding a config makes?

    Loading a configuration creates the empty directories a run needs, so a
    fresh machine has `blobs/`, `logs/` and friends before anything has ever
    happened. Archiving four empty folders and announcing a reset would be
    theatre.
    """
    if entry.is_dir():
        return any(entry.rglob("*"))
    return True


def perform(chosen: dict[str, Any], *, delete: bool = False,
            dry_run: bool = False) -> dict[str, Any]:
    """Move (or remove) what the plan chose. Nothing else is touched."""
    moved: list[str] = []
    if not dry_run and not delete and chosen["move"]:
        chosen["archive"].mkdir(parents=True, exist_ok=True)
    for entry in chosen["move"]:
        if dry_run:
            moved.append(entry.name)
            continue
        if delete:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        else:
            shutil.move(str(entry), str(chosen["archive"] / entry.name))
        moved.append(entry.name)
    return {"moved": moved,
            "kept": [e.name for e in chosen["keep"]],
            "archive": None if (delete or dry_run) else str(chosen["archive"]),
            "deleted": bool(delete) and not dry_run,
            "dry_run": bool(dry_run)}
