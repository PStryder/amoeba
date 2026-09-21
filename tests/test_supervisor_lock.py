"""Exactly one supervisor per state directory, including across a restart.

Clearing a dead holder's lock races two ways, and restarting the organism
hits both windows:

  * the previous supervisor releases its own lock between our seeing it and
    our unlinking it -- the unlink finds nothing. This one actually happened:
    a restart died with `FileNotFoundError: ... supervisor.lock` and the
    organism did not come up;
  * another supervisor creates a fresh lock between our unlink and our open --
    the exclusive create finds one. That used to escape as a bare
    FileExistsError instead of being judged like any other live holder.

The tests reproduce each window deterministically by performing the other
party's move at exactly the moment it would have to happen, rather than by
hoping two processes collide.
"""

from __future__ import annotations

import json
import os

import pytest

from amoeba import supervisor as sup_mod
from amoeba.errors import ResourceExhausted
from amoeba.supervisor import SingleInstanceLock

DEAD_PID = 2 ** 22 + 12345   # far above anything Windows or Linux hands out


def _write_holder(path, pid: int) -> None:
    path.write_text(json.dumps({"pid": pid, "started": 0}), encoding="utf-8")


def _holder_pid(path) -> int:
    return json.loads(path.read_text(encoding="utf-8"))["pid"]


def test_an_unheld_lock_is_taken(tmp_path):
    lock = SingleInstanceLock(tmp_path / "supervisor.lock")
    lock.acquire()
    try:
        assert _holder_pid(lock.path) == os.getpid()
    finally:
        lock.release()
    assert not lock.path.exists()


def test_a_live_holder_is_refused(tmp_path):
    path = tmp_path / "supervisor.lock"
    _write_holder(path, os.getpid())     # this process: certainly alive
    with pytest.raises(ResourceExhausted):
        SingleInstanceLock(path).acquire()
    assert _holder_pid(path) == os.getpid(), "the live holder's lock was touched"


def test_a_dead_holders_lock_is_taken_over(tmp_path):
    path = tmp_path / "supervisor.lock"
    _write_holder(path, DEAD_PID)
    lock = SingleInstanceLock(path)
    lock.acquire()
    try:
        assert _holder_pid(path) == os.getpid()
    finally:
        lock.release()


def test_a_holder_releasing_as_we_clear_it_does_not_kill_the_start(tmp_path):
    """The restart that failed tonight.

    The old supervisor was still shutting down: we saw its lock, judged it
    stale, and by the time we unlinked it the old process had removed it
    itself. The unlink raised FileNotFoundError and the new supervisor died.
    """
    path = tmp_path / "supervisor.lock"
    _write_holder(path, DEAD_PID)

    class ReleasedMeanwhile(SingleInstanceLock):
        def _is_stale(self) -> bool:
            stale = super()._is_stale()
            path.unlink()           # the holder's own release, landing now
            return stale

    lock = ReleasedMeanwhile(path)
    lock.acquire()
    try:
        assert _holder_pid(path) == os.getpid()
    finally:
        lock.release()


def test_a_rival_taking_the_lock_as_we_clear_it_wins_cleanly(tmp_path, monkeypatch):
    """Two supervisors both find a dead holder; one of them must lose politely.

    We unlink the stale lock and, before our exclusive create, a live rival
    creates its own. The create then finds a lock that is *not* stale, and the
    right outcome is the ordinary refusal -- not a FileExistsError escaping,
    and certainly not a second owner.
    """
    path = tmp_path / "supervisor.lock"
    _write_holder(path, DEAD_PID)
    real_unlink = os.unlink

    def unlink_then_rival_arrives(p, *a, **kw):
        real_unlink(p, *a, **kw)
        _write_holder(path, os.getpid())    # a live rival, in the gap

    monkeypatch.setattr(sup_mod.os, "unlink", unlink_then_rival_arrives)
    with pytest.raises(ResourceExhausted):
        SingleInstanceLock(path).acquire()
    monkeypatch.undo()
    assert _holder_pid(path) == os.getpid(), "the rival's lock was taken from it"


def test_a_lock_that_never_settles_gives_up_rather_than_spinning(tmp_path, monkeypatch):
    """Bounded: a pathological directory is reported, not looped on forever."""
    path = tmp_path / "supervisor.lock"
    _write_holder(path, DEAD_PID)
    real_unlink = os.unlink

    def unlink_then_another_dead_one(p, *a, **kw):
        real_unlink(p, *a, **kw)
        _write_holder(path, DEAD_PID)       # always another corpse

    monkeypatch.setattr(sup_mod.os, "unlink", unlink_then_another_dead_one)
    with pytest.raises(ResourceExhausted, match="kept changing hands"):
        SingleInstanceLock(path).acquire()
