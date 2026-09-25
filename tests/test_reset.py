"""A reset starts a new organism without pretending the old one never ran.

Everything in the state directory is the record of a mind that ran: what it
concluded, what it was told, what it did and when. So the default is to move
it aside intact under a timestamped name, and `--delete` has to say it means
it. Nothing happens at all while a supervisor owns the directory: deleting a
database under a live writer leaves a half-state nobody can reason about
afterwards.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from amoeba import reset
from amoeba.__main__ import main as cli


class _Cfg:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir


def _organism(tmp_path: Path) -> Path:
    """A state directory with the shape a run leaves behind."""
    state = tmp_path / "amoeba-state"
    (state / "blobs" / "ab").mkdir(parents=True)
    (state / "logs").mkdir()
    (state / "artifacts").mkdir()
    (state / "mind.sqlite3").write_text("the record")
    (state / "mind.sqlite3-wal").write_text("in flight")
    (state / "blobs" / "ab" / "abcd").write_text("a conclusion's evidence")
    (state / "logs" / "supervisor.log").write_text("what happened")
    (state / "supervisor.ready").write_text("{}")
    (state / "control.token").write_text("tok-control")
    (state / "scope.ego.token").write_text("tok-ego")
    (state / "operator.session").write_text("tok-operator")
    (state / "api_clients.json").write_text('{"client_a": "key"}')
    return state


def _names(state: Path) -> set[str]:
    return {p.name for p in state.iterdir()}


# ---------------------------------------------------------------------------
# What a reset moves, and what it leaves
# ---------------------------------------------------------------------------
def test_the_mind_is_archived_and_the_plumbing_is_kept(tmp_path):
    state = _organism(tmp_path)
    out = reset.perform(reset.plan(_Cfg(state)))

    assert _names(state) == {"control.token", "scope.ego.token",
                             "operator.session", "api_clients.json"}
    archive = Path(out["archive"])
    assert (archive / "mind.sqlite3").read_text() == "the record"
    assert (archive / "blobs" / "ab" / "abcd").read_text() == \
        "a conclusion's evidence"
    assert (archive / "logs" / "supervisor.log").exists()
    assert out["deleted"] is False


def test_rotating_credentials_takes_them_too(tmp_path):
    state = _organism(tmp_path)
    out = reset.perform(reset.plan(_Cfg(state), rotate_credentials=True))
    assert _names(state) == set()
    assert (Path(out["archive"]) / "control.token").read_text() == "tok-control"
    assert out["kept"] == []


def test_a_dry_run_moves_nothing(tmp_path):
    state = _organism(tmp_path)
    before = sorted(p.name for p in state.rglob("*"))
    out = reset.perform(reset.plan(_Cfg(state)), dry_run=True)
    assert out["dry_run"] is True and out["archive"] is None
    assert "mind.sqlite3" in out["moved"]
    assert sorted(p.name for p in state.rglob("*")) == before


def test_deleting_really_deletes(tmp_path):
    state = _organism(tmp_path)
    out = reset.perform(reset.plan(_Cfg(state)), delete=True)
    assert out["deleted"] is True and out["archive"] is None
    assert not (tmp_path / "amoeba-state.reset").exists()
    assert list(tmp_path.glob("amoeba-state.reset-*")) == []
    assert _names(state) == {"control.token", "scope.ego.token",
                             "operator.session", "api_clients.json"}


def test_nothing_outside_the_state_directory_is_touched(tmp_path):
    state = _organism(tmp_path)
    outside = tmp_path / "peters-files"
    outside.mkdir()
    (outside / "readings.csv").write_text("not amoeba's")
    reset.perform(reset.plan(_Cfg(state)))
    assert (outside / "readings.csv").read_text() == "not amoeba's"


@pytest.mark.parametrize("name, credential", [
    ("control.token", True), ("operator.session", True),
    ("api_clients.json", True), ("scope.neuocyte.token", True),
    ("mind.sqlite3", False), ("blobs", False), ("supervisor.ready", False),
])
def test_what_counts_as_plumbing(name, credential):
    assert reset.is_credential(name) is credential


# ---------------------------------------------------------------------------
# What a reset refuses
# ---------------------------------------------------------------------------
def test_a_running_organism_is_not_reset(tmp_path, capsys):
    """The supervisor's own lock decides, by its own liveness rule.

    Asserted on the *message*, not merely on the refusal. Since `owned()`
    started taking the lock for the whole operation there are two ways to be
    refused, and both print "refusing" -- so checking for that word stopped
    distinguishing them, and the pre-flight check became unobservable. What it
    is for is telling the operator which process to stop, which is what the
    User Guide promises and what the lock contention alone cannot say.
    """
    state = _organism(tmp_path)
    (state / "supervisor.lock").write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()}))
    assert reset.holder(_Cfg(state)) is not None

    code = cli(["reset", "--config", str(_config(tmp_path, state))])
    out = capsys.readouterr().out
    assert code == 2
    assert "refusing" in out
    assert f"pid {os.getpid()}" in out, (
        "the refusal did not name the process that owns the directory")
    assert (state / "mind.sqlite3").exists(), "a live organism was reset"


def test_a_lock_whose_holder_is_gone_does_not_stop_a_reset(tmp_path):
    state = _organism(tmp_path)
    (state / "supervisor.lock").write_text(
        json.dumps({"pid": 999_999, "started": time.time()}))
    assert reset.holder(_Cfg(state)) is None


def test_a_supervisor_cannot_start_while_a_reset_runs(tmp_path):
    """Checking for an owner and then moving the database is two steps.

    Reviewed on 2026-09-24: a supervisor starting inside that gap had its
    database archived out from under it. The lock a supervisor would have to
    take is taken by the reset instead, for the whole operation.
    """
    from amoeba.errors import ResourceExhausted
    from amoeba.supervisor import SingleInstanceLock

    state = _organism(tmp_path)
    cfg = _Cfg(state)
    assert reset.holder(cfg) is None, "nothing owns it yet"

    with reset.owned(cfg):
        # Exactly what a supervisor starting mid-reset would attempt.
        with pytest.raises(ResourceExhausted):
            SingleInstanceLock(state / "supervisor.lock").acquire()

    # And it is given back afterwards.
    lock = SingleInstanceLock(state / "supervisor.lock")
    lock.acquire()
    lock.release()


def test_a_reset_refuses_while_a_supervisor_holds_the_directory(tmp_path):
    from amoeba.errors import ResourceExhausted
    from amoeba.supervisor import SingleInstanceLock

    state = _organism(tmp_path)
    held = SingleInstanceLock(state / "supervisor.lock")
    held.acquire()
    try:
        with pytest.raises(ResourceExhausted):
            with reset.owned(_Cfg(state)):
                pass
    finally:
        held.release()
    assert (state / "mind.sqlite3").exists()


def test_the_lock_a_reset_holds_is_not_archived(tmp_path):
    """Archiving your own claim on the directory would end the exclusion."""
    state = _organism(tmp_path)
    with reset.owned(_Cfg(state)):
        chosen = reset.plan(_Cfg(state))
        assert all(e.name != "supervisor.lock" for e in chosen["move"])


def test_deleting_needs_to_be_meant(tmp_path, capsys):
    state = _organism(tmp_path)
    code = cli(["reset", "--config", str(_config(tmp_path, state)), "--delete"])
    assert code == 2
    assert "pass --yes to mean it" in capsys.readouterr().out
    assert (state / "mind.sqlite3").exists()


def test_an_empty_directory_is_not_an_error(tmp_path, capsys):
    state = tmp_path / "amoeba-state"
    state.mkdir()
    code = cli(["reset", "--config", str(_config(tmp_path, state))])
    assert code == 0
    assert "nothing to reset" in capsys.readouterr().out


def test_the_command_says_what_it_did_and_where_it_went(tmp_path, capsys):
    state = _organism(tmp_path)
    code = cli(["reset", "--config", str(_config(tmp_path, state))])
    said = capsys.readouterr().out
    assert code == 0
    assert "archived" in said and "mind.sqlite3" in said
    assert "readable at" in said
    assert "incarnation 1" in said
    assert "control.token" in said.split("kept")[1]


def _config(tmp_path: Path, state: Path) -> Path:
    path = tmp_path / "reset.toml"
    path.write_bytes(f'state_dir = "{state.as_posix()}"\n'.encode())
    return path
