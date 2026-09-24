"""Answering is not thinking.

Live, on 2026-09-22 at 18:56:57, Id asked for its own rejuvenation. The
Harness rebuilt the session, recorded the new handle, and did not tell Id --
the handover was the caller's job and only one of the three callers did it.
Id went on addressing a session that had been closed. Every turn for the next
thirty-seven hours failed with "unknown inference session": seventy-five of
them, not one success.

Nothing noticed. The process answered every health probe, so supervision saw
a healthy child; role turn failures were not counted as failures anywhere, so
the pulse showed none; and the only component whose job is to watch the
organism was the one that could not think. A role asked whether it is well is
the worst available witness, which is why this is measured from the Harness's
own record.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from amoeba import mailbox
from amoeba.store.events import EventKind


def _turn(mind, role, stop_reason, *, kind="heartbeat"):
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind=kind, source="scheduler",
                                  summary="review"),
        actor="harness", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role=role, incarnation=1,
                                profile_ref=f"{role}@1", profile_sha256="p",
                                environment_sha256="e", environment_blob="eb"),
        actor=role, bump_version=False)
    mind.writer.apply(
        lambda m: mailbox.complete(m, mind, turn_id=turn["turn_id"],
                                   stop_reason=stop_reason),
        actor="harness", bump_version=False)
    return turn["turn_id"]


# ---------------------------------------------------------------------------
# What the record knows
# ---------------------------------------------------------------------------
def test_consecutive_failures_are_counted_from_the_record(mind):
    for _ in range(3):
        _turn(mind, "id", "role_failure")
    streak = mailbox.failing_streak(mind.db.conn, "id")
    assert streak["turns"] == 3
    assert streak["stop_reason"] == "role_failure"
    assert streak["since"] is not None


def test_one_good_turn_ends_the_streak(mind):
    _turn(mind, "id", "role_failure")
    _turn(mind, "id", "backend_error")
    assert mailbox.failing_streak(mind.db.conn, "id")["turns"] == 2
    _turn(mind, "id", "model_stop")
    assert mailbox.failing_streak(mind.db.conn, "id")["turns"] == 0


@pytest.mark.parametrize("stop", ["context_pressure", "max_output_tokens",
                                  "tool_turn_limit_reached", "deadline_reached"])
def test_a_mind_meeting_a_known_limit_is_not_failing(mind, stop):
    """Homeostasis and ceilings are the organism working, not breaking."""
    for _ in range(4):
        _turn(mind, "id", stop)
    assert mailbox.failing_streak(mind.db.conn, "id")["turns"] == 0


def test_the_streak_is_per_role(mind):
    for _ in range(3):
        _turn(mind, "id", "role_failure")
    _turn(mind, "ego", "model_stop", kind="user_input")
    assert mailbox.failing_streak(mind.db.conn, "ego")["turns"] == 0


# ---------------------------------------------------------------------------
# What supervision does about it
# ---------------------------------------------------------------------------
class _Sup:
    """The supervisor's thinking check, with everything else stubbed out."""

    def __init__(self, mind, *, threshold=3, repairs=2):
        from amoeba.config import Config
        from amoeba.supervisor import Supervisor

        self.mind = mind
        self.cfg = Config()
        self.cfg.scheduler.role_failure_threshold_turns = threshold
        self.cfg.scheduler.role_repairs_per_hour = repairs
        self.log = logging.getLogger("test-sup")
        self._not_thinking: dict[str, Any] = {}
        self._role_repairs: dict[str, list[float]] = {}
        self.restarts: list[tuple[str, str]] = []
        self._check = Supervisor._supervise_thinking.__get__(self)
        self._emit_not_thinking = Supervisor._emit_not_thinking.__get__(self)

    def _restart_child(self, name, reason):
        self.restarts.append((name, reason))

    def check(self):
        self._check()


def _recorded(mind):
    import json

    return [json.loads(r["payload_inline"]) for r in mind.db.conn.execute(
        "SELECT payload_inline FROM events WHERE kind = ?",
        (EventKind.ROLE_NOT_THINKING,))]


def test_a_role_that_cannot_think_is_recorded_and_repaired(mind):
    sup = _Sup(mind)
    for _ in range(3):
        _turn(mind, "id", "role_failure")
    sup.check()

    assert [r[0] for r in sup.restarts] == ["id"]
    assert "failed 3 turns in a row" in sup.restarts[0][1]
    said = _recorded(mind)
    assert len(said) == 1
    assert said[0]["role"] == "id" and said[0]["consecutive_failed_turns"] == 3
    assert said[0]["repair"] == "restarting the role"


def test_a_healthy_role_is_left_alone(mind):
    sup = _Sup(mind)
    _turn(mind, "id", "role_failure")
    _turn(mind, "id", "model_stop")
    sup.check()
    assert sup.restarts == [] and _recorded(mind) == []


def test_one_streak_is_reported_once_however_often_it_is_checked(mind):
    sup = _Sup(mind)
    for _ in range(3):
        _turn(mind, "id", "role_failure")
    for _ in range(5):
        sup.check()
    assert len(_recorded(mind)) == 1, "a wedged role must not spam the record"
    assert len(sup.restarts) == 1


def test_repairs_are_bounded_and_the_alarm_outlasts_them(mind):
    """A role that keeps failing is left visibly unwell, not restarted forever."""
    sup = _Sup(mind, repairs=1)
    for streak in range(3):
        for _ in range(3):
            _turn(mind, "id", "role_failure")
            time.sleep(0.002)
        _turn(mind, "id", "model_stop")          # a success ends the streak
        _turn(mind, "id", "role_failure")        # and a new one begins
        _turn(mind, "id", "role_failure")
        _turn(mind, "id", "role_failure")
        sup.check()

    assert len(sup.restarts) == 1, "repairs were not bounded"
    said = _recorded(mind)
    assert len(said) >= 2, "the alarm stopped when the repairs ran out"
    assert said[-1]["repair"].startswith("repairs for this hour are spent")


# ---------------------------------------------------------------------------
# What the operator sees
# ---------------------------------------------------------------------------
def test_the_pulse_reports_a_role_that_cannot_think(mind):
    """Measured from the record, so a cheerful role cannot hide it."""
    from types import SimpleNamespace

    from amoeba.config import Config
    from amoeba.pulse import PulseCollector

    cfg = Config()
    cfg.scheduler.role_failure_threshold_turns = 3
    pulse = PulseCollector(SimpleNamespace(mind=mind, cfg=cfg))

    assert pulse._thinking("id") == {"consecutive_failed_turns": 0, "thinking": True}
    for _ in range(3):
        _turn(mind, "id", "role_failure")
    assert pulse._thinking("id") == {"consecutive_failed_turns": 3, "thinking": False}


def test_a_role_that_cannot_think_counts_as_a_failure(mind):
    """It used to count as nothing: the pulse showed zero failures throughout."""
    from amoeba.pulse import FAILURE_KINDS

    assert EventKind.ROLE_NOT_THINKING in FAILURE_KINDS
