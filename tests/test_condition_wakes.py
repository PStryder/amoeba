"""The organism is not blind when it is strained.

Id is woken by conclusions, messages, work it owns and the operator. Nothing
woke it for a failure, for resource pressure, or for the other role being
wedged -- so the heartbeat's interval was the detection latency for
everything the inward mind exists to catch. Worse, the heartbeat is deferred
while the pool is under pressure: the organism looked at itself least often
exactly when it was most strained.

These wakes measure and say nothing about what the measurement means. They
are bounded by a cooldown, they do not repeat news still sitting unread in
the mailbox, and they never wake a role about its own inability to think --
a mind that cannot complete a turn cannot complete a turn about that.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from amoeba import conditions, mailbox

QUIET = {"failures": {}, "pressure": (None, 0.0), "streaks": {"ego": 0, "id": 0},
         "failure_threshold": 5, "pressure_level": "critical", "failure_turns": 3}


def _detect(**over):
    return conditions.detect("id", **{**QUIET, **over})


# ---------------------------------------------------------------------------
# What is worth a turn
# ---------------------------------------------------------------------------
def test_a_burst_of_failures_wakes_the_inward_mind():
    found = _detect(failures={"work_failed": 4, "tool_rejected": 3})
    assert [c.key for c in found] == ["failure_burst"]
    assert "7 failures in the last five minutes" in found[0].text
    assert found[0].measured["total"] == 7


def test_a_failure_or_two_does_not():
    assert _detect(failures={"work_failed": 2}) == []


def test_pressure_wakes_it_at_the_configured_level():
    assert [c.key for c in _detect(pressure=("critical", 3.0))] == ["pressure_critical"]
    assert _detect(pressure=("high", 3.0)) == []
    assert [c.key for c in _detect(pressure=("high", 3.0), pressure_level="high")] \
        == ["pressure_high"]


def test_unknown_pressure_is_not_pressure():
    """An absent measurement is not evidence; a monitor being down wakes nobody."""
    assert _detect(pressure=(None, float("inf"))) == []


def test_the_other_role_being_wedged_wakes_it():
    found = _detect(streaks={"ego": 4, "id": 0})
    assert [c.key for c in found] == ["not_thinking_ego"]
    assert found[0].measured == {"role": "ego", "consecutive_failed_turns": 4}


def test_a_role_is_never_woken_about_its_own_wedging():
    """It could not answer, and the Harness has its own alarm for that (I126)."""
    assert _detect(streaks={"ego": 0, "id": 9},
                   failures={"work_failed": 9},
                   pressure=("critical", 1.0)) == []


def test_several_conditions_at_once_are_all_reported():
    found = _detect(failures={"work_failed": 6}, pressure=("critical", 2.0),
                    streaks={"ego": 5, "id": 0})
    assert {c.key for c in found} == {"failure_burst", "pressure_critical",
                                      "not_thinking_ego"}


def test_a_wake_measures_and_does_not_interpret():
    text = conditions.render(_detect(failures={"work_failed": 9})[0]).lower()
    for verdict in ("you should", "recommend", "urgent", "serious", "ignore",
                    "probably", "must"):
        assert verdict not in text
    assert "what it means is yours to say" in text


# ---------------------------------------------------------------------------
# What stops it becoming noise
# ---------------------------------------------------------------------------
def test_news_still_unread_is_not_repeated(mind):
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="attention", source="harness",
                                  source_ref="pressure_critical",
                                  summary="the shared context pool is at critical"
                                          " pressure (measured 2s ago).",
                                  payload={"condition": "pressure_critical"}),
        actor="harness", bump_version=False)
    assert mailbox.pending_of_kind(mind.db.conn, "id", "attention", "pressure_critical")
    assert not mailbox.pending_of_kind(mind.db.conn, "id", "attention", "failure_burst")
    assert not mailbox.pending_of_kind(mind.db.conn, "ego", "attention")


def test_a_consumed_condition_can_be_said_again(mind):
    from test_persistent_turns import _claim, _complete

    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="attention", source="harness",
                                  source_ref="failure_burst",
                                  summary="9 failures in the last five minutes",
                                  payload={"condition": "failure_burst"}),
        actor="harness", bump_version=False)
    turn = _claim(mind, "id")
    _complete(mind, turn["turn_id"], stop_reason="model_stop")
    assert not mailbox.pending_of_kind(mind.db.conn, "id", "attention", "failure_burst")


class _Sup:
    """The supervisor's condition pass, with everything else stubbed out."""

    def __init__(self, mind, **failures):
        from amoeba.config import Config
        from amoeba.supervisor import Supervisor

        self.mind = mind
        self.cfg = Config()
        self.log = logging.getLogger("test-sup")
        self._condition_woke: dict[tuple[str, str], float] = {}
        self.queued: list[dict] = []
        self.pulse = SimpleNamespace(failures_last=lambda window="last_5m": failures)
        self.homeostasis = SimpleNamespace(last_pressure=lambda: (None, 0.0))
        self._wake = Supervisor._wake_on_conditions.__get__(self)

    def methods(self):
        def role_enqueue_trigger(**kw):
            self.queued.append(kw)
            self.mind.writer.apply(
                lambda m: mailbox.enqueue(
                    m, role=kw["role"], kind=kw["kind"], source=kw["source"],
                    source_ref=kw.get("source_ref"),
                    summary=kw["summary"], payload=kw.get("payload")),
                actor="harness", bump_version=False)
            return {"queued": True}
        return {"role_enqueue_trigger": role_enqueue_trigger}

    def wake(self):
        self._wake()


def test_the_wake_is_queued_once_and_then_held_by_the_cooldown(mind):
    sup = _Sup(mind, work_failed=9)
    sup.wake()
    assert [q["kind"] for q in sup.queued] == ["attention"]
    assert sup.queued[0]["payload"]["condition"] == "failure_burst"
    assert sup.queued[0]["ambient"] is True

    for _ in range(4):
        sup.wake()
    assert len(sup.queued) == 1, "a bad hour became a wake storm"


def test_the_cooldown_holds_even_once_nothing_is_pending(mind):
    """The second guard, on its own.

    Two things stop a wake storm: a durable check for a trigger of this kind
    already waiting, and an in-memory cooldown. While the first wake is still
    pending the durable check does all the work, so the sibling test above
    passes with the cooldown removed entirely -- it was evidence for the pair
    and for neither part.

    Here the role consumes the trigger first, which is what happens in
    practice the moment Id takes a turn. Nothing is pending, the condition is
    still true, and only the cooldown stands between a bad hour and a storm.
    """
    sup = _Sup(mind, work_failed=9)
    sup.wake()
    assert len(sup.queued) == 1

    mind.db.conn.execute("UPDATE role_triggers SET status = 'consumed'")
    mind.db.conn.commit()

    for _ in range(4):
        sup.wake()
    assert len(sup.queued) == 1, (
        "the cooldown did not hold once the first wake had been consumed")


def test_turning_them_off_is_possible(mind):
    sup = _Sup(mind, work_failed=99)
    sup.cfg.scheduler.condition_wakes = False
    sup.wake()
    assert sup.queued == []


def test_a_condition_wake_is_never_deferred_by_pressure():
    """The gate holds back a discretionary review; pressure is why we woke.

    Read at the source: only `_schedule_heartbeats` consults the gate.
    """
    import inspect

    from amoeba.supervisor import Supervisor

    wake = inspect.getsource(Supervisor._wake_on_conditions)
    assert "_heartbeat_deferred_for_pressure" not in wake
    gate = inspect.getsource(Supervisor._heartbeat_deferred_for_pressure)
    assert "heartbeat" in gate.lower()


def test_news_already_waiting_is_not_said_twice_even_once_the_cooldown_lapses(mind):
    """The cooldown bounds how often; the mailbox bounds saying it again.

    With no cooldown at all, the only thing standing between a wedged
    organism and a queue full of identical triggers is the unread one.
    """
    sup = _Sup(mind, work_failed=9)
    sup.cfg.scheduler.condition_wake_cooldown_seconds = 0.0
    for _ in range(4):
        sup.wake()
    assert len(sup.queued) == 1, "the same unread condition was queued again"
