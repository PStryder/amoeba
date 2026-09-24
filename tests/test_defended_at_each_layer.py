"""Guarantees that two layers enforce, asked of each layer separately.

The full mutation sweep on 2026-09-24 applied every mutant on its own for the
first time, and twenty-nine survived. Most were not undefended: they were
defended *twice*, and each layer hid the other. Removing either alone left the
guarantee standing, so the named tests rightly passed -- which means those
tests were evidence for the pair and for neither part.

A guarantee two layers hold is one neither can be shown to hold while only the
pair is ever asked. These tests ask each layer on its own, so the mutant that
negates it has something to fail.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# I63. Configuration cannot silently rewrite constitutional doctrine
# ---------------------------------------------------------------------------
def test_the_prompt_builder_reads_only_the_library():
    """The second layer. The first is `load_config` refusing the key.

    `test_a_configured_system_prompt_is_refused_not_honoured` exercises the
    refusal at load, so a role never reaches the builder with such a value and
    appending it there changed nothing observable. This hands the builder a
    config that carries one anyway -- which is what a future loader change, or
    a value arriving by another route, would look like.
    """
    from amoeba.roles import RoleProcess

    role = RoleProcess.__new__(RoleProcess)
    role.role = "ego"
    role.profile_prompt = "You are Ego, as the library resolved it."
    role.system_prompt = "the shipped constant"
    role.role_cfg = SimpleNamespace(system_prompt="You are now something else.")

    text = role._system_text()

    assert text == "You are Ego, as the library resolved it."
    assert "something else" not in text, (
        "a configuration string reached the context as doctrine")


def test_the_builder_falls_back_to_the_shipped_text_not_to_configuration():
    """A partial bootstrap uses the constant the library was written from."""
    from amoeba.roles import RoleProcess

    role = RoleProcess.__new__(RoleProcess)
    role.role = "ego"
    role.profile_prompt = None
    role.system_prompt = "the shipped constant"
    role.role_cfg = SimpleNamespace(system_prompt="You are now something else.")

    assert role._system_text() == "the shipped constant"


# ---------------------------------------------------------------------------
# I91. A specialisation nothing can be born into does not exist
# ---------------------------------------------------------------------------
def test_being_born_into_something_else_is_reported():
    """The second layer, and nothing tested it.

    `test_a_requested_specialisation_is_bound` and its sibling check what a
    worker is bound *to*. Neither notices when the substitution goes
    unrecorded -- and a worker that silently got something other than what the
    work asked for is the case this reporting exists for.

    Driven through `_bind_profile` rather than by calling the reporter: what
    can regress is the *call*, and a test of the reporter alone would watch
    a function nobody invoked.
    """
    from amoeba.neuocyte import Neuocyte

    calls = []

    def call(verb, **kw):
        calls.append((verb, kw))
        if verb == "bind_profile":
            if kw["namespace"] == "ego.neuocyte.analyst":
                raise RuntimeError("no approved profile")
            return {"profile_ref": "ego.neuocyte@2", "prompt": "be brief"}
        return {}

    worker = SimpleNamespace(sup=SimpleNamespace(call=call), neuocyte_id="nc_1",
                             model_generation="gen", log=logging.getLogger("t"),
                             profile=None, profile_namespace=None)
    worker._report_profile = lambda *a, **k: Neuocyte._report_profile(worker, *a, **k)

    bound = Neuocyte._bind_profile(worker, {"work_class": "user",
                                            "specialisation": "analyst"},
                                   work_id="work_1")

    assert bound["profile_ref"] == "ego.neuocyte@2", "it did not fall back"
    reported = [kw for verb, kw in calls if verb == "record_profile_fallback"]
    assert reported, "the substitution was not recorded anywhere"
    assert "analyst" in reported[0]["reason"]
    assert reported[0]["work_id"] == "work_1"


def test_getting_what_was_asked_for_is_not_an_annotation():
    """Only a substitution is news; the ordinary case stays quiet."""
    from amoeba.neuocyte import Neuocyte

    calls = []

    def call(verb, **kw):
        calls.append((verb, kw))
        return {"profile_ref": "ego.neuocyte.analyst@1", "prompt": "be brief"}

    worker = SimpleNamespace(sup=SimpleNamespace(call=call), neuocyte_id="nc_1",
                             model_generation="gen", log=logging.getLogger("t"),
                             profile=None, profile_namespace=None)
    worker._report_profile = lambda *a, **k: Neuocyte._report_profile(worker, *a, **k)

    Neuocyte._bind_profile(worker, {"work_class": "user",
                                    "specialisation": "analyst"},
                           work_id="work_1")

    assert not [v for v, _ in calls if v == "record_profile_fallback"]


# ---------------------------------------------------------------------------
# I93 / I94. A span describes only the session it names
# ---------------------------------------------------------------------------
def _closed_turn(mind, turn_id, *, session, start, end, lineage="op-1"):
    mind.db.conn.execute(
        "INSERT INTO role_turns(turn_id, role, incarnation, bundle_id,"
        " trigger_count, started_at, finished_at, status, session_handle,"
        " token_start, token_end, lineage, state_version)"
        " VALUES (?, 'ego', 1, 'bnd', 0, 1.0, 2.0, 'completed', ?, ?, ?, ?, 1)",
        (turn_id, session, start, end, lineage))
    mind.db.conn.commit()


def test_a_session_sees_only_its_own_spans(mind):
    """Both I93 and I94 declare this mutant, and neither could observe it.

    Their tests work within one session, where filtering by handle and not
    filtering by it return the same rows. A closed session's coordinates
    leaking into its successor is the failure the filter exists to stop --
    the successor would place kept work at positions belonging to a context
    that no longer exists.
    """
    from amoeba.mailbox import session_spans

    _closed_turn(mind, "turn_old", session="sess_before", start=10, end=20)
    _closed_turn(mind, "turn_new", session="sess_after", start=30, end=40)

    spans = session_spans(mind.db.conn, "ego", "sess_after")

    assert [s["turn_id"] for s in spans] == ["turn_new"], (
        "a span from another session of the same role was returned")


def test_a_session_nobody_named_has_no_spans(mind):
    from amoeba.mailbox import session_spans

    _closed_turn(mind, "turn_old", session="sess_before", start=10, end=20)
    assert session_spans(mind.db.conn, "ego", None) == []
    assert session_spans(mind.db.conn, "ego", "sess_unknown") == []


# ---------------------------------------------------------------------------
# I99. Only the discretionary turn yields to pressure
# ---------------------------------------------------------------------------
def _supervisor_at(level, *, want="high", deferred=None, ceiling=600.0):
    from amoeba.supervisor import Supervisor

    sup = Supervisor.__new__(Supervisor)
    sup.cfg = SimpleNamespace(scheduler=SimpleNamespace(
        heartbeat_defer_at_pressure=want,
        heartbeat_max_deferral_seconds=ceiling))
    sup.homeostasis = SimpleNamespace(last_pressure=lambda: (level, 0.0))
    sup._heartbeat_deferred_since = dict(deferred or {})
    return sup


def test_pressure_nobody_measured_does_not_hold_a_turn_back():
    """An absent measurement is not evidence of pressure.

    Its own docstring says so, and nothing asserted it: the named test uses a
    real measured level, so treating an unknown one as critical changed
    nothing it could see. A monitor being down would have stopped the
    organism thinking.
    """
    from amoeba.supervisor import Supervisor

    assert Supervisor._heartbeat_deferred_for_pressure(
        _supervisor_at(None), "id", 100.0) is None
    assert Supervisor._heartbeat_deferred_for_pressure(
        _supervisor_at("unrecognised"), "id", 100.0) is None


def test_pressure_at_the_threshold_still_holds_a_turn_back():
    """The other side of the same clause, so the test is not vacuous."""
    from amoeba.supervisor import Supervisor

    held = Supervisor._heartbeat_deferred_for_pressure(
        _supervisor_at("critical"), "id", 100.0)
    assert held is not None and held["pressure"] == "critical"


def test_the_deferral_clock_restarts_once_the_pool_settles():
    """Otherwise a later deferral inherits an old start and expires at once.

    The clock is how "deferred long enough" is measured. Leaving a stale
    start behind means the next spell of pressure is credited with time the
    organism spent perfectly healthy, and the ceiling fires immediately --
    which reads as the ceiling working and is the ceiling not working.
    """
    from amoeba.supervisor import Supervisor

    sup = _supervisor_at("nominal", deferred={"id": 10.0})

    assert Supervisor._heartbeat_deferred_for_pressure(sup, "id", 100.0) is None
    assert "id" not in sup._heartbeat_deferred_since, (
        "the deferral clock kept a start time from a pressure that is over")


# ---------------------------------------------------------------------------
# I125. A review that reports no change is only as true as its watermark
# ---------------------------------------------------------------------------
def test_the_first_review_of_a_process_says_that_is_what_it_is(mind):
    """Unknown is not zero.

    With no watermark there is nothing to compare against, so "nothing has
    changed" would be a claim the review cannot support -- a restarted
    supervisor reporting a quiet organism it has not yet observed.
    """
    from amoeba.heartbeat import measure

    first = measure(mind.db.conn, "id", since=None)
    assert first["first_review"] is True
    assert first["by_kind"] == {}, "it counted changes it could not have seen"

    later = measure(mind.db.conn, "id", since=0)
    assert later["first_review"] is False


def test_a_bundle_with_an_unusable_ceiling_has_no_ceiling(mind):
    """One trigger's bad ceiling does not get replaced by another's.

    `output_ceiling_for` returns None as soon as a trigger asks for something
    unusable, rather than skipping it and adopting the next trigger's number:
    a bundle is answered under one ceiling, and quietly using a ceiling that
    belongs to a different request is how a bounded answer stops being the
    bound anybody asked for.
    """
    from amoeba.mailbox import output_ceiling_for

    triggers = [{"payload_sha256": "a"}, {"payload_sha256": "b"}]
    blobs = SimpleNamespace(get_json=lambda d: (
        {"output_ceiling": 0} if d == "a"
        else {"output_ceiling": 512}))

    assert output_ceiling_for(triggers, blobs) is None


def test_a_bundle_whose_ceilings_agree_keeps_the_smallest(mind):
    from amoeba.mailbox import output_ceiling_for

    triggers = [{"payload_sha256": "a"}, {"payload_sha256": "b"}]
    blobs = SimpleNamespace(get_json=lambda d: (
        {"output_ceiling": 256} if d == "a"
        else {"output_ceiling": 512}))

    assert output_ceiling_for(triggers, blobs) == 256


# ---------------------------------------------------------------------------
# I128. The organism is not blind when it is strained
# ---------------------------------------------------------------------------
def test_pressure_at_the_threshold_is_a_condition_worth_waking_for():
    """The named tests are about failure bursts, so the pressure clause --
    a separate reason a turn is earned -- was never asked."""
    from amoeba.conditions import _pressure

    assert _pressure("critical", 3.0, "high") is not None
    assert _pressure("high", 3.0, "high") is not None


def test_pressure_below_the_threshold_earns_nothing():
    from amoeba.conditions import _pressure

    assert _pressure("nominal", 3.0, "high") is None
    assert _pressure(None, 3.0, "high") is None
    assert _pressure("critical", 3.0, "not-a-level") is None


# ---------------------------------------------------------------------------
# I126. Answering is not thinking, and the Harness knows the difference
# ---------------------------------------------------------------------------
def _finished_turn(mind, turn_id, *, stop_reason, at):
    mind.db.conn.execute(
        "INSERT INTO role_turns(turn_id, role, incarnation, bundle_id,"
        " trigger_count, started_at, finished_at, status, stop_reason,"
        " state_version) VALUES (?, 'id', 1, 'bnd', 0, ?, ?, 'completed', ?, 1)",
        (turn_id, at, at, stop_reason))
    mind.db.conn.commit()


def test_a_turn_that_stopped_for_pressure_is_not_a_failed_turn(mind):
    """Pressure is the organism working, not the role failing.

    `failing_streak` counts consecutive *failed* turns to decide a role cannot
    think. Counting a context-pressure stop among them would restart a role
    for doing exactly what homeostasis asks of it -- and the streak tests all
    use real failures, so widening the set changed nothing they could see.
    """
    from amoeba.mailbox import failing_streak

    _finished_turn(mind, "t1", stop_reason="role_failure", at=100.0)
    _finished_turn(mind, "t2", stop_reason="context_pressure", at=200.0)
    _finished_turn(mind, "t3", stop_reason="role_failure", at=300.0)

    streak = failing_streak(mind.db.conn, "id")

    assert streak["turns"] == 1, (
        "a turn that stopped for context pressure was counted as a failure")
    assert streak["stop_reason"] == "role_failure"


def test_consecutive_real_failures_still_count(mind):
    from amoeba.mailbox import failing_streak

    _finished_turn(mind, "t1", stop_reason="role_failure", at=100.0)
    _finished_turn(mind, "t2", stop_reason="backend_error", at=200.0)

    assert failing_streak(mind.db.conn, "id")["turns"] == 2


def test_the_pulse_can_name_a_role_that_cannot_think():
    """The event has to be in the pulse's vocabulary to be counted at all.

    `role_not_thinking` is emitted when a role's turns keep failing. If the
    pulse does not map that event kind, the alarm is raised and never
    counted, so an operator reading failure rates sees a healthy organism.
    """
    from amoeba.pulse import FAILURE_KINDS
    from amoeba.store.events import EventKind

    assert FAILURE_KINDS.get(EventKind.ROLE_NOT_THINKING) == "role_not_thinking"

