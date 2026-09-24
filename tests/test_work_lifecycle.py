"""What a work item's ending means, and who is told about it.

An independent audit on 2026-09-24 found five ways the work lifecycle lost
track of an outcome. They share a shape: the decision that *happened* and the
record or notification of it were computed in different places, from different
inputs, and drifted apart.

* `fail` validated the fencing token and nothing else, while cancellation and
  completion cleared the lease without retiring its token -- so a late error
  report resurrected terminal work to `queued`. Cancelled work could run
  again; a committed success could lose its terminal status.
* The terminal transition was decided by the repository (`attempt` against the
  ceiling) and the notification by the caller's `requeue` flag, so the attempt
  that ended the work told nobody.
* An expired lease did not count against any ceiling, so an item whose worker
  died every time cycled forever with nothing decided.
* Dispatch chose the highest-priority *queued* item and the lease then refused
  it for being unready, so a blocked head of queue hid ready work behind it.
* A dependency that failed or was cancelled left its dependent queued forever.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest

from amoeba.errors import Fenced, InvalidInput
from amoeba.store.work_repo import MAX_WORK_ATTEMPTS


def _admit(mind, *, objective="look into the cache", origin="ego", **kw):
    work_id, _ = mind.work.admit(objective=objective, work_class="user",
                                 origin_actor=origin, **kw)
    return work_id


def _status(mind, work_id):
    return mind.db.conn.execute(
        "SELECT status FROM work_items WHERE work_id = ?", (work_id,)).fetchone()["status"]


def _events(mind, kind):
    return [r["event_id"] for r in mind.db.conn.execute(
        "SELECT event_id FROM events WHERE kind = ?", (kind,))]


# ---------------------------------------------------------------------------
# A. A terminal item is closed
# ---------------------------------------------------------------------------
def test_a_late_failure_cannot_reopen_cancelled_work(mind):
    """Cancellation races a worker's error report, and cancellation wins."""
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.cancel(work_id=work_id, actor="operator", reason="no longer needed")

    with pytest.raises(Fenced):
        mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                       fencing_token=lease["fencing_token"], failure="died")

    assert _status(mind, work_id) == "cancelled", "cancelled work was requeued"


def test_a_late_failure_cannot_undo_a_committed_success(mind):
    """Completion commits, its response is lost, the worker reports failure."""
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.complete(work_id=work_id, neuocyte_id="nc_1",
                       fencing_token=lease["fencing_token"], result={"finding": "x"})

    with pytest.raises(Fenced):
        mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                       fencing_token=lease["fencing_token"], failure="timed out")

    assert _status(mind, work_id) == "done", "a successful result lost its status"


def test_cancelling_retires_the_lease_it_cancelled(mind):
    """An outstanding worker's token stops being one the row accepts."""
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    before = lease["fencing_token"]
    mind.work.cancel(work_id=work_id, actor="operator", reason="stop")
    after = mind.db.conn.execute(
        "SELECT fencing_token FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()["fencing_token"]
    assert after > before, "the cancelled lease still holds authority"


# ---------------------------------------------------------------------------
# 2. A refused result leaves a trace
# ---------------------------------------------------------------------------
def test_a_refused_result_is_recorded_even_though_it_was_refused(mind):
    """The event was emitted inside the mutation that then raised.

    Rollback took it with it, so a rejected result left nothing behind and
    could not reach the counter that is meant to notice them.
    """
    work_id = _admit(mind)
    first = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.expire_leases(now=time.time() + 10_000)

    with pytest.raises(Fenced):
        mind.work.complete(work_id=work_id, neuocyte_id="nc_1",
                           fencing_token=first["fencing_token"], result={"late": True})

    assert _events(mind, "work.result_fenced"), "the rejection was not recorded"


# ---------------------------------------------------------------------------
# 1. The owner is told what was committed
# ---------------------------------------------------------------------------
def _sup(mind):
    from amoeba import supervisor_api, turn_api

    holder: dict[str, object] = {}
    cache: dict[str, object] = {}

    def all_methods():
        if not cache:
            cache.update(supervisor_api.build(holder["sup"]))
            cache.update(turn_api.build(holder["sup"]))
        return cache

    sup = SimpleNamespace(mind=mind, cfg=mind.cfg, log=logging.getLogger("t"),
                          methods=all_methods, note_trigger=lambda role: None,
                          note_turn_finished=lambda *a, **k: None,
                          next_heartbeat=lambda role: None,
                          release_work_sandbox=lambda *a, **k: None,
                          neuocytes={}, arbiter=SimpleNamespace())
    holder["sup"] = sup
    return sup


def _triggers(mind, kind):
    return [dict(r) for r in mind.db.conn.execute(
        "SELECT * FROM role_triggers WHERE kind = ?", (kind,))]


def test_the_attempt_that_ends_the_work_tells_its_owner(mind):
    """The neuocyte's default is `requeue=True`, and the third one is final."""
    verbs = _sup(mind).methods()
    work_id = _admit(mind)

    for _ in range(MAX_WORK_ATTEMPTS):
        lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
        verbs["fail_work"](work_id=work_id, neuocyte_id="nc_1",
                           fencing_token=lease["fencing_token"],
                           failure="could not read the file")

    assert _status(mind, work_id) == "failed"
    assert _triggers(mind, "work_failed"), (
        "the work was given up on and its owner was never told")


def test_a_failure_that_will_be_retried_tells_nobody(mind):
    """An attempt is not an outcome."""
    verbs = _sup(mind).methods()
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    out = verbs["fail_work"](work_id=work_id, neuocyte_id="nc_1",
                             fencing_token=lease["fencing_token"], failure="once")

    assert out["outcome"] == "queued"
    assert not _triggers(mind, "work_failed")


def test_only_a_persistent_role_can_own_work(mind):
    """The ownership filter, on its own.

    Asserted here rather than through `wake_owner_of_work`, because the
    enqueue below also refuses a non-role: each layer masked the other, and a
    guarantee two layers defend is one that neither can be shown to defend
    unless each is asked at its own level.
    """
    from amoeba.waking import owner_of_work

    assert owner_of_work(mind, _admit(mind, origin="ego")) is not None
    assert owner_of_work(mind, _admit(mind, origin="operator")) is None
    assert owner_of_work(mind, _admit(mind, origin="nc_7")) is None


def test_a_trigger_cannot_be_queued_for_something_that_is_not_a_role(mind):
    """The enqueue's own refusal, which is what actually stops the write."""
    from amoeba import mailbox

    with pytest.raises(InvalidInput):
        mind.writer.apply(
            lambda m: mailbox.enqueue(m, role="operator", kind="work_failed",
                                      source="harness", summary="x"),
            actor="test", bump_version=False)


def test_work_nobody_owns_wakes_nobody(mind):
    """Only a persistent role can be owed a notification.

    Work the operator or the supervisor started has no role waiting on it, and
    a trigger addressed to "operator" is a trigger nothing will ever collect.
    """
    from amoeba.waking import wake_owner_of_work

    sup = _sup(mind)
    work_id = _admit(mind, origin="operator")
    returned = wake_owner_of_work(sup, mind, work_id, kind="work_failed",
                                  summary="the operator's errand failed")

    assert returned is None
    assert not [dict(r) for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers WHERE source_ref = ?", (work_id,))], (
        "a trigger was queued for somebody who is not a role")


# ---------------------------------------------------------------------------
# 4b. An expired attempt is still an attempt
# ---------------------------------------------------------------------------
def test_leases_that_keep_expiring_do_not_cycle_forever(mind):
    """Ten rounds used to leave it queued on attempt ten, with nothing decided."""
    work_id = _admit(mind)
    for _ in range(MAX_WORK_ATTEMPTS + 2):
        if _status(mind, work_id) != "queued":
            break
        mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
        mind.work.expire_leases(now=time.time() + 10_000)

    row = mind.db.conn.execute(
        "SELECT status, attempt FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()
    assert row["status"] == "failed", f"still {row['status']} on attempt {row['attempt']}"
    assert row["attempt"] <= MAX_WORK_ATTEMPTS


# ---------------------------------------------------------------------------
# 4a. Dispatch serves what can actually be worked on
# ---------------------------------------------------------------------------
def test_a_ready_item_is_not_hidden_behind_a_blocked_one(mind):
    """Dispatch chose by priority alone; the lease then refused what it chose."""
    blocker = _admit(mind, objective="the dependency")
    blocked = _admit(mind, objective="waits for it", depends_on=[blocker], priority=10)
    ready = _admit(mind, objective="can run now", priority=5)

    assert mind.work.next_ready(work_class="user") in (blocker, ready)
    # With the blocker itself leased, the highest-priority queued item is the
    # blocked one, and the ready item must still be what dispatch serves.
    mind.work.lease(neuocyte_id="nc_1", work_id=blocker)
    assert mind.work.next_ready(work_class="user") == ready, (
        f"dispatch would have served {blocked}, which cannot be leased")


def test_work_cannot_depend_on_something_that_does_not_exist(mind):
    with pytest.raises(InvalidInput):
        _admit(mind, depends_on=["work_nope"])


def test_a_dependency_that_failed_ends_the_work_waiting_on_it(mind):
    """A failed dependency is an answer, not a wait."""
    blocker = _admit(mind, objective="the dependency")
    blocked = _admit(mind, objective="waits for it", depends_on=[blocker])

    mind.work.cancel(work_id=blocker, actor="operator", reason="not wanted")
    retired = mind.work.retire_blocked()

    assert retired == [blocked]
    assert _status(mind, blocked) == "failed"
    assert "never finish" in mind.db.conn.execute(
        "SELECT failure FROM work_items WHERE work_id = ?",
        (blocked,)).fetchone()["failure"]


def test_work_whose_dependency_is_still_running_is_left_alone(mind):
    blocker = _admit(mind, objective="the dependency")
    blocked = _admit(mind, objective="waits for it", depends_on=[blocker])
    mind.work.lease(neuocyte_id="nc_1", work_id=blocker)

    assert mind.work.retire_blocked() == []
    assert _status(mind, blocked) == "queued"
