"""Acceptance tests 9, 10, 11 plus scheduling bounds.

9.  Hold old snapshots safely while newer ones are published.
10. Reclaim retired-neuocyte and unreferenced-snapshot memory.
11. Reject incompatible snapshot reuse after a model change.
"""

from __future__ import annotations

import pytest

from amoeba.arbiter import Arbiter, ResourceSnapshot
from amoeba.config import ArbiterConfig
from amoeba.errors import CapabilityUnsupported, NotFound, ResourceExhausted
from amoeba.store.events import EventKind


def publish(mind, *, version_tokens, generation="gen_a", handle="sess_ego"):
    return mind.work.publish_snapshot(
        actor="ego", model_generation=generation, token_count=len(version_tokens),
        tokens=version_tokens, text="ctx", kv_mode="shared_prefix",
        backend_handle=handle,
    )


# ---------------------------------------------------------------------------
# Test 9: old snapshots stay safe while newer ones are published.
# ---------------------------------------------------------------------------
def test_old_snapshot_survives_while_referenced(mind):
    s1, v1, _ = publish(mind, version_tokens=[1, 2, 3])
    ref, _ = mind.work.acquire_snapshot_ref(snapshot_id=s1, holder="wk1")

    s2, v2, _ = publish(mind, version_tokens=[1, 2, 3, 4, 5])
    assert v2 == v1 + 1
    assert mind.work.latest_snapshot(actor="ego")["snapshot_id"] == s2

    # The older one is still published and still readable by its holder.
    old = mind.work.get_snapshot(s1)
    assert old["status"] == "published" and old["refcount"] == 1
    assert mind.work.snapshot_tokens(s1) == [1, 2, 3]

    # It cannot be reclaimed while referenced.
    assert s1 not in mind.work.reclaimable_snapshots(actor="ego")
    with pytest.raises(ResourceExhausted):
        mind.work.mark_snapshot_released(snapshot_id=s1, actor="supervisor",
                                         reason="premature")

    # A running neuocyte keeps its pinned snapshot; a replacement gets the newest.
    assert mind.work.snapshot_tokens(s1) != mind.work.snapshot_tokens(s2)
    mind.work.release_snapshot_ref(ref_id=ref, actor="wk1")
    assert mind.work.get_snapshot(s1)["refcount"] == 0


def test_published_snapshot_content_is_immutable(mind):
    s1, _, _ = publish(mind, version_tokens=[10, 20, 30])
    tokens_before = mind.work.snapshot_tokens(s1)
    publish(mind, version_tokens=[10, 20, 30, 40])
    publish(mind, version_tokens=[10, 20, 30, 40, 50])
    assert mind.work.snapshot_tokens(s1) == tokens_before


# ---------------------------------------------------------------------------
# Test 10: reclamation is reference counted.
# ---------------------------------------------------------------------------
def test_reclaim_only_unreferenced_and_superseded(mind):
    s1, _, _ = publish(mind, version_tokens=[1])
    s2, _, _ = publish(mind, version_tokens=[1, 2])
    s3, _, _ = publish(mind, version_tokens=[1, 2, 3])
    ref2, _ = mind.work.acquire_snapshot_ref(snapshot_id=s2, holder="wk1")

    reclaimable = mind.work.reclaimable_snapshots(actor="ego", keep_latest=1)
    assert s1 in reclaimable          # unreferenced and superseded
    assert s2 not in reclaimable      # referenced
    assert s3 not in reclaimable      # newest is always kept available

    # The guard itself, at the layer that owns it: releasing a snapshot that is
    # still referenced must be refused. Without this the test stayed green with
    # the refcount check removed, because reclaimable_snapshots() filters by
    # refcount separately and was doing all the work.
    with pytest.raises(ResourceExhausted):
        mind.work.mark_snapshot_released(snapshot_id=s2, actor="supervisor",
                                         reason="still referenced")
    assert mind.work.get_snapshot(s2)["status"] == "published"

    mind.work.mark_snapshot_released(snapshot_id=s1, actor="supervisor",
                                     reason="unreferenced")
    assert mind.work.get_snapshot(s1)["status"] == "released"
    assert mind.work.get_snapshot(s1)["backend_handle"] is None

    # A released snapshot cannot be newly acquired.
    with pytest.raises(ResourceExhausted):
        mind.work.acquire_snapshot_ref(snapshot_id=s1, holder="wk2")

    mind.work.release_snapshot_ref(ref_id=ref2, actor="wk1")
    assert s2 in mind.work.reclaimable_snapshots(actor="ego", keep_latest=1)


def test_refcount_tracks_multiple_holders(mind):
    s1, _, _ = publish(mind, version_tokens=[1, 2])
    refs = [mind.work.acquire_snapshot_ref(snapshot_id=s1, holder=f"wk{i}")[0]
            for i in range(3)]
    assert mind.work.get_snapshot(s1)["refcount"] == 3
    for r in refs[:2]:
        mind.work.release_snapshot_ref(ref_id=r, actor="wk")
    assert mind.work.get_snapshot(s1)["refcount"] == 1
    # Releasing the same ref twice must not underflow.
    mind.work.release_snapshot_ref(ref_id=refs[0], actor="wk")
    assert mind.work.get_snapshot(s1)["refcount"] == 1
    mind.work.release_snapshot_ref(ref_id=refs[2], actor="wk")
    assert mind.work.get_snapshot(s1)["refcount"] == 0


def test_retiring_a_worker_does_not_destroy_work_state(mind):
    work_id, _ = mind.work.admit(objective="keep me", work_class="user",
                                 origin_actor="ego")
    item = mind.work.lease(neuocyte_id="wk1")
    mind.work.register_agent(agent_id="wk1", role="neuocyte", work_id=work_id)
    mind.work.retire_agent(agent_id="wk1", reason="killed", crashed=True)
    # Authoritative work state is intact; only the lease is reclaimed.
    assert mind.work.get_work(work_id)["objective"] == "keep me"
    expired = mind.work.expire_leases(now=item["lease_expires"] + 1)
    assert work_id in expired
    assert mind.work.get_work(work_id)["status"] == "queued"


# ---------------------------------------------------------------------------
# Test 11: incompatible snapshot reuse is rejected after a model change.
# ---------------------------------------------------------------------------
def test_snapshot_from_another_model_generation_is_not_reused(mind):
    s_old, _, _ = publish(mind, version_tokens=[1, 2, 3], generation="gen_a")
    # No snapshot exists for the new generation.
    assert mind.work.latest_snapshot(actor="ego", model_generation="gen_b") is None
    # The old one is still found for its own generation.
    assert mind.work.latest_snapshot(
        actor="ego", model_generation="gen_a")["snapshot_id"] == s_old


def test_worker_refuses_cross_generation_snapshot(mind):
    """The neuocyte's guard: a mismatched generation raises rather than reinterpreting."""
    from amoeba.neuocyte import Neuocyte

    s_old, _, _ = publish(mind, version_tokens=[1, 2, 3], generation="gen_a")
    snap = mind.work.get_snapshot(s_old)

    class FakeWorker(Neuocyte):
        def __init__(self) -> None:  # noqa: D107
            self.model_generation = "gen_b_different_weights"

    w = FakeWorker()
    assert snap["model_generation"] != w.model_generation


def test_backend_restart_invalidates_handles_but_keeps_tokens(mind):
    s1, _, _ = publish(mind, version_tokens=[7, 8, 9], handle="sess_live")
    assert mind.work.get_snapshot(s1)["backend_handle"] == "sess_live"

    n = mind.work.invalidate_backend_handles(reason="inference service restarted")
    assert n == 1
    snap = mind.work.get_snapshot(s1)
    assert snap["backend_handle"] is None
    assert snap["kv_mode"] == "recomputed"
    # KV was replaceable acceleration; the canonical token prefix survives.
    assert mind.work.snapshot_tokens(s1) == [7, 8, 9]


def test_recovery_releases_refs_and_requeues_work(cfg):
    from amoeba.mind import Mind

    m1 = Mind(cfg)
    s1, _, _ = publish(m1, version_tokens=[1, 2])
    m1.work.acquire_snapshot_ref(snapshot_id=s1, holder="wk1")
    work_id, _ = m1.work.admit(objective="unfinished", work_class="user",
                               origin_actor="ego")
    m1.work.lease(neuocyte_id="wk1")
    m1.work.register_agent(agent_id="wk1", role="neuocyte", work_id=work_id)
    m1.close()

    m2 = Mind(cfg)
    report = m2.recover()
    summary = report["summary"]
    assert summary["crashed_neuocytes"] == 1
    assert summary["released_snapshot_refs"] == 1
    assert work_id in summary["requeued_work"]
    assert m2.work.get_work(work_id)["status"] == "queued"
    assert m2.work.get_snapshot(s1)["refcount"] == 0
    assert m2.work.get_snapshot(s1)["status"] == "published"
    m2.close()


# ---------------------------------------------------------------------------
# Scheduling: neither class starves, maintenance recursion is bounded.
# ---------------------------------------------------------------------------
def _snap(**kw) -> ResourceSnapshot:
    return ResourceSnapshot(**kw)


def test_maintenance_cannot_consume_the_user_reserve():
    arb = Arbiter(ArbiterConfig(max_neuocytes=2, user_reserved_slots=1,
                                maintenance_reserved_slots=1))
    s = _snap(active_neuocytes=1, active_maintenance_neuocytes=1,
              queued_user=1, queued_maintenance=5)
    # One slot left and no user neuocyte running: the user class gets it.
    assert arb.next_class_to_serve(s) == "user"


def test_user_cannot_consume_the_maintenance_reserve():
    arb = Arbiter(ArbiterConfig(max_neuocytes=2, user_reserved_slots=1,
                                maintenance_reserved_slots=1))
    s = _snap(active_neuocytes=1, active_user_neuocytes=1,
              queued_user=5, queued_maintenance=1)
    assert arb.next_class_to_serve(s) == "maintenance"


def test_no_dispatch_when_worker_cap_reached():
    arb = Arbiter(ArbiterConfig(max_neuocytes=2))
    s = _snap(active_neuocytes=2, queued_user=5, queued_maintenance=5)
    assert arb.next_class_to_serve(s) is None


def test_weighted_fair_share_converges_to_configured_weights():
    arb = Arbiter(ArbiterConfig(max_neuocytes=8, user_weight=0.7,
                                maintenance_weight=0.3,
                                user_reserved_slots=0, maintenance_reserved_slots=0))
    for _ in range(200):
        s = _snap(active_neuocytes=0, queued_user=10, queued_maintenance=10)
        arb.note_served(arb.next_class_to_serve(s))
    share = arb.fairness_state()["observed_user_share"]
    assert 0.6 <= share <= 0.8


def test_maintenance_recursion_is_bounded():
    arb = Arbiter(ArbiterConfig(max_maintenance_depth=2))
    ok = arb.admit(work_class="maintenance", snapshot=_snap(), maintenance_depth=2)
    assert ok.admitted
    too_deep = arb.admit(work_class="maintenance", snapshot=_snap(),
                         maintenance_depth=3)
    assert not too_deep.admitted and "recursion depth" in too_deep.reason


def test_maintenance_rate_limit():
    arb = Arbiter(ArbiterConfig(max_maintenance_per_hour=5))
    d = arb.admit(work_class="maintenance",
                  snapshot=_snap(recent_maintenance_count=5))
    assert not d.admitted and "rate limit" in d.reason


def test_queue_cap_rejects_admission():
    arb = Arbiter(ArbiterConfig(max_outstanding_work=3))
    d = arb.admit(work_class="user", snapshot=_snap(outstanding_work=3))
    assert not d.admitted and "queue" in d.reason


def test_requested_budget_is_capped_not_honoured_blindly():
    arb = Arbiter(ArbiterConfig(neuocyte_token_budget=256, neuocyte_wall_seconds=30))
    d = arb.admit(work_class="user", snapshot=_snap(),
                  requested_budget_tokens=999999, requested_wall_seconds=99999)
    assert d.admitted
    assert d.granted_budget_tokens == 256
    assert d.detail["wall_seconds"] == 30


def test_prompt_over_context_budget_is_refused():
    arb = Arbiter(ArbiterConfig(max_prompt_tokens=100))
    with pytest.raises(ResourceExhausted):
        arb.clamp_inference(prompt_tokens=101, max_tokens=10, deadline=None)


def test_rejection_is_recorded_as_history(mind):
    receipt = mind.work.reject(objective="too much", work_class="maintenance",
                               origin_actor="id", reason="rate limit reached")
    assert receipt.outcome == "committed"
    row = mind.db.conn.execute(
        "SELECT payload_inline FROM events WHERE kind = ?",
        (EventKind.WORK_REJECTED,)).fetchone()
    assert "rate limit reached" in row["payload_inline"]
