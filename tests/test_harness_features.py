"""Blackboard, sandboxed compute and homeostasis, through a live Harness.

These go over the real control plane against a running supervisor, so they
exercise the authority boundary as deployed: what a neuocyte can reach, what
only the Harness can do, and what gets a receipt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from conftest import PYTHON, ROOT, LiveStack


# ===========================================================================
# Cognitive blackboard
# ===========================================================================
def test_board_round_trip_over_the_control_plane(stack: LiveStack):
    a = stack.call("board_post", author="wk_1", author_kind="neuocyte",
                   post_type="finding", body="the knee is at 32 sessions",
                   title="scaling", confidence=0.8)
    assert a["receipt_id"] and a["board_naive"] is True

    read = stack.call("board_read", reader="wk_2", limit=10)
    assert read["count"] == 1
    assert read["reads_recorded"] is True
    assert "socially informed" in read["note"]
    cursor = read["cursor"]

    b = stack.call("board_post", author="wk_2", author_kind="neuocyte",
                   post_type="note", body="agreed", thread_id=a["thread_id"],
                   relations=[{"to_post": a["post_id"], "relation": "supports"}])
    assert b["board_naive"] is False        # wk_2 had read the board

    fresh = stack.call("board_read", reader="wk_3", since_seq=cursor, record=False)
    assert [p["post_id"] for p in fresh["posts"]] == [b["post_id"]]


def test_independence_and_corroboration_over_rpc(stack: LiveStack):
    claim = stack.call("board_post", author="wk_1", author_kind="neuocyte",
                       post_type="finding", body="prefix KV is shared")
    # wk_2 never reads: independent.
    indep = stack.call("board_post", author="wk_2", author_kind="neuocyte",
                       post_type="finding", body="measured: shared",
                       relations=[{"to_post": claim["post_id"],
                                   "relation": "supports"}])
    # wk_3 reads first: an echo.
    stack.call("board_read", reader="wk_3", limit=10)
    echo = stack.call("board_post", author="wk_3", author_kind="neuocyte",
                      post_type="note", body="concur",
                      relations=[{"to_post": claim["post_id"],
                                  "relation": "supports"}])

    verdict = stack.call("board_independence", post_a=claim["post_id"],
                         post_b=indep["post_id"])
    assert verdict["verdict"] == "independent"

    corr = stack.call("board_corroboration", post_id=claim["post_id"])
    assert corr["independent_support"] == [indep["post_id"]]
    assert corr["socially_informed_support"] == [echo["post_id"]]
    assert corr["independent_support_count"] == 1


def test_promotion_from_board_to_memory_is_an_explicit_harness_act(stack: LiveStack):
    p = stack.call("board_post", author="wk_1", author_kind="neuocyte",
                   post_type="finding", body="the pool taxes idle sessions",
                   confidence=0.9)
    # Posting alone changed no belief.
    assert stack.call("ego_recall", query="pool")["result"]["count"] == 0

    promoted = stack.call("board_promote_to_memory", post_id=p["post_id"],
                          actor="supervisor")
    assert promoted["memory_id"] and promoted["receipt_id"]
    assert "remains on the board" in promoted["note"]

    recalled = stack.call("ego_recall", query="idle sessions")["result"]["memories"]
    assert any("taxes idle sessions" in m["claim"] for m in recalled)
    # The post survives promotion with its own identity.
    assert stack.call("board_get_post", post_id=p["post_id"])["status"] == "open"


def test_promotion_records_how_much_support_was_real(stack: LiveStack):
    claim = stack.call("board_post", author="wk_1", author_kind="neuocyte",
                       post_type="finding", body="claim under test")
    stack.call("board_read", reader="wk_echo", limit=10)
    stack.call("board_post", author="wk_echo", author_kind="neuocyte", post_type="note",
               body="agreed", relations=[{"to_post": claim["post_id"],
                                          "relation": "supports"}])
    out = stack.call("board_promote_to_memory", post_id=claim["post_id"])
    assert out["independent_support"] == 0
    assert out["socially_informed_support"] == 1
    mem = stack.call("get_memory", memory_id=out["memory_id"])
    # Non-independent support is filed as opposing context, not as corroboration.
    assert any("not independent evidence" in (e["note"] or "")
               for e in mem["evidence"]["opposing"])


def test_board_posts_are_not_mind_state(stack: LiveStack):
    stack.call("board_post", author="wk_1", author_kind="neuocyte",
               post_type="finding", body="port is 9090")
    stack.call("board_post", author="wk_2", author_kind="neuocyte",
               post_type="finding", body="port is 8080")
    assert stack.call("ego_recall")["result"]["count"] == 0
    assert stack.call("board_stats")["posts"] == 2


# ===========================================================================
# Sandboxed compute
# ===========================================================================
def _sandbox_ok(stack: LiveStack) -> bool:
    return bool(stack.call("sandbox_capabilities").get("sandbox_available"))


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_sandbox_lifecycle_and_receipts(stack: LiveStack):
    if not _sandbox_ok(stack):
        pytest.skip("sandbox unavailable in this environment")
    created = stack.call("sandbox_create", owner="wk_1", work_id="work_1")
    sid = created["sandbox_id"]
    try:
        assert created["receipt_id"]
        assert created["capabilities"]["enforcement"] == "os_kernel"

        run = stack.call("sandbox_run", sandbox_id=sid, actor="wk_1",
                         code="print('computed', 6*7)")
        assert run["exit_code"] == 0 and "computed 42" in run["stdout"]
        assert run["receipt_id"]

        # The run is in the record.
        kinds = [e["kind"] for e in stack.call("history", limit=300)]
        assert "sandbox.created" in kinds and "sandbox.run" in kinds
    finally:
        out = stack.call("sandbox_destroy", sandbox_id=sid)
        assert out["receipt_id"]


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_sandbox_cannot_reach_the_host_through_the_harness(stack: LiveStack):
    if not _sandbox_ok(stack):
        pytest.skip("sandbox unavailable")
    sid = stack.call("sandbox_create", owner="wk_1")["sandbox_id"]
    try:
        net = stack.call("sandbox_run", sandbox_id=sid, actor="wk_1", code=(
            "import socket\n"
            "socket.create_connection(('1.1.1.1', 53), timeout=4)\n"
            "print('CONNECTED')\n"))
        assert "CONNECTED" not in net["stdout"]

        fs = stack.call("sandbox_run", sandbox_id=sid, actor="wk_1", code=(
            "import os; print('ENTRIES', len(os.listdir(r'C:\\\\Users')))"))
        assert "ENTRIES" not in fs["stdout"]
    finally:
        stack.call("sandbox_destroy", sandbox_id=sid)


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_artifact_promotion_is_proposal_then_harness_decision(stack: LiveStack):
    if not _sandbox_ok(stack):
        pytest.skip("sandbox unavailable")
    sid = stack.call("sandbox_create", owner="wk_1", work_id="work_1")["sandbox_id"]
    try:
        stack.call("sandbox_run", sandbox_id=sid, actor="wk_1", code=(
            "import json, pathlib\n"
            "pathlib.Path('checker.py').write_text('def check(x):\\n    return x > 0\\n')\n"
            "print('wrote checker')\n"))
        files = stack.call("sandbox_files", sandbox_id=sid)
        rel = next(f["path"] for f in files if f["path"].endswith("checker.py"))

        proposed = stack.call("artifact_propose", sandbox_id=sid, path=rel,
                              rationale="a deterministic validity check",
                              proposed_by="wk_1", work_id="work_1")
        assert proposed["status"] == "proposed"
        assert "nothing has been copied" in proposed["note"]

        workspace = Path(stack.cfg.workspace_dir)
        before = set(workspace.glob("*")) if workspace.exists() else set()

        promoted = stack.call("artifact_promote",
                              artifact_id=proposed["artifact_id"],
                              decided_by="supervisor")
        assert promoted["status"] == "promoted"
        assert promoted["sha256"] == proposed["sha256"]
        assert promoted["content_changed_since_proposal"] is False
        after = set(workspace.glob("*"))
        assert len(after) == len(before) + 1
        landed = (after - before).pop()
        assert "def check" in landed.read_text()

        kinds = [e["kind"] for e in stack.call("history", limit=400)]
        assert "artifact.proposed" in kinds and "artifact.promoted" in kinds
    finally:
        stack.call("sandbox_destroy", sandbox_id=sid)


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_the_harness_can_refuse_a_promotion(stack: LiveStack):
    if not _sandbox_ok(stack):
        pytest.skip("sandbox unavailable")
    sid = stack.call("sandbox_create", owner="wk_1")["sandbox_id"]
    try:
        stack.call("sandbox_write", sandbox_id=sid, path="work/notes.txt",
                   content="speculative")
        art = stack.call("artifact_propose", sandbox_id=sid, path="work/notes.txt",
                         rationale="maybe useful", proposed_by="wk_1")
        out = stack.call("artifact_reject", artifact_id=art["artifact_id"],
                         reason="not reproducible")
        assert out["status"] == "rejected" and out["receipt_id"]
        listing = stack.call("artifact_list", status="rejected")
        assert any(a["artifact_id"] == art["artifact_id"] for a in listing)
    finally:
        stack.call("sandbox_destroy", sandbox_id=sid)


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_path_escape_through_the_harness_is_refused(stack: LiveStack):
    if not _sandbox_ok(stack):
        pytest.skip("sandbox unavailable")
    sid = stack.call("sandbox_create", owner="wk_1")["sandbox_id"]
    try:
        for bad in ("../../escape.txt", "C:\\Windows\\System32\\drivers\\etc\\hosts"):
            with pytest.raises(Exception):
                stack.call("sandbox_write", sandbox_id=sid, path=bad, content="x")
    finally:
        stack.call("sandbox_destroy", sandbox_id=sid)


# ===========================================================================
# Context homeostasis
# ===========================================================================
def test_context_report_is_measured_not_estimated(stack: LiveStack):
    r = stack.call("context_report")
    assert r["backend_available"] is True
    assert r["pool_capacity"] > 0
    assert r["pressure"] in ("nominal", "elevated", "high", "critical")
    assert isinstance(r["sessions"], list)


def test_assessment_performs_no_action(stack: LiveStack):
    before = stack.call("context_report")
    out = stack.call("context_assess")
    after = stack.call("context_report")
    assert "performs no action" in out["note"]
    assert after["sessions"] and len(after["sessions"]) == len(before["sessions"])


def test_id_health_surfaces_context_board_and_sandbox(stack: LiveStack):
    h = stack.call("id_health")
    assert "context" in h and "report" in h["context"]
    assert "board" in h and "posts" in h["board"]
    assert "sandbox" in h and "sandbox_available" in h["sandbox"]


def test_id_can_request_rejuvenation_and_the_harness_decides(stack: LiveStack):
    out = stack.call("request_rejuvenation", role="id",
                     reason="context growing", requested_by="id")
    assert "performed" in out
    kinds = [e["kind"] for e in stack.call("history", limit=400)]
    assert "rejuvenation.requested" in kinds
    if out["performed"]:
        assert out["new_session_id"] != out["old_session_id"]
        assert "verbatim" in out["reconstitution"]
        assert "rejuvenation.performed" in kinds
    else:
        assert out["refused_because"]
        assert "rejuvenation.refused" in kinds


def test_summarise_mode_is_refused_through_the_harness(stack: LiveStack):
    with pytest.raises(Exception):
        stack.call("context_rejuvenate", role="ego", reason="x", mode="summarise")


# ===========================================================================
# Board-naive neuocytes: the experiment the read tracking exists for
# ===========================================================================
def test_work_can_be_admitted_board_naive(stack: LiveStack):
    stack.call("board_post", author="wk_seed", author_kind="neuocyte",
               post_type="finding", body="a seeded finding others might echo")
    admitted = stack.call("admit_work", objective="independently determine X",
                          work_class="user", origin_actor="ego",
                          board_access="none")
    assert admitted["admitted"] and admitted["board_access"] == "none"
    item = stack.call("get_work", work_id=admitted["work_id"])
    assert item["board_access"] == "none"


def test_a_naive_worker_posts_without_having_read_the_board(stack: LiveStack):
    stack.call("board_post", author="wk_seed", author_kind="neuocyte",
               post_type="finding", body="seeded claim")
    admitted = stack.call("admit_work", objective="determine the answer",
                          work_class="user", origin_actor="ego",
                          board_access="none")
    work_id = admitted["work_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        item = stack.call("get_work", work_id=work_id)
        if item["status"] in ("done", "failed"):
            break
        time.sleep(1)
    item = stack.call("get_work", work_id=work_id)
    if item["status"] != "done":
        pytest.skip(f"neuocyte did not complete: {item['status']}")
    result = item["result"]
    assert result["board_access"] == "none"
    assert result["board_posts_seen"] == []
    # board_access 'none' also means it does not publish.
    assert result["board_post_id"] is None
