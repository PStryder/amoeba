"""Acceptance tests 1, 2, 3, 4, 14 against a running multi-process mind.

1.  Kill all disposable workers; state survives and unfinished work resumes.
2.  Restart Ego, Id, the inference service and the supervisor independently.
3.  Call Ego and Id directly through a real MCP client.
4.  Audit an Ego conclusion through Id.
14. Report concurrency evidence without conflating batching with kernel overlap.

These run on the deterministic backend so they exercise lifecycle and recovery
without a GPU. Output from that backend is a hash function, not inference, and
the tests assert that the system says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import PYTHON, ROOT, LiveStack, start_stack


# ---------------------------------------------------------------------------
# Test 1: killing every worker preserves state and resumes work.
# ---------------------------------------------------------------------------
def test_killing_all_workers_preserves_state_and_resumes_work(stack: LiveStack):
    mem = stack.call("remember", kind="belief", claim="survives worker death",
                     confidence=0.9, created_by="operator")
    version_before = mem["state_version"]

    admitted = [
        stack.call("admit_work", objective=f"long task {i}", work_class="user",
                   origin_actor="ego")
        for i in range(4)
    ]
    assert all(a["admitted"] for a in admitted)
    work_ids = [a["work_id"] for a in admitted]

    # Let the scheduler dispatch some of them.
    deadline = time.time() + 30
    while time.time() < deadline:
        stats = stack.call("queue_stats")
        if stats["by_status"].get("leased", 0) > 0:
            break
        time.sleep(0.3)

    killed = stack.call("kill_all_workers", reason="acceptance test")
    assert isinstance(killed["killed_workers"], list)

    # Durable state is intact.
    assert stack.call("get_memory", memory_id=mem["memory_id"])["claim"] == \
        "survives worker death"
    assert stack.call("status")["state_version"] >= version_before

    # Every work item still exists and none is silently lost.
    statuses = {w: stack.call("get_work", work_id=w)["status"] for w in work_ids}
    assert set(statuses) == set(work_ids)
    assert all(s in ("queued", "leased", "done", "failed") for s in statuses.values())

    # Unfinished work drains once workers come back.
    deadline = time.time() + 90
    while time.time() < deadline:
        remaining = [w for w in work_ids
                     if stack.call("get_work", work_id=w)["status"] in
                     ("queued", "leased")]
        if not remaining:
            break
        time.sleep(1)
    final = {w: stack.call("get_work", work_id=w)["status"] for w in work_ids}
    assert all(s in ("done", "failed") for s in final.values()), final
    # At least one produced a committed result rather than merely being dropped.
    assert any(stack.call("get_work", work_id=w).get("result") for w in work_ids)


def test_stale_worker_cannot_commit_after_replacement(stack: LiveStack):
    a = stack.call("admit_work", objective="fence me", work_class="user",
                   origin_actor="ego")
    work_id = a["work_id"]
    first = stack.call("lease_work", worker_id="ghost")
    if first is None or first["work_id"] != work_id:
        pytest.skip("scheduler leased a different item first")
    time.sleep(stack.cfg.arbiter.lease_seconds + 2)
    second = None
    deadline = time.time() + 30
    while time.time() < deadline and second is None:
        item = stack.call("lease_work", worker_id="replacement")
        if item and item["work_id"] == work_id:
            second = item
            break
        time.sleep(0.5)
    if second is None:
        pytest.skip("replacement lease not obtained in time")
    assert second["fencing_token"] > first["fencing_token"]
    with pytest.raises(Exception):
        stack.call("complete_work", work_id=work_id, worker_id="ghost",
                   fencing_token=first["fencing_token"], result={"stale": True})


# ---------------------------------------------------------------------------
# Test 2: independent restarts.
# ---------------------------------------------------------------------------
def _child_pid(stack: LiveStack, name: str) -> int:
    return stack.call("health")["children"][name]["reported"]["pid"]


def _wait_child(stack: LiveStack, name: str, *, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        child = stack.call("health")["children"].get(name, {})
        if child.get("reported"):
            return child["reported"]
        time.sleep(0.5)
    raise AssertionError(f"{name} did not come back")


@pytest.mark.parametrize("role", ["ego", "id"])
def test_role_restarts_independently(stack: LiveStack, role: str):
    before = _wait_child(stack, role)
    mem = stack.call("remember", kind="belief", claim=f"{role} restart marker",
                     confidence=0.7, created_by="operator")

    subprocess.run(["taskkill", "/PID", str(before["pid"]), "/T", "/F"],
                   capture_output=True, check=False)

    # Wait for the actual signal -- a strictly newer incarnation -- rather than
    # for the first differing pid. A restart is only complete once the new
    # process has registered, and sampling health before that races.
    after = None
    deadline = time.time() + 150
    while time.time() < deadline:
        candidate = stack.call("health")["children"].get(role, {}).get("reported")
        if candidate and candidate["incarnation"] > before["incarnation"]:
            after = candidate
            break
        time.sleep(0.5)
    assert after is not None, (
        f"{role} did not come back with a newer incarnation "
        f"(before={before['incarnation']})"
    )

    # A new incarnation, a new private session, same durable identity.
    assert after["pid"] != before["pid"]
    assert after["session_id"] != before["session_id"]
    assert stack.call("get_memory", memory_id=mem["memory_id"])["claim"] == \
        f"{role} restart marker"

    # The restart is in the record.
    kinds = [e["kind"] for e in stack.call("history", limit=400)]
    assert "agent.started" in kinds
    assert "agent.crashed" in kinds


def test_inference_restart_invalidates_handles_but_keeps_snapshots(stack: LiveStack):
    snap = stack.call("ensure_snapshot")
    snapshot_id = snap["snapshot_id"]
    tokens_before = stack.call("snapshot_tokens", snapshot_id=snapshot_id)
    assert tokens_before

    before = _wait_child(stack, "inference")
    subprocess.run(["taskkill", "/PID", str(before["pid"]), "/T", "/F"],
                   capture_output=True, check=False)

    after = None
    deadline = time.time() + 180
    while time.time() < deadline:
        candidate = stack.call("health")["children"].get("inference", {}).get("reported")
        if candidate and candidate["pid"] != before["pid"]:
            after = candidate
            break
        time.sleep(0.5)
    assert after is not None, "inference service was not restarted"
    assert after["incarnation"] > before["incarnation"]

    # KV handles died with the process; the canonical token prefix survives, so
    # the snapshot is still usable by recomputation.
    record = next(s for s in stack.call("list_snapshots")
                  if s["snapshot_id"] == snapshot_id)
    assert record["backend_handle"] is None
    assert record["kv_mode"] == "recomputed"
    assert stack.call("snapshot_tokens", snapshot_id=snapshot_id) == tokens_before
    assert stack.call("health")["status"] == "alive"


def test_supervisor_restart_recovers_state(tmp_path: Path):
    s1 = start_stack(tmp_path)
    try:
        mem = s1.call("remember", kind="belief", claim="crosses a supervisor restart",
                      confidence=0.95, created_by="operator")
        work = s1.call("admit_work", objective="unfinished across restart",
                       work_class="user", origin_actor="ego")
        version = s1.call("status")["state_version"]
        cfg_path = s1.cfg.source_path
    finally:
        s1.stop()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env["SYNTHETIC_MIND_STDERR_LOG"] = "0"
    proc = subprocess.Popen(
        [PYTHON, "-m", "synthetic_mind.supervisor", "--config", str(cfg_path)],
        env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    from synthetic_mind.rpc import wait_for_port

    assert wait_for_port(s1.cfg.supervisor_host, s1.cfg.supervisor_port, timeout=180)
    s2 = LiveStack(s1.cfg, proc)
    try:
        s2.wait_for_children(timeout=180)
        assert s2.call("get_memory", memory_id=mem["memory_id"])["claim"] == \
            "crosses a supervisor restart"
        assert s2.call("status")["state_version"] >= version
        # Unfinished work was requeued rather than lost.
        item = s2.call("get_work", work_id=work["work_id"])
        assert item["status"] in ("queued", "leased", "done")
        integ = s2.call("verify_integrity", deep=True)
        assert integ["hash_chain_ok"] is True
        assert integ["missing_content_count"] == 0
        kinds = [e["kind"] for e in s2.call("history", limit=500)]
        assert "supervisor.recovery" in kinds
    finally:
        s2.stop()


def test_second_supervisor_refuses_the_same_state_dir(stack: LiveStack):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    env["SYNTHETIC_MIND_STDERR_LOG"] = "0"
    out = subprocess.run(
        [PYTHON, "-m", "synthetic_mind.supervisor", "--config",
         str(stack.cfg.source_path)],
        env=env, cwd=str(ROOT), capture_output=True, timeout=120,
    )
    assert out.returncode != 0


# ---------------------------------------------------------------------------
# Test 4: Id audits an Ego conclusion through the record.
# ---------------------------------------------------------------------------
def test_id_audits_ego_conclusion_without_asking_ego(stack: LiveStack):
    turn = stack.call("ego_converse", message="State one fact about this mind.",
                      idempotency_key="audit-subject-1")
    conclusion_id = turn["result"]["conclusion_id"]

    audit = stack.call("id_audit", conclusion_id=conclusion_id,
                       focus="is the claim supported by recorded evidence?")
    result = audit["result"]
    assert result["asked_ego_to_defend_itself"] is False
    assert result["verdict"] in ("supported", "contested", "unsupported", "inconclusive")
    assert audit["receipt_id"] and audit["state_version"]

    # The audit resolved the conclusion to its original inputs and configuration.
    reviewed = result["evidence_reviewed"]
    assert reviewed["ego_consulted"] is False
    assert reviewed["resolved_from"] == "durable record only"
    assert reviewed["hash_chain_ok"] is True
    assert reviewed["operation_id"] == turn["operation_id"]
    kinds = [e["kind"] for e in reviewed["events"]]
    assert "input.received" in kinds and "conclusion.recorded" in kinds
    assert reviewed["conclusion"]["model_identity"]

    # The verdict is recorded durably.
    audits = stack.call("history", kinds=["audit.recorded"], limit=20)
    assert audits


def test_contested_audit_opens_a_disagreement_rather_than_overwriting(stack: LiveStack):
    concl = stack.call("record_conclusion", claim="an unsupported assertion",
                       produced_by="ego", evidence=[])
    before = stack.call("get_conclusion", conclusion_id=concl["conclusion_id"])
    assert before["review_status"] == "unreviewed"

    stack.call("record_audit", target_kind="conclusion",
               target_id=concl["conclusion_id"], verdict="contested",
               findings=["no evidence references"], actor="id")
    after = stack.call("get_conclusion", conclusion_id=concl["conclusion_id"])
    # The original claim is untouched; only its review status changed.
    assert after["claim"] == before["claim"]
    assert after["review_status"] == "audited_contested"

    stack.call("open_disagreement", subject_kind="conclusion",
               subject_id=concl["conclusion_id"], claim_a=before["claim"],
               actor_a="ego", claim_b="unsupported", actor_b="id")
    disagreements = stack.call("id_disagreements", scope="open")
    assert disagreements["result"]["count"] >= 1
    assert "majority-truth" in " ".join(disagreements["limitations"])


# ---------------------------------------------------------------------------
# Test 14: capability reporting does not overclaim.
# ---------------------------------------------------------------------------
def test_health_reports_capabilities_without_conflating_them(stack: LiveStack):
    health = stack.call("id_health")
    caps = health["capabilities"]
    assert caps["is_simulated"] is True
    assert caps["physical_overlap_verified"] is False
    assert caps["concurrency_mode"] == "serialized"
    assert caps["weight_ownership"] == "none_no_weights_loaded"
    assert caps["kv_mode"] == "simulated"
    assert "SIMULATED" in caps["banner"]
    # Limits and resource accounting are reported, not implied.
    assert health["limits"]["max_workers"] >= 1
    assert "active_workers" in health["resources"]


def test_health_stays_answerable_when_inference_is_down(stack: LiveStack):
    before = _wait_child(stack, "inference")
    subprocess.run(["taskkill", "/PID", str(before["pid"]), "/T", "/F"],
                   capture_output=True, check=False)
    # Immediately after the kill, health must still answer.
    health = stack.call("health")
    assert health["status"] == "alive"
    child = health["children"]["inference"]
    assert child["reported"] is None or child.get("error") or child["running"] is False
    _wait_child(stack, "inference", timeout=180)


def test_simulated_backend_is_labelled_on_every_cognitive_result(stack: LiveStack):
    turn = stack.call("ego_converse", message="anything", idempotency_key="sim-label-1")
    assert any("SIMULATED" in lim for lim in turn["limitations"])
    assert turn["result"]["is_simulated"] is True
    assert turn["result"]["answer"].startswith("[SIMULATED]")


def test_maintenance_workers_get_no_ego_snapshot(stack: LiveStack):
    admitted = stack.call("admit_work", objective="prune stale references",
                          work_class="maintenance", origin_actor="id")
    assert admitted["admitted"]
    work_id = admitted["work_id"]
    deadline = time.time() + 90
    while time.time() < deadline:
        item = stack.call("get_work", work_id=work_id)
        if item["status"] in ("done", "failed"):
            break
        time.sleep(1)
    item = stack.call("get_work", work_id=work_id)
    if item["status"] != "done":
        pytest.skip(f"maintenance work did not complete: {item['status']}")
    result = item["result"]
    assert result["kind"] == "maintenance_finding"
    assert result["received_ego_snapshot"] is False
    assert result["snapshot_id"] is None
    assert isinstance(result["state_references"], list)


def test_side_channel_signal_changes_no_state(stack: LiveStack):
    before = stack.call("status")["state_version"]
    ack = stack.call("side_channel", to_role="ego", kind="urgency",
                     payload={"level": "high"}, from_role="id")
    assert ack["accepted"] is True
    assert "no state changed" in ack["note"]
    assert stack.call("status")["state_version"] == before


# ---------------------------------------------------------------------------
# Test 3: a real MCP client calls both halves directly.
# ---------------------------------------------------------------------------
@pytest.mark.timeout(300)
def test_real_mcp_client_calls_both_halves(stack: LiveStack):
    import anyio

    async def run() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
                                             env.get("PYTHONPATH", "")])
        env["SYNTHETIC_MIND_STDERR_LOG"] = "0"
        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "synthetic_mind.mcp_api", "--config",
                  str(stack.cfg.source_path)],
            env=env, cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                ego = await session.call_tool(
                    "ego_converse",
                    {"message": "Name one invariant you maintain.",
                     "idempotency_key": "mcp-client-1"},
                )
                idh = await session.call_tool("id_health", {})
                recall = await session.call_tool("ego_recall", {"query": "invariant"})
                return {
                    "server_name": init.serverInfo.name,
                    "instructions": init.instructions or "",
                    "tool_names": names,
                    "ego": ego.structuredContent or json.loads(ego.content[0].text),
                    "id_health": idh.structuredContent
                    or json.loads(idh.content[0].text),
                    "recall": recall.structuredContent
                    or json.loads(recall.content[0].text),
                }

    out = anyio.run(run)

    assert out["server_name"] == "synthetic-mind"
    # Both halves are directly callable and discovery lists the cognitive verbs.
    for expected in ("ego_converse", "ego_investigate", "ego_recall", "ego_status",
                     "id_introspect", "id_health", "id_audit", "id_disagreements",
                     "id_maintenance"):
        assert expected in out["tool_names"], out["tool_names"]
    # No worker, cache or snapshot controls are exposed.
    assert not any(bad in name for name in out["tool_names"]
                   for bad in ("worker", "snapshot", "kv", "fork", "lease", "session"))

    ego = out["ego"]
    for field in ("schema_version", "operation_id", "status", "receipt_id",
                  "state_version", "result", "limitations"):
        assert field in ego, (field, ego)
    assert ego["schema_version"] == "1.0.0"
    assert ego["status"] == "completed"
    assert ego["operation_id"] and ego["receipt_id"]
    assert any("SIMULATED" in lim for lim in ego["limitations"])

    health = out["id_health"]["result"] if "result" in out["id_health"] \
        and out["id_health"].get("result") else out["id_health"]
    assert isinstance(health, dict)

    recall = out["recall"]
    assert "MAINTAINED" in " ".join(recall["limitations"])


@pytest.mark.timeout(300)
def test_mcp_client_disconnect_does_not_kill_the_mind(stack: LiveStack):
    import anyio

    before = stack.call("health")
    before_pids = {k: v["pid"] for k, v in before["children"].items()}

    async def run() -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
                                             env.get("PYTHONPATH", "")])
        env["SYNTHETIC_MIND_STDERR_LOG"] = "0"
        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "synthetic_mind.mcp_api", "--config",
                  str(stack.cfg.source_path)],
            env=env, cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("ego_status", {})
        # exiting the context terminates the facade process

    anyio.run(run)
    time.sleep(2)

    after = stack.call("health")
    assert after["status"] == "alive"
    assert {k: v["pid"] for k, v in after["children"].items()} == before_pids
    assert after["state_version"] >= before["state_version"]
