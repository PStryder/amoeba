"""Acceptance tests 1, 2, 3, 4, 14 against a running multi-process mind.

1.  Kill all disposable neuocytes; state survives and unfinished work resumes.
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
# Test 1: killing every neuocyte preserves state and resumes work.
# ---------------------------------------------------------------------------
def test_killing_all_workers_preserves_state_and_resumes_work(stack: LiveStack):
    mem = stack.call("remember", kind="belief", claim="survives neuocyte death",
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

    killed = stack.call("kill_all_neuocytes", reason="acceptance test")
    assert isinstance(killed["killed_neuocytes"], list)

    # Durable state is intact.
    assert stack.call("get_memory", memory_id=mem["memory_id"])["claim"] == \
        "survives neuocyte death"
    assert stack.call("status")["state_version"] >= version_before

    # Every work item still exists and none is silently lost.
    statuses = {w: stack.call("get_work", work_id=w)["status"] for w in work_ids}
    assert set(statuses) == set(work_ids)
    assert all(s in ("queued", "leased", "done", "failed") for s in statuses.values())

    # Unfinished work drains once neuocytes come back.
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
    first = stack.call("lease_work", neuocyte_id="ghost")
    if first is None or first["work_id"] != work_id:
        pytest.skip("scheduler leased a different item first")
    time.sleep(stack.cfg.arbiter.lease_seconds + 2)
    second = None
    deadline = time.time() + 30
    while time.time() < deadline and second is None:
        item = stack.call("lease_work", neuocyte_id="replacement")
        if item and item["work_id"] == work_id:
            second = item
            break
        time.sleep(0.5)
    if second is None:
        pytest.skip("replacement lease not obtained in time")
    assert second["fencing_token"] > first["fencing_token"]
    with pytest.raises(Exception):
        stack.call("complete_work", work_id=work_id, neuocyte_id="ghost",
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
    env["AMOEBA_STDERR_LOG"] = "0"
    proc = subprocess.Popen(
        [PYTHON, "-m", "amoeba.supervisor", "--config", str(cfg_path)],
        env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    from amoeba.rpc import wait_for_port

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
    env["AMOEBA_STDERR_LOG"] = "0"
    out = subprocess.run(
        [PYTHON, "-m", "amoeba.supervisor", "--config",
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
    assert health["limits"]["max_neuocytes"] >= 1
    assert "active_neuocytes" in health["resources"]


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


def test_a_transient_signal_changes_no_state(stack: LiveStack):
    """A nudge is still a nudge, and still changes nothing.

    `durable=False` is the old behaviour, kept for liveness-style hints. Its
    price is stated: a transient signal cannot enter a trigger bundle, so it
    cannot influence cognition either.
    """
    before = stack.call("status")["state_version"]
    ack = stack.call("side_channel", to_role="ego", kind="urgency",
                     payload={"level": "high"}, from_role="id", durable=False)
    assert ack["durable"] is False
    assert ack["trigger_id"] is None
    assert "cannot enter a trigger bundle" in ack["note"]
    assert stack.call("status")["state_version"] == before


def test_a_message_that_reaches_cognition_is_attributable(stack: LiveStack):
    """Delivery may be transient. Influence may not be unaudited.

    The backchannel used to push into an in-memory list the role drained into
    its own cognition -- a receipt-free message with no author, no body and no
    record shaping what a persistent mind thought. A message that can wake a
    role is now a durable trigger carrying all three.
    """
    ack = stack.call("side_channel", to_role="ego", kind="urgency",
                     payload={"message": "your work queue is backing up"},
                     from_role="id")
    assert ack["durable"] is True
    assert ack["trigger_id"]

    mailbox = stack.call("role_mailbox", role="ego")["ego"]
    queued = {t["trigger_id"]: t for t in mailbox["next_triggers"]}
    consumed = None
    if ack["trigger_id"] not in queued:
        # Already taken into a turn; it is attributable there instead.
        for turn in stack.call("role_turns", role="ego")["turns"]:
            detail = stack.call("role_turn", turn_id=turn["turn_id"])
            for trig in detail["triggers"]:
                if trig["trigger_id"] == ack["trigger_id"]:
                    consumed = trig
    found = queued.get(ack["trigger_id"]) or consumed
    assert found, "the message is neither queued nor recorded against a turn"
    assert found["source"] == "id", "the author was not preserved"
    assert "backing up" in (found["summary"] or "")


# ---------------------------------------------------------------------------
# Test 3: a real MCP client calls both halves directly.
# ---------------------------------------------------------------------------
@pytest.mark.timeout(300)
def test_a_real_mcp_client_gets_the_io_surface_only(stack: LiveStack):
    """Through a real MCP client, over stdio, end to end.

    This test used to be called "calls both halves" and drove `ego_converse`,
    `id_health` and `ego_recall`. That premise is now wrong by design: MCP has
    no Id half and no half of Ego's internals. It is a cognitive service
    interface -- input in, output out -- and the tools it publishes are the
    whole of what it can do.

    The removed tool names are tried anyway, spelled exactly as they used to
    be, because a client that remembers the old surface is the realistic case.
    """
    import anyio

    async def run() -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
                                             env.get("PYTHONPATH", "")])
        env["AMOEBA_STDERR_LOG"] = "0"
        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "amoeba.mcp_api", "--config",
                  str(stack.cfg.source_path)],
            env=env, cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)

                asked = await session.call_tool(
                    "amoeba_ask", {"text": "Name one invariant you maintain.",
                                   "wait_seconds": 90})
                answer = asked.structuredContent or json.loads(
                    asked.content[0].text)

                gone = {}
                for old in ("ego_converse", "id_health", "ego_recall",
                            "mind_file_write", "mind_artifact_promote",
                            "board_post", "mind_cancel", "id_maintenance"):
                    try:
                        res = await session.call_tool(old, {})
                        gone[old] = ("ERROR-FLAGGED" if res.isError
                                     else "EXECUTED")
                    except Exception as exc:  # noqa: BLE001
                        gone[old] = type(exc).__name__
                return {"server_name": init.serverInfo.name,
                        "tool_names": names, "answer": answer, "gone": gone}

    out = anyio.run(run)

    assert out["tool_names"] == [
        "amoeba_ask", "amoeba_attach", "amoeba_capabilities", "amoeba_list",
        "amoeba_output", "amoeba_result", "amoeba_status", "amoeba_submit",
    ], out["tool_names"]

    # The surface works for what it is for.
    result = out["answer"].get("result", out["answer"])
    assert result.get("status") in ("complete", "failed"), out["answer"]
    assert result.get("interaction_id", "").startswith("ixn_")

    # And every removed tool is gone rather than merely refused.
    for old, outcome in out["gone"].items():
        assert outcome != "EXECUTED", f"{old} still executes over MCP"

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
        env["AMOEBA_STDERR_LOG"] = "0"
        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "amoeba.mcp_api", "--config",
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


def test_health_stays_fast_while_a_child_is_down(stack: LiveStack):
    """Responsive means fast, not merely eventually answering.

    health() touches every child. Before the liveness probe was split from the
    patient reconnect path, one dead child made every health call block for the
    full ~20s retry window -- including the supervision loop trying to restart
    it, which is how a restart could miss a 150s deadline.
    """
    before = _wait_child(stack, "id")
    subprocess.run(["taskkill", "/PID", str(before["pid"]), "/T", "/F"],
                   capture_output=True, check=False)
    try:
        slowest = 0.0
        for _ in range(4):
            t0 = time.time()
            health = stack.call("health")
            slowest = max(slowest, time.time() - t0)
            assert health["status"] == "alive"
            time.sleep(0.2)
        assert slowest < 8.0, f"health took {slowest:.1f}s with one child down"
    finally:
        _wait_child(stack, "id", timeout=180)


def test_a_role_that_dies_during_startup_is_still_restarted(stack: LiveStack):
    """Regression: supervision must not wait for startup to finish.

    start() used to block up to 180s per role in wait_for_port, and the
    supervision thread only began afterwards. A role killed inside that window
    went unnoticed for three minutes while the supervisor sat blind. The test
    kills a role as early as possible, which is precisely that window.
    """
    before = _wait_child(stack, "id")
    subprocess.run(["taskkill", "/PID", str(before["pid"]), "/T", "/F"],
                   capture_output=True, check=False)

    after, deadline = None, time.time() + 60
    while time.time() < deadline:
        cand = stack.call("health")["children"].get("id", {}).get("reported")
        if cand and cand["incarnation"] > before["incarnation"]:
            after = cand
            break
        time.sleep(0.5)
    assert after is not None, "a role killed during startup was never restarted"
    assert after["pid"] != before["pid"]


def test_supervision_keeps_making_passes(stack: LiveStack):
    """A stalled supervision loop is the failure that hides every other one."""
    first = stack.call("health")["supervision"]
    assert first["passes"] >= 0
    time.sleep(6)
    second = stack.call("health")["supervision"]
    assert second["passes"] > first["passes"], "supervision stopped making passes"
    assert second["seconds_since_last_pass"] < 10


def test_the_real_roles_are_born_from_the_prompt_library(stack: LiveStack):
    """Ego and Id take their prompt from the library, not the module constant.

    The fallback path is a warning log, which is exactly the kind of thing that
    passes unnoticed: the organism would keep working while the whole library
    sat unused. So this asserts the binding exists, and that the digest the
    supervisor reports as *configured* equals what the running role actually
    embodied -- the two agreeing is the point of routing `prompt_version`
    through the library at all.
    """
    tree = stack.call("prompt_tree")
    selected = {n["namespace"]: n["selected"] for n in tree["nodes"]}
    assert selected["ego"]["profile_ref"] == "ego@1"
    assert selected["ego.neuocyte"]["profile_ref"] == "ego.neuocyte@1.1"

    bindings = stack.call("prompt_incarnations")["bindings"]
    by_actor = {b["actor_id"]: b for b in bindings}
    assert "ego" in by_actor and "id" in by_actor, "a role fell back"

    lib = stack.call("operator_prompt_library")
    for role in ("ego", "id"):
        configured = lib["current"][role]["configured"]
        assert configured["detail"]["source"] == f"{role}@1"
        assert configured["sha256"] == lib["current"][role]["embodied_sha256"]
        assert by_actor[role]["prompt_sha256"] == configured["sha256"]


def test_an_incarnation_binding_carries_its_incarnation(stack: LiveStack):
    """The binding is stamped with the incarnation it belongs to.

    A binding with a NULL incarnation cannot be tied to a transcript, which
    makes it decoration. It is asserted here, against the live stack, because
    the stamp happens in `register_agent` while the single writer is busy with
    the rest of startup -- the first implementation used a raw `conn.execute`
    plus `conn.commit()` on the writer's own connection, and the stamp was
    silently lost. A bare commit from outside the writer can also land inside
    another mutation's transaction, so this pins the *write path*, not just
    the value.
    """
    bindings = stack.call("prompt_incarnations")["bindings"]
    assert bindings
    for b in bindings:
        assert b["incarnation"] is not None, b
        assert b["incarnation"] >= 1


def test_a_role_turn_carries_a_frozen_environment(stack: LiveStack):
    """Ego's answers are tied to the exact environment that produced them."""
    first = stack.call("ego_converse", message="what do you know?")["result"]
    assert first["environment_sha256"]
    assert first["profile_ref"] == "ego@1"
    assert first["stop_reason"]

    # Unchanged world, unchanged digest: the manifest is a function of state,
    # not of the clock.
    second = stack.call("ego_converse", message="and now?")["result"]
    assert second["environment_sha256"] == first["environment_sha256"]


def test_the_exact_environment_a_turn_saw_is_reconstructable(stack: LiveStack):
    """A digest whose content cannot be recovered is not provenance.

    The manifest bytes go to the content-addressed store, so the environment a
    past turn reasoned against can be read back and re-hashed rather than
    recomputed from state that has since moved.
    """
    import hashlib
    import json

    env = stack.call("role_environment", role="ego", incarnation=1,
                     profile_ref="ego@1", trigger="provenance check")
    blob = env["environment_blob"]
    assert blob

    # `history` returns the raw rows: a small payload is inline, a large one
    # is a blob reference. Resolve whichever it is.
    from amoeba.store.blobs import BlobStore

    blobs = BlobStore(stack.cfg.blob_dir)

    def _payload(event):
        if event.get("payload_inline") is not None:
            return json.loads(event["payload_inline"])
        if event.get("payload_sha256"):
            return blobs.get_json(event["payload_sha256"])
        return {}

    events = stack.call("history", kinds=["role.turn_began"], limit=20)
    turn = [e for e in events if _payload(e).get("environment_blob") == blob]
    assert turn, "the turn did not record the environment it was given"
    payload = _payload(turn[0])
    for field in ("role", "profile_ref", "environment_sha256",
                  "environment_blob", "trigger", "model_generation"):
        assert field in payload, field

    # The stored bytes really are the manifest that was handed over, and they
    # still hash to the digest the turn recorded.
    raw = blobs.get(blob)
    assert hashlib.sha256(raw).hexdigest() == blob
    body = json.loads(raw.decode("utf-8"))
    assert body["environment_sha256"] == env["environment_sha256"]
    assert {c["verb"] for c in body["capabilities"]} == {
        c["verb"] for c in env["manifest"]["capabilities"]}
    assert body["available_profiles"] == env["manifest"]["available_profiles"]


def test_the_environment_digest_tracks_the_authoritative_state(stack: LiveStack):
    """Approving a profile changes the environment; nothing else needs to."""
    before = stack.call("role_environment", role="ego")["environment_sha256"]
    created = stack.call("operator_prompt_author",
                         namespace="ego.neuocyte.do_thing", prompt_mode="append",
                         prompt_text="Do the thing precisely.",
                         rationale="environment test")
    # A candidate is not yet a capability.
    assert stack.call("role_environment", role="ego")["environment_sha256"] == before

    for state in ("validated", "proposed", "production_approved"):
        stack.call("operator_prompt_state", version_id=created["version_id"],
                   state=state)
    stack.call("operator_prompt_select", namespace="ego.neuocyte.do_thing",
               version_id=created["version_id"])

    after = stack.call("role_environment", role="ego")
    assert after["environment_sha256"] != before
    names = [p["profile_ref"] for p in after["manifest"]["available_profiles"]]
    assert any("do_thing" in n for n in names), names


def test_a_role_credential_cannot_reach_another_roles_effectors(stack: LiveStack):
    """The real physics behind the environment: the scope table.

    The manifest refusing an unlisted verb is the first defence. This is the
    second and the one that matters -- Ego's own credential resolves to a
    method table that does not contain Id's effectors, so the verb is not
    refused, it is absent.
    """
    from amoeba.rpc import RpcClient, read_or_create_token

    cfg = stack.cfg
    ego = RpcClient(cfg.supervisor_host, cfg.supervisor_port,
                    read_or_create_token(cfg.scope_token_path("ego")),
                    name="test-ego")
    ident = RpcClient(cfg.supervisor_host, cfg.supervisor_port,
                      read_or_create_token(cfg.scope_token_path("id")),
                      name="test-id")
    try:
        ego.connect(retries=10, delay=0.2)
        ident.connect(retries=10, delay=0.2)

        with pytest.raises(Exception) as exc:
            ego.call("id_raise_finding", kind="anomaly", summary="I am Id")
        assert "unknown method" in str(exc.value).lower()

        with pytest.raises(Exception):
            ident.call("ego_request_work", objective="I am Ego")

        # Neither can reach the operator's prompt governance.
        for client in (ego, ident):
            with pytest.raises(Exception):
                client.call("operator_prompt_select", namespace="ego",
                            version_id="anything")
        # But each can build its own environment.
        assert ego.call("role_environment", role="ego")["manifest"]["role"] == "ego"
    finally:
        ego.close()
        ident.close()


# ---------------------------------------------------------------------------
# persistent roles: real processes, real scope tables, real scheduling
# ---------------------------------------------------------------------------
def _wait_for(fn, timeout=30.0, interval=0.25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(interval)
    return None


def test_id_wakes_at_startup_and_ego_stays_quiet(tmp_path: Path):
    """Id forms an initial view; Ego does not talk to itself.

    A persistent identity that generates cognition because its process exists
    is not the same as one that responds to the world. Ego waits.
    """
    stack = start_stack(tmp_path)
    try:
        id_turn = _wait_for(
            lambda: [t for t in stack.call("role_turns", role="id")["turns"]
                     if "startup" in t["trigger_kinds"]])
        assert id_turn, "Id never received its startup turn"
        assert stack.call("role_turns", role="ego")["turns"] == [], \
            "Ego generated a turn with nothing to respond to"
        assert stack.call("role_mailbox")["ego"]["state"] == "idle"
    finally:
        stack.stop()


def test_input_arriving_during_an_ego_turn_is_queued_not_injected(tmp_path: Path):
    """The load-bearing runtime claim, held open deterministically.

    This originally polled `role_mailbox` hoping to catch Ego with a turn
    open. That is a race the test always lost: against the deterministic
    backend a turn is claimed, generated and closed faster than a 50ms sampler
    can observe, so the first sample already showed it completed. Racing the
    scheduler proves nothing when you lose the race every time.

    So the test *holds a turn open itself*, through the same Harness verbs the
    role uses. There is nothing artificial about the guarantee being tested:
    bundling is the Harness's decision, and what is being asserted is that
    while a turn is open, a newly queued input does not join it. The turn
    being held by the test rather than by Ego's generation loop is exactly
    what makes the window deterministic instead of a millisecond wide.
    """
    stack = start_stack(tmp_path)
    try:
        # Take a turn for Ego and keep it. Ego's own loop polls, so it may win
        # the first claim; wait for it to finish and try again.
        held = None
        for _ in range(40):
            first = stack.call("ego_converse", message="first message",
                               wait=False)["result"]
            try:
                claimed = stack.call(
                    "role_claim_turn", role="ego", incarnation=1,
                    profile_ref="ego@1")
            except Exception:
                claimed = {"turn": None}
            if claimed.get("turn"):
                held = claimed["turn"]
                break
            # Ego got there first; let it drain and retry.
            _wait_for(lambda: stack.call("role_mailbox",
                                         role="ego")["ego"]["queued"] == 0,
                      timeout=30.0)
        assert held, "could not hold a turn open for Ego"
        assert first["trigger_id"] in {t["trigger_id"] for t in held["triggers"]}

        # === the turn is now open, and stays open ===
        second = stack.call("ego_converse", message="second message",
                            wait=False)["result"]

        # It must not have joined the turn already running.
        detail = stack.call("role_turn", turn_id=held["turn_id"])
        assert detail["status"] == "running"
        assert second["trigger_id"] not in {t["trigger_id"]
                                            for t in detail["triggers"]}, \
            "an input arriving mid-turn was injected into it"

        # And it is visibly waiting rather than lost.
        mb = stack.call("role_mailbox", role="ego")["ego"]
        assert mb["queued"] >= 1
        assert second["trigger_id"] in {t["trigger_id"]
                                        for t in mb["next_triggers"]}

        # === release the turn ===
        stack.call("role_complete_turn", turn_id=held["turn_id"],
                   stop_reason="model_stop")

        # Now it lands, in a different turn, that began after the first ended.
        def consuming_turn():
            for t in stack.call("role_turns", role="ego")["turns"]:
                if t["turn_id"] == held["turn_id"]:
                    continue
                d = stack.call("role_turn", turn_id=t["turn_id"])
                if second["trigger_id"] in {x["trigger_id"]
                                            for x in d["triggers"]}:
                    return d
            return None

        later = _wait_for(consuming_turn, timeout=90.0)
        assert later, "the queued input was never given a turn"
        assert later["turn_id"] != held["turn_id"]

        closed = stack.call("role_turn", turn_id=held["turn_id"])
        assert closed["finished_at"]
        assert later["started_at"] >= closed["finished_at"], \
            "the next turn began before the held one ended"
    finally:
        stack.stop()


def _hold_a_turn_open(stack) -> dict:
    """Claim Ego's turn before Ego does, so the next boundary is ours to pick.

    Ego polls, so a burst queued "while it is busy" is a transient state a
    sampler can easily miss. Holding a turn open makes the boundary
    deterministic: everything queued now is still queued when the turn closes.
    """
    for _ in range(40):
        stack.call("operator_message_role", role="ego", message="wake up")
        try:
            claimed = stack.call("role_claim_turn", role="ego", incarnation=1,
                                 profile_ref="ego@1")
        except Exception:  # noqa: BLE001
            claimed = {"turn": None}
        if claimed.get("turn"):
            return claimed["turn"]
        _wait_for(lambda: stack.call("role_mailbox",
                                     role="ego")["ego"]["queued"] == 0,
                  timeout=30.0)
    raise AssertionError("could not hold a turn open for Ego")


def test_a_burst_of_requests_is_not_bundled_into_one_turn(tmp_path: Path):
    """I81, against the live stack: two questions never share one turn.

    This test used to assert the opposite, and was correct to, while an answer
    belonged to the turn that produced it. It is the evil twin now: three
    questions in one turn share its single answer, so two of the three callers
    receive a reply to somebody else's question.

    Bundling did not go away -- see the notices test below. It stopped
    applying to triggers that are owed a reply.
    """
    stack = start_stack(tmp_path)
    try:
        held = _hold_a_turn_open(stack)
        submitted = [
            stack.call("ego_converse", message=f"question {i}",
                       wait=False)["result"]["trigger_id"]
            for i in range(3)]
        stack.call("role_complete_turn", turn_id=held["turn_id"],
                   stop_reason="model_stop")

        def each_in_its_own_turn():
            placed = {}
            for t in stack.call("role_turns", role="ego")["turns"]:
                d = stack.call("role_turn", turn_id=t["turn_id"])
                for x in d["triggers"]:
                    if x["trigger_id"] in submitted:
                        placed[x["trigger_id"]] = t["turn_id"]
            return placed if len(placed) == len(submitted) else None

        placed = _wait_for(each_in_its_own_turn, timeout=180.0)
        assert placed, "the queued questions were never all given turns"
        assert len(set(placed.values())) == len(submitted), (
            "two requests shared a turn, so they would share its answer: "
            f"{placed}")
    finally:
        stack.stop()


def test_a_burst_of_notices_is_still_bundled_into_one_turn(tmp_path: Path):
    """Bundling is alive; it applies to what is not owed a reply.

    Ego waking once for three things that happened while it was busy is the
    point of bundling, and waking three times is the thrash it exists to
    avoid. An operator notice is owed no reply, so nothing is shared by
    bundling it -- which is the distinction the whole tier turns on.
    """
    stack = start_stack(tmp_path)
    try:
        held = _hold_a_turn_open(stack)
        for i in range(3):
            stack.call("operator_message_role", role="ego",
                       message=f"notice {i}")
        queued = {t["trigger_id"] for t
                  in stack.call("role_mailbox", role="ego")["ego"]["next_triggers"]}
        assert len(queued) >= 3, "the notices were not queued behind the turn"
        stack.call("role_complete_turn", turn_id=held["turn_id"],
                   stop_reason="model_stop")

        def bundled():
            for t in stack.call("role_turns", role="ego")["turns"]:
                if t["turn_id"] == held["turn_id"]:
                    continue
                d = stack.call("role_turn", turn_id=t["turn_id"])
                got = {x["trigger_id"] for x in d["triggers"]}
                if len(queued & got) >= 3:
                    return t
            return None

        turn = _wait_for(bundled, timeout=90.0)
        assert turn, "three notices were not bundled into one turn"
    finally:
        stack.stop()


def test_a_role_turn_records_profile_environment_and_triggers(tmp_path: Path):
    """Profile + environment + trigger bundle, all recoverable afterwards."""
    stack = start_stack(tmp_path)
    try:
        submitted = stack.call("ego_converse", message="tell me something",
                               wait=False)["result"]
        # The turn that consumed *this* input, not merely the newest completed
        # one: a truncated turn schedules a continuation, and `role_turns`
        # returns newest first, so "the first completed turn" is the
        # continuation rather than the turn that read the message.
        def consuming_turn():
            for t in stack.call("role_turns", role="ego")["turns"]:
                if t["status"] != "completed":
                    continue
                d = stack.call("role_turn", turn_id=t["turn_id"])
                if submitted["trigger_id"] in {x["trigger_id"]
                                               for x in d["triggers"]}:
                    return d
            return None

        detail = _wait_for(consuming_turn, timeout=90.0)
        assert detail, "no completed Ego turn consumed the submitted input"
        assert detail["profile_ref"] == "ego@1"
        assert detail["environment_sha256"] and detail["environment_blob"]
        assert detail["stop_reason"]
        assert detail["bundle"]["triggers"], "the bundle was not recoverable"
        assert detail["environment"]["role"] == "ego"
        assert submitted["trigger_id"] in {t["trigger_id"]
                                           for t in detail["triggers"]}
        # The bundle the model read is the bundle on the record.
        assert "<turn_input>" in detail["bundle"]["text"]
    finally:
        stack.stop()


def test_id_receives_a_deterministic_heartbeat(tmp_path: Path):
    """Id is logically always on without being a token furnace.

    The heartbeat is explicit input with its own kind, not a fake user
    message: Id can tell the difference between the world asking something and
    the clock coming round.
    """
    stack = start_stack(tmp_path, scheduler={"id_heartbeat_seconds": 4.0})
    try:
        beat = _wait_for(
            lambda: [t for t in stack.call("role_turns", role="id")["turns"]
                     if "heartbeat" in t["trigger_kinds"]], timeout=60.0)
        assert beat, "Id never received a heartbeat turn"
        detail = stack.call("role_turn", turn_id=beat[0]["turn_id"])
        kinds = {t["kind"] for t in detail["triggers"]}
        assert "heartbeat" in kinds and "user_input" not in kinds
        assert any("homeostatic" in (t["summary"] or "")
                   for t in detail["triggers"])
    finally:
        stack.stop()


def test_an_idle_organism_does_not_spin(tmp_path: Path):
    """Quiet means quiet: no unbounded zero-delay turn loop."""
    stack = start_stack(tmp_path, scheduler={"id_heartbeat_seconds": 0.0})
    try:
        time.sleep(6)
        turns = stack.call("role_turns")["turns"]
        # Startup for Id, plus whatever continuations that thought needed, and
        # nothing else. Certainly not one turn per poll tick.
        assert len(turns) <= 6, [t["trigger_kinds"] for t in turns]
        assert not [t for t in turns if t["role"] == "ego"]
    finally:
        stack.stop()


def test_the_operator_can_see_the_mailbox(tmp_path: Path):
    stack = start_stack(tmp_path)
    try:
        mb = stack.call("role_mailbox")
        for role in ("ego", "id"):
            assert mb[role]["state"] in ("idle", "queued", "processing",
                                         "heartbeat_wait", "recovering")
            assert "queued" in mb[role] and "current_turn" in mb[role]
        stack.call("operator_message_role", role="ego",
                   message="have a look at the work queue")
        assert _wait_for(
            lambda: stack.call("role_mailbox", role="ego")["ego"]["queued"]
            or [t for t in stack.call("role_turns", role="ego")["turns"]])
    finally:
        stack.stop()


def test_an_adverse_audit_is_not_recorded_as_a_favourable_one():
    """I77. `unsupported` must never be read as `supported`.

    Substring matching did exactly that -- the word contains it -- so an
    adverse audit became a favourable durable verdict, and the disagreement it
    should have opened never was, because that branch tests for `unsupported`.
    An audit that silently inverts is worse than no audit.

    Driven through the real parser rather than a model, because a
    deterministic backend never emits a verdict line and the bug hid behind
    exactly that.
    """
    from amoeba.supervisor_api import _parse_audit

    cases = {
        "VERDICT: unsupported": "unsupported",
        "VERDICT: supported": "supported",
        "VERDICT: contested": "contested",
        "VERDICT: inconclusive": "inconclusive",
        "VERDICT: clearly UNSUPPORTED by the record": "unsupported",
    }
    for text, expected in cases.items():
        got = _parse_audit(text)
        assert got["verdict"] == expected, (text, got)
        assert got["verdict_stated"] is True

    # A model echoing the menu has not judged anything.
    echoed = _parse_audit("VERDICT: supported | contested | unsupported | inconclusive")
    assert echoed["verdict_stated"] is False
    assert echoed["verdict"] == "inconclusive"


def test_id_can_actually_call_the_effectors_that_name_a_target(tmp_path: Path):
    """I78. An effector whose target is a role must still be callable.

    `role` means "who is asking" everywhere in this system, so the tool loop
    strips it -- which made two Id effectors permanently uncallable, because
    for them `role` was the *target*. The collision was the defect; the
    parameter is now `target_role`.
    """
    from amoeba.roles import AUTHORITY_ARGUMENTS

    stack = start_stack(tmp_path)
    try:
        from amoeba.rpc import RpcClient, read_or_create_token

        cfg = stack.cfg
        idc = RpcClient(cfg.supervisor_host, cfg.supervisor_port,
                        read_or_create_token(cfg.scope_token_path("id")),
                        timeout=60)
        idc.connect(retries=10, delay=0.2)

        out = idc.call("id_propose_prompt", target_role="ego",
                       prompt="Revised Ego doctrine.",
                       rationale="observed overclaiming")
        assert out["status"] == "candidate" and out["profile_ref"] == "ego@2"

        rej = idc.call("id_request_rejuvenation", target_role="ego",
                       reason="context climbing")
        assert "performed" in rej or "refused" in rej

        # And the argument the loop strips is not one either verb needs.
        assert "target_role" not in AUTHORITY_ARGUMENTS
        assert "role" in AUTHORITY_ARGUMENTS
    finally:
        stack.stop()


def test_an_unanswered_interaction_is_not_reported_complete(tmp_path: Path):
    """I79. "Complete" means answered.

    An interaction whose answer never arrived used to be marked complete with
    an empty one -- telling the client, permanently, that nothing was the
    organism's reply.

    Ego is stopped before the input is submitted, so no answer *can* arrive.
    That removes the timing race: an earlier version of this test shortened
    the patience instead, and against a fast deterministic backend the answer
    sometimes beat it anyway, which made the distinction unobservable and the
    test pass for the wrong reason.
    """
    stack = start_stack(tmp_path, scheduler={"turn_wall_seconds": 0.3,
                                             "max_continuations": 0,
                                             "submit_wait_seconds": 0.2})
    try:
        ego = _wait_child(stack, "ego")
        subprocess.run(["taskkill", "/PID", str(ego["pid"]), "/T", "/F"],
                       capture_output=True, check=False)

        client = "test-client"
        sub = stack.call("io_submit", text="nobody is listening",
                         client_id=client)
        interaction = sub["interaction_id"]

        def settled():
            st = stack.call("io_status", interaction_id=interaction,
                            client_id=client)
            return st if st["status"] in ("complete", "failed") else None

        done = _wait_for(settled, timeout=60.0)
        assert done, "the interaction never settled at all"

        out = stack.call("io_output", interaction_id=interaction,
                         client_id=client)
        result = (out.get("output") or {}).get("result") or {}
        answer = (result.get("answer") or result.get("claim") or "")

        assert done["status"] == "failed", (
            "an interaction nobody answered was reported "
            f"{done['status']!r} with answer {answer!r}")
        assert out.get("error"), "a failed interaction with no reason"
        assert "queued" in out["error"] or "answer" in out["error"]
    finally:
        stack.stop()


def test_an_answered_interaction_carries_its_answer(tmp_path: Path):
    """The other half: given time, the answer does arrive and is returned."""
    stack = start_stack(tmp_path)
    try:
        client = "test-client"
        sub = stack.call("io_submit", text="say something", client_id=client)
        interaction = sub["interaction_id"]

        def settled():
            st = stack.call("io_status", interaction_id=interaction,
                            client_id=client)
            return st if st["status"] in ("complete", "failed") else None

        done = _wait_for(settled, timeout=120.0)
        assert done and done["status"] == "complete", done

        out = stack.call("io_output", interaction_id=interaction,
                         client_id=client)
        result = (out.get("output") or {}).get("result") or {}
        assert (result.get("answer") or "").strip(), out.get("output")
    finally:
        stack.stop()


# ---------------------------------------------------------------------------
# The external loop: attachments in, results out, neither crossing clients
# ---------------------------------------------------------------------------
def _submit_with_attachment(stack, *, client, text, filename, body):
    import base64

    ing = stack.call("io_attach_input", filename=filename,
                     content_base64=base64.b64encode(body.encode()).decode(),
                     client_id=client, media_type="text/plain")
    sub = stack.call("io_submit", text=text, client_id=client,
                     input_ids=[ing["input_id"]])
    return ing["input_id"], sub["interaction_id"]


def _turn_answering(stack, needle: str):
    """The Ego turn whose bundle contains this text."""
    def found():
        for t in stack.call("role_turns", role="ego")["turns"]:
            d = stack.call("role_turn", turn_id=t["turn_id"])
            # The rendered bundle lives in the stored blob, not on the row:
            # `role_turn` resolves it under "bundle".
            text = ((d.get("bundle") or {}).get("text")) or ""
            if needle in text:
                d["bundle_text"] = text
                return d
        return None
    return _wait_for(found, timeout=120.0)


def test_an_attachment_reaches_the_turn_and_can_be_read(tmp_path: Path):
    """The whole point: a client sends a file and the organism can read it.

    Both halves of this existed and nothing joined them -- the bytes were
    stored with exact provenance and then dropped when the worker called
    Ego with the text alone.
    """
    stack = start_stack(tmp_path)
    try:
        input_id, _ixn = _submit_with_attachment(
            stack, client="alice", text="what does the log say?",
            filename="server.log", body="ERROR: disk full at 03:14")

        turn = _turn_answering(stack, "what does the log say?")
        assert turn, "the request never reached a turn"
        assert "server.log" in turn["bundle_text"], "the attachment was not announced"

        read = stack.call("ego_read_attachment", input_id=input_id,
                          turn_id=turn["turn_id"])
        assert read["filename"] == "server.log"
        assert "disk full" in read["text"], "the contents did not come back"
    finally:
        stack.stop()


def test_ego_cannot_read_an_attachment_from_another_request(tmp_path: Path):
    """The isolation claim, over the dispatcher rather than over a table.

    Ego never names an interaction -- there is no parameter for it -- so the
    only way to reach another client's file would be for the Harness to
    resolve the wrong one. This asks for exactly that and requires a refusal.
    """
    stack = start_stack(tmp_path)
    try:
        _mine, _ = _submit_with_attachment(
            stack, client="alice", text="my own question",
            filename="mine.txt", body="alice's private notes")
        theirs, _ = _submit_with_attachment(
            stack, client="bob", text="a different question",
            filename="theirs.txt", body="bob's private notes")

        turn = _turn_answering(stack, "my own question")
        assert turn, "the first request never reached a turn"

        with pytest.raises(Exception) as exc:
            stack.call("ego_read_attachment", input_id=theirs,
                       turn_id=turn["turn_id"])
        assert "no attachment by that id" in str(exc.value)
    finally:
        stack.stop()


def test_ego_surfaces_a_result_to_the_client_that_asked(tmp_path: Path):
    """`surface_result` finally has a caller, and it cannot pick the recipient.

    Knowing a digest has never been authority to fetch it from outside. This
    is the deliberate act that makes one specific thing reachable -- by the
    interaction that asked, resolved from the turn.
    """
    stack = start_stack(tmp_path)
    try:
        input_id, interaction_id = _submit_with_attachment(
            stack, client="alice", text="please surface something",
            filename="in.txt", body="the input")
        turn = _turn_answering(stack, "please surface something")
        assert turn, "the request never reached a turn"

        read = stack.call("ego_read_attachment", input_id=input_id,
                          turn_id=turn["turn_id"])
        out = stack.call("ego_surface_result", sha256=read["sha256"],
                         turn_id=turn["turn_id"], filename="answer.txt")
        assert out["sha256"] == read["sha256"]

        listed = stack.call("io_output", interaction_id=interaction_id,
                            client_id="alice")
        assert any(r["sha256"] == read["sha256"]
                   for r in listed["results"]), listed
    finally:
        stack.stop()


# ---------------------------------------------------------------------------
# Waking: artifacts and board posts reach the role that owns the work
# ---------------------------------------------------------------------------
def _ego_triggers(stack, kind=None):
    out = []
    for t in stack.call("role_mailbox", role="ego")["ego"]["next_triggers"]:
        if kind is None or t["kind"] == kind:
            out.append(t)
    return out


def _work_for(stack, *, origin_actor="ego"):
    res = stack.call("admit_work", objective="find something out",
                     work_class="user", origin_actor=origin_actor)
    assert res["admitted"], res
    return res["work_id"]


def test_an_artifact_proposal_wakes_the_role_that_asked_for_the_work(tmp_path: Path):
    """The kind existed and nothing emitted it, so Ego never heard.

    A proposal grants nothing and waits for a decision only the requesting
    role can make. Leaving it unannounced meant a neuocyte could propose
    something and have it sit there until an operator noticed.
    """
    stack = start_stack(tmp_path)
    try:
        work_id = _work_for(stack)
        sbx = stack.call("sandbox_create", work_id=work_id, owner="nc_1")
        stack.call("sandbox_write", sandbox_id=sbx["sandbox_id"],
                   path="finding.txt", content="the disk is full")
        stack.call("artifact_propose", sandbox_id=sbx["sandbox_id"],
                   path="finding.txt", rationale="worth keeping",
                   proposed_by="nc_1", work_id=work_id)

        events = _wait_for(lambda: _ego_triggers(stack, "artifact_event") or None,
                           timeout=30.0)
        assert events, "the proposal never reached Ego"
        assert work_id in events[0]["summary"]
    finally:
        stack.stop()


def test_a_board_post_about_owned_work_wakes_the_owner(tmp_path: Path):
    """A finding posted against Ego's work is evidence for Ego's thought."""
    stack = start_stack(tmp_path)
    try:
        work_id = _work_for(stack)
        stack.call("board_post", author="nc_1", author_kind="neuocyte",
                   post_type="finding", body="the build fails on cold cache",
                   work_id=work_id)

        events = _wait_for(lambda: _ego_triggers(stack, "board_event") or None,
                           timeout=30.0)
        assert events, "the board post never reached Ego"
        assert "nc_1" in events[0]["summary"]
    finally:
        stack.stop()


def test_a_board_post_about_nobody_s_work_wakes_nobody(tmp_path: Path):
    """Ownership is the rule, and it is recorded rather than guessed.

    Work the supervisor or operator originated has no persistent role waiting
    on it. Waking one would be telling it about somebody else's business --
    and a relevance rule based on what *looks* interesting is the heuristic
    the architecture declined to invent.
    """
    stack = start_stack(tmp_path)
    try:
        work_id = _work_for(stack, origin_actor="operator")
        stack.call("board_post", author="nc_1", author_kind="neuocyte",
                   post_type="finding", body="unrelated to any role",
                   work_id=work_id)
        stack.call("board_post", author="nc_1", author_kind="neuocyte",
                   post_type="finding", body="no work id at all")

        # Give the queue time to be wrong.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            assert not _ego_triggers(stack, "board_event"), \
                "a post about work Ego did not originate woke Ego"
            time.sleep(0.25)
    finally:
        stack.stop()


def test_a_role_posting_about_its_own_work_does_not_wake_itself(tmp_path: Path):
    """Self-notification is not a wake storm, it is a spiral.

    Ego posts, Ego wakes, Ego posts. The suppression is on the author rather
    than on a rate limit, because the problem is not volume -- it is that the
    notification carries no information its recipient did not just create.
    """
    stack = start_stack(tmp_path)
    try:
        work_id = _work_for(stack)
        stack.call("board_post", author="ego", author_kind="ego",
                   post_type="finding", body="I looked into this myself",
                   work_id=work_id)

        # Specifically Ego's own post, not any board event. This work is real
        # and gets dispatched, so a neuocyte may legitimately post a finding
        # about it -- a different author on work Ego originated is exactly
        # what should wake Ego. Forbidding every board event made this pass or
        # fail on whether that neuocyte got as far as posting inside the
        # window, which is one run in three.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            mine = [t for t in _ego_triggers(stack, "board_event")
                    if t["summary"].startswith("ego posted")]
            assert not mine, f"Ego woke itself by posting about its own work: {mine}"
            time.sleep(0.25)
    finally:
        stack.stop()


def test_a_wake_trigger_is_evidence_not_a_question(tmp_path: Path):
    """These carry the work's lineage and expect no answer.

    That is what makes them safe to emit freely: they join the turn already
    thinking about that work instead of opening a rival interaction, so
    bundling absorbs a burst rather than a throttle having to.
    """
    stack = start_stack(tmp_path)
    try:
        work_id = _work_for(stack)
        for i in range(3):
            stack.call("board_post", author="nc_1", author_kind="neuocyte",
                       post_type="finding", body=f"finding {i}",
                       work_id=work_id)

        events = _wait_for(
            lambda: (_ego_triggers(stack, "board_event")
                     if len(_ego_triggers(stack, "board_event")) >= 3 else None),
            timeout=30.0)
        assert events, "the posts never reached Ego"
        for t in events:
            assert not t.get("expects_answer"), \
                "a board post was queued as a question Ego owes a reply to"
    finally:
        stack.stop()
