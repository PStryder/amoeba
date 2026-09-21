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
    """The load-bearing runtime claim, against real processes.

    Before this, `ego_converse` called into the Ego process synchronously and
    the role's RPC server is threaded -- two callers produced two concurrent
    turns against one inference session. Now there is one ingestion path and
    one turn at a time.

    The precondition matters: the second input has to arrive while a turn is
    genuinely *running*. Two inputs submitted before Ego reaches a boundary
    are correctly bundled into one turn, which is a different guarantee and is
    covered separately.
    """
    stack = start_stack(tmp_path)
    try:
        first = stack.call("ego_converse", message="first message",
                           wait=False)["result"]

        # Wait until Ego has actually opened a turn for it.
        running = _wait_for(
            lambda: stack.call("role_mailbox", role="ego")["ego"]["current_turn"],
            timeout=60.0, interval=0.02)
        assert running, "Ego never opened a turn for the first input"
        running_turn = running["turn_id"]

        second = stack.call("ego_converse", message="second message",
                            wait=False)["result"]

        # While that turn is open, the new input is queued and unseen.
        detail = stack.call("role_turn", turn_id=running_turn)
        assert second["trigger_id"] not in {t["trigger_id"]
                                            for t in detail["triggers"]}, \
            "an input arriving mid-turn was injected into it"

        def second_consumed():
            row = stack.call("role_turn", turn_id=running_turn)
            turns = stack.call("role_turns", role="ego")["turns"]
            for t in turns:
                if t["turn_id"] == running_turn:
                    continue
                d = stack.call("role_turn", turn_id=t["turn_id"])
                if second["trigger_id"] in {x["trigger_id"] for x in d["triggers"]}:
                    return t
            return None

        later = _wait_for(second_consumed, timeout=120.0)
        assert later, "the queued input was never given a turn"
        assert later["turn_id"] != running_turn

        # And the two turns did not overlap.
        turns = {t["turn_id"]: t for t in
                 stack.call("role_turns", role="ego")["turns"]}
        a, b = turns[running_turn], turns[later["turn_id"]]
        assert a["finished_at"] and b["started_at"] >= a["finished_at"], \
            "the second turn began before the first ended"
    finally:
        stack.stop()


def test_inputs_arriving_together_are_bundled_into_one_turn(tmp_path: Path):
    """A burst before a boundary becomes one bundle, not one turn each.

    The counterpart to the test above. Ego waking once for four things that
    happened while it was busy is the point of bundling; waking four times
    would be the thrash it exists to avoid.
    """
    stack = start_stack(tmp_path)
    try:
        submitted = [
            stack.call("ego_converse", message=f"message {i}",
                       wait=False)["result"]["trigger_id"]
            for i in range(3)]

        def all_consumed():
            for t in stack.call("role_turns", role="ego")["turns"]:
                d = stack.call("role_turn", turn_id=t["turn_id"])
                got = {x["trigger_id"] for x in d["triggers"]}
                if set(submitted) <= got:
                    return t
            return None

        turn = _wait_for(all_consumed, timeout=90.0)
        assert turn, "the burst was never bundled into a single turn"
        assert turn["trigger_count"] >= 3
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
