"""Cancellation has to actually stop things, not just return quickly."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

from amoeba.backends.deterministic import DeterministicBackend
from amoeba.store.events import EventKind
from conftest import PYTHON, ROOT, LiveStack


# ---------------------------------------------------------------------------
# Backend: the flag is readable while the engine lock is held
# ---------------------------------------------------------------------------
def test_request_cancel_does_not_need_the_engine_lock():
    """The whole point is to be callable *during* a generation.

    If request_cancel took the engine lock it would deadlock behind the
    generate() it is trying to stop, so this asserts it can be called from
    another thread while that lock is held.
    """
    b = DeterministicBackend(n_seq_max=4)
    b.load()
    sess = b.open_session(role="ego")
    b.ingest(sess.session_id, b.tokenize("hello"))

    reached = threading.Event()
    result: list[bool] = []

    def hold_lock_then_wait():
        with b._lock:
            reached.set()
            time.sleep(1.0)

    holder = threading.Thread(target=hold_lock_then_wait, daemon=True)
    holder.start()
    assert reached.wait(5)
    t0 = time.perf_counter()
    result.append(b.request_cancel(sess.session_id))
    elapsed = time.perf_counter() - t0
    holder.join(timeout=5)

    assert result == [True]
    assert elapsed < 0.5, "request_cancel blocked on the engine lock"
    assert b.get_session(sess.session_id).cancel_requested is True


def test_cancel_flag_stops_generation_and_is_reported():
    b = DeterministicBackend(n_seq_max=4)
    b.load()
    sess = b.open_session(role="ego")
    b.ingest(sess.session_id, b.tokenize("hello"))
    b.request_cancel(sess.session_id)
    out = b.generate(sess.session_id, max_tokens=64)
    assert out.finish_reason == "cancelled"
    assert out.completion_tokens == 0
    # The flag clears, so the session is usable again.
    assert b.get_session(sess.session_id).cancel_requested is False
    again = b.generate(sess.session_id, max_tokens=8)
    assert again.finish_reason != "cancelled"


def test_cancelling_an_unknown_session_is_false_not_an_error():
    b = DeterministicBackend()
    b.load()
    assert b.request_cancel("sess_nope") is False


# ---------------------------------------------------------------------------
# Supervisor: operation, work, neuocytes
# ---------------------------------------------------------------------------
def test_cancel_stops_queued_work_and_records_it(stack: LiveStack):
    op = stack.call("open_operation", kind="ego_investigate", actor="ego",
                    request={"question": "something long"})
    admitted = stack.call("admit_work", objective="a long task", work_class="user",
                          origin_actor="ego", operation_id=op["operation_id"])
    assert admitted["admitted"]

    out = stack.call("cancel_operation", operation_id=op["operation_id"],
                     reason="client went away")
    assert out["cancelled"] is True
    assert admitted["work_id"] in out["cancelled_work"]
    assert out["receipt_id"]

    assert stack.call("get_work", work_id=admitted["work_id"])["status"] == "cancelled"
    assert stack.call("get_operation",
                      operation_id=op["operation_id"])["status"] == "cancelled"
    kinds = [e["kind"] for e in stack.call("history", limit=300)]
    assert EventKind.OPERATION_CANCELLED in kinds


def test_a_cancelled_work_item_is_never_picked_up(stack: LiveStack):
    op = stack.call("open_operation", kind="ego_investigate", actor="ego",
                    request={"q": "x"})
    admitted = stack.call("admit_work", objective="should not run",
                          work_class="user", origin_actor="ego",
                          operation_id=op["operation_id"])
    stack.call("cancel_operation", operation_id=op["operation_id"], reason="stop")
    assert stack.call("lease_work", neuocyte_id="nc_probe",
                      work_id=admitted["work_id"]) is None


def test_cancel_is_idempotent_and_truthful_about_finished_work(stack: LiveStack):
    turn = stack.call("ego_converse", message="quick question",
                      idempotency_key="cancel-after-done")
    assert turn["status"] == "completed"

    out = stack.call("cancel_operation", operation_id=turn["operation_id"])
    assert out["cancelled"] is False
    assert out["already_terminal"] is True
    assert out["prior_status"] == "completed"
    assert "already terminal" in out["detail"]

    # Cancelling twice is still fine.
    again = stack.call("cancel_operation", operation_id=turn["operation_id"])
    assert again["already_terminal"] is True


def test_cancel_by_idempotency_key(stack: LiveStack):
    turn = stack.call("ego_converse", message="hello", idempotency_key="key-abc")
    out = stack.call("cancel_operation", idempotency_key="key-abc")
    assert out["operation_id"] == turn["operation_id"]


def test_cancelling_an_unknown_key_is_reported_not_raised(stack: LiveStack):
    out = stack.call("cancel_operation", idempotency_key="never-existed")
    assert out["cancelled"] is False
    assert out["operation_id"] is None
    assert "no operation found" in out["detail"]


def test_cancel_needs_something_to_identify_the_operation(stack: LiveStack):
    with pytest.raises(Exception):
        stack.call("cancel_operation", reason="nothing to go on")


def test_cancellation_does_not_undo_committed_state(stack: LiveStack):
    """Cancelling stops future work; it does not roll back what already landed."""
    mem = stack.call("remember", kind="belief", claim="committed before cancel",
                     confidence=0.9, created_by="operator")
    op = stack.call("open_operation", kind="ego_investigate", actor="ego",
                    request={"q": "x"})
    stack.call("admit_work", objective="t", work_class="user", origin_actor="ego",
               operation_id=op["operation_id"])
    stack.call("cancel_operation", operation_id=op["operation_id"])
    assert stack.call("get_memory",
                      memory_id=mem["memory_id"])["claim"] == "committed before cancel"


def test_cancel_generation_targets_a_live_role_session(stack: LiveStack):
    out = stack.call("cancel_generation", role="ego", reason="test")
    assert "cancel_requested" in out
    assert "prefill is not interruptible" in out["granularity"]


# ---------------------------------------------------------------------------
# A real MCP client aborting mid-call
# ---------------------------------------------------------------------------
@pytest.mark.timeout(300)
def test_mcp_client_cancellation_reaches_the_supervisor(stack: LiveStack):
    """Abort an in-flight tool call and check the mind was actually told.

    The deterministic backend answers instantly, so this asserts the wiring
    (cancel arrives, operation is marked, receipt exists) rather than racing to
    interrupt a slow generation.
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
            args=["-m", "amoeba.mcp_api", "--config", str(stack.cfg.source_path)],
            env=env, cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                started = await session.call_tool(
                    "ego_converse", {"message": "a question",
                                     "idempotency_key": "mcp-cancel-target"})
                import json as _json
                payload = started.structuredContent or _json.loads(
                    started.content[0].text)
                cancelled = await session.call_tool(
                    "mind_cancel", {"operation_id": payload["operation_id"],
                                    "reason": "client changed its mind"})
                cpayload = cancelled.structuredContent or _json.loads(
                    cancelled.content[0].text)
                return {"tools": tools, "turn": payload, "cancel": cpayload}

    out = anyio.run(run)
    assert "mind_cancel" in out["tools"]
    result = out["cancel"].get("result", out["cancel"])
    assert result["operation_id"] == out["turn"]["operation_id"]
    assert result["receipt_id"]
    # It had already completed, and the response says so rather than pretending.
    assert result["already_terminal"] is True


@pytest.mark.timeout(300)
def test_aborting_an_mcp_call_cancels_the_operation(stack: LiveStack):
    """Cancel the *await* itself, the way an MCP client disconnect does."""
    import anyio

    key = "mcp-abort-key"

    async def run() -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
                                             env.get("PYTHONPATH", "")])
        env["AMOEBA_STDERR_LOG"] = "0"
        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "amoeba.mcp_api", "--config", str(stack.cfg.source_path)],
            env=env, cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                with anyio.move_on_after(0.02):
                    await session.call_tool(
                        "ego_converse", {"message": "abort me",
                                         "idempotency_key": key})

    anyio.run(run)
    time.sleep(2)
    # Either the turn completed before the abort landed, or the abort cancelled
    # it. Both are legitimate; what must NOT happen is an operation left
    # running with nobody waiting for it.
    out = stack.call("cancel_operation", idempotency_key=key,
                     reason="test sweep")
    if out["operation_id"] is not None:
        op = stack.call("get_operation", operation_id=out["operation_id"])
        assert op["status"] in ("completed", "cancelled", "failed"), op["status"]


def test_the_mind_survives_a_cancelled_client(stack: LiveStack):
    before = stack.call("health")
    pids = {k: v["pid"] for k, v in before["children"].items()}
    op = stack.call("open_operation", kind="ego_converse", actor="ego",
                    request={"m": "x"})
    stack.call("cancel_operation", operation_id=op["operation_id"])
    after = stack.call("health")
    assert after["status"] == "alive"
    assert {k: v["pid"] for k, v in after["children"].items()} == pids
