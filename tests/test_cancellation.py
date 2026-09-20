"""Cancellation has to actually stop things, not just return quickly."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

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
def test_an_mcp_client_has_no_cancellation_verb(stack: LiveStack):
    """Cancellation is control, so it is not in the external vocabulary.

    This test used to drive `mind_cancel` over MCP and assert the cancel
    reached the supervisor. That surface is gone on purpose: a client that
    wants something stopped says so as input, and Amoeba decides what follows.

    What remains true, and is tested directly in
    `test_call_cancellable_issues_a_cancel_when_the_await_is_cancelled`, is
    that an *aborted MCP call* still tells the supervisor to stop the work --
    that is the transport honouring its own protocol, not the client holding a
    control verb.
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
                attempts = {}
                for verb in ("mind_cancel", "cancel_operation", "cancel_work",
                             "kill_all_neuocytes"):
                    try:
                        res = await session.call_tool(verb, {"operation_id": "x"})
                        attempts[verb] = ("ERROR-FLAGGED" if res.isError
                                          else "EXECUTED")
                    except Exception as exc:  # noqa: BLE001
                        attempts[verb] = type(exc).__name__
                return {"tools": tools, "attempts": attempts}

    out = anyio.run(run)
    assert not any("cancel" in t for t in out["tools"]), out["tools"]
    assert not any("kill" in t for t in out["tools"])
    for verb, outcome in out["attempts"].items():
        assert outcome != "EXECUTED", f"an MCP client executed {verb}"

def test_call_cancellable_issues_a_cancel_when_the_await_is_cancelled():
    """The facade's actual contract, tested directly.

    Driving this through a real MCP client meant cancelling `call_tool` from
    the client side, which tears down the SDK's stream and raises
    BrokenResourceError in teardown -- that exercises the SDK, not this code.
    What matters here is narrow and checkable: when the awaited call is
    cancelled, the facade must issue cancel_operation for the same idempotency
    key, and must do so from a shielded scope so the cleanup is not itself
    cancelled.
    """
    import anyio

    from amoeba.config import Config
    from amoeba.mcp_api import Facade

    cfg = Config()
    cfg.state_dir = Path(tempfile.mkdtemp())
    cfg.ensure_dirs()
    facade = Facade(cfg)

    calls: list[tuple[str, dict]] = []
    started = threading.Event()

    def fake_call(method: str, **params):
        calls.append((method, params))
        if method == "cancel_operation":
            return {"cancelled": True}
        started.set()
        time.sleep(3)                       # a long in-flight call
        return {"never": "reached"}

    facade.call = fake_call                 # type: ignore[assignment]

    async def run() -> None:
        with anyio.move_on_after(0.75):
            await facade.call_cancellable("ego_converse",
                                          idempotency_key="key-under-test",
                                          message="hello")

    anyio.run(run)

    methods = [m for m, _ in calls]
    assert "ego_converse" in methods
    assert "cancel_operation" in methods, f"no cancel was issued: {methods}"
    cancel = next(p for m, p in calls if m == "cancel_operation")
    assert cancel["idempotency_key"] == "key-under-test"
    assert cancel["actor"] == "mcp_client"
    assert "client cancelled" in cancel["reason"]


def test_call_cancellable_still_propagates_the_cancellation():
    """Cleanup must not swallow the cancellation: the task has to end."""
    import anyio

    from amoeba.config import Config
    from amoeba.mcp_api import Facade

    cfg = Config()
    cfg.state_dir = Path(tempfile.mkdtemp())
    cfg.ensure_dirs()
    facade = Facade(cfg)
    facade.call = lambda method, **kw: (  # type: ignore[assignment]
        {"ok": True} if method == "cancel_operation" else time.sleep(3))

    finished = []

    async def run() -> None:
        with anyio.move_on_after(0.5):
            await facade.call_cancellable("ego_converse", idempotency_key="k")
            finished.append("returned normally")

    anyio.run(run)
    assert finished == [], "a cancelled call must not return a value"


def test_the_mind_survives_a_cancelled_client(stack: LiveStack):
    before = stack.call("health")
    pids = {k: v["pid"] for k, v in before["children"].items()}
    op = stack.call("open_operation", kind="ego_converse", actor="ego",
                    request={"m": "x"})
    stack.call("cancel_operation", operation_id=op["operation_id"])
    after = stack.call("health")
    assert after["status"] == "alive"
    assert {k: v["pid"] for k, v in after["children"].items()} == pids


# ---------------------------------------------------------------------------
# Review-pass regressions
# ---------------------------------------------------------------------------
def test_a_cancel_arriving_before_generate_is_honoured_not_discarded():
    """The realistic race: cancel lands during the prefill.

    A turn is ingest_text (a long, uninterruptible prefill) then generate.
    A cancel arriving during the prefill sets the flag; generate used to clear
    it on entry and carry on, silently discarding exactly the cancellation this
    feature exists to deliver.
    """
    b = DeterministicBackend(n_seq_max=4)
    b.load()
    sess = b.open_session(role="ego")
    b.ingest(sess.session_id, b.tokenize("a long prompt"))
    b.request_cancel(sess.session_id)          # arrives before generate starts
    out = b.generate(sess.session_id, max_tokens=64)
    assert out.finish_reason == "cancelled"
    assert out.completion_tokens == 0


def test_a_consumed_cancel_does_not_persist():
    """A session cancelled once must be usable again, in either decode mode."""
    b = DeterministicBackend(n_seq_max=4)
    b.load()
    sess = b.open_session(role="ego")
    b.ingest(sess.session_id, b.tokenize("hello"))
    b.request_cancel(sess.session_id)
    assert b.generate(sess.session_id, max_tokens=8).finish_reason == "cancelled"
    assert b.get_session(sess.session_id).cancel_requested is False
    assert b.generate(sess.session_id, max_tokens=8).finish_reason != "cancelled"


def test_cancelling_one_role_does_not_stop_the_other(stack: LiveStack):
    """Blast radius. Cancelling an Ego operation must not stop Id.

    cancel_operation used to ask both roles to stop generating regardless of
    which owned the operation, so cancelling a conversation would abort an
    unrelated audit that happened to be in flight.
    """
    turn = stack.call("open_operation", kind="ego_converse", actor="ego",
                      request={"message": "x"})
    out = stack.call("cancel_operation", operation_id=turn["operation_id"])
    stopped = {g["role"] for g in out["generations_stopped"]}
    assert "id" not in stopped, f"cancelling an ego operation stopped Id: {stopped}"

    id_op = stack.call("open_operation", kind="id_audit", actor="id",
                       request={"focus": "y"})
    out2 = stack.call("cancel_operation", operation_id=id_op["operation_id"])
    assert "ego" not in {g["role"] for g in out2["generations_stopped"]}
