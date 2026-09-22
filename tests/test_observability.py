"""What the operator is shown is what was measured, and it is attributed.

Three gaps from the second pressure run, each a view that looked informed and
was not.

- Every `role.tool_invoked` event carried `operation_id=None`, so the one query
  that asks what an operation did never showed what its roles reached for.
- The overview's role context and inference KV were null on every pulse. The
  pulse read them off `health` replies that do not carry them; the same nulls
  fed `context_pressure`, which is what Id watches.
- `prompt_incarnations` omitted `harness_constraints`, so a binding whose
  ceiling the Harness supplied looked identical to one whose profile stated it.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from amoeba.rpc import RpcClient, read_or_create_token
from conftest import start_stack

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="live stack fixtures are Windows-only here")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = start_stack(tmp_path_factory.mktemp("observe"),
                    scheduler={"id_startup_turn": False})
    s.wait_for_children(timeout=90)
    yield s
    s.stop()


def _converse(stack, message: str) -> dict:
    env = stack.call("ego_converse", message=message, wait=False)
    deadline = time.time() + 120
    while stack.call("role_answer", trigger_id=env["result"]["trigger_id"])[
            "status"] not in ("completed", "incomplete", "unanswerable"):
        assert time.time() < deadline
        time.sleep(0.3)
    return env


def test_a_role_tool_call_is_on_its_operations_history(stack):
    cfg = stack.cfg
    inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                    read_or_create_token(cfg.token_path), timeout=60)
    inf.connect(retries=20, delay=0.5)
    call = json.dumps({"name": "board_read", "arguments": {"limit": 3}})
    inf.call("script_responses", responses=[
        {"role": "ego", "text": f"<tool_call>{call}</tool_call>",
         "finish_reason": "stop", "continues": True},
        {"role": "ego", "text": "Read it.", "finish_reason": "stop",
         "continues": True}])
    env = _converse(stack, "look at the board")
    events = stack.call("history", operation_id=env["operation_id"],
                        kinds=["role.tool_invoked"], limit=20)
    tools = [json.loads(e["payload_inline"])["tool"] for e in events]
    assert "board_read" in tools, \
        "the tool call is not attributed to the operation that made it"


def test_the_overview_shows_measured_context(stack):
    _converse(stack, "anything")
    time.sleep(2.5)                                    # past the pulse's cache
    overview = stack.call("operator_overview")
    measured = stack.call("context_report")
    ego = overview["roles"]["ego"]
    held = {s["session_id"]: s for s in measured["sessions"]}[ego["session_id"]]
    assert isinstance(ego["context_tokens"], int) and ego["context_tokens"] > 0, ego
    assert ego["max_context_tokens"] == held["budget_tokens"]
    assert overview["context_pressure"]["ego"]["occupancy"] is not None
    inference = overview["inference"]
    assert inference["n_ctx"] == measured["pool_capacity"]
    assert isinstance(inference["kv_tokens_used"], int), inference
    assert inference["kv_tokens_total"] == measured["pool_capacity"]


def test_a_binding_shows_what_the_harness_supplied(stack):
    bindings = stack.call("prompt_incarnations", detail="full")["bindings"]
    assert bindings
    for b in bindings:
        assert "harness_constraints" in b, b
        assert isinstance(json.loads(b["harness_constraints"]), dict)
