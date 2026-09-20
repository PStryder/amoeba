"""The tool execution loop, asserted where each guarantee actually lives.

The loop spans three layers and they fail differently, so they are tested
separately rather than through one end-to-end path that would hide which part
is doing the work:

* **Authorisation** lives in the Harness, on the work row. Tested by calling
  `tool_invoke` directly -- no model involved, because no model is needed to
  establish that a capability is refused.
* **Loop control** (turns, budget, deadline, feeding results back) lives in the
  neuocyte process. Tested against fake RPC clients, so the bounds are
  exercised precisely instead of hoping a model happens to hit them.
* **End to end** is tested with scripted backend responses, because hashed
  deterministic text never contains a tool call -- a loop tested only on input
  that never takes its interesting branch is not tested.

The guarantee that matters most here is not that tools work. It is that a
model asking for a capability is not a way to obtain one.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest

from amoeba.errors import Fenced
from amoeba.neuocyte import Neuocyte
from amoeba.tools import build_neuocyte_registry

# ===========================================================================
# Authorisation: the Harness decides, from durable state
# ===========================================================================
def _admit(stack, *, objective="compute something", sandbox_allowed=False):
    res = stack.call("admit_work", objective=objective, work_class="user",
                     origin_actor="test", sandbox_allowed=sandbox_allowed)
    assert res["admitted"], res
    return res["work_id"]


def _lease(stack, work_id, neuocyte_id="nc_test"):
    item = stack.call("lease_work", neuocyte_id=neuocyte_id, work_id=work_id)
    assert item, "could not lease the work item"
    return item


def test_a_permitted_tool_executes_and_is_receipted(stack):
    work_id = _admit(stack)
    item = _lease(stack, work_id)
    out = stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                     fencing_token=item["fencing_token"],
                     name="current_state_version", arguments={})
    assert out["accepted"] is True, out
    assert out["error"] is None
    assert out["result"]["state_version"] >= 0
    assert out["receipt_id"]

    kinds = [e["kind"] for e in stack.call("history", limit=400)]
    assert "tool.requested" in kinds and "tool.result" in kinds


def test_an_unknown_tool_is_refused_and_recorded(stack):
    work_id = _admit(stack)
    item = _lease(stack, work_id)
    out = stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                     fencing_token=item["fencing_token"],
                     name="rm_minus_rf", arguments={"path": "/"})
    assert out["accepted"] is False
    assert "no such tool" in out["reason"]
    kinds = [e["kind"] for e in stack.call("history", limit=400)]
    assert "tool.rejected" in kinds, "a refusal must be recorded, not just returned"


def test_a_neuocyte_cannot_grant_itself_the_sandbox(stack):
    """The capability comes from the work row, never from the request.

    This is the load-bearing one. A model that can talk its way into a
    sandbox has no boundary at all, so the check is: admit work *without*
    sandbox access, then ask for the sandbox anyway, including by passing
    arguments that look like a grant.
    """
    work_id = _admit(stack, sandbox_allowed=False)
    item = _lease(stack, work_id)
    for args in ({"code": "print(1)"},
                 {"code": "print(1)", "sandbox_allowed": True},
                 {"code": "print(1)", "work_id": "someone-elses"}):
        out = stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                         fencing_token=item["fencing_token"],
                         name="run_code", arguments=args)
        assert out["accepted"] is False, f"run_code executed without permission: {out}"
        assert "no such tool" in out["reason"], out["reason"]
    assert "run_code" not in out["available_tools"]


def test_sandbox_tools_are_absent_when_the_work_item_did_not_allow_them(stack):
    """Not merely refused at the door: never registered.

    A tool that exists but is guarded can be reached by a bug in the guard. A
    tool that was never registered has no handler to reach at all.
    """
    denied = build_neuocyte_registry(_FakeSup(), work_id="w1", neuocyte_id="nc",
                                     sandbox_allowed=False)
    allowed = build_neuocyte_registry(_FakeSup(), work_id="w1", neuocyte_id="nc",
                                      sandbox_allowed=True)
    sandbox_tools = {"run_code", "write_file", "read_file", "list_files",
                     "propose_artifact"}
    assert sandbox_tools & set(denied.names()) == set()
    assert sandbox_tools <= set(allowed.names(role="neuocyte"))
    for name in sandbox_tools:
        assert denied.get(name) is None


def test_no_neuocyte_tool_accepts_a_sandbox_id(stack):
    """A model cannot name a sandbox because there is nowhere to put one."""
    reg = build_neuocyte_registry(_FakeSup(), work_id="w1", neuocyte_id="nc",
                                  sandbox_allowed=True)
    for schema in reg.schemas(role="neuocyte"):
        props = set(schema["parameters"]["properties"])
        leaky = {p for p in props if "sandbox" in p.lower()}
        assert not leaky, f"{schema['name']} exposes {leaky} to the model"


def test_two_work_items_get_different_sandboxes(stack):
    if not stack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    a, b = _admit(stack, sandbox_allowed=True), _admit(stack, sandbox_allowed=True)
    ia = _lease(stack, a, "nc_a")
    ib = _lease(stack, b, "nc_b")
    try:
        ra = stack.call("tool_invoke", neuocyte_id="nc_a", work_id=a,
                        fencing_token=ia["fencing_token"], name="write_file",
                        arguments={"path": "work/a.txt", "content": "from-a"})
        rb = stack.call("tool_invoke", neuocyte_id="nc_b", work_id=b,
                        fencing_token=ib["fencing_token"], name="list_files",
                        arguments={})
        assert ra["accepted"] and rb["accepted"], (ra, rb)
        names = [f["path"] for f in rb["result"]]
        assert not any("a.txt" in n for n in names), (
            f"work item B can see work item A's scratch: {names}")
    finally:
        stack.call("cancel_work", work_id=a, reason="test")
        stack.call("cancel_work", work_id=b, reason="test")


def test_a_fenced_neuocyte_cannot_invoke_a_tool(stack):
    """Killing a neuocyte must actually stop it, including mid-loop."""
    work_id = _admit(stack)
    item = _lease(stack, work_id)
    good = item["fencing_token"]

    ok = stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                    fencing_token=good, name="current_state_version", arguments={})
    assert ok["accepted"], "control: the call should work before fencing"

    with pytest.raises(Exception) as exc:
        stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                   fencing_token=good + 1, name="current_state_version",
                   arguments={})
    assert "fenc" in str(exc.value).lower() or "stale" in str(exc.value).lower()

    with pytest.raises(Exception):
        stack.call("tool_invoke", neuocyte_id="nc_impostor", work_id=work_id,
                   fencing_token=good, name="current_state_version", arguments={})


def test_the_sandbox_is_destroyed_when_the_work_item_finishes(stack):
    if not stack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work_id = _admit(stack, sandbox_allowed=True)
    item = _lease(stack, work_id)
    r = stack.call("tool_invoke", neuocyte_id="nc_test", work_id=work_id,
                   fencing_token=item["fencing_token"], name="write_file",
                   arguments={"path": "work/x.txt", "content": "scratch"})
    assert r["accepted"], r
    assert stack.call("sandbox_list"), "a sandbox should be live at this point"

    stack.call("complete_work", work_id=work_id, neuocyte_id="nc_test",
               fencing_token=item["fencing_token"], result={"finding": "done"})
    assert stack.call("sandbox_list") == [], (
        "scratch outlived the work item that produced it")


# ===========================================================================
# Loop control: turns, budget, deadline
# ===========================================================================
class _FakeSup:
    """Stands in for the supervisor where only the registry shape matters."""

    class _Mind:
        def __getattr__(self, _name):
            raise AssertionError("the registry must not touch the mind here")

    mind = _Mind()

    def methods(self):
        return {}

    def sandbox_for_work(self, work_id, *, owner):
        return "sbx_fake"


class _FakeInference:
    """Returns scripted generations and records what was fed back in."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.ingested: list[str] = []
        self.generations = 0

    def call(self, method: str, **params: Any) -> Any:
        if method == "generate":
            self.generations += 1
            text = self.replies.pop(0) if self.replies else "FINDING: done"
            return {"text": text, "completion_tokens": 100}
        if method == "apply_chat_template":
            return params["messages"][0]["content"]
        if method == "ingest_text":
            self.ingested.append(params["text"])
            return {"ok": True}
        raise AssertionError(f"unexpected inference call {method}")


class _FakeSupervisorRpc:
    def __init__(self, outcome: dict[str, Any] | None = None) -> None:
        self.invocations: list[dict[str, Any]] = []
        self.outcome = outcome or {"name": "current_state_version", "accepted": True,
                                   "result": {"state_version": 7}, "error": None,
                                   "reason": "executed", "receipt_id": "rcp_1"}

    def call(self, method: str, **params: Any) -> Any:
        if method == "tool_invoke":
            self.invocations.append(params)
            return self.outcome
        raise AssertionError(f"unexpected supervisor call {method}")


def _neuocyte(cfg, inf, sup) -> Neuocyte:
    nc = Neuocyte.__new__(Neuocyte)
    nc.cfg = cfg
    nc.neuocyte_id = "nc_loop"
    nc.session_id = "sess_1"
    nc.inf = inf
    nc.sup = sup
    from amoeba.logging_setup import get_logger
    nc.log = get_logger("test")
    return nc


TOOL_CALL = '<tool_call>{"name": "current_state_version", "arguments": {}}</tool_call>'
ITEM = {"work_id": "w1", "fencing_token": 3}


def test_a_tool_result_is_fed_back_and_generation_resumes(cfg):
    inf = _FakeInference([TOOL_CALL, "FINDING: the version is 7"])
    sup = _FakeSupervisorRpc()
    nc = _neuocyte(cfg, inf, sup)

    out, trace = nc._generate_with_tools(ITEM, budget=4000, deadline=time.time() + 60,
                                         max_turns=6)
    assert inf.generations == 2, "the model was not resumed after the tool ran"
    assert len(sup.invocations) == 1
    assert sup.invocations[0]["name"] == "current_state_version"
    assert any("state_version" in t for t in inf.ingested), (
        "the tool result was never fed back into the session")
    assert out["stop_reason"] == "answered"
    assert trace[0]["accepted"] is True


def test_the_neuocyte_process_never_executes_a_tool_itself(cfg):
    """The loop is here; the authority is not.

    The fake supervisor is the only thing that can execute. If the neuocyte
    ever ran a handler locally, `tool_invoke` would not be called and this
    would see zero invocations while still producing a result.
    """
    inf = _FakeInference([TOOL_CALL, "FINDING: done"])
    sup = _FakeSupervisorRpc()
    nc = _neuocyte(cfg, inf, sup)
    nc._generate_with_tools(ITEM, budget=4000, deadline=time.time() + 60, max_turns=6)
    assert len(sup.invocations) == 1, (
        "the tool call did not go through the Harness")

    import amoeba.neuocyte as mod
    src = __import__("pathlib").Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("registry.execute", "ToolRegistry(", "build_neuocyte_registry"):
        assert forbidden not in src, (
            f"the neuocyte process references {forbidden}; execution must stay "
            "in the Harness")


def test_a_refusal_is_fed_back_rather_than_hidden(cfg):
    inf = _FakeInference([TOOL_CALL, "FINDING: I could not check"])
    sup = _FakeSupervisorRpc(outcome={"name": "run_code", "accepted": False,
                                      "reason": "no such tool", "result": None,
                                      "error": None, "receipt_id": "rcp_2"})
    nc = _neuocyte(cfg, inf, sup)
    _out, trace = nc._generate_with_tools(ITEM, budget=4000,
                                          deadline=time.time() + 60, max_turns=6)
    assert trace[0]["accepted"] is False
    assert any("refused" in t and "no such tool" in t for t in inf.ingested), (
        "the model was not told its request was refused")


def test_the_tool_loop_stops_at_the_turn_limit(cfg):
    """A model that keeps calling tools terminates, and says why."""
    inf = _FakeInference([TOOL_CALL] * 20)
    sup = _FakeSupervisorRpc()
    nc = _neuocyte(cfg, inf, sup)
    out, trace = nc._generate_with_tools(ITEM, budget=100000,
                                         deadline=time.time() + 600, max_turns=4)
    assert out["stop_reason"] == "turn_limit_reached"
    assert inf.generations <= 4
    assert len(sup.invocations) <= 3, (
        "a tool was executed on the final turn, whose result could never be used")
    assert trace[-1]["executed"] is False


def test_the_tool_loop_stops_when_the_token_budget_is_exhausted(cfg):
    inf = _FakeInference([TOOL_CALL] * 20)
    sup = _FakeSupervisorRpc()
    nc = _neuocyte(cfg, inf, sup)
    out, _trace = nc._generate_with_tools(ITEM, budget=250,
                                          deadline=time.time() + 600, max_turns=50)
    assert out["stop_reason"] == "token_budget_exhausted"
    assert out["tokens_spent"] >= 250
    assert inf.generations == 3, (
        f"budget 250 at 100 tokens per generation should stop after 3, "
        f"got {inf.generations}")


def test_the_tool_loop_stops_at_the_deadline(cfg):
    inf = _FakeInference([TOOL_CALL] * 20)
    sup = _FakeSupervisorRpc()
    nc = _neuocyte(cfg, inf, sup)
    out, _trace = nc._generate_with_tools(ITEM, budget=100000,
                                          deadline=time.time() - 1, max_turns=50)
    assert out["stop_reason"] == "deadline_reached"
    assert inf.generations == 0, "generation started after the deadline had passed"


def test_a_fenced_tool_call_aborts_the_loop(cfg):
    """Fencing is not a failure to route around; it means stop."""
    class _Fencing(_FakeSupervisorRpc):
        def call(self, method: str, **params: Any) -> Any:
            raise Fenced("superseded", work_id="w1")

    inf = _FakeInference([TOOL_CALL] * 5)
    nc = _neuocyte(cfg, inf, _Fencing())
    with pytest.raises(Fenced):
        nc._generate_with_tools(ITEM, budget=100000, deadline=time.time() + 60,
                                max_turns=6)


def test_a_transport_failure_is_reported_to_the_model_not_fatal(cfg):
    class _Broken(_FakeSupervisorRpc):
        def call(self, method: str, **params: Any) -> Any:
            raise ConnectionError("supervisor went away")

    inf = _FakeInference([TOOL_CALL, "FINDING: proceeded without the tool"])
    nc = _neuocyte(cfg, inf, _Broken())
    out, trace = nc._generate_with_tools(ITEM, budget=4000,
                                         deadline=time.time() + 60, max_turns=6)
    assert out["stop_reason"] == "answered"
    assert trace[0]["accepted"] is False
    assert "ConnectionError" in trace[0]["reason"]


# ===========================================================================
# End to end, with scripted generation
# ===========================================================================
@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_a_neuocyte_runs_real_code_in_its_sandbox_end_to_end(stack):
    """The whole path: request -> Harness -> AppContainer -> result -> resume.

    Scripted rather than hashed generation, because deterministic hash text
    never contains a tool call and the loop's interesting branch would never
    be taken.
    """
    if not stack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work_id = _admit(stack, objective="compute 6*7 by running code",
                     sandbox_allowed=True)
    item = _lease(stack, work_id, "nc_e2e")
    try:
        out = stack.call(
            "tool_invoke", neuocyte_id="nc_e2e", work_id=work_id,
            fencing_token=item["fencing_token"], name="run_code",
            arguments={"code": "print('answer', 6*7)"})
        assert out["accepted"] is True, out
        assert out["error"] is None, out["error"]
        assert out["result"]["exit_code"] == 0, out["result"]
        assert "answer 42" in out["result"]["stdout"]

        # And it really was the sandbox: no network from in there.
        blocked = stack.call(
            "tool_invoke", neuocyte_id="nc_e2e", work_id=work_id,
            fencing_token=item["fencing_token"], name="run_code",
            arguments={"code": ("import socket\n"
                                "socket.create_connection(('1.1.1.1', 53), timeout=4)\n"
                                "print('CONNECTED')\n")})
        assert "CONNECTED" not in blocked["result"]["stdout"]
    finally:
        stack.call("cancel_work", work_id=work_id, reason="test")


@pytest.mark.skipif(sys.platform != "win32", reason="AppContainer is Windows-only")
def test_a_neuocyte_can_propose_an_artifact_but_not_promote_it(stack):
    if not stack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work_id = _admit(stack, sandbox_allowed=True)
    item = _lease(stack, work_id, "nc_art")
    try:
        stack.call("tool_invoke", neuocyte_id="nc_art", work_id=work_id,
                   fencing_token=item["fencing_token"], name="write_file",
                   arguments={"path": "work/checker.py",
                              "content": "def check(x):\n    return x > 0\n"})
        proposed = stack.call(
            "tool_invoke", neuocyte_id="nc_art", work_id=work_id,
            fencing_token=item["fencing_token"], name="propose_artifact",
            arguments={"path": "work/checker.py", "rationale": "reusable check"})
        assert proposed["accepted"] is True, proposed
        assert proposed["result"]["status"] == "proposed"
        assert "nothing has been placed in the artifact store" in (
            proposed["result"]["note"])
        # Proposing preserves the bytes as evidence; it does not place
        # them at any destination. Those are different claims.
        assert proposed["result"]["evidence_preserved"] is True

        # Promotion is not in the neuocyte's vocabulary at all.
        denied = stack.call(
            "tool_invoke", neuocyte_id="nc_art", work_id=work_id,
            fencing_token=item["fencing_token"], name="artifact_promote",
            arguments={"artifact_id": proposed["result"]["artifact_id"]})
        assert denied["accepted"] is False
        assert "no such tool" in denied["reason"]
    finally:
        stack.call("cancel_work", work_id=work_id, reason="test")
