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
        if method == "ingest_messages":
            self.ingested.append(params["messages"][0]["content"])
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


# ===========================================================================
# A result the model cannot see whole says so
# ===========================================================================
def test_a_tool_result_that_fits_is_shown_whole(cfg):
    """The bound is a ceiling, not a formatter. Nothing is added below it."""
    from amoeba.tools import deliver_tool_result

    out = deliver_tool_result({"state_version": 7})
    assert out["complete"] is True
    assert out["sha256"] is None and out["result_ref"] is None
    assert out["text"] == '{"state_version": 7}'


def test_an_oversized_result_is_projected_whole_and_says_what_it_left_out(cfg):
    """Never chopped: whole items, how many of how many, and where the rest is.

    The old bound cut the serialized JSON at 2000 characters -- half a value
    in the context, priced in the wrong unit. A model has to learn that the
    view is partial, how much it did not see, and how to get it.
    """
    import json as _json

    from amoeba.tools import deliver_tool_result

    payload = {"items": [{"id": f"w{i}", "note": "x" * 80} for i in range(60)],
               "state_version": 3}
    digest = "a" * 64
    out = deliver_tool_result(payload, budget_tokens=400, count=lambda t: len(t) // 3,
                              store=lambda _text: digest)
    view = _json.loads(out["text"])                       # whole JSON, always
    assert out["complete"] is False and view["complete"] is False
    assert out["sha256"] == digest and view["result_ref"] == digest[:16]
    assert view["list"]["of"] == 60 and 0 < view["list"]["returned"] < 60
    assert view["list"]["next_offset"] == view["list"]["returned"]
    shown = view["result"]["items"]
    assert shown == payload["items"][:len(shown)], "an item was cut or altered"
    assert view["result"]["state_version"] == 3
    assert "result_read" in view["retrieve"]
    assert out["tokens"] <= 400


def test_a_value_too_large_to_show_is_named_not_cut(cfg):
    import json as _json

    from amoeba.tools import deliver_tool_result

    payload = {"namespace": "id", "prompt_text": "doctrine " * 800}
    out = deliver_tool_result(payload, budget_tokens=120, count=lambda t: len(t) // 3,
                              store=lambda _text: "c" * 64)
    view = _json.loads(out["text"])
    marker = view["result"]["prompt_text"]
    assert marker["omitted"] == "string" and marker["path"] == "prompt_text"
    assert marker["chars"] == len(_json.dumps(payload["prompt_text"]))
    assert view["result"]["namespace"] == "id"
    assert view["omitted"] == ["prompt_text"]


def test_a_projection_claims_no_copy_it_was_not_given(cfg):
    """Pointing at a copy nobody stored would be worse than saying nothing."""
    import json as _json

    from amoeba.tools import deliver_tool_result

    out = deliver_tool_result({"items": ["y" * 100 for _ in range(60)]},
                              budget_tokens=300)
    view = _json.loads(out["text"])
    assert out["complete"] is False
    assert out["sha256"] is None and view["result_ref"] is None
    assert view["retrieve"] == "narrow the call"
    assert out["counted"] == "characters"


def test_the_copy_a_projection_names_is_really_there(mind):
    """The reference has to resolve, against a real blob store."""
    import json as _json

    from amoeba.tools import deliver_tool_result

    payload = {"items": [{"id": f"w{i}", "note": "z" * 80} for i in range(60)]}
    out = deliver_tool_result(
        payload, budget_tokens=300,
        store=lambda text: mind.blobs.put(text.encode("utf-8")))
    assert out["complete"] is False and out["sha256"]
    stored = mind.blobs.get(out["sha256"]).decode("utf-8")
    assert _json.loads(stored) == payload, "the stored copy is not the result"


def test_the_neuocyte_feeds_back_the_bounded_text_it_was_given(cfg):
    """The process renders; it does not re-cut.

    The limit used to live in both cognitive processes as a bare `[:2000]`,
    free to drift from each other and from the Harness. Cutting again here
    would also remove the notice, which sits at the end of the text -- the
    truncation would become silent again at the last step.
    """
    inf = _FakeInference([TOOL_CALL, "FINDING: done"])
    bounded = '{"complete": false, "result_ref": null, "retrieve": "narrow the call", "result": "SHOWN"}'
    sup = _FakeSupervisorRpc({"name": "current_state_version", "accepted": True,
                              "result": {"items": ["ignored"]},
                              "result_text": bounded, "result_complete": False,
                              "result_chars": 9000, "result_sha256": "b" * 64,
                              "error": None, "reason": "executed",
                              "receipt_id": "rcp_1"})
    nc = _neuocyte(cfg, inf, sup)
    nc._generate_with_tools(ITEM, budget=4000, deadline=time.time() + 60,
                            max_turns=6)

    fed = "\n".join(inf.ingested)
    assert '"complete": false' in fed, "the bounded view never reached the model"
    assert "ignored" not in fed, "the process re-serialized instead of rendering"


# ===========================================================================
# Specialist neuocytes: asked for by Ego, governed by the library
# ===========================================================================
class _BindingSup:
    """Answers `bind_profile` for some namespaces and refuses the rest."""

    def __init__(self, approved: set[str]) -> None:
        self.approved = approved
        self.attempts: list[str] = []
        self.recorded: list[dict] = []

    def call(self, method: str, **params):
        if method == "bind_profile":
            ns = params["namespace"]
            self.attempts.append(ns)
            if ns not in self.approved:
                raise RuntimeError(f"no approved version for {ns}")
            return {"profile_ref": f"{ns}@1", "text": f"you are {ns}"}
        if method == "record_profile_fallback":
            self.recorded.append(params)
            return {"recorded": True}
        raise AssertionError(f"unexpected call {method}")


def _binding_neuocyte(cfg, sup):
    nc = Neuocyte.__new__(Neuocyte)
    nc.cfg = cfg
    nc.neuocyte_id = "nc_spec"
    nc.sup = sup
    nc.model_generation = "gen_1"
    nc.profile = None
    from amoeba.logging_setup import get_logger
    nc.log = get_logger("test")
    return nc


def test_a_requested_specialisation_is_bound(cfg):
    """The whole point: `ego.neuocyte.research` can finally be born into.

    It could be authored, versioned, approved, selected and advertised, and
    then birth bound `ego.neuocyte` regardless because the namespace came from
    `work_class` alone. A specialisation nothing can be born into does not
    exist.
    """
    sup = _BindingSup({"ego.neuocyte.research", "ego.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"work_class": "user",
                                         "specialisation": "research"}, work_id="w1")
    assert bound["profile_ref"] == "ego.neuocyte.research@1"
    assert sup.attempts == ["ego.neuocyte.research"], (
        "the base was tried even though the specialist bound")
    assert sup.recorded == [], "a successful bind needs no annotation"


def test_an_unapproved_specialisation_falls_back_and_says_so(cfg):
    """Refusing the work would be the worse failure; doing it silently is next.

    The library not having an approved `research` profile is a reason to run
    on the baseline, not a reason to abandon the job. But a specialisation
    that has quietly stopped applying looks exactly like one nobody asked for,
    so the fallback is recorded against the work item.
    """
    sup = _BindingSup({"ego.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"work_class": "user",
                                         "specialisation": "research"}, work_id="w2")
    assert bound["profile_ref"] == "ego.neuocyte@1", "the work did not run"
    assert sup.attempts == ["ego.neuocyte.research", "ego.neuocyte"]
    assert len(sup.recorded) == 1
    assert sup.recorded[0]["work_id"] == "w2"
    assert "ego.neuocyte.research" in sup.recorded[0]["reason"]


def test_maintenance_work_specialises_under_id(cfg):
    """A neuocyte descends from the mind that spawned it, specialised or not.

    Maintenance work forked from Id must not reach into Ego's family tree by
    naming a leaf, which is why Ego names a leaf and never a namespace.
    """
    sup = _BindingSup({"id.neuocyte.integrity"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"work_class": "maintenance",
                                         "specialisation": "integrity"}, work_id="w3")
    assert bound["profile_ref"] == "id.neuocyte.integrity@1"
    assert sup.attempts == ["id.neuocyte.integrity"]


def test_work_with_no_specialisation_binds_the_base_directly(cfg):
    """The ordinary case gains no extra round trip."""
    sup = _BindingSup({"ego.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"work_class": "user"}, work_id="w4")
    assert bound["profile_ref"] == "ego.neuocyte@1"
    assert sup.attempts == ["ego.neuocyte"]
    assert sup.recorded == []


def test_no_profile_at_all_is_survivable_and_reported(cfg):
    """An empty library is not a reason to stop working."""
    sup = _BindingSup(set())
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"work_class": "user",
                                         "specialisation": "research"}, work_id="w5")
    assert bound is None
    assert nc.profile is None
    assert sup.attempts == ["ego.neuocyte.research", "ego.neuocyte"]
    assert len(sup.recorded) == 1, "falling back to nothing went unrecorded"


def test_ego_cannot_name_a_namespace_only_a_leaf(stack):
    """Ego states intent; it does not address the family tree.

    A specialisation that could contain dots would let Ego ask for
    `id.neuocyte` -- or a root -- by writing one. The parameter takes a single
    name and the Harness builds the namespace, so reaching sideways is not
    refused so much as unsayable.
    """
    import pytest as _pytest

    for bad in ("ego.neuocyte.research", "../root", "id.neuocyte", "a b"):
        with _pytest.raises(Exception) as exc:
            stack.call("ego_request_work", objective="look into something",
                       specialisation=bad)
        assert "single name" in str(exc.value) or "not a path" in str(exc.value), \
            f"{bad!r} was not refused as a path"


def test_a_specialisation_reaches_the_work_item(stack):
    """Ego asks at request time and the row remembers, so birth can read it."""
    out = stack.call("ego_request_work", objective="research the build logs",
                     specialisation="research")
    admitted = [a for a in out.get("admitted", []) if a.get("admitted")]
    assert admitted, out
    work_id = admitted[0]["work_id"]
    row = stack.call("get_work", work_id=work_id)
    assert row["specialisation"] == "research"
