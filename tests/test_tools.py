"""Model-generated tool calls must pass through the harness.

A model can *request* a tool call. It cannot perform one. Every request is
parsed out of generated text, validated against a declared schema, checked
against the caller's role permissions, executed by the harness under a declared
timeout, and recorded.
"""

from __future__ import annotations

import pytest

from amoeba.errors import InvalidInput
from amoeba.tools import (
    ToolParam, ToolRegistry, ToolSpec, build_default_registry,
    parse_tool_calls, strip_tool_calls,
)


@pytest.fixture()
def registry(mind) -> ToolRegistry:
    return build_default_registry(mind)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_parses_a_well_formed_request():
    text = ('I will look this up.\n'
            '<tool_call>{"name": "recall_memory", "arguments": {"query": "port"}}</tool_call>')
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "recall_memory"
    assert calls[0].arguments == {"query": "port"}
    assert strip_tool_calls(text) == "I will look this up."


def test_malformed_json_is_surfaced_not_silently_dropped():
    calls = parse_tool_calls('<tool_call>{not json at all}</tool_call>')
    assert len(calls) == 1
    assert calls[0].name == "<unparseable>"
    assert calls[0].raw


def test_non_object_payload_is_rejected():
    calls = parse_tool_calls('<tool_call>{"name": "x", "arguments": [1,2,3]}</tool_call>')
    assert calls[0].arguments == {"<invalid_arguments>": [1, 2, 3]}


def test_request_count_is_bounded():
    text = '<tool_call>{"name":"recall_memory","arguments":{}}</tool_call>' * 20
    assert len(parse_tool_calls(text, limit=4)) == 4


def test_prose_without_a_tool_block_requests_nothing():
    assert parse_tool_calls("Let me call recall_memory for you.") == []


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------
def test_unknown_tool_is_refused(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "no such tool" in out.reason


def test_unknown_argument_is_refused(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"qeury":"typo"}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "unknown argument" in out.reason


def test_missing_required_argument_is_refused(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"get_conclusion","arguments":{}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "missing required argument" in out.reason


def test_wrong_type_is_refused(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"limit":"five"}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "must be integer" in out.reason


def test_boolean_is_not_accepted_as_an_integer(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"limit":true}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False


def test_out_of_range_value_is_refused(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"limit":9999}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "above maximum" in out.reason


def test_overlong_string_is_refused(registry: ToolRegistry):
    long = "x" * 5000
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"query":"%s"}}</tool_call>'
        % long)[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is False
    assert "max length" in out.reason


# ---------------------------------------------------------------------------
# permissions
# ---------------------------------------------------------------------------
def test_role_permissions_are_enforced(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"read_history","arguments":{"limit":5}}</tool_call>')[0]
    # Raw history is evidence for audit; Ego reads maintained memory instead.
    denied = registry.execute(call, role="ego")
    assert denied.accepted is False
    assert "may not call this tool" in denied.reason

    allowed = registry.execute(call, role="id")
    assert allowed.accepted is True


def test_prompt_block_only_advertises_permitted_tools(registry: ToolRegistry):
    ego_block = registry.prompt_block(role="ego")
    id_block = registry.prompt_block(role="id")
    assert "read_history" not in ego_block
    assert "read_history" in id_block
    assert "recall_memory" in ego_block


def test_schemas_are_role_scoped(registry: ToolRegistry):
    ego_names = {s["name"] for s in registry.schemas(role="ego")}
    id_names = {s["name"] for s in registry.schemas(role="id")}
    assert "read_history" in id_names and "read_history" not in ego_names


def test_declared_schema_shape(registry: ToolRegistry):
    schema = next(s for s in registry.schemas() if s["name"] == "recall_memory")
    assert schema["parameters"]["type"] == "object"
    assert "query" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["limit"]["maximum"] == 20


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def test_permitted_call_executes_against_real_state(registry: ToolRegistry, mind):
    mind.memory.remember(kind="belief", claim="the arbiter owns hard limits",
                         confidence=0.9, created_by="operator")
    call = parse_tool_calls(
        '<tool_call>{"name":"recall_memory","arguments":{"query":"arbiter"}}</tool_call>')[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is True and out.error is None
    assert out.result[0]["claim"] == "the arbiter owns hard limits"
    assert out.duration_seconds >= 0


def test_handler_exception_is_captured_not_raised(registry: ToolRegistry):
    call = parse_tool_calls(
        '<tool_call>{"name":"get_conclusion","arguments":{"conclusion_id":"nope"}}</tool_call>'
    )[0]
    out = registry.execute(call, role="ego")
    assert out.accepted is True          # the request was well formed
    assert out.error is not None         # but the handler failed
    assert "NotFound" in out.error


def test_over_budget_execution_is_flagged():
    import time

    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="slow", description="sleeps past its declared budget",
        params=[], handler=lambda _context=None: time.sleep(0.05) or "done",
        timeout_seconds=0.001,
    ))
    call = parse_tool_calls('<tool_call>{"name":"slow","arguments":{}}</tool_call>')[0]
    out = reg.execute(call, role="ego")
    assert out.reason == "executed_over_budget"
    assert "exceeded declared timeout" in out.error


def test_no_registered_tool_can_reach_a_shell_or_the_network(registry: ToolRegistry):
    """The registry is the whole attack surface for model-generated strings."""
    import inspect

    for name in registry.names():
        spec = registry.get(name)
        source = inspect.getsource(spec.handler)
        for forbidden in ("subprocess", "os.system", "eval(", "exec(",
                          "socket", "urlopen", "requests.", "shutil.rmtree"):
            assert forbidden not in source, (name, forbidden)


def test_default_registry_is_read_only(registry: ToolRegistry):
    for name in registry.names():
        assert registry.get(name).mutates_state is False


def test_ego_converse_records_tool_requests_without_executing_them(stack):
    """ego_converse parses and reports requests; it does not run them."""
    turn = stack.call("ego_converse", message="anything",
                      idempotency_key="tool-report-1")
    assert "tool_requests" in turn["result"]
    assert isinstance(turn["result"]["tool_requests"], list)


def test_the_execution_loop_is_wired_and_the_docs_say_so():
    """The inverse of the guard this replaces.

    This file used to assert that nothing called `ToolRegistry.execute`, so
    that the capability could not be re-claimed in docs before it existed. It
    exists now, so the assertion flips: the registry must actually be reached
    from production code, and the docs must no longer say it is not.

    Keeping the check rather than deleting it is the point. The failure mode it
    guards against is symmetric -- docs drifting away from the code in either
    direction.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "amoeba"
    callers = []
    for py in src.rglob("*.py"):
        if py.name == "tools.py":
            continue
        if "build_neuocyte_registry" in py.read_text(encoding="utf-8"):
            callers.append(py.name)
    assert "harness_api.py" in callers, (
        "the tool registry is not reached from the Harness; the execution "
        "loop is not actually wired")

    docs = pathlib.Path(__file__).resolve().parents[1] / "docs"
    impl = (docs / "IMPLEMENTATION.md").read_text(encoding="utf-8")
    row = [ln for ln in impl.splitlines() if "Tool *execution* loop" in ln]
    assert row and "**Wired**" in row[0], (
        "the capability table still says the loop is not wired")
    arch = " ".join((docs / "ARCHITECTURE.md").read_text(encoding="utf-8").split())
    assert "no production code path calls `ToolRegistry.execute`" not in arch, (
        "ARCHITECTURE.md still claims nothing executes tools")


def test_execution_stays_out_of_the_neuocyte_process():
    """Wiring the loop must not move the authority into the neuocyte.

    The neuocyte parses a request and sends it to the Harness. If it ever
    builds a registry or calls `execute` itself, the separation that makes
    `sandbox_allowed` meaningful is gone.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "amoeba"
    worker = (src / "neuocyte.py").read_text(encoding="utf-8")
    for forbidden in ("build_neuocyte_registry", "ToolRegistry", ".execute("):
        assert forbidden not in worker, (
            f"neuocyte.py references {forbidden}; tool execution belongs to "
            "the Harness, not the process holding the model output")


def test_ego_converse_reports_tool_requests_as_unexecuted(stack):
    """The limitation is load-bearing: a client must not assume tools ran."""
    turn = stack.call("ego_converse", message="anything",
                      idempotency_key="tool-not-run")
    assert isinstance(turn["result"]["tool_requests"], list)
    if turn["result"]["tool_requests"]:
        assert any("does not execute tools" in lim for lim in turn["limitations"])
