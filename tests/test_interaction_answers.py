"""Answers belong to interactions, not to turns.

The live failure this defends: asked to review its own previous answer, Ego
produced four bounded turns, each cut off at exactly 384 tokens. The operator
received 1777 characters beginning "Continuing from where the previous
analysis left off: --- Claim 4 (continued)" and ending mid-list at
`get_conclusion`. Claims one to three had been generated, recorded against
their turns, and never delivered, because the answer of record was written
from whichever turn *ended* the thought -- and the fourth turn only ended it
because the continuation limit said so, which was then reported as an ordinary
answer.

Four things were wrong, and each has its own tests here:

  * the answer was the last turn's text instead of every turn's, in order;
  * running out of continuations was reported as a finished answer;
  * Ego had no governed output ceiling at all -- a hardcoded 384 decided it --
    and a global 512 backstop would have clamped any real one;
  * a continuation reached the model as a new "please continue" message with
    the whole environment re-rendered in front of it, so every piece of a
    long answer opened with its own preamble and cost ~925 tokens before it
    said a word.

Turns are allowed to end. The user's answer is not allowed to disappear
between them.
"""

from __future__ import annotations

import http.client
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

from amoeba import mailbox
from amoeba.arbiter import Arbiter
from amoeba.config import ArbiterConfig
from amoeba.errors import ResourceExhausted
from amoeba.promptlib import bootstrap
from amoeba.promptlib.model import (FALLBACK_OUTPUT_CEILINGS,
                                    fallback_output_ceiling)
from amoeba.promptlib.resolver import Resolver
from amoeba.promptlib.store import PromptStore
from amoeba.rpc import RpcClient, read_or_create_token
from conftest import start_stack
from test_persistent_turns import (_answer_of, _claim, _complete,
                                   _queue_lineage, _request)
from test_prompt_library import _approve_and_select, _author, _Sup

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")
needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node to drive the panel")

# Distinctive, so a dropped, duplicated or reordered piece shows up in the
# assembled text rather than hiding inside something that reads plausibly.
ALPHA, BRAVO, CHARLIE = "Alpha one: the first claim. ", "Bravo two: the second. ", "Charlie three."


def _answer_record(mind, trigger_id):
    row = mind.db.conn.execute(
        "SELECT answer_status, answer_sha256 FROM role_triggers"
        " WHERE trigger_id = ?", (trigger_id,)).fetchone()
    return (row["answer_status"],
            mind.blobs.get_json(row["answer_sha256"]) if row["answer_sha256"] else None)


def _piece(text, *, resumed):
    """What a role reports for one bounded turn: its raw segment, and how it began."""
    return {"answer": text.strip(), "segment": text, "resumed": resumed}


def _cut(mind, turn, text, *, resumed=False, **kw):
    """Close a turn the way the output ceiling closes it."""
    return _complete(mind, turn["turn_id"], stop_reason="max_output_tokens",
                     result=_piece(text, resumed=resumed), **kw)


# ===========================================================================
# assembly
# ===========================================================================
def test_an_answer_spanning_three_turns_arrives_whole_and_in_order(mind):
    """The regression itself, at the layer where the answer is written.

    Three bounded turns, the first two cut off by their ceiling and the third
    ending because the model finished. Every piece must reach the answer, in
    order, once -- and nothing may be delivered before the third.
    """
    req = _request(mind, summary="review your previous answer")

    t1 = _claim(mind, "ego")
    out = _cut(mind, t1, ALPHA, max_continuations=3)
    assert out["continuation"], "the interrupted thought was not continued"
    assert _answer_of(mind, req["trigger_id"]) == (None, None), \
        "a turn that was cut off was treated as the end of the interaction"

    t2 = _claim(mind, "ego")
    assert t2["parent_turn"] == t1["turn_id"]
    out = _cut(mind, t2, BRAVO, resumed=True, max_continuations=3)
    assert out["continuation"]
    assert _answer_of(mind, req["trigger_id"]) == (None, None), \
        "a second cut-off turn was treated as the end of the interaction"

    t3 = _claim(mind, "ego")
    assert t3["parent_turn"] == t2["turn_id"]
    _complete(mind, t3["turn_id"], stop_reason="model_stop",
              result=_piece(CHARLIE, resumed=True), max_continuations=3)

    status, record = _answer_record(mind, req["trigger_id"])
    assert status == "answered"
    assert record["answer"] == (ALPHA + BRAVO + CHARLIE).strip()
    assert record["complete"] is True and record["ended_because"] == "model_stop"
    # Which turns it came from, in order: provenance back to each producer.
    assert [s["turn_id"] for s in record["segments"]] == [
        t1["turn_id"], t2["turn_id"], t3["turn_id"]]
    assert [s["ordinal"] for s in record["segments"]] == [0, 1, 2]

    # One thought, one owner: every turn serves the same interaction.
    lineages = {r["lineage"] for r in mind.db.conn.execute(
        "SELECT lineage FROM role_turns WHERE turn_id IN (?,?,?)",
        (t1["turn_id"], t2["turn_id"], t3["turn_id"]))}
    assert len(lineages) == 1


def test_each_piece_appears_exactly_once(mind):
    """No piece dropped, doubled or overwritten by a later one."""
    req = _request(mind)
    pieces = [f"piece-{i} " for i in range(5)]
    turn = _claim(mind, "ego")
    for i, text in enumerate(pieces[:-1]):
        _cut(mind, turn, text, resumed=i > 0, max_continuations=8)
        turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result=_piece(pieces[-1], resumed=True), max_continuations=8)

    answer = _answer_record(mind, req["trigger_id"])[1]["answer"]
    for text in pieces:
        assert answer.count(text.strip()) == 1, (text, answer)
    positions = [answer.index(t.strip()) for t in pieces]
    assert positions == sorted(positions), "pieces arrived out of order"


def test_a_resumed_piece_joins_exactly_where_it_was_cut(mind):
    """Pieces join byte for byte, whether resumed or asked for visibly.

    Stripping each piece before joining would weld "self" to " knowledge";
    inserting a separator would add text the model did not write, after
    telling it that its output is appended directly.
    """
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, "no direct observation of internal states or self",
         max_continuations=3)
    t2 = _claim(mind, "ego")
    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result=_piece("-knowledge, so this is inferred.", resumed=True),
              max_continuations=3)
    assert _answer_of(mind, req["trigger_id"])[1] == (
        "no direct observation of internal states or self-knowledge, so this "
        "is inferred.")

    other = _request(mind)
    t3 = _claim(mind, "ego")
    _cut(mind, t3, "asked visibly, the first half ", max_continuations=3)
    t4 = _claim(mind, "ego")
    _complete(mind, t4["turn_id"], stop_reason="model_stop",
              result=_piece("and then the second.", resumed=False),
              max_continuations=3)
    assert _answer_of(mind, other["trigger_id"])[1] == \
        "asked visibly, the first half and then the second."


def test_a_tool_call_cut_in_half_leaves_no_machinery_in_the_answer(mind):
    """The boundary can land inside a tool call; the answer must not show it."""
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, 'Let me check. <tool_call>{"name": "get_mem', max_continuations=3)
    t2 = _claim(mind, "ego")
    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result=_piece('ory", "arguments": {}}</tool_call> The record agrees.',
                            resumed=True), max_continuations=3)
    answer = _answer_of(mind, req["trigger_id"])[1]
    assert "tool_call" not in answer and "get_mem" not in answer, answer
    assert answer.startswith("Let me check.") and answer.endswith("The record agrees.")


# ===========================================================================
# ownership
# ===========================================================================
def test_a_rival_request_cannot_enter_the_chain_or_its_answer(mind):
    """A question arriving mid-thought waits for its own turn and its own answer."""
    first = _request(mind, summary="the long question")
    t1 = _claim(mind, "ego")
    _cut(mind, t1, ALPHA, max_continuations=3)

    rival = _request(mind, summary="an unrelated question")
    t2 = _claim(mind, "ego")
    assert rival["trigger_id"] not in {t["trigger_id"] for t in t2["triggers"]}, \
        "a continuation turn admitted an unrelated request"
    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result=_piece(BRAVO, resumed=True), max_continuations=3)

    assert _answer_of(mind, first["trigger_id"])[1] == (ALPHA + BRAVO).strip()
    assert _answer_of(mind, rival["trigger_id"]) == (None, None)

    t3 = _claim(mind, "ego")
    _complete(mind, t3["turn_id"], stop_reason="model_stop",
              result=_piece("the rival's own answer", resumed=False))
    rival_answer = _answer_of(mind, rival["trigger_id"])[1]
    assert rival_answer == "the rival's own answer"
    assert ALPHA.strip() not in rival_answer and BRAVO.strip() not in rival_answer


def test_fragments_are_attached_to_their_own_interaction(mind):
    """Two multi-turn interactions in a row; neither receives the other's pieces."""
    a = _request(mind, summary="question A")
    ta = _claim(mind, "ego")
    _cut(mind, ta, "A-first ", max_continuations=3)
    ta2 = _claim(mind, "ego")
    _complete(mind, ta2["turn_id"], stop_reason="model_stop",
              result=_piece("A-second", resumed=True), max_continuations=3)

    b = _request(mind, summary="question B")
    tb = _claim(mind, "ego")
    _cut(mind, tb, "B-first ", max_continuations=3)
    tb2 = _claim(mind, "ego")
    _complete(mind, tb2["turn_id"], stop_reason="model_stop",
              result=_piece("B-second", resumed=True), max_continuations=3)

    assert _answer_of(mind, a["trigger_id"])[1] == "A-first A-second"
    assert _answer_of(mind, b["trigger_id"])[1] == "B-first B-second"


def test_retrying_an_abandoned_continuation_does_not_duplicate_a_piece(mind):
    """A crash mid-continuation is retried, and its answer is still said once."""
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, ALPHA, max_continuations=3)

    dead = _claim(mind, "ego")
    mind.writer.apply(lambda m: mailbox.abandon(m, mind, turn_id=dead["turn_id"],
                                                reason="role died"),
                      actor="harness", bump_version=False)
    retry = _claim(mind, "ego")
    assert retry["parent_turn"] == t1["turn_id"]
    _complete(mind, retry["turn_id"], stop_reason="model_stop",
              result=_piece(BRAVO, resumed=True), max_continuations=3)

    status, record = _answer_record(mind, req["trigger_id"])
    assert record["answer"] == (ALPHA + BRAVO).strip()
    assert dead["turn_id"] not in [s["turn_id"] for s in record["segments"]]


# ===========================================================================
# terminal truth
# ===========================================================================
def test_running_out_of_continuations_is_incomplete_not_complete(mind):
    """A model that never finishes: the loop ends, and says it ended early.

    Every piece so far is kept, in order, and the status says plainly that
    this is not a finished answer -- rather than the old behaviour, which kept
    only the last piece and called it "answered".
    """
    req = _request(mind)
    turn = _claim(mind, "ego")
    said, turns = [], 0
    for i in range(20):
        text = f"segment-{i} "
        said.append(text)
        turns += 1
        out = _cut(mind, turn, text, resumed=i > 0, max_continuations=3)
        if out["continuation"] is None:
            assert out["continuation_limit_reached"] is True
            break
        turn = _claim(mind, "ego")
    assert turns == 4, "the chain did not stop at its bound"

    status, record = _answer_record(mind, req["trigger_id"])
    assert status == "incomplete"
    assert record["complete"] is False
    assert record["ended_because"] == "continuation_limit"
    assert record["answer"] == "".join(said).strip()


def test_a_stop_that_is_not_the_model_finishing_is_not_complete(mind):
    """A deadline ends the thought too, and does not conclude it."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="deadline_reached",
              result=_piece("as far as it got", resumed=False))
    status, record = _answer_record(mind, req["trigger_id"])
    assert status == "incomplete" and record["ended_because"] == "deadline_reached"


def test_a_thought_that_said_nothing_is_unanswerable(mind):
    req = _request(mind)
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result=_piece("   ", resumed=False))
    assert _answer_of(mind, req["trigger_id"]) == ("unanswerable", None)


# ===========================================================================
# resuming the cut-off message
# ===========================================================================
def _cut_at(mind, turn, text, *, session="sess-1", start=100, end=140, **kw):
    return _complete(mind, turn["turn_id"], stop_reason="max_output_tokens",
                     result=_piece(text, resumed=False), session_handle=session,
                     token_start=start, token_end=end, **kw)


def test_a_bare_continuation_is_offered_its_parents_exact_position(mind):
    _request(mind)
    t1 = _claim(mind, "ego")
    _cut_at(mind, t1, "cut off mid", max_continuations=3)
    t2 = _claim(mind, "ego")
    assert t2["resume"] == {"parent_turn": t1["turn_id"],
                            "session_handle": "sess-1", "token_end": 140,
                            "carry": ""}


def test_a_continuation_carrying_evidence_asks_visibly_instead(mind):
    """Evidence may ride with a continuation, and has to be shown.

    A resumed generation has nowhere to show it, so the turn falls back to a
    visible ask -- rather than resuming and silently dropping the evidence.
    """
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut_at(mind, t1, "cut off mid", max_continuations=3)
    _queue_lineage(mind, t1["lineage"], kind="work_completed", source="harness",
                   summary="the work it delegated came back")
    t2 = _claim(mind, "ego")
    assert len(t2["triggers"]) == 2, "the evidence did not reach the continuation"
    assert t2["resume"] is None, "a turn with evidence in it tried to resume"
    assert req  # the request is still owed; this is about the turn's shape


def test_a_turn_that_ran_in_between_forbids_resuming(mind):
    """Only a continuation of the role's *last* turn can pick up where it was."""
    _request(mind)
    t1 = _claim(mind, "ego")
    _cut_at(mind, t1, "cut off mid", max_continuations=3)
    # Something else ran on this role after the parent was cut off.
    def between(m):
        m.sql("UPDATE role_turns SET started_at = started_at - 1000"
              " WHERE turn_id = ?", (t1["turn_id"],))
        m.sql("INSERT INTO role_turns(turn_id, role, incarnation, trigger_kinds,"
              " trigger_count, started_at, status, state_version)"
              " VALUES ('turn_between', 'ego', 1, '[]', 0, ?, 'completed', 0)",
              (time.time(),))

    mind.writer.apply(between, actor="test", bump_version=False)
    t2 = _claim(mind, "ego")
    assert t2["resume"] is None


def test_a_carry_is_the_tool_call_the_ceiling_cut_in_half(mind):
    _request(mind)
    t1 = _claim(mind, "ego")
    _cut_at(mind, t1, 'checking <tool_call>{"name": "get_mem', max_continuations=3)
    t2 = _claim(mind, "ego")
    assert t2["resume"]["carry"] == '<tool_call>{"name": "get_mem'


# ===========================================================================
# ceilings
# ===========================================================================
EXPECTED = {"ego": 3072, "id": 1024, "ego.neuocyte": 512, "id.neuocyte": 384}


@pytest.fixture()
def library(mind):
    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")
    return store, Resolver(store)


def test_each_mind_is_granted_its_own_ceiling_by_its_governed_profile(library):
    """The values a freshly installed organism actually binds."""
    _, resolver = library
    for namespace, ceiling in EXPECTED.items():
        resolved = resolver.resolve_selected(namespace)
        assert resolved.model_vars.get("max_output_tokens") == ceiling, namespace
        assert resolved.backend_kwargs().get("max_tokens") == ceiling, namespace


def test_changing_one_ceiling_does_not_move_another(mind, library):
    """Ego's room is Ego's. A worker forked from Ego keeps the worker's."""
    store, resolver = library
    _, created = mind.writer.apply(
        lambda m: store.create_runtime_version(
            m, namespace="ego", prompt_mode="replace",
            prompt_text=resolver.resolve_selected("ego").prompt_text,
            model_vars={"temperature": 0.7, "top_p": 0.95,
                        "max_output_tokens": 2048},
            parent_version=None, origin="operator", created_by="test",
            state="candidate"),
        actor="test")
    ego_v2 = resolver.resolve_version("ego", created["local_version"])
    assert ego_v2.model_vars["max_output_tokens"] == 2048
    for namespace in ("id", "ego.neuocyte", "id.neuocyte"):
        assert resolver.resolve_selected(namespace).model_vars[
            "max_output_tokens"] == EXPECTED[namespace], namespace


def test_the_fallback_ceilings_match_the_shipped_headers():
    """The fallback is a copy of governed values, and must not drift from them.

    It is a code constant rather than a read of the shipped file so that
    editing a shipped file makes a candidate rather than silently reaching a
    running mind -- which is exactly why something has to keep the two equal.
    """
    shipped = {p["namespace"]: p["model_vars"].get("max_output_tokens")
               for p in bootstrap.load_prompt_files()}
    for namespace, ceiling in FALLBACK_OUTPUT_CEILINGS.items():
        assert shipped.get(namespace) == ceiling, namespace
    assert FALLBACK_OUTPUT_CEILINGS == EXPECTED


def test_a_silent_specialist_inherits_the_worker_ceiling_not_egos():
    assert fallback_output_ceiling("ego.neuocyte.reviewer") == 512
    assert fallback_output_ceiling("id.neuocyte.gc") == 384
    assert fallback_output_ceiling("ego") == 3072


def test_the_backstop_clamps_no_governed_ceiling():
    """The global cap is a backstop; at 512 it silently overruled Ego's 3072."""
    cfg = ArbiterConfig()
    assert cfg.max_completion_tokens >= max(FALLBACK_OUTPUT_CEILINGS.values())
    clamp = Arbiter(cfg).clamp_inference(prompt_tokens=100, max_tokens=3072,
                                         deadline=None, budget_tokens=16384)
    assert clamp["max_tokens"] == 3072


def test_the_shipped_config_backstop_clamps_no_governed_ceiling():
    from amoeba.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    assert cfg.arbiter.max_completion_tokens >= max(FALLBACK_OUTPUT_CEILINGS.values())


# ===========================================================================
# headroom
# ===========================================================================
def test_a_generation_is_admitted_only_if_its_allowance_fits():
    """A 3072-token ceiling is not real if the room was never reserved.

    Checking the prompt alone admitted a session a few hundred tokens short of
    its budget, which then generated straight through it.
    """
    arbiter = Arbiter(ArbiterConfig())
    ok = arbiter.clamp_inference(prompt_tokens=16384 - 3072, max_tokens=3072,
                                 deadline=None, budget_tokens=16384)
    assert ok["max_tokens"] == 3072
    with pytest.raises(ResourceExhausted) as refused:
        arbiter.clamp_inference(prompt_tokens=16384 - 3072 + 1, max_tokens=3072,
                                deadline=None, budget_tokens=16384)
    # Worded so a role routes it to rejuvenation rather than to the crash
    # path: `roles.CONTEXT_PRESSURE_MARKERS` matches on the message.
    from amoeba.roles import _is_context_pressure

    assert _is_context_pressure(refused.value)


# ===========================================================================
# the running organism, and the page the operator reads it on
# ===========================================================================
def _inference(stack) -> RpcClient:
    cfg = stack.cfg
    client = RpcClient(cfg.supervisor_host, cfg.inference_port,
                       read_or_create_token(cfg.token_path), timeout=60)
    client.connect(retries=20, delay=0.5)
    return client


def _await_answer(stack, trigger_id, timeout=180):
    deadline = time.time() + timeout
    while True:
        state = stack.call("role_answer", trigger_id=trigger_id)
        if state["status"] in ("completed", "incomplete", "unanswerable", "expired"):
            return state
        assert state["answer"] == "", "text was delivered before the answer was"
        assert time.time() < deadline, "no terminal answer"
        time.sleep(0.2)


def _chain(stack, last_turn):
    turns, current = [], last_turn
    while current:
        detail = stack.call("role_turn", turn_id=current)
        turns.append(detail)
        current = detail.get("parent_turn")
    return list(reversed(turns))


@pytest.fixture()
def answering(tmp_path):
    # Id stays quiet so nothing else draws from the scripted queue, and the
    # chain may run three continuations, as production allows.
    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False,
                                             "max_continuations": 3})
    stack.wait_for_children(timeout=90)
    yield stack
    stack.stop()


@live
def test_a_three_turn_answer_reaches_the_operator_whole(answering):
    """The live chain: cut off twice, finished on the third, delivered entire."""
    _inference(answering).call("script_responses", responses=[
        {"text": ALPHA, "finish_reason": "length"},
        {"text": BRAVO, "finish_reason": "length", "continues": True},
        {"text": CHARLIE, "finish_reason": "stop", "continues": True},
    ])
    env = answering.call("ego_converse", message="review your previous answer",
                         wait=False)
    trigger_id = env["result"]["trigger_id"]
    state = _await_answer(answering, trigger_id)

    assert state["status"] == "completed"
    assert state["answer"] == f"[SIMULATED] {ALPHA}{BRAVO}{CHARLIE}".strip()

    turn = answering.call("role_turns", role="ego")["turns"][0]["turn_id"]
    chain = _chain(answering, turn)
    assert [t["stop_reason"] for t in chain] == [
        "max_output_tokens", "max_output_tokens", "model_stop"]
    assert chain[0]["trigger_kinds"] == ["user_input"]
    assert all(t["trigger_kinds"] == ["continuation"] for t in chain[1:])
    # Resumed, not re-asked: each continuation starts exactly where its parent
    # ended and appends nothing but what it generated -- no environment, no
    # "please continue". Asked visibly, each would have re-ingested hundreds.
    for parent, child in zip(chain, chain[1:]):
        assert child["token_start"] == parent["token_end"]
    assert chain[1]["token_end"] - chain[1]["token_start"] == len(BRAVO.split())
    assert chain[2]["token_end"] - chain[2]["token_start"] == len(CHARLIE.split())


@live
def test_running_out_of_continuations_reaches_the_client_as_incomplete(tmp_path):
    """The model never finishes: everything it said arrives, marked unfinished."""
    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False,
                                             "max_continuations": 1})
    try:
        stack.wait_for_children(timeout=90)
        _inference(stack).call("script_responses", responses=[
            {"text": "first part, ", "finish_reason": "length"},
            {"text": "second part", "finish_reason": "length", "continues": True},
        ])
        env = stack.call("ego_converse", message="go on forever", wait=False)
        state = _await_answer(stack, env["result"]["trigger_id"])
        assert state["status"] == "incomplete"
        assert state["ended_because"] == "continuation_limit"
        assert state["answer"] == "[SIMULATED] first part, second part"
    finally:
        stack.stop()


@live
def test_a_long_answer_is_not_cut_by_characters_anywhere(answering):
    """No 4000-character slice survives between the model and the operator."""
    long_text = " ".join(f"word{i}" for i in range(2400))    # ~19k characters
    _inference(answering).call("script_responses", responses=[
        {"text": long_text, "finish_reason": "stop"}])
    env = answering.call("ego_converse", message="say a great deal", wait=False)
    state = _await_answer(answering, env["result"]["trigger_id"])
    assert state["status"] == "completed"
    assert state["answer"] == f"[SIMULATED] {long_text}"
    assert len(state["answer"]) > 15000


@live
@needs_node
def test_converse_renders_the_live_answer_whole(answering):
    """The operator's path end to end: HTTP, the served page, the real payload.

    The answer is produced by a live organism across three bounded turns,
    retrieved over the operator HTTP surface exactly as the browser does, and
    rendered by the page the server actually serves.
    """
    cfg = answering.cfg
    base = urlparse(f"http://{cfg.api_host}:{cfg.api_port}")
    token = cfg.operator_session_path.read_text(encoding="utf-8").strip()
    long_tail = " ".join(f"t{i}" for i in range(900))
    _inference(answering).call("script_responses", responses=[
        {"text": ALPHA, "finish_reason": "length"},
        {"text": BRAVO, "finish_reason": "length", "continues": True},
        {"text": CHARLIE + " " + long_tail, "finish_reason": "stop",
         "continues": True},
    ])

    conn = http.client.HTTPConnection(base.hostname, base.port, timeout=60)

    def call(method, params):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                           "params": params})
        conn.request("POST", "/operator/rpc", body=body, headers={
            "Content-Type": "application/json", "X-Amoeba-Operator": token,
            "Content-Length": str(len(body))})
        r = conn.getresponse()
        return {"status": r.status, "ctype": r.getheader("Content-Type"),
                "body": r.read().decode("utf-8")}

    try:
        conn.request("GET", "/")
        page = conn.getresponse().read().decode("utf-8")
        opened = call("ego_converse", {"message": "the hard one", "wait": False})
        trigger_id = json.loads(opened["body"])["result"]["result"]["trigger_id"]
        deadline = time.time() + 180
        while True:
            final = call("role_answer", {"trigger_id": trigger_id})
            # Terminal states only, exactly as the page waits. Between the
            # turns of a chain the request is "consumed" -- its first turn
            # has read it -- and the answer is still being written; this
            # loop once stopped there, and passed only while the whole chain
            # happened to finish inside one poll.
            if json.loads(final["body"])["result"]["status"] in (
                    "completed", "incomplete", "unanswerable", "expired"):
                break
            assert time.time() < deadline
            time.sleep(0.2)
    finally:
        conn.close()

    expected = f"[SIMULATED] {ALPHA}{BRAVO}{CHARLIE} {long_tail}"
    assert json.loads(final["body"])["result"]["answer"] == expected

    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    harness = Path(__file__).with_name("dashboard_panel_harness.js")
    scenario = {"panel": "converse", "session": token,
                "responses": [opened, final],
                "actions": [{"type": "type", "text": "the hard one"},
                            {"type": "key", "key": "Enter"}]}
    with tempfile.TemporaryDirectory() as tmp:
        js, scn = Path(tmp) / "page.js", Path(tmp) / "scenario.json"
        js.write_text(script, encoding="utf-8")
        scn.write_text(json.dumps(scenario), encoding="utf-8")
        r = subprocess.run([shutil.which("node"), str(harness), str(js), str(scn)],
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=120)
    assert r.returncode == 0, r.stderr
    shown = json.loads(r.stdout)["transcript"]
    assert [t["who"] for t in shown] == ["operator", "ego"]
    assert shown[1]["text"] == expected, "the page did not show the whole answer"
    assert shown[1]["cls"] == "said"



# ===========================================================================
# the continuation is told what happens to its output
# ===========================================================================
def test_the_continuation_instruction_says_its_output_is_appended(mind):
    """When a continuation has to be asked visibly, it is told the truth.

    Nothing afterwards rewrites what the model writes, so it has to know its
    output lands directly after the previous output -- or it will introduce
    the continuation, recap, and restart, as it did live.
    """
    _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, ALPHA, max_continuations=3)
    t2 = _claim(mind, "ego")
    said = t2["triggers"][0]["summary"]
    assert said in t2["text"], "the instruction is not what the model reads"
    assert "appended directly" in said
    for forbidden in ("introduce the continuation", "recap", "restart a section",
                      "repeat"):
        assert forbidden in said, forbidden
    assert len(said) <= mailbox.MAX_SUMMARY, "the instruction would be cut"


def test_a_preamble_the_model_writes_anyway_is_kept(mind):
    """No heuristic rewrites a reply. What the model said is what is delivered."""
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, "Claim 1: inferred. ", max_continuations=3)
    t2 = _claim(mind, "ego")
    preamble = "Continuing from where the previous analysis left off:\n\n---\n\n"
    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result=_piece(preamble + "Claim 2: measured.", resumed=False),
              max_continuations=3)
    assert _answer_of(mind, req["trigger_id"])[1] == (
        "Claim 1: inferred. " + preamble + "Claim 2: measured.")


# ===========================================================================
# a conclusion is the answer, not a fragment of it
# ===========================================================================
def _conclusions(mind):
    return [dict(r) for r in mind.db.conn.execute(
        "SELECT conclusion_id, claim, operation_id, produced_by"
        " FROM conclusions ORDER BY created_at")]


def _evidence_notes(mind, conclusion_id):
    return [r["note"] for r in mind.db.conn.execute(
        "SELECT note FROM conclusion_evidence WHERE conclusion_id = ?"
        " ORDER BY id", (conclusion_id,))]


def test_a_fragment_is_not_a_conclusion_and_the_whole_answer_is(mind):
    """Live, one question left four conclusions -- three of them half-sentences.

    A bounded turn ending is not Ego concluding anything. The conclusion is
    recorded once, when the interaction's answer is whole and finished, and it
    cites every turn that produced it.
    """
    req = _request(mind)
    t1 = _claim(mind, "ego")
    _cut(mind, t1, ALPHA, max_continuations=3)
    assert _conclusions(mind) == [], "a max_output_tokens fragment was concluded"
    t2 = _claim(mind, "ego")
    _cut(mind, t2, BRAVO, resumed=True, max_continuations=3)
    assert _conclusions(mind) == [], "a continuation fragment was concluded"
    t3 = _claim(mind, "ego")
    _complete(mind, t3["turn_id"], stop_reason="model_stop",
              result=_piece(CHARLIE, resumed=True), max_continuations=3)

    concluded = _conclusions(mind)
    assert len(concluded) == 1
    assert concluded[0]["claim"] == (ALPHA + BRAVO + CHARLIE).strip()
    assert concluded[0]["produced_by"] == "ego"
    notes = _evidence_notes(mind, concluded[0]["conclusion_id"])
    assert notes == [f"turn {t1['turn_id']}", f"turn {t2['turn_id']}",
                     f"turn {t3['turn_id']}", f"trigger {req['trigger_id']}"]
    # And the answer knows which conclusion it became.
    assert _answer_record(mind, req["trigger_id"])[1]["conclusion_id"] == \
        concluded[0]["conclusion_id"]


def test_an_incomplete_answer_is_not_a_conclusion(mind):
    """A thought the continuation limit stopped is not a claim Ego made."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    while True:
        out = _cut(mind, turn, "still going ", max_continuations=2)
        if out["continuation"] is None:
            break
        turn = _claim(mind, "ego")
    assert _answer_of(mind, req["trigger_id"])[0] == "incomplete"
    assert _conclusions(mind) == []


def test_ids_answers_are_not_egos_conclusions(mind):
    req = _request(mind, role="id")
    turn = _claim(mind, "id")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"text": "the pool is nominal", "segment": "the pool is nominal"})
    assert _answer_of(mind, req["trigger_id"])[0] == "answered"
    assert _conclusions(mind) == []


# ===========================================================================
# one canonical ceiling; the platform cap refuses rather than clamps
# ===========================================================================
def test_the_platform_cap_refuses_rather_than_clamps():
    """At 512, as a silent clamp, it gave Ego 512 of a governed 3072."""
    from amoeba.errors import InvalidInput

    cfg = ArbiterConfig()
    cfg.max_completion_tokens = 512
    with pytest.raises(InvalidInput) as refused:
        Arbiter(cfg).clamp_inference(prompt_tokens=10, max_tokens=3072,
                                     deadline=None, budget_tokens=16384)
    assert "platform cap" in str(refused.value)
    from amoeba.roles import _is_context_pressure

    assert not _is_context_pressure(refused.value), \
        "a contradiction must not be mistaken for a full context"


def test_a_silent_profile_is_bound_with_the_shipped_ceiling_on_the_record(mind, library):
    """An approved root that predates the setting gets the shipped value -- visibly.

    The live Ego ran on such a root. The number it was given now appears in
    the binding as supplied by the Harness, not as a choice anyone made.
    """
    from amoeba import prompt_api

    store, resolver = library
    silent = _author(mind, store, "ego", mode="replace",
                     text=resolver.resolve_selected("ego").prompt_text,
                     model_vars={"temperature": 0.7, "top_p": 0.95})
    _approve_and_select(mind, store, "ego", silent["version_id"])
    bound = prompt_api.build(_Sup(mind))["bind_profile"](
        namespace="ego", actor_id="ego", actor_kind="ego")
    assert bound["backend_arguments"]["max_tokens"] == 3072
    row = mind.db.conn.execute(
        "SELECT effective_settings, harness_constraints FROM incarnation_profiles"
        " WHERE binding_id = ?", (bound["binding_id"],)).fetchone()
    assert json.loads(row["harness_constraints"]) == {"max_output_tokens": 3072}
    assert json.loads(row["effective_settings"])["max_output_tokens"] == 3072


def test_a_stated_ceiling_is_never_narrowed_by_the_harness(mind, library):
    from amoeba import prompt_api

    bound = prompt_api.build(_Sup(mind))["bind_profile"](
        namespace="ego", actor_id="ego", actor_kind="ego")
    assert bound["backend_arguments"]["max_tokens"] == 3072
    row = mind.db.conn.execute(
        "SELECT harness_constraints FROM incarnation_profiles WHERE binding_id = ?",
        (bound["binding_id"],)).fetchone()
    assert json.loads(row["harness_constraints"]) == {}


def test_binding_refuses_a_profile_above_the_platform_cap(mind, library):
    from amoeba import prompt_api
    from amoeba.errors import InvalidInput

    store, resolver = library
    greedy = _author(mind, store, "ego", mode="replace",
                     text=resolver.resolve_selected("ego").prompt_text,
                     model_vars={"temperature": 0.7, "max_output_tokens": 999_999})
    _approve_and_select(mind, store, "ego", greedy["version_id"])
    with pytest.raises(InvalidInput) as refused:
        prompt_api.build(_Sup(mind))["bind_profile"](
            namespace="ego", actor_id="ego", actor_kind="ego")
    assert "platform cap" in str(refused.value)


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def _startup(mind, cap):
    import copy

    from amoeba.supervisor import Supervisor

    cfg = copy.deepcopy(mind.cfg)
    cfg.arbiter.max_completion_tokens = cap
    stub = type("S", (), {})()
    stub.cfg, stub.mind, stub.log = cfg, mind, _Log()
    Supervisor._validate_output_ceilings(stub)
    return stub.log


def test_startup_refuses_ceilings_that_contradict_the_platform_cap(mind, library):
    """Never silently clamp 3072 to 512: refuse to start, and say why."""
    with pytest.raises(RuntimeError) as refused:
        _startup(mind, 512)
    message = str(refused.value)
    assert "platform cap of 512" in message and "ego@" in message
    assert "max_completion_tokens" in message


def test_startup_accepts_a_consistent_configuration(mind, library):
    assert _startup(mind, 3072).warnings == []


def test_startup_says_aloud_when_a_selected_profile_states_no_ceiling(mind, library):
    store, resolver = library
    silent = _author(mind, store, "ego", mode="replace",
                     text=resolver.resolve_selected("ego").prompt_text,
                     model_vars={"temperature": 0.7, "top_p": 0.95})
    _approve_and_select(mind, store, "ego", silent["version_id"])
    warned = _startup(mind, 3072).warnings
    assert len(warned) == 1 and "ego@" in warned[0] and "approve" in warned[0]


# ===========================================================================
# the live answer is one conclusion
# ===========================================================================
@live
def test_the_live_three_turn_answer_is_one_conclusion(answering):
    _inference(answering).call("script_responses", responses=[
        {"text": ALPHA, "finish_reason": "length"},
        {"text": BRAVO, "finish_reason": "length", "continues": True},
        {"text": CHARLIE, "finish_reason": "stop", "continues": True},
    ])
    env = answering.call("ego_converse", message="one answer, three turns",
                         wait=False)
    state = _await_answer(answering, env["result"]["trigger_id"])
    assert state["status"] == "completed"
    recorded = answering.call("history", operation_id=env["operation_id"],
                              kinds=["conclusion.recorded"], limit=50)
    assert len(recorded) == 1, "one answer became several conclusions"
    payload = json.loads(recorded[0]["payload_inline"])
    concl = answering.call("get_conclusion",
                           conclusion_id=payload["conclusion_id"])
    assert concl["claim"] == state["answer"]
    assert sum(n["note"].startswith("turn ") for n in concl["evidence"]) == 3
