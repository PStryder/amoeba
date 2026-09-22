"""A role can use what it is offered, and is not killed for thinking.

Both defects here were found by the first live behavioural test, and both had
survived every earlier suite.

**The roles were never told how to call a tool.** The role tool loop parses
exactly `<tool_call>{"name": ..., "arguments": ...}</tool_call>`. Neuocytes
are shown that form; Ego and Id were shown a list of verbs and arguments and
nothing about how to invoke one. Across the organism's recorded life not one
Ego or Id turn executed a tool. Asked about an incident, Ego wrote
`board_post(author="ego@1", ...)` as a Python call; nothing parsed it, and the
raw text was delivered to the operator as the answer. Every test of the loop
scripted a correctly formatted reply, so the parser was tested and whether
anyone was ever told the format was not.

**A busy role was restarted as dead.** `RpcClient` holds one lock for a whole
request. A role's `health` handler asked inference for its health through the
very client the turn thread holds for the length of a generation, so the
supervisor's two-second probe waited on the generation, timed out, and after
the grace period restarted the role mid-turn. At 384 output tokens a
generation took ~2.4s and never tripped it; at Ego's 3072 it trips on most
long answers -- the ceiling increase turned a latent defect into a live one.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import pytest

from amoeba import roles
from amoeba.roles import ENVIRONMENT_BLOCK, RoleProcess
from amoeba.tools import TOOL_CALL_RE, parse_tool_calls
from conftest import start_stack

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")


class _Arbiter:
    max_tool_turns = 3
    neuocyte_wall_seconds = 60.0


class _Cfg:
    arbiter = _Arbiter()


def _bare_role() -> RoleProcess:
    """A RoleProcess with only what `_turn` and `health` touch -- no processes."""
    role = object.__new__(RoleProcess)
    role.cfg = _Cfg()
    role.environment = None
    role.environment_blob = None
    role.profile_ref = "ego@1"
    role.session_id = "sess_test"
    role.incarnation = 1
    role.started_at = time.time()
    role.model_generation = "gen_test"
    role.turns = 0
    role._signals = []
    role.capabilities = {}
    role.role = "ego"
    return role


# ---------------------------------------------------------------------------
# F1: taught what the parser accepts
# ---------------------------------------------------------------------------
def _taught(role_text: str) -> str:
    lines = [ln for ln in role_text.splitlines() if ln.startswith("<tool_call>")]
    assert len(lines) == 1, "the role is not shown exactly one call form"
    return lines[0]


def test_a_role_is_taught_the_syntax_its_parser_accepts():
    """What the model is handed, through the real `_turn`, and what it means.

    Captured from the text `_turn` actually gives the model on its first
    generation, then filled in the way the template says and handed to the
    role loop's own parser. If either side changes without the other, this
    fails -- which is the whole point: a parser nobody is told about is a
    capability nobody has.
    """
    role = _bare_role()
    seen: dict[str, Any] = {}

    def infer(text, **_):
        seen.setdefault("first", text)
        return {"text": "done", "finish_reason": "stop"}

    role._infer = infer
    role._can_resume = lambda resume: False
    role._turn("a question", environment={
        "manifest": {"environment_sha256": "e"}, "text": "DECLARED VERBS",
        "environment_blob": None})

    shown = seen["first"]
    assert "DECLARED VERBS" in shown
    form = _taught(shown)
    call = (form.replace("<verb>", "board_stats")
                .replace('"<argument>": <value>', '"limit": 5'))
    requests = parse_tool_calls(call)
    assert [(r.name, r.arguments) for r in requests] == [("board_stats", {"limit": 5})]
    assert TOOL_CALL_RE.search(call), "the taught form is not what the loop matches"


def test_the_role_is_told_to_wait_for_the_result_not_to_stop():
    """The governed doctrine says wait; the Harness block used to say stop."""
    text = ENVIRONMENT_BLOCK.format(environment="x")
    assert "wait for the Harness result" in text
    assert "and stop" not in text


def test_the_python_call_ego_actually_wrote_is_not_a_request():
    """What Ego wrote live is still prose to the parser -- by design.

    The fix is telling the model the form, not guessing at other forms: a
    parser that tried to read arbitrary text as calls would execute things
    nobody asked for.
    """
    assert parse_tool_calls('board_post(author="ego@1", body="x")') == []


# ---------------------------------------------------------------------------
# F2: health does not wait on a turn
# ---------------------------------------------------------------------------
class _Busy:
    """An inference client a running turn is holding."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lock.acquire()

    def call(self, *_a, **_k):
        # Blocks for as long as the turn holds it -- bounded, so a regression
        # fails the test that found it instead of hanging every test after it.
        if not self.lock.acquire(timeout=3.0):
            raise TimeoutError("waited on a busy turn's inference client")
        self.lock.release()
        return {}


class _Idle:
    def __init__(self):
        self.calls = 0
        self.closed = 0

    def call(self, *_a, **_k):
        self.calls += 1
        return {"status": "alive"}

    def close(self):
        self.closed += 1


def test_health_answers_while_a_turn_holds_the_inference_client():
    """The probe that killed a busy Ego, reduced to its cause."""
    role = _bare_role()
    role.inf = _Busy()
    role.inf_probe = _Idle()
    out: dict[str, Any] = {}
    t = threading.Thread(target=lambda: out.setdefault("health", role.health()),
                         daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert "health" in out, "health waited on the turn's inference client"
    assert out["health"]["inference_reachable"] is True
    assert role.inf_probe.calls == 1


def test_a_failed_probe_leaves_no_reply_behind_for_the_next():
    """A timed-out call leaves its answer in the socket; the next would read it."""
    class _TimesOut(_Idle):
        def call(self, *_a, **_k):
            raise TimeoutError("probe timed out")

    role = _bare_role()
    role.inf = _Busy()
    role.inf_probe = _TimesOut()
    health = role.health()
    assert health["inference_reachable"] is False
    assert role.inf_probe.closed == 1, "the probe connection was left mid-reply"


@live
def test_a_long_generation_does_not_get_its_role_restarted(tmp_path):
    """Hold one Ego generation open well past the probe grace, for real.

    The supervisor probes each child every pass and restarts one that has been
    unreachable for CHILD_GRACE_SECONDS. A generation longer than that used to
    make a busy Ego look dead.
    """
    from amoeba.rpc import RpcClient, read_or_create_token
    from amoeba.supervisor import CHILD_GRACE_SECONDS, PROBE_TIMEOUT_SECONDS

    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False})
    try:
        stack.wait_for_children(timeout=90)
        before = stack.call("health")["children"]["ego"]["pid"]
        cfg = stack.cfg
        inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                        read_or_create_token(cfg.token_path), timeout=60)
        inf.connect(retries=20, delay=0.5)
        # Long enough that a restart is certain if probes fail: the first
        # failure lands up to one pass after the generation starts, then the
        # grace runs, then one more pass acts. grace + probe + 4 was too
        # short -- the generation finished first, and this test passed with
        # the defect present, which is how it was caught.
        hold = CHILD_GRACE_SECONDS + 10.0
        assert hold > CHILD_GRACE_SECONDS + 2 * PROBE_TIMEOUT_SECONDS + 4
        inf.call("script_responses", responses=[
            {"text": "a considered answer", "finish_reason": "stop",
             "delay_seconds": hold}])

        env = stack.call("ego_converse", message="take your time", wait=False)
        trigger_id = env["result"]["trigger_id"]
        deadline = time.time() + hold + 60
        while True:
            state = stack.call("role_answer", trigger_id=trigger_id)
            if state["status"] in ("completed", "incomplete", "unanswerable", "expired"):
                break
            assert time.time() < deadline
            time.sleep(0.5)

        assert state["status"] == "completed", state
        assert state["answer"] == "[SIMULATED] a considered answer"
        abandoned = stack.call("history", kinds=["role.turn_abandoned"], limit=20)
        assert abandoned == [], "a busy role was restarted mid-turn"
        after = stack.call("health")["children"]["ego"]["pid"]
        assert after == before, "Ego was restarted while it was thinking"
    finally:
        stack.stop()



# ---------------------------------------------------------------------------
# A malformed call is not an answer
# ---------------------------------------------------------------------------
# Teaching the syntax is the fix; this is what keeps one typo from turning
# the operator's answer back into a service hatch. What Ego wrote live:
LIVE_ATTEMPT = ('board_post(\n  author="ego@1",\n  author_kind="role",\n'
                '  post_type="investigation",\n  body="Initiating investigation"\n)')
OFFERED = ["board_post", "recall", "get_memory", "recall_memory"]


@pytest.mark.parametrize("text, expected", [
    (LIVE_ATTEMPT, "board_post"),
    ('{"name": "recall", "arguments": {}}', "recall"),
    ('I will check. <tool_call>{"name": "recall"', "<tool_call>"),
    ("recall_memory(query='x')", "recall_memory"),       # longest verb wins
    ("Use `get_memory(memory_id)` to inspect a belief.", None),
    ("- `get_memory(memory_id)` -> inspect maintained beliefs", None),
    ("The recall of beliefs is weighted by confidence.", None),
    ("unknown_verb(x=1)", None),                          # not offered: prose
])
def test_a_malformed_attempt_is_recognised_structurally(text, expected):
    from amoeba.tools import malformed_call

    assert malformed_call(text, OFFERED) == expected


class _Session:
    """What `_feed_tool_result` writes into, recorded instead of sent."""

    def __init__(self):
        self.fed = []

    def call(self, method, **kw):
        if method == "apply_chat_template":
            return kw["messages"][0]["content"]
        if method == "ingest_text":
            self.fed.append(kw["text"])
        return {}


def _looping_role(replies):
    role = _bare_role()
    role.inf = _Session()
    queue = list(replies)
    role._infer = lambda text, **_: {"text": queue.pop(0), "finish_reason": "stop"}
    role._can_resume = lambda resume: False
    return role


_ENV = {"manifest": {"environment_sha256": "e",
                     "capabilities": [{"verb": v} for v in OFFERED]},
        "text": "VERBS", "environment_blob": None}


def test_a_malformed_attempt_is_refused_and_the_model_may_try_again():
    """One typo costs a retry, not the operator's answer."""
    role = _looping_role([LIVE_ATTEMPT, "Here is my assessment."])
    out = role._turn("investigate", environment=_ENV)

    assert out["text"] == "Here is my assessment."
    assert out["malformed_call"] is None
    assert out["tool_call_count"] == 0, "a malformed attempt is not a request"
    assert [t["malformed"] for t in out["tool_calls"]] == [True]
    told = role.inf.fed[0]
    assert "not executed" in told and "not delivered as your reply" in told
    assert '<tool_call>{"name": "board_post"' in told, "not shown the right form"


def test_a_turn_that_ends_still_malformed_is_flagged_not_answered():
    role = _looping_role([LIVE_ATTEMPT] * 6)
    out = role._turn("investigate", environment=_ENV)
    assert out["stop_reason"] == "tool_turn_limit_reached"
    assert out["malformed_call"] == "board_post"


def test_a_malformed_piece_is_withheld_from_the_answer_and_recorded(mind):
    from test_persistent_turns import _answer_of, _claim, _complete, _request

    req = _request(mind)
    t1 = _claim(mind, "ego")
    _complete(mind, t1["turn_id"], stop_reason="tool_turn_limit_reached",
              result={"answer": LIVE_ATTEMPT, "segment": LIVE_ATTEMPT,
                      "malformed_call": "board_post"}, max_continuations=3)
    t2 = _claim(mind, "ego")
    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result={"answer": "The assessment.", "segment": "The assessment."},
              max_continuations=3)
    status, answer = _answer_of(mind, req["trigger_id"])
    assert (status, answer) == ("answered", "The assessment.")
    assert "board_post" not in answer
    row = mind.db.conn.execute("SELECT answer_sha256 FROM role_triggers"
                               " WHERE trigger_id = ?", (req["trigger_id"],)).fetchone()
    record = mind.blobs.get_json(row["answer_sha256"])
    assert record["withheld"] == ["malformed call: board_post"]
    # Kept, not lost: the attempt is still in the turn that wrote it.
    assert [s.get("withheld") for s in record["segments"]] == [
        "malformed call: board_post", None]


def test_an_answer_that_was_only_a_malformed_call_says_so(mind):
    """Not delivered as prose, and not reported as silence either."""
    from test_persistent_turns import _answer_of, _claim, _complete, _request

    req = _request(mind)
    t1 = _claim(mind, "ego")
    _complete(mind, t1["turn_id"], stop_reason="tool_turn_limit_reached",
              result={"answer": LIVE_ATTEMPT, "segment": LIVE_ATTEMPT,
                      "malformed_call": "board_post"}, max_continuations=0)
    status, answer = _answer_of(mind, req["trigger_id"])
    assert status == "unanswerable" and answer == ""
    row = mind.db.conn.execute("SELECT answer_sha256 FROM role_triggers"
                               " WHERE trigger_id = ?", (req["trigger_id"],)).fetchone()
    record = mind.blobs.get_json(row["answer_sha256"])
    assert record["ended_because"] == "malformed_call"


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="needs node")
@pytest.mark.parametrize("state, expect_in, expect_cls", [
    ({"status": "unanswerable", "answer": "", "withheld": ["malformed call: board_post"],
      "ended_because": "malformed_call"},
     "could not read", "failed"),
    ({"status": "completed", "answer": "The assessment.",
      "withheld": ["malformed call: board_post"]},
     "The assessment.", "said"),
])
def test_converse_never_shows_the_raw_call(state, expect_in, expect_cls):
    import json

    from test_converse_panel import _drive, _json, _queued, _type_and_send

    out = _drive("converse", [_queued("trg_1"), _json(trigger_id="trg_1", **state)],
                 _type_and_send("investigate"))
    reply = out["transcript"][1]
    assert expect_in in reply["text"] and reply["cls"] == expect_cls
    everything = json.dumps(out["transcript"])
    assert "board_post" not in everything and "author=" not in everything


@live
def test_a_live_malformed_attempt_is_corrected_not_delivered(tmp_path):
    """The live failure, end to end: the typo is refused, the retry is the answer."""
    from amoeba.rpc import RpcClient, read_or_create_token

    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False})
    try:
        stack.wait_for_children(timeout=90)
        cfg = stack.cfg
        inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                        read_or_create_token(cfg.token_path), timeout=60)
        inf.connect(retries=20, delay=0.5)
        inf.call("script_responses", responses=[
            {"text": LIVE_ATTEMPT, "finish_reason": "stop", "continues": True},
            {"text": "Here is my assessment.", "finish_reason": "stop",
             "continues": True}])
        env = stack.call("ego_converse", message="investigate", wait=False)
        deadline = time.time() + 120
        while True:
            state = stack.call("role_answer", trigger_id=env["result"]["trigger_id"])
            if state["status"] in ("completed", "incomplete", "unanswerable", "expired"):
                break
            assert time.time() < deadline
            time.sleep(0.3)
        assert state["status"] == "completed"
        assert state["answer"] == "Here is my assessment."
        turn = stack.call("role_turns", role="ego")["turns"][0]
        assert turn["tool_call_count"] == 0
    finally:
        stack.stop()


# ---------------------------------------------------------------------------
# D7: an unchanged declaration is not re-sent into a session that holds it
# ---------------------------------------------------------------------------
def _env(sha="sha_one"):
    return {"manifest": {"environment_sha256": sha}, "text": "DECLARED VERBS",
            "environment_blob": None}


def _shown_per_turn(role, envs, sessions):
    seen = []
    role._infer = lambda text, **_: (seen.append(text), {"text": "ok",
                                                         "finish_reason": "stop"})[1]
    role._can_resume = lambda resume: False
    outs = []
    for env, session in zip(envs, sessions):
        role.session_id = session
        outs.append(role._turn("q", environment=env))
    return seen, outs


def test_a_session_reads_an_unchanged_declaration_once():
    role = _bare_role()
    seen, outs = _shown_per_turn(role, [_env(), _env(), _env()],
                                 ["s1", "s1", "s1"])
    assert "DECLARED VERBS" in seen[0]
    for later in seen[1:]:
        assert "DECLARED VERBS" not in later, "an unchanged declaration was re-sent"
        assert "unchanged" in later
        # The call form is never the thing that goes missing.
        _taught(later)
    assert [o["environment_rendered"] for o in outs] == ["full", "reference", "reference"]


def test_a_new_session_or_a_changed_declaration_gets_the_whole_thing():
    """Every rejuvenation makes a new session; it must not inherit a reference."""
    role = _bare_role()
    seen, outs = _shown_per_turn(role, [_env("a"), _env("a"), _env("b")],
                                 ["s1", "s2", "s2"])
    assert all("DECLARED VERBS" in text for text in seen)
    assert [o["environment_rendered"] for o in outs] == ["full", "full", "full"]


@live
def test_a_second_turn_does_not_pay_for_the_declaration_again(tmp_path):
    """Measured, not assumed: the second turn's session span shrinks by the block."""
    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False})
    try:
        stack.wait_for_children(timeout=90)
        spans = []
        for message in ("first", "second"):
            env = stack.call("ego_converse", message=message, wait=False)
            deadline = time.time() + 120
            while stack.call("role_answer", trigger_id=env["result"]["trigger_id"])[
                    "status"] not in ("completed", "incomplete", "unanswerable"):
                assert time.time() < deadline
                time.sleep(0.3)
            turn = stack.call("role_turns", role="ego")["turns"][0]
            d = stack.call("role_turn", turn_id=turn["turn_id"])
            spans.append(d["token_end"] - d["token_start"])
        declared = len(stack.call("role_environment", role="ego")["text"].split())
        assert spans[1] < spans[0] - declared * 0.8, (spans, declared)
    finally:
        stack.stop()
