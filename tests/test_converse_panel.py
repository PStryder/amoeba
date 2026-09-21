"""Converse and Consult Id, driven rather than read.

Both panels were a textarea, a button and `JSON.stringify(result, null, 2)`
in a `<pre>`. The operator saw an envelope, not an answer, and could not tell
from it whether the answer had arrived -- because the envelope's `status` is
the *operation's* status, and the operation completes the moment the request
is queued.

What the panels do now, and what these tests hold them to:

  * queue the message (`wait: false`), keep that request's trigger id, and
    poll `role_answer` for that id until an answer of record exists. A thought
    that spans continuation turns is still answering the message that began
    it, and nothing about turns reaches the operator;
  * never show an answer that has not terminally arrived;
  * read status and content type before parsing, so an HTML error page is
    reported as an HTML error page rather than as
    `Unexpected token '<', "<!DOCTYPE "...`;
  * keep the transcript in the page and nowhere else. It is a viewport onto a
    persistent Ego, not a stored conversation replayed back at it.

These drive the real script through the real handlers, because every property
above is a property of the *request sequence* and is invisible to anything
that only inspects the page source.
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

from test_external_interfaces import net  # noqa: F401

live = pytest.mark.skipif(
    sys.platform != "win32",
    reason="live stack fixtures are Windows-only here")
needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node to drive the panel")

HARNESS = Path(__file__).with_name("dashboard_panel_harness.js")
TOKEN = "operator-token"


def _script() -> str:
    from amoeba.dashboard import DASHBOARD_HTML

    return re.search(r"<script>(.*?)</script>", DASHBOARD_HTML, re.S).group(1)


def _json(**payload) -> dict:
    return {"status": 200, "ctype": "application/json",
            "body": json.dumps({"jsonrpc": "2.0", "id": 1, "result": payload})}


def _queued(trigger_id: str) -> dict:
    """What `ego_converse` returns: the operation completed, the turn has not."""
    return _json(schema_version=1, operation_id="op_x", status="completed",
                 result={"trigger_id": trigger_id, "status": "queued",
                         "conversation_id": None},
                 limitations=["queued only; the answer is not in this response"])


def _answer(status: str, text: str = "") -> dict:
    return _json(trigger_id="trg_1", status=status, answer=text,
                 is_simulated=False)


def _drive(panel: str, responses: list, actions: list, **extra) -> dict:
    scenario = {"panel": panel, "session": TOKEN, "responses": responses,
                "actions": actions, **extra}
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "dashboard.js"
        js.write_text(_script(), encoding="utf-8")
        scn = Path(tmp) / "scenario.json"
        scn.write_text(json.dumps(scenario), encoding="utf-8")
        r = subprocess.run([shutil.which("node"), str(HARNESS), str(js),
                            str(scn)],
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=120)
    assert r.returncode == 0, r.stderr.strip()
    return json.loads(r.stdout)


def _type_and_send(text: str) -> list:
    return [{"type": "type", "text": text},
            {"type": "key", "key": "Enter"}]


# ---------------------------------------------------------------------------
# Reaching the right mind
# ---------------------------------------------------------------------------
@needs_node
def test_converse_opens_on_an_empty_transcript():
    """Ephemeral by design: a reload starts blank and that is correct.

    Ego's continuity is Ego's own -- its context, its maintained memory, the
    durable record. A transcript stored and replayed at it would be a second
    memory telling a slightly different story.
    """
    out = _drive("converse", [], [])
    assert out["transcript"] == []
    assert out["composer"] == ""
    assert out["requests"] == [], "the panel asked for history it should not keep"


@needs_node
def test_a_message_reaches_ego_and_its_answer_is_collected_by_request():
    out = _drive("converse",
                 [_queued("trg_1"), _answer("completed", "I am here.")],
                 _type_and_send("are you there?"))

    assert [r["method"] for r in out["requests"]] == ["ego_converse", "role_answer"]
    assert out["requests"][0]["params"] == {"message": "are you there?",
                                            "wait": False}
    # The answer is fetched for the request that was made, not for a turn, a
    # conversation, or whatever happened to be latest.
    assert out["requests"][1]["params"] == {"trigger_id": "trg_1"}
    assert out["requests"][0]["token"] == TOKEN

    assert [(t["who"], t["text"]) for t in out["transcript"]] == [
        ("operator", "are you there?"), ("ego", "I am here.")]
    assert out["composer"] == "", "the composer should clear once it is sent"


@needs_node
def test_consult_id_keeps_its_own_surface_and_its_own_verb():
    """Two minds, two panels. Merging them would merge their semantics."""
    envelope = _json(consult_id="cons_1", receipt_id="r",
                     answer={"schema_version": 1, "status": "completed",
                             "result": {"trigger_id": "trg_9",
                                        "status": "queued"}})
    out = _drive("consult id", [envelope, _answer("completed", "Nothing is wedged.")],
                 _type_and_send("anything stuck?"))

    assert [r["method"] for r in out["requests"]] == ["operator_consult_id",
                                                      "role_answer"]
    assert out["requests"][0]["params"] == {"question": "anything stuck?",
                                            "wait": False}
    assert out["requests"][1]["params"] == {"trigger_id": "trg_9"}
    assert [(t["who"], t["text"]) for t in out["transcript"]] == [
        ("operator", "anything stuck?"), ("id", "Nothing is wedged.")]
    # The standing caveat is still on the page.
    assert out["pills"] == [
        "an input into Id's reasoning; it carries no capability"]


# ---------------------------------------------------------------------------
# An answer arrives when it arrives
# ---------------------------------------------------------------------------
@needs_node
def test_nothing_is_shown_as_an_answer_until_it_terminally_is():
    """The operation completing is not the answer arriving.

    `ego_converse` returns as soon as the message is queued, and the envelope
    says `status: completed` about the *operation*. Treating that as the reply
    would print an empty answer under every message.
    """
    out = _drive("converse",
                 [_queued("trg_1"), _answer("queued"), _answer("claimed"),
                  _answer("completed", "It took some thinking.")],
                 _type_and_send("a hard one"))

    assert [r["method"] for r in out["requests"]] == [
        "ego_converse", "role_answer", "role_answer", "role_answer"]
    reply = out["transcript"][1]
    assert reply["text"] == "It took some thinking."
    assert reply["cls"] == "said"


@needs_node
def test_while_it_is_working_the_page_says_so_and_nothing_else():
    """One inline line, and no answer under it yet."""
    out = _drive("converse", [_queued("trg_1"), _answer("queued")],
                 _type_and_send("a hard one"), maxRequests=2)

    assert out["transcript"][1]["text"] == "Ego is thinking…"
    assert out["transcript"][1]["cls"] == "waiting"
    assert out["sendDisabled"] is True, "a second send should not be possible"


@needs_node
def test_a_thought_that_ends_without_an_answer_says_so():
    out = _drive("converse", [_queued("trg_1"), _answer("unanswerable")],
                 _type_and_send("hm"))
    reply = out["transcript"][1]
    assert reply["cls"] == "failed"
    assert "without an answer" in reply["text"]


# ---------------------------------------------------------------------------
# Errors that name what happened
# ---------------------------------------------------------------------------
@needs_node
def test_an_html_error_page_is_not_parsed_as_a_result():
    """The exact failure the operator saw, and the exact thing not to say.

    A desynchronised connection makes the stock server answer with an HTML
    error page. Parsing it as JSON produced `Unexpected token '<', "<!DOCTYPE
    "... is not valid JSON`, which describes the parser's problem and sends
    whoever reads it to debug the wrong layer.
    """
    html = {"status": 501, "ctype": "text/html;charset=utf-8",
            "body": "<!DOCTYPE HTML>\n<html><body>Unsupported method</body></html>"}
    out = _drive("converse", [html], _type_and_send("hello"))

    assert "Unexpected token" not in out["notice"]
    assert "JSON.parse" not in out["notice"]
    assert "501" in out["notice"] and "text/html" in out["notice"]
    # Diagnosis is kept, just not in the operator's face.
    assert any("501" in line for line in out["consoleErrors"]), out["consoleErrors"]


@needs_node
def test_a_message_that_never_left_is_not_lost():
    """The submission failed, so the text stays where the operator put it."""
    out = _drive("converse", [{"status": 500, "ctype": "text/plain",
                               "body": "boom"}],
                 _type_and_send("something I spent a while writing"))

    assert out["composer"] == "something I spent a while writing"
    assert out["transcript"] == [], "it was never sent, so it is not in the log"
    assert out["sendDisabled"] is False, "the operator can try again"


@needs_node
def test_an_expired_session_says_what_to_do_about_it():
    refused = {"status": 401, "ctype": "application/json",
               "body": json.dumps({"jsonrpc": "2.0", "id": 1, "error": {
                   "code": -32000,
                   "message": "operator session required in X-Amoeba-Operator"}})}
    out = _drive("converse", [refused], _type_and_send("hello"))

    assert "token" in out["notice"] and "session" in out["notice"]
    assert "Unexpected token" not in out["notice"]
    assert out["tokenPrompt"] is True, "it should offer somewhere to put one"
    assert out["composer"] == "hello", "the message was not delivered"


@needs_node
def test_an_unreachable_supervisor_is_not_reported_as_a_parse_failure():
    out = _drive("converse", [], _type_and_send("hello"))
    assert "could not reach Amoeba" in out["notice"]


# ---------------------------------------------------------------------------
# The composer
# ---------------------------------------------------------------------------
@needs_node
def test_enter_sends_and_shift_enter_does_not():
    out = _drive("converse", [_queued("trg_1"), _answer("completed", "ok")],
                 [{"type": "type", "text": "first line"},
                  {"type": "key", "key": "Enter", "shift": True}])
    assert out["requests"] == [], "Shift+Enter is a newline, not a send"
    assert out["composer"] == "first line"


@needs_node
def test_a_multiline_message_keeps_its_shape():
    text = "First paragraph.\n\nSecond paragraph,\n  indented continuation."
    out = _drive("converse",
                 [_queued("trg_1"), _answer("completed", "line one\n\nline two")],
                 _type_and_send(text))
    assert out["transcript"][0]["text"] == text
    assert out["transcript"][1]["text"] == "line one\n\nline two"


@needs_node
def test_an_empty_composer_sends_nothing():
    out = _drive("converse", [_queued("trg_1")],
                 [{"type": "type", "text": "   \n  "},
                  {"type": "key", "key": "Enter"}])
    assert out["requests"] == []


# ---------------------------------------------------------------------------
# What the operator is not shown
# ---------------------------------------------------------------------------
@needs_node
def test_the_transcript_carries_no_machinery():
    """Converse is Ego's face, not another observability panel."""
    out = _drive("converse",
                 [_queued("trg_abc123"),
                  _answer("completed", "Forty-two.")],
                 _type_and_send("the question"))
    blob = " ".join(t["text"] + " " + t["who"] for t in out["transcript"])
    for leak in ("trg_", "turn_", "op_", "lineage", "trigger", "neuocyte",
                 "blackboard", "conclusion", "sha256", "receipt"):
        assert leak not in blob.lower(), f"{leak!r} reached the transcript"


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------
@needs_node
def test_a_reader_who_scrolled_up_is_left_alone():
    """Yanking the viewport away from something being read is worse than
    making somebody scroll back down."""
    out = _drive("converse", [_queued("trg_1"), _answer("completed", "ok")],
                 _type_and_send("hello"),
                 scroll={"scrollTop": 0, "scrollHeight": 4000,
                         "clientHeight": 400})
    assert out["scrollTop"] == 0, "the viewport followed a reader who had left it"


@needs_node
def test_a_reader_at_the_bottom_is_carried_along():
    out = _drive("converse", [_queued("trg_1"), _answer("completed", "ok")],
                 _type_and_send("hello"),
                 scroll={"scrollTop": 3600, "scrollHeight": 4000,
                         "clientHeight": 400})
    assert out["scrollTop"] == 4000


# ---------------------------------------------------------------------------
# The same sequence, against the running organism
# ---------------------------------------------------------------------------
@live
def test_the_browsers_request_sequence_works_against_a_live_amoeba(net):
    """Queue, let go, come back for the answer -- over real HTTP.

    The panel tests above prove the page does the right sequence. This proves
    the sequence is the right one: the same three calls, against a running
    supervisor, ending in an answer of record for the request that was made.
    """
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=60)

    def call(method, params):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                           "params": params})
        conn.request("POST", "/operator/rpc", body=body, headers={
            "Content-Type": "application/json",
            "X-Amoeba-Operator": net.operator_session,
            "Content-Length": str(len(body))})
        r = conn.getresponse()
        assert r.status == 200, r.status
        payload = json.loads(r.read())
        assert "error" not in payload, payload["error"]
        return payload["result"]

    try:
        env = call("ego_converse", {"message": "Say something.", "wait": False})
        trigger_id = env["result"]["trigger_id"]
        assert env["result"]["status"] == "queued"

        deadline = time.time() + 180
        while True:
            state = call("role_answer", {"trigger_id": trigger_id})
            assert state["trigger_id"] == trigger_id
            if state["status"] in ("completed", "unanswerable", "expired"):
                break
            assert state["answer"] == "", "an answer before it was answered"
            assert time.time() < deadline, "no answer in three minutes"
            time.sleep(0.5)
    finally:
        conn.close()

    assert state["status"] == "completed", state
    assert state["answer"].strip(), "completed with nothing to show"


@live
def test_consult_id_answers_at_all(net):
    """It never has. `operator_consult_id` passed `operation_id` to
    `id_introspect`, which has no such parameter, so every call raised
    TypeError -- a whole operator surface that no test ever invoked."""
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=60)

    def call(method, params):
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                           "params": params})
        conn.request("POST", "/operator/rpc", body=body, headers={
            "Content-Type": "application/json",
            "X-Amoeba-Operator": net.operator_session,
            "Content-Length": str(len(body))})
        payload = json.loads(conn.getresponse().read())
        assert "error" not in payload, payload["error"]
        return payload["result"]

    try:
        out = call("operator_consult_id",
                   {"question": "Is anything wedged?", "wait": False})
        trigger_id = out["answer"]["result"]["trigger_id"]
        assert out["consult_id"].startswith("cons")

        deadline = time.time() + 180
        while True:
            state = call("role_answer", {"trigger_id": trigger_id})
            if state["status"] in ("completed", "unanswerable", "expired"):
                break
            assert time.time() < deadline, "Id did not answer in three minutes"
            time.sleep(0.5)
    finally:
        conn.close()

    assert state["status"] == "completed", state
