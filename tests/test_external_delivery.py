"""What a client sent arrives whole, and an answer that exists is delivered.

Three defects found on main at e6c2654, all on the path between an external
client and cognition:

1. An external request was answered by a turn -- durably -- but publishing
   that answer onto the interaction was done by the daemon thread that
   submitted it. A thread does not survive a restart, so a recovered thought
   completed and `io_output` said `output: null` forever. The record knew the
   whole time.
2. An investigation was handed neither `interaction_id` nor `attachments`,
   so its turn could not resolve the request's own files, and could not
   return one through `ego_surface_result`. A conversation got both.
3. The door accepts 32,000 characters; a conversation stored 16,000 of them
   and an investigation 8,000. Trailing instructions -- which is where
   constraints live -- were dropped from otherwise valid requests, silently.
"""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace

import pytest

from amoeba import io_api, mailbox, supervisor_api
from amoeba.errors import InvalidInput, ResourceExhausted


def _sup(mind, methods=None, **extra):
    """The verbs these paths actually reach for, and nothing else running."""
    from amoeba import turn_api

    holder: dict[str, object] = {}
    cache: dict[str, object] = {}

    def all_methods():
        if not cache:
            cache.update(supervisor_api.build(holder["sup"]))
            cache.update(turn_api.build(holder["sup"]))
            cache.update(io_api.build(holder["sup"]))
            cache.update(methods or {})
        return cache

    sup = SimpleNamespace(mind=mind, cfg=mind.cfg, log=logging.getLogger("t"),
                          methods=all_methods, note_trigger=lambda role: None,
                          note_turn_finished=lambda *a, **k: None,
                          next_heartbeat=lambda role: None,
                          arbiter=SimpleNamespace(), **extra)
    holder["sup"] = sup
    return sup


def _interaction(mind, client="client_a", kind="converse", status="running"):
    interaction_id = f"ixn_{kind}_{int(time.time() * 1000000) % 10**9}"
    mind.writer.apply(
        lambda m: m.sql(
            "INSERT INTO interactions(interaction_id, client_id, surface, kind,"
            " input_sha256, status, created_at, state_version)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (interaction_id, client, "api", kind, "d" * 64, status,
             time.time(), m.prior_version + 1)),
        actor="test", bump_version=False)
    return interaction_id


def _answered_trigger(mind, *, answer="the organism's reply", status="answered"):
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=True, lineage="op-x"),
        actor="test", bump_version=False)
    from test_persistent_turns import _claim, _complete

    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": answer} if status == "answered" else None)
    return [r["trigger_id"] for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers WHERE target_role = 'ego'")][-1]


def _row(mind, interaction_id):
    return dict(mind.db.conn.execute(
        "SELECT * FROM interactions WHERE interaction_id = ?",
        (interaction_id,)).fetchone())


# ---------------------------------------------------------------------------
# 1. The answer belongs to the client, whatever happened to the thread
# ---------------------------------------------------------------------------
def test_the_interaction_records_which_trigger_answers_it(mind):
    """The association was only ever in a thread's local variables."""
    interaction_id = _interaction(mind)
    verbs = _sup(mind).methods()
    out = verbs["ego_converse"](message="what is the state?",
                                interaction_id=interaction_id, wait=False)
    assert _row(mind, interaction_id)["trigger_id"] == out["result"]["trigger_id"]


def test_an_investigation_records_its_trigger_too(mind):
    interaction_id = _interaction(mind, kind="investigate")
    verbs = _sup(mind).methods()
    out = verbs["ego_investigate"](question="why did gw-3 drift?",
                                   interaction_id=interaction_id, wait=False)
    assert _row(mind, interaction_id)["trigger_id"] == out["result"]["trigger_id"]


def test_an_answer_reaches_a_client_whose_thread_is_gone(mind):
    """The reported defect: restart, thought completes, output stays null."""
    interaction_id = _interaction(mind)
    trigger_id = _answered_trigger(mind, answer="gw-3 drifted by 4.2s")
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)

    # No thread anywhere: this is what a restarted supervisor looks like.
    verbs = _sup(mind).methods()
    out = verbs["io_reconcile"]()

    assert out["count"] == 1 and out["delivered"][0]["from"] == "the record"
    row = _row(mind, interaction_id)
    assert row["status"] == "complete"
    assert "gw-3 drifted" in row["output_preview"]
    delivered = verbs["io_output"](interaction_id=interaction_id,
                                   client_id="client_a")
    assert delivered["output"]["result"]["answer"] == "gw-3 drifted by 4.2s"


def test_delivering_twice_does_not_move_a_finished_interaction(mind):
    interaction_id = _interaction(mind)
    trigger_id = _answered_trigger(mind)
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)
    verbs = _sup(mind).methods()
    assert verbs["io_reconcile"]()["count"] == 1
    first = _row(mind, interaction_id)["completed_at"]
    assert verbs["io_reconcile"]()["count"] == 0, "published an answer twice"
    assert _row(mind, interaction_id)["completed_at"] == first


def test_an_interaction_nobody_answered_is_left_alone(mind):
    interaction_id = _interaction(mind)
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=True, lineage="op-open"),
        actor="test", bump_version=False)
    trigger_id = [r["trigger_id"] for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers")][-1]
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)

    verbs = _sup(mind).methods()
    assert verbs["io_reconcile"]()["count"] == 0
    assert _row(mind, interaction_id)["status"] == "running"


def test_reconciling_is_not_an_external_capability():
    """Delivery is the Harness's job; a client cannot ask for somebody's."""
    from amoeba.scopes import EXTERNAL_IO

    assert "io_reconcile" not in EXTERNAL_IO


# ---------------------------------------------------------------------------
# 2. An investigation is told what it was asked with
# ---------------------------------------------------------------------------
def test_an_investigation_carries_its_request_context(mind):
    interaction_id = _interaction(mind, kind="investigate")
    attachment = {"input_id": "inp_1", "filename": "readings.csv",
                  "media_type": "text/csv", "bytes": 12, "sha256": "a" * 64}
    verbs = _sup(mind).methods()
    out = verbs["ego_investigate"](question="what do these readings show?",
                                   interaction_id=interaction_id,
                                   attachments=[attachment], wait=False)
    trigger = dict(mind.db.conn.execute(
        "SELECT * FROM role_triggers WHERE trigger_id = ?",
        (out["result"]["trigger_id"],)).fetchone())
    payload = mind.blobs.get_json(trigger["payload_sha256"])
    assert payload["interaction_id"] == interaction_id
    assert payload["attachments"] == [attachment]


def test_the_investigation_branch_passes_what_it_was_given(mind):
    """The io_api side: what reaches `ego_investigate` for an investigation."""
    import inspect

    source = inspect.getsource(io_api.build)
    branch = source.split('if kind == "investigate":', 1)[1].split("else:", 1)[0]
    assert "interaction_id=interaction_id" in branch
    assert "attachments=attachments" in branch


def test_two_requests_can_hold_the_same_attachment(mind):
    """Exercising the live surface, two submissions naming one input lost it.

    The input row carried a single interaction id and `io_submit` moved it, so
    the second request took the file away from the first and neither turn
    could resolve it. An input is admitted once and referred to since.
    """
    import base64

    told: list[tuple[str, list[str]]] = []

    def fake_converse(*, message, conversation_id=None, interaction_id=None,
                      attachments=(), wait_seconds=None, **kw):
        told.append((interaction_id, [a["input_id"] for a in attachments]))
        return {"result": {"answer": "ok", "status": "completed"}}

    verbs = _sup(mind, methods={"ego_converse": fake_converse}).methods()
    admitted = verbs["io_attach_input"](
        filename="notes.txt", client_id="client_a", media_type="text/plain",
        content_base64=base64.b64encode(b"the reading was 4.2s").decode())
    input_id = admitted["input_id"]

    first = verbs["io_submit"](text="what does it say?", client_id="client_a",
                               input_ids=[input_id])
    second = verbs["io_submit"](text="and again?", client_id="client_a",
                                input_ids=[input_id])

    deadline = time.time() + 30
    while time.time() < deadline and len(told) < 2:
        time.sleep(0.05)

    carried = dict(told)
    assert carried.get(first["interaction_id"]) == [input_id], (
        "the first request lost its attachment to the second")
    assert carried.get(second["interaction_id"]) == [input_id]


def test_an_attachment_is_referred_to_not_moved(mind):
    """The record of where a file came from is history, and is not rewritten."""
    import base64

    def fake_converse(**kw):
        return {"result": {"answer": "ok", "status": "completed"}}

    verbs = _sup(mind, methods={"ego_converse": fake_converse}).methods()
    admitted = verbs["io_attach_input"](
        filename="notes.txt", client_id="client_a",
        content_base64=base64.b64encode(b"x").decode())
    verbs["io_submit"](text="q", client_id="client_a",
                       input_ids=[admitted["input_id"]])
    time.sleep(0.2)

    row = mind.db.conn.execute(
        "SELECT interaction_id FROM interaction_inputs WHERE input_id = ?",
        (admitted["input_id"],)).fetchone()
    assert row["interaction_id"] is None, (
        "the input row was rewritten to point at the request that used it")


# ---------------------------------------------------------------------------
# What comes back, and what may go in
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, result", [
    ("converse", {"answer": "the drift was 4.2 seconds", "status": "completed"}),
    ("investigate", {"claim": "the drift is real", "status": "completed",
                     "answer": {"nested": "shape"}}),
])
def test_an_answer_is_in_the_same_place_whatever_was_asked(mind, kind, result):
    """An investigation's reply used to be somewhere a conversation's never was."""
    interaction_id = _interaction(mind, kind=kind)
    verbs = _sup(mind).methods()
    payload = {"result": result, "operation_id": None}
    digest = mind.blobs.put_json(payload)
    mind.writer.apply(lambda m: (
        m.register_blob(digest, 0, "application/json", "external_output"),
        m.sql("UPDATE interactions SET status = 'complete', output_sha256 = ?"
              " WHERE interaction_id = ?", (digest, interaction_id))),
        actor="test", bump_version=False)

    out = verbs["io_output"](interaction_id=interaction_id, client_id="client_a")
    assert isinstance(out["answer"], str) and out["answer"].strip(), (
        f"a {kind} left the client nothing to read at the top level")
    assert out["output"]["result"] == result, "the recorded thought is unchanged"


@pytest.mark.parametrize("bad", ["notes\x00.txt", "bell\x07.txt", "del\x7f.txt",
                                 "line\nbreak.txt"])
def test_a_filename_may_not_carry_control_characters(mind, bad):
    """A NUL truncates the name wherever it is handed to C: the record and the
    filesystem would then disagree about what arrived."""
    import base64

    verbs = _sup(mind).methods()
    with pytest.raises(InvalidInput):
        verbs["io_attach_input"](
            filename=bad, client_id="client_a",
            content_base64=base64.b64encode(b"x").decode())


# ---------------------------------------------------------------------------
# What a client is told it may call, and what it is told when it calls wrong
# ---------------------------------------------------------------------------
def test_every_advertised_verb_says_how_to_call_it(mind):
    """`io_await` was advertised with no way to learn it takes a timeout."""
    verbs = _sup(mind).methods()
    caps = verbs["io_capabilities"](client_id="client_a")

    assert set(caps["verbs"]) == set(caps["calls"]), (
        "a verb is advertised that discovery cannot describe")
    await_args = {p["name"]: p for p in caps["calls"]["io_await"]["takes"]}
    assert "timeout_seconds" in await_args, "io_await's timeout is undiscoverable"
    assert await_args["timeout_seconds"]["required"] is False
    assert await_args["timeout_seconds"]["default"] == 30.0
    assert await_args["interaction_id"]["required"] is True


def test_discovery_does_not_ask_for_what_the_credential_decides(mind):
    """`client_id` is bound from the key; naming it invites a call that fails."""
    verbs = _sup(mind).methods()
    caps = verbs["io_capabilities"](client_id="client_a")
    for name, call in caps["calls"].items():
        assert all(p["name"] != "client_id" for p in call["takes"]), name


def test_a_described_call_is_one_the_check_accepts(mind):
    """Discovery and dispatch read the same annotations, so they agree."""
    from amoeba.argcheck import call_problem

    verbs = _sup(mind).methods()
    caps = verbs["io_capabilities"](client_id="client_a")
    for name, call in caps["calls"].items():
        required = {p["name"]: "x" for p in call["takes"] if p["required"]}
        problem = call_problem(verbs[name], {**required, "client_id": "client_a"},
                               method=name)
        assert problem is None, f"{name}: {problem}"


def test_a_call_that_cannot_be_made_names_the_verb_not_the_harness(mind):
    """Live, omitting `text` was answered with `build.<locals>.io_submit()`."""
    from amoeba.argcheck import call_problem

    verbs = _sup(mind).methods()
    problem = call_problem(verbs["io_submit"], {"client_id": "client_a"},
                           method="io_submit")
    assert problem and "text" in problem
    assert "<locals>" not in problem and "build" not in problem


def test_a_list_argument_given_as_a_string_is_refused_before_it_is_walked(mind):
    """`input_ids="inp_1"` iterated characters and refused `no such input: "i"`."""
    from amoeba.argcheck import call_problem

    verbs = _sup(mind).methods()
    problem = call_problem(verbs["io_submit"],
                           {"text": "hello", "input_ids": "inp_1",
                            "client_id": "client_a"}, method="io_submit")
    assert problem and "input_ids" in problem


def test_an_argument_the_verb_does_not_take_is_named(mind):
    from amoeba.argcheck import call_problem

    verbs = _sup(mind).methods()
    problem = call_problem(verbs["io_status"],
                           {"interaction_id": "ixn_1", "client_id": "c",
                            "patience": 5}, method="io_status")
    assert problem and "patience" in problem


# ---------------------------------------------------------------------------
# 3. What the door accepts, cognition receives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("verb, field, kwargs", [
    ("ego_converse", "message", {"message": "x" * 20000}),
    ("ego_investigate", "question", {"question": "y" * 20000}),
])
def test_a_long_request_is_not_quietly_shortened(mind, verb, field, kwargs):
    """20,000 characters is under the door's limit, so all of it is the request."""
    verbs = _sup(mind).methods()
    out = verbs[verb](wait=False, **kwargs)
    trigger = dict(mind.db.conn.execute(
        "SELECT * FROM role_triggers WHERE trigger_id = ?",
        (out["result"]["trigger_id"],)).fetchone())
    payload = mind.blobs.get_json(trigger["payload_sha256"])
    assert len(payload[field]) == 20000, "the end of the request was dropped"


def test_an_investigation_keeps_its_constraints(mind):
    """Constraints live at the end, which is exactly what used to be cut."""
    verbs = _sup(mind).methods()
    constraints = "do not touch production. " * 200
    out = verbs["ego_investigate"](question="look into the drift",
                                   constraints=constraints, wait=False)
    trigger = dict(mind.db.conn.execute(
        "SELECT * FROM role_triggers WHERE trigger_id = ?",
        (out["result"]["trigger_id"],)).fetchone())
    assert mind.blobs.get_json(trigger["payload_sha256"])["constraints"] == constraints


def test_what_is_rendered_says_where_the_rest_is(mind):
    """Bounded for reading, never for storing -- and it says so (I86)."""
    verbs = _sup(mind).methods()
    out = verbs["ego_converse"](message="z" * 20000, wait=False)
    trigger = dict(mind.db.conn.execute(
        "SELECT * FROM role_triggers WHERE trigger_id = ?",
        (out["result"]["trigger_id"],)).fetchone())
    body = mailbox.trigger_body(trigger, mind.blobs)
    assert len(body) < 20000
    assert "truncated at" in body and trigger["payload_sha256"][:12] in body


def test_more_than_the_door_allows_is_refused_not_trimmed():
    from amoeba.io_api import MAX_INPUT_CHARS, _text

    assert len(_text("k" * MAX_INPUT_CHARS, "text")) == MAX_INPUT_CHARS
    with pytest.raises(ResourceExhausted):
        _text("k" * (MAX_INPUT_CHARS + 1), "text")
    with pytest.raises(InvalidInput):
        _text("   ", "text")


def test_a_thread_publishing_first_is_not_overwritten_by_the_reconciler(mind):
    """The real race: the waiting thread answers between the read and the write.

    The reconciler selects an interaction that is still running, and by the
    time it writes, the thread that was waiting has already published. The
    second write must land on nothing rather than move a finished
    interaction's answer and completion time.
    """
    interaction_id = _interaction(mind)
    trigger_id = _answered_trigger(mind, answer="the thread got there first")
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)

    sup = _sup(mind)
    verbs = sup.methods()
    real_answer = verbs["role_answer"]

    def answer_then_publish(**kw):
        state = real_answer(**kw)
        # What the waiting thread would have done, a moment earlier.
        mind.writer.apply(
            lambda m: m.sql(
                "UPDATE interactions SET status = 'complete',"
                " output_preview = 'published by the thread', completed_at = ?"
                " WHERE interaction_id = ?", (111.0, interaction_id)),
            actor="test", bump_version=False)
        return state

    verbs["role_answer"] = answer_then_publish
    verbs["io_reconcile"]()

    row = _row(mind, interaction_id)
    assert row["output_preview"] == "published by the thread"
    assert row["completed_at"] == 111.0, "a finished interaction was republished"
