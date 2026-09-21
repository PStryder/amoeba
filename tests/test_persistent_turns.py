"""Persistent identities running bounded turns.

Ego and Id are not infinite generations. They are identities that survive turn
boundaries, and the Harness decides when each gets another bounded turn, what
woke it, what inputs it is admitted, and what happens when it ends.

The claims worth defending:

* one turn at a time per role, structurally rather than by convention;
* events wake cognition, they do not interrupt it -- anything arriving during
  a turn waits for the next boundary;
* queued is not seen, so submitting something is not the same as a mind having
  considered it;
* Ego wakes because something relevant happened, never because its process
  exists; Id stays responsible through events, continuations and a heartbeat;
* a turn that was interrupted gets a continuation, and that chain is bounded;
* a role that dies mid-turn does not silently swallow its inputs.

The mailbox is tested directly because that is the layer the guarantees live
at; the runtime behaviour is asserted against a live stack, where real
processes and the real scope tables answer.
"""

from __future__ import annotations

import json
import time

import pytest

from amoeba import mailbox
from amoeba.errors import InvalidInput
from amoeba.mind import Mind


def _queue(mind: Mind, role: str, kind: str = "user_input", *,
           source: str = "operator", summary: str = "something",
           source_ref: str | None = None) -> dict:
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind=kind, source=source,
                                  summary=summary, source_ref=source_ref),
        actor="test", bump_version=False)
    return out


def _claim(mind: Mind, role: str) -> dict | None:
    _, out = mind.writer.apply(
        lambda m: mailbox.claim(
            m, mind, role=role, incarnation=1, profile_ref=f"{role}@1",
            profile_sha256="p", environment_sha256="e", environment_blob="eb"),
        actor=role, bump_version=False)
    return out


def _complete(mind: Mind, turn_id: str, stop_reason: str = "model_stop",
              **kw) -> dict:
    _, out = mind.writer.apply(
        lambda m: mailbox.complete(m, mind, turn_id=turn_id,
                                   stop_reason=stop_reason, **kw),
        actor="harness", bump_version=False)
    return out


# ---------------------------------------------------------------------------
# one turn at a time
# ---------------------------------------------------------------------------
def test_a_role_cannot_have_two_turns_at_once(mind):
    """I64. A persistent role runs one bounded turn at a time.

    Two concurrent turns against one role would share an inference session and
    interleave two thoughts into one context. Before this work that was not
    merely possible, it was what happened: `ego_converse` called into the Ego
    process synchronously, and the role's RPC server is threaded, so two
    callers produced two simultaneous turns.

    Defended twice. There is one turn thread in the role process, and the
    database carries a partial unique index over open turns, so the second
    claim is refused even if something else tried.
    """
    _queue(mind, "ego")
    first = _claim(mind, "ego")
    assert first is not None

    _queue(mind, "ego", summary="arrived later")
    with pytest.raises(InvalidInput) as exc:
        _claim(mind, "ego")
    assert "one bounded turn at a time" in str(exc.value)

    # The database refuses it too, independently of that check.
    with pytest.raises(Exception):
        mind.db.conn.execute(
            "INSERT INTO role_turns(turn_id, role, started_at, status,"
            " state_version) VALUES ('t2','ego',0,'running',1)")


def test_ego_and_id_run_concurrently(mind):
    """Serialization is per role, not global. The organism stays concurrent."""
    _queue(mind, "ego")
    _queue(mind, "id", kind="heartbeat", source="scheduler")
    ego_turn = _claim(mind, "ego")
    id_turn = _claim(mind, "id")
    assert ego_turn and id_turn
    assert mailbox.open_turn(mind.db.conn, "ego")["turn_id"] == ego_turn["turn_id"]
    assert mailbox.open_turn(mind.db.conn, "id")["turn_id"] == id_turn["turn_id"]


# ---------------------------------------------------------------------------
# events wake cognition; they do not interrupt it
# ---------------------------------------------------------------------------
def test_input_arriving_during_a_turn_waits_for_the_next_one(mind):
    """I65. The trigger bundle is frozen before generation starts.

    The whole point of a mailbox. If something arriving mid-turn could join
    the turn already running, the model would be reasoning against an input
    set that does not match what provenance recorded, and a conversational
    ordering would be a race.
    """
    first = _queue(mind, "ego", summary="message A")
    turn = _claim(mind, "ego")
    assert [t["trigger_id"] for t in turn["triggers"]] == [first["trigger_id"]]

    later = _queue(mind, "ego", summary="message B")

    # Nothing reopens the running turn.
    reread = mind.db.conn.execute(
        "SELECT trigger_count FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone()
    assert reread["trigger_count"] == 1
    assert "message B" not in turn["text"]
    assert mind.db.conn.execute(
        "SELECT status FROM role_triggers WHERE trigger_id = ?",
        (later["trigger_id"],)).fetchone()["status"] == "queued"

    # And it is waiting at the next boundary.
    _complete(mind, turn["turn_id"])
    second = _claim(mind, "ego")
    assert [t["trigger_id"] for t in second["triggers"]] == [later["trigger_id"]]
    assert "message B" in second["text"]


def test_queued_is_not_seen(mind):
    """Submitting is not the same as a mind having considered it."""
    queued = _queue(mind, "ego")
    row = mind.db.conn.execute(
        "SELECT status, turn_id, consumed_at FROM role_triggers"
        " WHERE trigger_id = ?", (queued["trigger_id"],)).fetchone()
    assert row["status"] == "queued"
    assert row["turn_id"] is None and row["consumed_at"] is None

    turn = _claim(mind, "ego")
    mid = mind.db.conn.execute(
        "SELECT status, consumed_at FROM role_triggers WHERE trigger_id = ?",
        (queued["trigger_id"],)).fetchone()
    # Claimed is still not consumed: a mind that dies now has not thought.
    assert mid["status"] == "claimed" and mid["consumed_at"] is None

    _complete(mind, turn["turn_id"])
    after = mind.db.conn.execute(
        "SELECT status, consumed_at FROM role_triggers WHERE trigger_id = ?",
        (queued["trigger_id"],)).fetchone()
    assert after["status"] == "consumed" and after["consumed_at"] is not None


def test_trigger_order_is_deterministic_and_bundled(mind):
    """A burst becomes one bundle, in arrival order, bounded."""
    ids = [_queue(mind, "ego", summary=f"m{i}")["trigger_id"] for i in range(5)]
    turn = _claim(mind, "ego")
    assert [t["trigger_id"] for t in turn["triggers"]] == ids
    assert turn["text"].index("m0") < turn["text"].index("m4")


def test_a_burst_larger_than_the_bundle_leaves_the_rest_queued(mind):
    """Bounded input, and it says what it left behind rather than dropping it."""
    for i in range(mailbox.MAX_BUNDLE + 4):
        _queue(mind, "ego", summary=f"m{i}")
    turn = _claim(mind, "ego")
    assert len(turn["triggers"]) == mailbox.MAX_BUNDLE
    assert turn["left_behind"] == 4
    assert "further trigger(s) queued" in turn["text"]
    _complete(mind, turn["turn_id"])
    assert mailbox.pending_count(mind.db.conn, "ego") == 4


def test_the_bundle_preserves_every_member_identity(mind):
    """A bundle is not permission to erase individual causal provenance."""
    a = _queue(mind, "ego", kind="user_input", summary="a user asked")
    b = _queue(mind, "ego", kind="work_completed", source="harness",
               source_ref="wk_1", summary="work finished")
    turn = _claim(mind, "ego")
    kinds = {t["kind"] for t in turn["triggers"]}
    assert kinds == {"user_input", "work_completed"}
    assert {t["trigger_id"] for t in turn["triggers"]} == {a["trigger_id"],
                                                          b["trigger_id"]}
    # Causal type survives into what the model reads.
    assert "[user_input]" in turn["text"] and "[work_completed]" in turn["text"]
    assert "ref=wk_1" in turn["text"]


def test_an_idle_role_gets_no_turn(mind):
    """Nothing queued is not an error; it is what an idle organism looks like."""
    assert _claim(mind, "ego") is None
    assert mailbox.open_turn(mind.db.conn, "ego") is None


# ---------------------------------------------------------------------------
# what the model is actually given
# ---------------------------------------------------------------------------
def test_a_turn_shows_the_request_not_a_preview(mind):
    """I76. A role reads what was said, not the first 400 characters of it.

    The summary is a bounded label for operator listings; the body is the
    request. Rendering only the summary meant Ego answered questions it was
    never fully asked -- and constraints live at the end of a message far more
    often than in its opening.
    """
    constraint = "CONSTRAINT: reply only in French."
    message = "A" * 600 + " " + constraint
    mind.writer.apply(
        lambda m: mailbox.enqueue(
            m, role="ego", kind="user_input", source="operator",
            summary=message[:mailbox.MAX_SUMMARY],
            payload={"message": message}),
        actor="test", bump_version=False)

    turn = _claim(mind, "ego")
    assert constraint in turn["text"], "the request was truncated to its preview"
    assert len(turn["text"]) > mailbox.MAX_SUMMARY


def test_an_oversized_body_says_that_it_was_truncated(mind):
    """A budget is fine; a silent one is not."""
    message = "B" * (mailbox.MAX_BODY_CHARS + 5000)
    mind.writer.apply(
        lambda m: mailbox.enqueue(
            m, role="ego", kind="user_input", source="operator",
            summary="big", payload={"message": message}),
        actor="test", bump_version=False)
    turn = _claim(mind, "ego")
    assert "[truncated at" in turn["text"]
    assert "5000 more in" in turn["text"], "it does not say what was withheld"


def test_a_trigger_with_no_payload_still_renders(mind):
    """Harness-synthesised triggers carry their whole content in the summary."""
    _queue(mind, "ego", kind="heartbeat", source="scheduler",
           summary="periodic review")
    turn = _claim(mind, "ego")
    assert "periodic review" in turn["text"]


# ---------------------------------------------------------------------------
# stop reasons and continuation
# ---------------------------------------------------------------------------
def test_stop_reasons_are_recorded_distinctly(mind):
    """I66. Turn-end reasons are first class, not collapsed into "it ended"."""
    seen = {}
    for reason in ("model_stop", "deadline_reached", "cancelled",
                   "tool_turn_limit_reached", "backend_error"):
        _queue(mind, "ego")
        turn = _claim(mind, "ego")
        _complete(mind, turn["turn_id"], stop_reason=reason)
        row = mind.db.conn.execute(
            "SELECT stop_reason FROM role_turns WHERE turn_id = ?",
            (turn["turn_id"],)).fetchone()
        seen[reason] = row["stop_reason"]
    assert seen == {r: r for r in seen}

    with pytest.raises(InvalidInput):
        _queue(mind, "ego")
        turn = _claim(mind, "ego")
        _complete(mind, turn["turn_id"], stop_reason="vibes")


def test_a_non_terminal_stop_schedules_a_continuation(mind):
    """I67. The Harness decides a truncated thought is unfinished.

    Not the model. A thought cut off by an output ceiling cannot be relied on
    to ask for its own continuation, because being cut off is what stopped it;
    requiring a magic phrase to survive truncation would make the guarantee
    depend on the thing that failed.
    """
    _queue(mind, "ego")
    turn = _claim(mind, "ego")
    out = _complete(mind, turn["turn_id"], stop_reason="max_output_tokens")

    assert out["continuation"] is not None
    assert out["continuation"]["kind"] == "continuation"
    nxt = _claim(mind, "ego")
    assert nxt["parent_turn"] == turn["turn_id"], "causal linkage lost"
    assert nxt["triggers"][0]["kind"] == "continuation"
    assert "stopped early" in nxt["text"]

    # It is a new bounded turn, not an invisible extension of the old one.
    assert nxt["turn_id"] != turn["turn_id"]


def test_a_terminal_stop_leaves_the_role_idle(mind):
    _queue(mind, "ego")
    turn = _claim(mind, "ego")
    out = _complete(mind, turn["turn_id"], stop_reason="model_stop")
    assert out["continuation"] is None
    assert mailbox.pending_count(mind.db.conn, "ego") == 0
    assert _claim(mind, "ego") is None


def test_the_continuation_chain_is_bounded(mind):
    """I68. A thought that keeps truncating does not continue forever.

    Found by running it: every turn ended `max_output_tokens`, each scheduled
    a successor, and the organism burned its context until inference refused
    the prompt. An unbounded continuation policy is a token furnace.
    """
    _queue(mind, "ego")
    turn = _claim(mind, "ego")
    depth = 0
    while True:
        out = _complete(mind, turn["turn_id"], stop_reason="max_output_tokens",
                        max_continuations=3)
        if out["continuation"] is None:
            assert out["continuation_limit_reached"] is True
            break
        depth += 1
        assert depth <= 4, "continuation chain did not terminate"
        turn = _claim(mind, "ego")
    assert depth == 3
    assert _claim(mind, "ego") is None, "the chain left work queued"


def test_continuation_depth_counts_only_the_chain(mind):
    """An ordinary trigger starts a fresh chain however long the history."""
    _queue(mind, "ego")
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="max_output_tokens")
    second = _claim(mind, "ego")
    assert mailbox.continuation_depth(mind.db.conn, second["turn_id"]) == 1
    _complete(mind, second["turn_id"], stop_reason="model_stop")

    _queue(mind, "ego", summary="a new conversation")
    third = _claim(mind, "ego")
    assert mailbox.continuation_depth(mind.db.conn, third["turn_id"]) == 0


# ---------------------------------------------------------------------------
# an answer belongs to the request that asked
# ---------------------------------------------------------------------------
def _request(mind, role="ego", summary="a question"):
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind="user_input",
                                  source="operator", summary=summary,
                                  expects_answer=True),
        actor="test", bump_version=False)
    return out


def _lineage_request(mind, lineage, summary):
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary=summary,
                                  expects_answer=True, lineage=lineage),
        actor="test", bump_version=False)
    return out


def _queue_lineage(mind, lineage, *, kind, source, summary, ambient=False):
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind=kind, source=source,
                                  summary=summary, lineage=lineage,
                                  ambient=ambient),
        actor="test", bump_version=False)
    return out

def _answer_of(mind, trigger_id):
    row = mind.db.conn.execute(
        "SELECT answer_status, answer_sha256 FROM role_triggers"
        " WHERE trigger_id = ?", (trigger_id,)).fetchone()
    if row["answer_status"] != "answered" or not row["answer_sha256"]:
        return row["answer_status"], None
    return row["answer_status"], mind.blobs.get_json(row["answer_sha256"])["answer"]


def test_two_requests_never_share_one_turn(mind):
    """I80. A request is answered on its own, or it is not an answer.

    Bundling is right for *events* -- waking once for four things that
    happened while busy is the point. It is wrong for *requests*: two in one
    turn would share that turn's single answer, and a caller would receive a
    reply to somebody else's question. Requests are serialised; everything
    else still bundles around them.
    """
    a = _request(mind, summary="question A")
    b = _request(mind, summary="question B")
    _queue(mind, "ego", kind="work_completed", source="harness",
           summary="an event that may ride along")

    first = _claim(mind, "ego")
    ids = {t["trigger_id"] for t in first["triggers"]}
    assert a["trigger_id"] in ids
    assert b["trigger_id"] not in ids, "two requests shared a turn"
    # The non-answering event still bundles alongside.
    assert len(first["triggers"]) == 2
    assert first["left_behind"] == 1

    _complete(mind, first["turn_id"], result={"answer": "answer to A"})
    second = _claim(mind, "ego")
    assert {t["trigger_id"] for t in second["triggers"]} == {b["trigger_id"]}


def test_each_request_gets_its_own_answer(mind):
    a = _request(mind, summary="question A")
    b = _request(mind, summary="question B")

    t1 = _claim(mind, "ego")
    _complete(mind, t1["turn_id"], result={"answer": "answer to A"})
    t2 = _claim(mind, "ego")
    _complete(mind, t2["turn_id"], result={"answer": "answer to B"})

    assert _answer_of(mind, a["trigger_id"]) == ("answered", "answer to A")
    assert _answer_of(mind, b["trigger_id"]) == ("answered", "answer to B")


def test_a_continued_thought_answers_the_request_that_started_it(mind):
    """I81. A thought spread over turns still answers the question asked.

    The answer used to be whatever the *first* turn produced, because the
    trigger was consumed there. Everything the continuation went on to think
    was produced and then unreachable.
    """
    req = _request(mind, summary="something that needs more than one turn")

    first = _claim(mind, "ego")
    out = _complete(mind, first["turn_id"], stop_reason="max_output_tokens",
                    result={"answer": "a partial thought"})
    assert out["continuation"], "no continuation was scheduled"
    # Still unanswered: the thought is not finished.
    assert _answer_of(mind, req["trigger_id"]) == (None, None)

    second = _claim(mind, "ego")
    assert second["parent_turn"] == first["turn_id"]
    _complete(mind, second["turn_id"], stop_reason="model_stop",
              result={"answer": "the finished thought"})

    assert _answer_of(mind, req["trigger_id"]) == ("answered",
                                                   "the finished thought")


def test_a_continuation_still_receives_its_supporting_evidence(mind):
    """A continuation is held back from new *requests*, not from evidence.

    The thought it resumes usually needs exactly what arrived while it was
    running: the work it delegated, an artifact that landed, a message from
    the other role. Withholding those would make the rule about keeping a turn
    uninformed, when it is only about who is owed a reply.
    """
    request = _request(mind, summary="investigate the locking bug")
    first = _claim(mind, "ego")
    out = _complete(mind, first["turn_id"], stop_reason="max_output_tokens",
                    result={"answer": "delegated, waiting on results"})
    assert out["continuation"]

    # Evidence for the thought in flight, plus a rival request.
    work = _queue(mind, "ego", kind="work_completed", source="harness",
                  source_ref="wk_7", summary="the work you asked for finished")
    note = _queue(mind, "ego", kind="role_message", source="id",
                  summary="Id: that lock is held across an await")
    rival = _request(mind, summary="an unrelated question")

    second = _claim(mind, "ego")
    ids = {t["trigger_id"] for t in second["triggers"]}

    assert work["trigger_id"] in ids, "the continuation was starved of evidence"
    assert note["trigger_id"] in ids
    assert rival["trigger_id"] not in ids, "it adopted an unrelated request"

    _complete(mind, second["turn_id"], stop_reason="model_stop",
              result={"answer": "it is a race on the lock"})
    assert _answer_of(mind, request["trigger_id"]) == (
        "answered", "it is a race on the lock")
    assert _answer_of(mind, rival["trigger_id"]) == (None, None)


def test_a_continuation_does_not_adopt_a_new_request(mind):
    """I82. No answer satisfies an interaction merely by sharing a turn.

    The back door. A continuation carries no request of its own, so a rule
    that only counted requests would admit a *new* one beside it -- and the
    answer, walking the continuation chain, would find both and satisfy the
    newcomer with the older thought's reply. Two interactions in one turn,
    reached sideways.
    """
    first = _request(mind, summary="the original question")
    t1 = _claim(mind, "ego")
    out = _complete(mind, t1["turn_id"], stop_reason="max_output_tokens",
                    result={"answer": "a partial thought"})
    assert out["continuation"]

    # A second caller arrives while the first thought is unfinished.
    second = _request(mind, summary="an unrelated question")

    t2 = _claim(mind, "ego")
    ids = {t["trigger_id"] for t in t2["triggers"]}
    assert second["trigger_id"] not in ids, \
        "a continuation turn adopted an unrelated request"

    _complete(mind, t2["turn_id"], stop_reason="model_stop",
              result={"answer": "the finished thought"})

    # The original got its answer; the newcomer is still waiting for its own.
    assert _answer_of(mind, first["trigger_id"]) == ("answered",
                                                     "the finished thought")
    assert _answer_of(mind, second["trigger_id"]) == (None, None)

    # And it gets a different one.
    t3 = _claim(mind, "ego")
    assert {t["trigger_id"] for t in t3["triggers"]} == {second["trigger_id"]}
    _complete(mind, t3["turn_id"], result={"answer": "a different answer"})
    assert _answer_of(mind, second["trigger_id"]) == ("answered",
                                                      "a different answer")


def test_evidence_from_another_interaction_does_not_ride_along(mind):
    """I83. Supporting evidence is lineage-scoped unless it is ambient.

    The evil cousin of the answer-sharing bug, and worse because both of the
    reply-routing invariants stay green while it happens. Nobody owes a work
    completion a reply, so reply routing alone lets client B's delegated
    result ride into client A's continuation turn -- and Ego composes A's
    answer with B's evidence in front of it.

    `expects_answer` governs reply ownership. `lineage` governs information
    ownership. They are different questions, and this is what happens when
    only the first is asked.
    """
    alpha = _lineage_request(mind, "I17", "what is the status of Alpha?")
    t1 = _claim(mind, "ego")
    assert t1["lineage"] == "I17"
    out = _complete(mind, t1["turn_id"], stop_reason="max_output_tokens",
                    result={"answer": "looking into Alpha"})
    assert out["continuation"]

    # Client B's work lands while A's thought is unfinished.
    beta_work = _queue_lineage(mind, "I18", kind="work_completed",
                               source="harness",
                               summary="Beta credentials rotated to XYZ")
    # A's own evidence, and something every turn may see.
    alpha_work = _queue_lineage(mind, "I17", kind="work_completed",
                                source="harness",
                                summary="Alpha build finished")
    announcement = _queue_lineage(mind, None, kind="operator_message",
                                  source="operator", ambient=True,
                                  summary="maintenance window at 02:00")

    t2 = _claim(mind, "ego")
    ids = {t["trigger_id"] for t in t2["triggers"]}

    assert beta_work["trigger_id"] not in ids, \
        "another interaction's evidence entered this turn"
    assert "Beta credentials" not in t2["text"]
    assert alpha_work["trigger_id"] in ids, "its own evidence was withheld"
    assert announcement["trigger_id"] in ids, "an ambient trigger was withheld"


def test_an_ambient_trigger_is_seen_by_any_lineage(mind):
    """Explicitly global, so it is admitted anywhere -- and only explicitly."""
    _lineage_request(mind, "I17", "a question")
    ann = _queue_lineage(mind, None, kind="operator_message", source="operator",
                         ambient=True, summary="a global announcement")
    turn = _claim(mind, "ego")
    assert ann["trigger_id"] in {t["trigger_id"] for t in turn["triggers"]}


def test_unowned_evidence_does_not_default_to_everyone(mind):
    """Absence of a lineage is not ambience.

    This one tags the stray with nothing at all, which is the case the name
    promises and the one that was actually broken: an untagged trigger was
    admitted into any turn, so "unowned" behaved exactly like "ambient" while
    the architecture said they were different. Tagging it with a rival lineage
    instead -- as this test first did -- only repeats
    `test_evidence_from_another_interaction_does_not_ride_along`.
    """
    _lineage_request(mind, "I17", "a question")
    stray = _queue_lineage(mind, None, kind="work_completed", source="harness",
                           summary="a result nobody tagged")
    turn = _claim(mind, "ego")
    assert stray["trigger_id"] not in {t["trigger_id"] for t in turn["triggers"]}


def test_unowned_evidence_is_admitted_by_a_turn_that_owns_nothing(mind):
    """Held back is not dropped.

    The rule keeps untagged evidence out of a turn that is already serving
    somebody. A turn serving nobody has no lineage to violate, so the same
    trigger goes straight in -- otherwise "not ambient" would quietly mean
    "never delivered".
    """
    stray = _queue_lineage(mind, None, kind="work_completed", source="harness",
                           summary="a result nobody tagged")
    turn = _claim(mind, "ego")
    assert stray["trigger_id"] in {t["trigger_id"] for t in turn["triggers"]}


def test_a_thought_that_ends_without_an_answer_says_so(mind):
    """A caller must be able to stop waiting."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="backend_error", result={})
    status, answer = _answer_of(mind, req["trigger_id"])
    assert status == "unanswerable" and answer is None


def test_an_exhausted_continuation_chain_releases_its_waiter(mind):
    """Nobody will finish it, so nobody should keep waiting for it."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    for _ in range(8):
        out = _complete(mind, turn["turn_id"], stop_reason="max_output_tokens",
                        result={"answer": "still going"}, max_continuations=2)
        if out["continuation"] is None:
            break
        turn = _claim(mind, "ego")
    status, answer = _answer_of(mind, req["trigger_id"])
    assert status == "answered", "the waiter was left hanging"
    assert answer == "still going", "the partial thought was discarded"


def test_an_event_expects_no_answer(mind):
    """Only a request is owed one; a wake-up is not."""
    _queue(mind, "ego", kind="work_completed", source="harness",
           summary="work finished")
    turn = _claim(mind, "ego")
    out = _complete(mind, turn["turn_id"], result={"answer": "noted"})
    assert out["answered"] == []
    assert out["awaiting"] == []


def test_a_requeued_request_is_owed_its_answer_again(mind):
    """A crash mid-turn must not leave a request marked answered."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    mind.writer.apply(lambda m: mailbox.recover(m, mind, role="ego"),
                      actor="supervisor", bump_version=False)
    row = mind.db.conn.execute(
        "SELECT status, answer_status FROM role_triggers WHERE trigger_id = ?",
        (req["trigger_id"],)).fetchone()
    assert row["status"] == "queued" and row["answer_status"] is None


# ---------------------------------------------------------------------------
# crash and recovery
# ---------------------------------------------------------------------------
def test_a_role_that_dies_mid_turn_does_not_swallow_its_inputs(mind):
    """I69. Claimed but never consumed is not the same as thought about.

    At-least-once, deliberately, with trigger identity preserved so a replay
    is visible rather than looking like a new event. Pretending a failed
    delivery was consumed would lose the one thing the mailbox exists for.
    """
    queued = _queue(mind, "ego", summary="please think about this")
    turn = _claim(mind, "ego")
    assert turn is not None

    # The process dies here. Recovery re-opens the turn.
    _, rec = mind.writer.apply(lambda m: mailbox.recover(m, mind),
                               actor="supervisor", bump_version=False)
    assert rec["recovered_turns"]
    assert queued["trigger_id"] in rec["recovered_turns"][0]["requeued"]

    row = mind.db.conn.execute(
        "SELECT status, deliveries FROM role_triggers WHERE trigger_id = ?",
        (queued["trigger_id"],)).fetchone()
    assert row["status"] == "queued"
    assert row["deliveries"] == 1, "the redelivery is not visible"

    # The role can take a turn again; the open turn no longer blocks it.
    again = _claim(mind, "ego")
    assert again is not None
    assert [t["trigger_id"] for t in again["triggers"]] == [queued["trigger_id"]]


def test_a_trigger_that_keeps_killing_the_role_expires(mind):
    """An undying poison message would be worse than a lost one."""
    queued = _queue(mind, "ego")
    for _ in range(mailbox.MAX_DELIVERIES):
        turn = _claim(mind, "ego")
        assert turn is not None
        mind.writer.apply(lambda m: mailbox.recover(m, mind), actor="sup",
                          bump_version=False)
    row = mind.db.conn.execute(
        "SELECT status FROM role_triggers WHERE trigger_id = ?",
        (queued["trigger_id"],)).fetchone()
    assert row["status"] == "expired"
    assert _claim(mind, "ego") is None


def test_a_completed_turn_does_not_reconsume_its_triggers(mind):
    queued = _queue(mind, "ego")
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"])
    with pytest.raises(InvalidInput):
        _complete(mind, turn["turn_id"])
    assert mailbox.pending_count(mind.db.conn, "ego") == 0


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_a_turn_records_what_caused_it(mind):
    """I70. Profile + environment + trigger bundle is reconstructable.

    Recorded as content-addressed bytes rather than recomputed, so a past
    turn stays explicable after the mailbox, the library and the world have
    all moved on.
    """
    a = _queue(mind, "ego", kind="user_input", summary="what is going on")
    b = _queue(mind, "ego", kind="work_completed", source="harness",
               source_ref="wk_9", summary="work done")
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], tool_call_count=2,
              result={"answer": "something"})

    row = dict(mind.db.conn.execute(
        "SELECT * FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone())
    assert row["profile_ref"] == "ego@1"
    assert row["environment_sha256"] == "e"
    assert row["environment_blob"] == "eb"
    assert row["trigger_count"] == 2
    assert json.loads(row["trigger_kinds"]) == ["user_input", "work_completed"]
    assert row["tool_call_count"] == 2

    # The exact bundle the model saw is recoverable, and still hashes right.
    body = mind.blobs.get_json(row["bundle_blob"])
    assert [t["trigger_id"] for t in body["triggers"]] == [a["trigger_id"],
                                                           b["trigger_id"]]
    assert body["text"] == turn["text"]


def test_later_changes_do_not_rewrite_a_historical_turn(mind):
    _queue(mind, "ego")
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"])
    before = dict(mind.db.conn.execute(
        "SELECT * FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone())

    for i in range(3):
        _queue(mind, "ego", summary=f"later {i}")
        nxt = _claim(mind, "ego")
        _complete(mind, nxt["turn_id"])

    after = dict(mind.db.conn.execute(
        "SELECT * FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone())
    assert after == before


def test_queued_at_and_consumed_by_remain_distinguishable(mind):
    queued = _queue(mind, "ego")
    time.sleep(0.01)
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"])
    row = mind.db.conn.execute(
        "SELECT created_at, claimed_at, consumed_at, turn_id FROM role_triggers"
        " WHERE trigger_id = ?", (queued["trigger_id"],)).fetchone()
    assert row["created_at"] < row["claimed_at"] <= row["consumed_at"]
    assert row["turn_id"] == turn["turn_id"]


# ---------------------------------------------------------------------------
# policy: who wakes, and why
# ---------------------------------------------------------------------------
def test_the_scheduler_is_substrate_not_a_cognitive_component():
    """I71. No scheduler neuocyte, no arbiter agent.

    The control plane stays deterministic. A model-driven scheduler would put
    cognition in charge of when cognition happens, which is the one loop this
    architecture keeps open.
    """
    import inspect

    from amoeba import mailbox as mb
    from amoeba import turn_api

    for module in (mb, turn_api):
        source = inspect.getsource(module)
        for banned in ("open_session", "apply_chat_template", "generate(",
                       "lease_work"):
            assert banned not in source, (module.__name__, banned)


def test_the_turn_verbs_are_not_offered_to_the_model():
    """A mind that could claim its own next turn would be scheduling itself."""
    from amoeba import scopes, turn_api

    for role in ("ego", "id"):
        offered = set(scopes.model_facing_verbs(role))
        for verb in turn_api.ROLE_TURN_VERBS + ("role_enqueue_trigger",):
            assert verb not in offered, (role, verb)
        # The operator's view of the mailbox is not a role capability either.
        for verb in turn_api.OPERATOR_TURN_VERBS:
            assert verb not in offered, (role, verb)


def test_the_mailbox_is_absent_from_the_external_surface():
    from amoeba import scopes, turn_api
    from amoeba.io_api import EXTERNAL_VERBS

    verbs = set(turn_api.ROLE_TURN_VERBS) | set(turn_api.OPERATOR_TURN_VERBS) \
        | {"role_enqueue_trigger"}
    assert not (verbs & set(EXTERNAL_VERBS))
    assert not (verbs & set(scopes.EXTERNAL_IO))


def test_unknown_roles_and_kinds_are_refused(mind):
    with pytest.raises(InvalidInput):
        _queue(mind, "operator")
    with pytest.raises(InvalidInput):
        _queue(mind, "ego", kind="whatever")


# ---------------------------------------------------------------------------
# a role must not be wedged by a turn it never closed
# ---------------------------------------------------------------------------
def test_recovery_can_target_one_role(mind):
    """I73. A restarted role gets its own turns recovered, not everyone's.

    Recovery used to run only when the whole supervisor started, while
    supervision restarts individual roles routinely. A turn its owner never
    closed then blocked *every* future turn for that role -- one open turn per
    role is a database constraint -- and the role came back, heartbeated,
    reported healthy, and never thought again while its mailbox filled.
    """
    _queue(mind, "ego", summary="for ego")
    _queue(mind, "id", kind="heartbeat", source="scheduler", summary="for id")
    ego_turn = _claim(mind, "ego")
    id_turn = _claim(mind, "id")
    assert ego_turn and id_turn

    # Ego died. Id did not.
    _, out = mind.writer.apply(
        lambda m: mailbox.recover(m, mind, role="ego"), actor="supervisor",
        bump_version=False)
    assert len(out["recovered_turns"]) == 1

    assert mailbox.open_turn(mind.db.conn, "ego") is None
    assert mailbox.open_turn(mind.db.conn, "id")["turn_id"] == id_turn["turn_id"]
    assert mailbox.pending_count(mind.db.conn, "ego") == 1

    # And the replacement incarnation can work again.
    again = _claim(mind, "ego")
    assert again is not None and again["turn_id"] != ego_turn["turn_id"]

    # The capability is useless unless a restart actually reaches for it, and
    # that wiring is the part that was missing rather than the function.
    import inspect

    from amoeba.supervisor import Supervisor

    restart = inspect.getsource(Supervisor._restart_child)
    assert "_recover_role_turns" in restart, \
        "restarting a role does not recover the turn it never closed"
    tick = inspect.getsource(Supervisor._scheduler_tick)
    assert "_expire_stale_turns" in tick, \
        "nothing sweeps a turn left open by a role that hung rather than died"

    # The capability is useless unless a restart actually reaches for it, and
    # that wiring is the part that was missing rather than the function.
    import inspect

    from amoeba.supervisor import Supervisor

    restart = inspect.getsource(Supervisor._restart_child)
    assert "_recover_role_turns" in restart, \
        "restarting a role does not recover the turn it never closed"
    tick = inspect.getsource(Supervisor._scheduler_tick)
    assert "_expire_stale_turns" in tick, \
        "nothing sweeps a turn left open by a role that hung rather than died"


def test_a_turn_nobody_closed_does_not_wedge_the_role(mind):
    """The backstop for a role that is alive but stuck.

    A crash is recoverable because the process is visibly gone. A hang is not:
    nothing dies, nothing is restarted, and the open turn blocks the role just
    as thoroughly.
    """
    import time as _time

    _queue(mind, "ego", summary="please think")
    turn = _claim(mind, "ego")
    assert turn is not None

    # Not yet stale: a turn in progress must not be taken away from it.
    _, fresh = mind.writer.apply(
        lambda m: mailbox.expire_stale_turns(m, mind, max_seconds=3600),
        actor="supervisor", bump_version=False)
    assert fresh["expired_turns"] == []
    assert mailbox.open_turn(mind.db.conn, "ego") is not None

    # Old enough that the role has already ignored its own deadline.
    mind.db.conn.execute(
        "UPDATE role_turns SET started_at = ? WHERE turn_id = ?",
        (_time.time() - 10_000, turn["turn_id"]))
    mind.db.conn.commit()
    _, swept = mind.writer.apply(
        lambda m: mailbox.expire_stale_turns(m, mind, max_seconds=300),
        actor="supervisor", bump_version=False)

    assert len(swept["expired_turns"]) == 1
    assert mailbox.open_turn(mind.db.conn, "ego") is None
    assert mailbox.pending_count(mind.db.conn, "ego") == 1
    assert _claim(mind, "ego") is not None, "the role is still wedged"


def test_a_role_cannot_forge_attribution_in_a_mailbox():
    """I74. Identity is the credential, in the mailbox too.

    `role_enqueue_trigger` takes `source` as an argument, so a role holding it
    could write into the other role's mailbox attributed to anyone -- Ego
    queueing "the operator says approve this" into Id's cognition. Ego is the
    component most exposed to a confident user, which is exactly why it must
    not hold a verb whose attribution it chooses.

    Roles reach each other through `ego_message_id` / `id_message_ego`, which
    attribute the sender themselves.
    """
    from amoeba import scopes

    for role in ("ego", "id", "neuocyte", "external_io"):
        assert "role_enqueue_trigger" not in scopes.verbs_for(role), role
    # The messaging effectors that replace it are still there.
    assert "ego_message_id" in scopes.EGO
    assert "id_message_ego" in scopes.ID


def test_the_harness_refuses_a_capability_from_a_turn_that_is_not_running(mind):
    """I75, against the real fence rather than a fake that reimplements it.

    The role-side test for this uses a stub supervisor, so it proves the role
    asks correctly and proves nothing about whether the Harness refuses. This
    exercises `role_tool_invoke` itself: a turn that hung, was swept and had
    its inputs handed to a replacement must not be able to act, and being
    refused at commit was never enough -- the result could not land, but the
    side effects could.
    """
    from amoeba import turn_api

    invoked: list[dict] = []

    class _Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
            self.log = __import__("logging").getLogger("test")

        def methods(self):
            def ego_request_work(**kw):
                invoked.append(kw)
                return {"admitted": [{"admitted": True}]}
            return {"ego_request_work": ego_request_work}

        def note_trigger(self, role): pass
        def note_turn_finished(self, *a, **k): pass
        def role_activity(self, role): return "idle"
        def next_heartbeat(self, role): return None

    verbs = turn_api.build(_Sup())

    _queue(mind, "ego", summary="the original request")
    turn = _claim(mind, "ego")

    # While it is running, the capability works.
    ok = verbs["role_tool_invoke"](turn_id=turn["turn_id"],
                                   name="ego_request_work",
                                   arguments={"objective": "legitimate"})
    assert ok["accepted"] is True and len(invoked) == 1

    # The turn hangs and is swept.
    mind.db.conn.execute("UPDATE role_turns SET started_at = 0 WHERE turn_id = ?",
                         (turn["turn_id"],))
    mind.db.conn.commit()
    mind.writer.apply(
        lambda m: mailbox.expire_stale_turns(m, mind, max_seconds=60),
        actor="supervisor", bump_version=False)
    assert mailbox.open_turn(mind.db.conn, "ego") is None

    # Now it wakes up. It must not be able to act.
    zombie = verbs["role_tool_invoke"](turn_id=turn["turn_id"],
                                       name="ego_request_work",
                                       arguments={"objective": "act anyway"})
    assert zombie["accepted"] is False, "a swept turn reached an effector"
    assert "no longer act" in zombie["reason"]
    assert len(invoked) == 1, "the effector ran for a turn that was swept"

    # A turn id that never existed buys nothing either.
    assert verbs["role_tool_invoke"](
        turn_id="turn_invented", name="ego_request_work",
        arguments={})["accepted"] is False

    # And a running turn still cannot exceed what its role is offered.
    _queue(mind, "ego", summary="next")
    live = _claim(mind, "ego")
    over = verbs["role_tool_invoke"](turn_id=live["turn_id"],
                                     name="id_raise_finding", arguments={})
    assert over["accepted"] is False
    assert "not a capability offered to ego" in over["reason"]


def test_no_verb_generates_cognition_outside_the_mailbox():
    """I72. One ingestion path: cognition happens in claimed turns only.

    A verb that calls into a role process to make it think bypasses the
    mailbox entirely -- it runs a generation against the same inference
    session a claimed turn may already be using, which is the concurrency bug
    the whole turn model exists to remove. `converse`, `investigate`,
    `introspect` and `audit` all did exactly that.

    Status, health and snapshot calls into a role are deliberately allowed:
    they read, they do not generate.
    """
    import re
    from pathlib import Path as _P

    cognitive = {"converse", "investigate", "introspect", "audit",
                 "health_report", "propose_maintenance"}
    client_call = re.compile(r"""client\(\s*["'](?:ego|id|role|to_role)""")
    method_call = re.compile(r"""\.call\(\s*["']([a-z_]+)["']""")

    root = _P(__file__).resolve().parents[1] / "src" / "amoeba"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "roles.py":          # the role process itself
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if not client_call.search(line):
                continue
            found = method_call.search(line)
            if found and found.group(1) in cognitive:
                offenders.append(f"{path.name}:{lineno} {line.strip()[:70]}")
    assert not offenders, (
        "these make a role think outside a claimed turn:\n"
        + "\n".join(offenders))


# ---------------------------------------------------------------------------
# Ownership has to be declared where triggers are produced, not assumed.
# ---------------------------------------------------------------------------
def test_every_trigger_producer_declares_ownership():
    """A trigger that names neither a lineage nor ambience starves silently.

    The bundler keeps untagged, non-ambient evidence out of a turn that is
    already serving an interaction. That is the intended rule, and it makes
    forgetting to declare ownership invisible at runtime: the trigger simply
    waits, and on a busy role it may wait a long time.

    So the declaration is required here instead, where forgetting is a failing
    test rather than a stall. Every call to `role_enqueue_trigger` in the
    source must pass `lineage` or `ambient` -- saying "this belongs to nobody
    in particular" is fine, but it has to be said.
    """
    import ast
    from pathlib import Path

    src_dir = Path(__file__).resolve().parents[1] / "src" / "amoeba"
    offenders = []
    for path in sorted(src_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # `f(...)`, `obj.f(...)` and `table["f"](...)` all count.
            name = None
            fn = node.func
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            elif isinstance(fn, ast.Subscript) and isinstance(fn.slice, ast.Constant):
                name = fn.slice.value if isinstance(fn.slice.value, str) else None
            if name != "role_enqueue_trigger":
                continue
            kw = {k.arg for k in node.keywords}
            if not ({"lineage", "ambient"} & kw):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "trigger producers that declare neither lineage nor ambient: "
        + ", ".join(offenders)
        + ". Pass lineage= for an interaction's own evidence, or ambient=True "
          "for something every turn may see.")


# ---------------------------------------------------------------------------
# The round trip: a request delegates work, and the result comes home.
# ---------------------------------------------------------------------------
def _queue_op(mind, *, lineage, operation_id, **kw):
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", lineage=lineage,
                                  operation_id=operation_id, **kw),
        actor="test", bump_version=False)
    return out


def test_a_continuation_keeps_the_operation_it_continues(mind):
    """The second half of a thought is accountable under the same name.

    A continuation that carries no operation delegates work anonymously, and
    the result returns tagged with nothing -- so it never enters the turn that
    asked for it. It also breaks the audit trail: a conclusion recorded in the
    second half of a thought could not be resolved back to the request.
    """
    _queue_op(mind, lineage="op-1", operation_id="op-1", kind="user_input",
              source="operator", summary="a long question", expects_answer=True)
    first = _claim(mind, "ego")
    assert first["lineage"] == "op-1"

    _complete(mind, first["turn_id"], stop_reason="max_output_tokens")
    second = _claim(mind, "ego")
    row = mind.db.conn.execute(
        "SELECT operation_id, lineage FROM role_turns WHERE turn_id = ?",
        (second["turn_id"],)).fetchone()
    assert row["operation_id"] == "op-1"
    assert row["lineage"] == "op-1"


def test_a_delegated_result_returns_to_the_thought_that_asked(mind):
    """The whole point of lineage, exercised end to end at the mailbox.

    Ego is asked something, delegates, and is cut off. The work finishes and
    arrives tagged with the operation that requested it. The continuation --
    which is still finishing that same thought -- must receive it, while a
    result belonging to somebody else waits for a turn of its own.
    """
    _queue_op(mind, lineage="op-A", operation_id="op-A", kind="user_input",
              source="operator", summary="find out about the build",
              expects_answer=True)
    first = _claim(mind, "ego")
    _complete(mind, first["turn_id"], stop_reason="max_output_tokens")

    ours = _queue_op(mind, lineage="op-A", operation_id="op-A",
                     kind="work_completed", source="harness",
                     summary="the build finished")
    theirs = _queue_op(mind, lineage="op-B", operation_id="op-B",
                       kind="work_completed", source="harness",
                       summary="somebody else's build finished")

    second = _claim(mind, "ego")
    admitted = {t["trigger_id"] for t in second["triggers"]}
    assert ours["trigger_id"] in admitted, "the continuation lost its own evidence"
    assert theirs["trigger_id"] not in admitted, "a stranger's result rode along"


def test_a_turns_operation_reaches_what_the_role_does(mind):
    """Work delegated in a turn is accountable to the turn's operation.

    The role cannot supply this: `operation_id` is stripped from
    model-supplied arguments along with every other authority argument. So the
    Harness supplies it from the turn, and without that, work admitted during
    a turn carries no operation -- which is what made its result come back
    untagged and unable to reach the thought that delegated it.
    """
    from amoeba import turn_api

    seen: list[dict] = []

    class _Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
            self.log = __import__("logging").getLogger("test")

        def methods(self):
            def ego_request_work(*, objective, operation_id=None, **kw):
                seen.append({"objective": objective,
                             "operation_id": operation_id})
                return {"admitted": [{"admitted": True}]}
            return {"ego_request_work": ego_request_work}

        def note_trigger(self, role): return None

        def next_heartbeat(self, role): return None

    verbs = turn_api.build(_Sup())

    _queue_op(mind, lineage="op-C", operation_id="op-C", kind="user_input",
              source="operator", summary="look into it", expects_answer=True)
    turn = _claim(mind, "ego")

    out = verbs["role_tool_invoke"](
        turn_id=turn["turn_id"], name="ego_request_work",
        arguments={"objective": "check the build logs"})
    assert out["accepted"] is True
    assert seen == [{"objective": "check the build logs",
                     "operation_id": "op-C"}]


def test_the_harness_does_not_override_an_operation_the_caller_gave(mind):
    """Supplying a default is not the same as seizing the argument.

    The injection fills a gap; a verb called with an explicit operation keeps
    it. Otherwise the Harness would silently rewrite the accountability of a
    call that already knew its own.
    """
    from amoeba import turn_api

    seen: list[str | None] = []

    class _Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
            self.log = __import__("logging").getLogger("test")

        def methods(self):
            def ego_request_work(*, objective, operation_id=None, **kw):
                seen.append(operation_id)
                return {"admitted": []}
            return {"ego_request_work": ego_request_work}

        def note_trigger(self, role): return None

        def next_heartbeat(self, role): return None

    verbs = turn_api.build(_Sup())
    _queue_op(mind, lineage="op-D", operation_id="op-D", kind="user_input",
              source="operator", summary="look into it", expects_answer=True)
    turn = _claim(mind, "ego")
    verbs["role_tool_invoke"](
        turn_id=turn["turn_id"], name="ego_request_work",
        arguments={"objective": "x", "operation_id": "explicit"})
    assert seen == ["explicit"]


# ===========================================================================
# Retention: the working set is forgettable, the record is not
# ===========================================================================
from amoeba import retention  # noqa: E402


def _old_turn(mind, *, age: float, expects_answer: bool = False,
              answer: str | None = "done"):
    """A closed turn with a consumed trigger, aged into the past."""
    trig = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=expects_answer,
                                  lineage="op-x"),
        actor="test", bump_version=False)[1]
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": answer} if answer else None)
    past = time.time() - age
    mind.db.conn.execute(
        "UPDATE role_turns SET started_at = ?, finished_at = ? WHERE turn_id = ?",
        (past, past, turn["turn_id"]))
    mind.db.conn.execute(
        "UPDATE role_triggers SET created_at = ?, consumed_at = ?"
        " WHERE trigger_id = ?", (past, past, trig["trigger_id"]))
    mind.db.conn.commit()
    return trig, turn


def _prune(mind, *, older_than_seconds: float):
    _, out = mind.writer.apply(
        lambda m: retention.prune(m, mind,
                                  older_than_seconds=older_than_seconds),
        actor="test", bump_version=False)
    return out


def test_an_old_consumed_trigger_and_its_closed_turn_are_forgotten(mind):
    """The working set is an index into the record, not the record."""
    trig, turn = _old_turn(mind, age=200.0)
    out = _prune(mind, older_than_seconds=100.0)

    assert out["triggers_pruned"] == 1 and out["turns_pruned"] == 1
    assert mind.db.conn.execute(
        "SELECT COUNT(*) FROM role_triggers WHERE trigger_id = ?",
        (trig["trigger_id"],)).fetchone()[0] == 0
    assert mind.db.conn.execute(
        "SELECT COUNT(*) FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone()[0] == 0


def test_pruning_touches_no_evidence_table(mind):
    """The thing this must never do, asserted against every listed table.

    Counted before and after rather than spot-checked, because the failure
    being guarded against is a DELETE somebody adds later to a table nobody
    thought to re-read this test for.
    """
    _old_turn(mind, age=200.0)
    before = {t: mind.db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in retention.EVIDENCE_TABLES}
    _prune(mind, older_than_seconds=100.0)
    after = {t: mind.db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in retention.EVIDENCE_TABLES}
    shrunk = {t: (before[t], after[t]) for t in before if after[t] < before[t]}
    assert not shrunk, f"pruning removed evidence: {shrunk}"


def test_the_event_chain_still_verifies_after_pruning(mind):
    """Deleting a link in a hash chain does not free space, it destroys proof.

    The working-set tables are not in the chain, so this should hold
    trivially -- which is exactly why it is worth asserting: the day someone
    adds `events` to the prunable list, this is what says no.
    """
    from amoeba.store.events import verify_chain

    _old_turn(mind, age=200.0)
    _prune(mind, older_than_seconds=100.0)
    ok, first_bad = verify_chain(mind.db.conn)
    assert ok, f"the chain broke at {first_bad}"


def test_a_request_still_owed_an_answer_is_never_old_enough(mind):
    """Age is not a reason to stop owing somebody a reply.

    A request whose row vanished would leave its caller waiting on a trigger
    id that no longer exists -- and `_await_turn` reports a missing trigger as
    `NotFound`, so the caller would be told its question never existed.
    """
    trig = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="unanswered",
                                  expects_answer=True, lineage="op-y"),
        actor="test", bump_version=False)[1]
    turn = _claim(mind, "ego")
    # Terminal stop with no answer would mark it unanswerable, so leave the
    # turn open and age the trigger instead: still owed, and old.
    past = time.time() - 10_000.0
    mind.db.conn.execute(
        "UPDATE role_triggers SET created_at = ?, consumed_at = ?, status = 'consumed'"
        " WHERE trigger_id = ?", (past, past, trig["trigger_id"]))
    mind.db.conn.commit()

    out = _prune(mind, older_than_seconds=100.0)
    assert out["triggers_pruned"] == 0
    assert mind.db.conn.execute(
        "SELECT COUNT(*) FROM role_triggers WHERE trigger_id = ?",
        (trig["trigger_id"],)).fetchone()[0] == 1
    assert turn is not None


def test_an_open_turn_is_never_pruned_however_old(mind):
    """A running turn is the present, whatever its timestamp says.

    A turn wedged open for a week is exactly the situation where somebody is
    investigating, and deleting the row would take the evidence away mid-look.

    The turn here holds no triggers at all. That is deliberate: the first
    version of this test claimed a turn normally, which left a claimed trigger
    pointing at it -- and *that* is what kept the turn, not the status check
    this test is named after. Removing the status check did not fail it, which
    is the definition of a test proving nothing.
    """
    past = time.time() - 10_000.0
    mind.db.conn.execute(
        "INSERT INTO role_turns(turn_id, role, incarnation, bundle_id,"
        " trigger_count, started_at, status, state_version)"
        " VALUES ('turn_wedged', 'ego', 1, 'bnd_x', 0, ?, 'running', 1)",
        (past,))
    mind.db.conn.commit()

    out = _prune(mind, older_than_seconds=100.0)
    assert out["turns_pruned"] == 0, "a running turn was pruned"
    assert mind.db.conn.execute(
        "SELECT status FROM role_turns WHERE turn_id = 'turn_wedged'"
    ).fetchone()["status"] == "running"


def test_a_turn_a_continuation_still_points_at_is_kept(mind):
    """Deleting a parent breaks the chain `awaiting_answer` walks.

    A continuation finds the request that started its thought by following
    `parent_turn`. Prune the parent and the chain ends early, so the original
    request is never answered -- the exact failure Tier 2 existed to remove,
    reintroduced by housekeeping.
    """
    # A heartbeat, not a request: a request would still be owed an answer,
    # and *that* would keep the parent alive regardless of the continuation.
    # The first version of this test did exactly that, so removing the guard
    # below changed nothing and the test still passed.
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="heartbeat",
                                  source="scheduler", summary="review",
                                  ambient=True),
        actor="test", bump_version=False)
    first = _claim(mind, "ego")
    _complete(mind, first["turn_id"], stop_reason="max_output_tokens")
    second = _claim(mind, "ego")
    assert second, "no continuation was scheduled"

    past = time.time() - 10_000.0
    mind.db.conn.execute("UPDATE role_turns SET started_at = ?, finished_at = ?"
                         " WHERE turn_id = ?", (past, past, first["turn_id"]))
    mind.db.conn.execute("UPDATE role_triggers SET created_at = ?, consumed_at = ?",
                         (past, past))
    mind.db.conn.commit()

    _prune(mind, older_than_seconds=100.0)
    assert mind.db.conn.execute(
        "SELECT COUNT(*) FROM role_turns WHERE turn_id = ?",
        (first["turn_id"],)).fetchone()[0] == 1, \
        "the parent of a live continuation was pruned"


def test_the_footprint_reports_without_collecting(mind):
    """Measurement first. Blob bytes are counted and never reclaimed."""
    _old_turn(mind, age=200.0)
    before = mind.db.conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]

    out = retention.footprint(mind.db.conn)
    assert out["tables"]["events"]["evidence"] is True
    assert out["tables"]["events"]["prunable"] is False
    assert out["tables"]["role_turns"]["prunable"] is True
    assert out["blobs"]["count"] == before
    assert "never reclaimed" in out["blobs"]["note"]

    after = mind.db.conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    assert after == before, "reporting the footprint deleted something"


def test_no_scope_can_prune(mind):
    """What an organism may forget is policy, not cognition.

    A mind that could prune its own turn history could remove the record of
    what it did in the same motion, so `store_prune` appears in no scope --
    not Ego's, not Id's, not a neuocyte's. Looking is allowed; forgetting is
    not.
    """
    from amoeba import scopes

    for table in (scopes.EGO, scopes.ID, scopes.NEUOCYTE):
        assert "store_prune" not in table
    assert "store_footprint" in scopes.ID
    assert "store_footprint" not in scopes.EGO
    assert "store_footprint" not in scopes.NEUOCYTE


# ===========================================================================
# Attachments: a request says what came with it
# ===========================================================================
def test_a_request_tells_ego_what_was_sent_with_it(mind):
    """Files are named in the bundle, not inlined into it.

    Inlining would hand a role bytes it has no way to decline -- binary, or
    large enough to fill the context it has to answer in. Naming them means it
    learns what arrived and reads only what it needs.
    """
    _, trig = mind.writer.apply(
        lambda m: mailbox.enqueue(
            m, role="ego", kind="user_input", source="operator",
            summary="look at this", expects_answer=True, lineage="op-att",
            payload={"message": "what does this log say?",
                     "attachments": [
                         {"input_id": "inp_1", "filename": "server.log",
                          "media_type": "text/plain", "bytes": 4096},
                         {"input_id": "inp_2", "filename": "heap.bin",
                          "media_type": "application/octet-stream",
                          "bytes": 900}]}),
        actor="test", bump_version=False)
    turn = _claim(mind, "ego")

    assert "what does this log say?" in turn["text"], "the body was lost"
    assert "server.log" in turn["text"] and "heap.bin" in turn["text"]
    assert "inp_1" in turn["text"], "the id Ego needs to read it is missing"
    assert "ego_read_attachment" in turn["text"], "nothing says how to read one"
    assert "2 file(s)" in turn["text"]
    assert trig["trigger_id"]


def test_a_request_with_no_attachments_says_nothing_about_them(mind):
    """The common case stays clean.

    A note about the absence of files in every heartbeat would be noise, and
    noise in a prompt is not free.
    """
    mind.writer.apply(
        lambda m: mailbox.enqueue(
            m, role="ego", kind="user_input", source="operator",
            summary="a plain question", expects_answer=True, lineage="op-p",
            payload={"message": "no files here"}),
        actor="test", bump_version=False)
    turn = _claim(mind, "ego")
    assert "no files here" in turn["text"]
    assert "ego_read_attachment" not in turn["text"]
    assert "file(s)" not in turn["text"]
