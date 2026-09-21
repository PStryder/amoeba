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
