"""An answer carries what it did, and an invocation is not what it did.

Live on 2026-09-25 Ego told a client "A worker has been delegated to compute
this sum via code execution in a sandbox". There is no `ego_request_work`
anywhere in that turn: it posted to the board, read the board, read three
results, and answered. The delegation never happened, and nothing in the
record contradicted the claim where the client could see it.

The Harness does not read the prose. It attaches what the operation actually
did, so an unbacked claim is visibly unbacked (I139).

The distinction these tests exist to hold is that a call is not an effect.
In the same live record Ego called `board_post` five times; four were refused
-- on argument shape, then on a missing relation key, then on a foreign key
constraint -- and one posted. Counting invocations would have reported five
postings, or at best one that was indistinguishable from the four failures.
"""

from __future__ import annotations

import json

import pytest

from amoeba.actions import OBJECT_KEY, TURN_BOOKKEEPING, calls_of, effects_of
from amoeba.errors import InvalidInput
from amoeba.store.events import EventKind


def _emit(mind, kind, payload, *, actor, operation_id=None):
    mind.writer.apply(
        lambda m: m.emit(kind, payload, operation_id=operation_id),
        actor=actor, operation_id=operation_id, bump_version=False)


# ---------------------------------------------------------------------------
# A call is not an effect
# ---------------------------------------------------------------------------
def test_a_refused_call_is_not_an_effect(mind):
    """The live case: four refusals and one post is one post."""
    op = "op_probe"
    for reason in ("'evidence' must be a list of objects",
                   "'relations' must be a list of objects",
                   "a relation needs 'to_post' and 'relation'",
                   "FOREIGN KEY constraint failed"):
        _emit(mind, EventKind.ROLE_TOOL_INVOKED,
              {"turn_id": "turn_1", "role": "ego", "tool": "board_post",
               "accepted": False, "reason": reason},
              actor="ego", operation_id=op)
    _emit(mind, EventKind.BOARD_POSTED, {"post_id": "post_real", "author": "ego"},
          actor="ego", operation_id=op)
    _emit(mind, EventKind.ROLE_TOOL_INVOKED,
          {"turn_id": "turn_1", "role": "ego", "tool": "board_post",
           "accepted": True, "reason": None},
          actor="ego", operation_id=op)

    effects = effects_of(mind.db.conn, operation_id=op, actor="ego")
    assert [e["effect"] for e in effects] == ["board.posted"]
    assert effects[0]["post_id"] == "post_real", "the effect did not name what it made"

    made, refused = calls_of(mind.db.conn, ["turn_1"], actor="ego")
    assert len(made) == 1 and len(refused) == 4
    assert "FOREIGN KEY" in refused[-1]["reason"]


def test_an_accepted_call_that_effected_nothing_shows_no_effect(mind):
    """`accepted` means the handler returned, not that anything was committed.

    This is the gap a naive grounding would miss entirely: the call looks
    successful and the state transition never happened.
    """
    op = "op_empty"
    _emit(mind, EventKind.ROLE_TOOL_INVOKED,
          {"turn_id": "t", "role": "ego", "tool": "ego_request_work",
           "accepted": True, "reason": None},
          actor="ego", operation_id=op)

    assert effects_of(mind.db.conn, operation_id=op, actor="ego") == []
    made, refused = calls_of(mind.db.conn, ["t"], actor="ego")
    assert [c["tool"] for c in made] == ["ego_request_work"] and refused == []


def test_asking_for_work_is_not_being_given_it(mind):
    """`work.requested_by_ego` and `work.admitted` are attempt and effect.

    Both are effects of the operation and both are reported, because both
    happened -- but they are different events naming different moments, and a
    request that was never admitted is not a delegated worker.
    """
    op = "op_req"
    _emit(mind, EventKind.WORK_REQUESTED_BY_EGO, {"work_id": None, "objective": "x"},
          actor="ego", operation_id=op)

    kinds = [e["effect"] for e in effects_of(mind.db.conn, operation_id=op, actor="ego")]
    assert kinds == ["work.requested_by_ego"]
    assert "work.admitted" not in kinds, "a request was reported as an admission"


# ---------------------------------------------------------------------------
# The live claim, and what the client now sees
# ---------------------------------------------------------------------------
def test_the_live_claim_has_nothing_under_it(mind):
    """Ego's own answer, replayed: it read and posted, and delegated nothing."""
    op = "op_live"
    _emit(mind, EventKind.BOARD_POSTED, {"post_id": "post_x", "author": "ego"},
          actor="ego", operation_id=op)
    for tool in ("board_read", "result_read", "result_read", "result_read"):
        _emit(mind, EventKind.ROLE_TOOL_INVOKED,
              {"turn_id": "t", "role": "ego", "tool": tool, "accepted": True},
              actor="ego", operation_id=op)

    effects = effects_of(mind.db.conn, operation_id=op, actor="ego")
    assert [e["effect"] for e in effects] == ["board.posted"]
    assert not any(e["effect"].startswith("work.") for e in effects), (
        '"a worker has been delegated" would have had a receipt beside it')


def test_turn_bookkeeping_is_not_reported_as_something_done(mind):
    """Beginning and ending a turn is the Harness, not the role."""
    op = "op_noise"
    for kind in ("role.turn_began", "role.trigger_claimed", "role.turn_ended",
                 "context.measured", "output.emitted"):
        _emit(mind, kind, {"turn_id": "t"}, actor="ego", operation_id=op)

    assert effects_of(mind.db.conn, operation_id=op, actor="ego") == []


def test_an_unclassified_event_is_reported_rather_than_hidden(mind):
    """The default runs towards noise, never towards a false accusation.

    Omitting something a mind really did turns a true statement into an
    apparently unsupported one. A kind nobody has classified therefore
    reports itself.
    """
    op = "op_new"
    _emit(mind, "something.nobody_classified", {"detail": 1},
          actor="ego", operation_id=op)

    effects = effects_of(mind.db.conn, operation_id=op, actor="ego")
    assert [e["effect"] for e in effects] == ["something.nobody_classified"]


def test_another_actors_effects_are_not_credited_to_this_one(mind):
    """Id acting under the same operation is not Ego having acted."""
    op = "op_shared"
    _emit(mind, EventKind.BOARD_POSTED, {"post_id": "p_id"}, actor="id",
          operation_id=op)

    assert effects_of(mind.db.conn, operation_id=op, actor="ego") == []
    assert len(effects_of(mind.db.conn, operation_id=op, actor="id")) == 1


def test_effects_outside_the_operation_are_not_borrowed(mind):
    """An answer accounts for its own operation, not for whatever else ran."""
    _emit(mind, EventKind.BOARD_POSTED, {"post_id": "p_other"}, actor="ego",
          operation_id="op_other")

    assert effects_of(mind.db.conn, operation_id="op_mine", actor="ego") == []


def test_an_operationless_thought_claims_no_receipts(mind):
    """No operation is no grounding, and silence is the honest answer."""
    assert effects_of(mind.db.conn, operation_id=None, actor="ego") == []


# ---------------------------------------------------------------------------
# The classification stays honest as the system grows
# ---------------------------------------------------------------------------
def test_every_object_key_names_a_real_event_kind():
    """A mapping for an event that cannot happen is a mapping nobody checks."""
    known = {v for k, v in vars(EventKind).items()
             if not k.startswith("_") and isinstance(v, str)}
    unknown = sorted(set(OBJECT_KEY) - known)
    assert not unknown, f"OBJECT_KEY names event kinds that do not exist: {unknown}"


def test_bookkeeping_names_real_event_kinds():
    known = {v for k, v in vars(EventKind).items()
             if not k.startswith("_") and isinstance(v, str)}
    unknown = sorted(TURN_BOOKKEEPING - known)
    assert not unknown, f"TURN_BOOKKEEPING names kinds that do not exist: {unknown}"


def test_bookkeeping_and_effects_do_not_overlap():
    """A kind cannot be both the Harness narrating and the role doing."""
    both = sorted(TURN_BOOKKEEPING & set(OBJECT_KEY))
    assert not both, f"declared as both bookkeeping and an effect: {both}"


# ---------------------------------------------------------------------------
# The wire: a real answer, read the way a client reads it
# ---------------------------------------------------------------------------
def test_a_clients_answer_carries_what_the_thought_actually_did(mind):
    """End to end, through the surface the live claim was made on.

    The other tests prove the rule; this proves something applies it. Without
    it the accounting could be perfect and never reach the answer, which is
    exactly the state the organism was in on 2026-09-25.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_external_delivery import _interaction, _row, _sup  # noqa: E402
    from test_persistent_turns import _claim, _complete  # noqa: E402

    from amoeba import mailbox

    op = "op_receipts"
    interaction_id = _interaction(mind)
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=True, lineage=op,
                                  operation_id=op),
        actor="test", bump_version=False)

    turn = _claim(mind, "ego")
    # It posts, and it does not delegate -- the live shape exactly.
    _emit(mind, EventKind.BOARD_POSTED, {"post_id": "post_live", "author": "ego"},
          actor="ego", operation_id=op)
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": "A worker has been delegated to compute this."})

    trigger_id = [r["trigger_id"] for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers WHERE target_role = 'ego'")][-1]
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)

    verbs = _sup(mind).methods()
    verbs["io_reconcile"]()
    out = verbs["io_output"](interaction_id=interaction_id, client_id="client_a")

    assert "delegated" in (out["answer"] or ""), "the claim under test is gone"
    assert "effected" in out, "the client cannot see what the thought did"
    kinds = [e["effect"] for e in out["effected"]]
    assert "board.posted" in kinds, "a real effect was not reported"
    assert not any(k.startswith("work.") for k in kinds), (
        'the claim "a worker has been delegated" is reported as unbacked')
