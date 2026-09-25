"""What a role actually *did*, taken from the record rather than from its prose.

Live on 2026-09-25 Ego answered a client with "A worker has been delegated to
compute this sum via code execution in a sandbox". No `ego_request_work`
appears anywhere in that turn: it posted to the board, read the board, read
three results, and answered. The delegation never happened.

That is not a wrong belief about the world. It is the model narrating an
intention in the past tense, and the Harness having no opinion about it. The
shape reaches much further than workers -- "I saved the file", "I notified
Id", "I cancelled the operation", "I posted the finding" -- and each is a
claim about a state transition that either happened or did not.

The Harness does not read the prose and does not try to. Policing text would
put a classifier on the critical path of every answer, where a misfire either
blocks a correct reply or mangles it. Instead an answer carries what its
operation actually did, so a claim with no receipt beside it is *visibly*
unbacked. This is the same move as I138: a post may say whatever it likes in
its body, and what it cannot do is assert a root it did not acquire.

## An invocation is not an effect

The obvious grounding -- "it called the tool, so it did the thing" -- is
wrong, and the same live record shows why. Ego called `board_post` five times:

    07:42:43  board_post  refused: 'evidence' must be a list of objects
    07:42:45  board_post  refused: 'relations' must be a list of objects
    07:42:47  board_post  refused: a relation needs 'to_post' and 'relation'
    07:42:49  board_post  refused: FOREIGN KEY constraint failed
    07:42:51  board.posted  post_id=post_01M3C606GS3Y30Q6WX18ABBV8E
    07:42:51  board_post  accepted

Four of those five calls posted nothing. `accepted` is weaker than it looks
too: it means the handler returned without raising, not that anything was
committed. What proves a post exists is `board.posted`, a separate durable
event naming the post it created. So three things are kept apart:

    attempted  the call was made          -- role.tool_invoked
    refused    the call was turned away   -- role.tool_invoked, accepted=False
    effected   something came into being  -- the domain event, with its id

`work.requested_by_ego` and `work.admitted` are this distinction already
written down: Ego asking for work is an attempt, and the Harness admitting it
is the effect. Only the second is a delegated worker.

## Why the default runs the other way here

An effect is any event in the operation attributed to the role that is not
declared turn bookkeeping. Unlisted counts as an *effect* -- the opposite of
the default `FRESH_ACQUISITION` takes, and deliberately so. Omitting something
a mind really did turns a true statement into an apparently unsupported one,
which is the one failure this must never produce. A new event kind therefore
reports itself until somebody says it is noise.
"""

from __future__ import annotations

from typing import Any

# The Harness narrating the turn: claiming, beginning, ending, measuring.
# None of it is something the role *did* to anything, and all of it would
# otherwise drown the list. Everything not named here counts as an effect.
TURN_BOOKKEEPING = frozenset({
    "role.turn_began", "role.turn_ended", "role.turn_abandoned",
    "role.trigger_queued", "role.trigger_claimed", "role.trigger_consumed",
    "role.trigger_answered", "role.trigger_recovered",
    "role.environment_built", "role.continuation_scheduled",
    "role.tool_invoked", "role.not_thinking", "role.structure_refused",
    "context.measured", "context.pressure",
    "input.received", "output.emitted",
    "interaction.accepted", "interaction.completed", "interaction.failed",
    "interaction.wait_expired",
    "tool.requested", "tool.result", "tool.rejected",
})

# Which field of an effect's payload names the thing that came into being, so
# a reader is given the identifier and not just the verb. An effect whose kind
# is absent here is still reported -- it simply arrives without an id.
OBJECT_KEY = {
    "board.posted": "post_id",
    "board.related": "to_post",
    "board.status_changed": "post_id",
    "board.promoted_to_memory": "memory_id",
    "work.admitted": "work_id",
    "work.requested_by_ego": "work_id",
    "work.cancelled": "work_id",
    "work.message_sent": "work_id",
    "work.rejected": "work_id",
    "conclusion.recorded": "conclusion_id",
    "conclusion.withdrawn": "conclusion_id",
    "memory.created": "memory_id",
    "memory.superseded": "memory_id",
    "memory.retracted": "memory_id",
    "artifact.proposed": "artifact_id",
    "interaction.result_surfaced": "result_id",
    "id.finding_raised": "finding_id",
    "ego.review_requested": "review_id",
}


def _payload(row: Any) -> dict[str, Any]:
    import json
    raw = row["payload_inline"]
    if not raw:
        return {}
    try:
        got = json.loads(raw)
    except Exception:  # noqa: BLE001 -- a malformed payload is not an effect's fault
        return {}
    return got if isinstance(got, dict) else {}


def effects_of(conn: Any, *, operation_id: str | None, actor: str,
               ) -> list[dict[str, Any]]:
    """The state transitions this role caused under this operation.

    Read from the event chain, which is what proves them. The answer already
    finds the operation's conclusions this way; its actions are the same
    question asked of every kind of effect rather than one.
    """
    if not operation_id:
        return []
    out: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT seq, kind, payload_inline FROM events"
        " WHERE operation_id = ? AND actor_id = ? ORDER BY seq",
        (operation_id, actor),
    ):
        kind = row["kind"]
        if kind in TURN_BOOKKEEPING:
            continue
        effect: dict[str, Any] = {"effect": kind, "event_seq": row["seq"]}
        key = OBJECT_KEY.get(kind)
        if key:
            value = _payload(row).get(key)
            if value is not None:
                effect[key] = value
        out.append(effect)
    return out


def calls_of(conn: Any, turn_ids: list[str], *, actor: str
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every tool call these turns made, split into the ones that were refused.

    Reported beside the effects because the three questions are different: a
    call that was made, a call that was turned away, and a thing that came
    into being. Four of Ego's five `board_post` calls were refusals.
    """
    if not turn_ids:
        return [], []
    marks = ",".join("?" for _ in turn_ids)
    made: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for row in conn.execute(
        f"SELECT seq, payload_inline FROM events WHERE kind = 'role.tool_invoked'"
        f" AND actor_id = ? ORDER BY seq",
        (actor,),
    ):
        got = _payload(row)
        if got.get("turn_id") not in turn_ids:
            continue
        entry = {"tool": got.get("tool"), "event_seq": row["seq"]}
        if got.get("accepted"):
            made.append(entry)
        else:
            refused.append({**entry, "reason": got.get("reason")})
    return made, refused
