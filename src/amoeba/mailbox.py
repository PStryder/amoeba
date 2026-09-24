"""The durable role mailbox: what wakes a persistent role, and when.

Ego and Id are persistent *identities*. Their cognition happens in bounded
turns. This module owns the part the model does not:

    something happens  ->  trigger queued        (durable, not yet seen)
    a turn begins      ->  eligible triggers bundled and claimed
    the turn commits   ->  those triggers consumed
    the role dies      ->  claimed-but-unfinished triggers become eligible again

Three distinctions this exists to keep:

**Queued is not seen.** A trigger becomes a cognitive input only when the
Harness puts it in a specific turn's bundle. Submitting something does not mean
a mind has considered it, and the two states are separately visible so nobody
has to guess which one a caller is looking at.

**Events wake cognition; they do not interrupt it.** Anything arriving while a
turn is running stays queued for the next boundary. There is no path that
injects input into an active inference sequence, because the bundle is frozen
before generation starts and nothing reopens it.

**One turn at a time, per role.** Enforced by a partial unique index on
``role_turns``, so a second concurrent turn is a database constraint violation
rather than a convention. Ego and Id still run concurrently with each other,
and neuocytes are untouched.

====================================================================
A TURN MAY CONSUME MANY TRIGGERS, BUT AN INTERACTION BECOMES COMPLETE
ONLY WHEN A TERMINAL RESULT EXPLICITLY ADDRESSED TO THAT INTERACTION
HAS BEEN DURABLY PRODUCED.

NO RESULT FROM ONE INTERACTION MAY SATISFY ANOTHER INTERACTION MERELY
BECAUSE THEIR TRIGGERS SHARED A TURN.
====================================================================

Everything below about requests, bundling and answers exists to hold those
two. An answer used to be a property of a *turn*: whoever was waiting on any
trigger that turn consumed received that turn's result. Two callers therefore
received the same reply, and a thought continued into a second turn returned
only what the first one had managed to say.

Three rules keep them now:

* a trigger records whether anyone is *waiting* on it -- a request is owed an
  answer, an event that merely wakes a role is not;
* a turn admits **at most one answer-bearing request**, and a turn continuing
  an unanswered thought admits **no further answer-bearing request**, so two
  interactions can never be in flight together in one turn. Everything that
  merely *informs* a turn still bundles into it -- a work result, an artifact,
  a message from the other role -- because a continuation usually needs
  exactly that evidence to finish the thought it resumes;
* the answer is written against the request, on a terminal stop, following the
  continuation chain back to the question that started it.

Nothing here is a cognitive component. The Harness decides *when* a role gets
another bounded turn; the role decides only what to think within one.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Sequence

from .errors import InvalidInput, NotFound
from .ids import new_id, sha256_hex
from .store.events import EventKind
from .store.writer import Mutation
from .tools import TOOL_CALL_RE

if TYPE_CHECKING:
    from .mind import Mind

ROLES = ("ego", "id")

TRIGGER_KINDS = (
    "user_input",          # external or operator conversational input
    "work_completed",      # work this role is responsible for finished
    "work_failed",
    "work_cancelled",
    "artifact_event",      # a proposal or promotion on relevant work
    "board_event",         # a relevant blackboard post
    "role_message",        # a targeted message from the other role
    "operator_message",    # the operator talking to this role
    "continuation",        # the Harness deciding a turn was not finished
    "heartbeat",           # Id's periodic homeostatic review
    "startup",             # the organism came up
    "conclusion_recorded", # Ego put a claim into the auditable record
)

# Why a bounded turn ended. Not collapsed into "it ended", because these drive
# different deterministic continuation behaviour.
STOP_REASONS = (
    "model_stop",               # the model finished its thought
    "max_output_tokens",        # truncated by the per-turn output ceiling
    "token_budget_exhausted",
    "tool_turn_limit_reached",
    "deadline_reached",
    "cancelled",
    "context_pressure",
    "backend_error",
    "role_failure",
    "no_environment",
)

# Stop reasons the Harness treats as "this thought was interrupted, not
# finished". A continuation turn is scheduled for these without the model
# having to ask, because a truncated thought cannot be relied on to request
# its own continuation -- the truncation is exactly what stopped it.
NON_TERMINAL = frozenset({
    "max_output_tokens", "token_budget_exhausted", "tool_turn_limit_reached",
    "context_pressure",
})

TERMINAL = frozenset({"model_stop", "cancelled", "deadline_reached",
                      "backend_error", "role_failure", "no_environment"})

MAX_BUNDLE = 16
"""Most triggers admitted to one turn.

Bounded so a burst cannot produce an unboundedly large cognitive input. The
remainder stay queued in order and are picked up by the next turn, which is
visible rather than silent: the bundle reports what it left behind.
"""

MAX_SUMMARY = 400
"""The bounded one-line description carried on the trigger row.

This is what an operator sees in a queue listing. It is deliberately NOT what
the model reads: a request truncated to a preview loses its own constraints,
and Ego would answer a question it was never fully asked."""

MAX_BODY_CHARS = 8000

MAX_ATTACHMENTS_LISTED = 20
"""How many attachments are named in a bundle before the list is summarised.

A bound on the rendering, not on what arrived: the count is always stated, so
a role is never told about fewer files than it was sent without being told
that is what happened.
"""
"""How much of a trigger's full body is rendered into a turn.

The body lives in the content store, in full, whatever this is set to. What
this bounds is how much of it is spent on context, and when it does truncate
the rendering says so rather than quietly handing over a fragment."""

BODY_FIELDS = ("message", "body", "question", "text", "summary")
"""Payload keys that carry what a sender actually said, in preference order."""
MAX_DELIVERIES = 3
"""After this many failed deliveries a trigger is expired rather than retried
forever. A trigger that reliably kills the role that reads it would otherwise
be an undying poison message."""


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# queueing
# ---------------------------------------------------------------------------
def enqueue(m: Mutation, *, role: str, kind: str, source: str,
            summary: str, source_ref: str | None = None,
            payload: dict[str, Any] | None = None,
            correlation_id: str | None = None,
            causal_parent: str | None = None,
            operation_id: str | None = None,
            expects_answer: bool = False,
            lineage: str | None = None,
            ambient: bool = False) -> dict[str, Any]:
    """Record that something happened which a role may need to think about.

    Deliberately cheap and always durable. The decision about whether this
    *deserves* a turn is the scheduler's, and the decision about whether it
    enters cognition is the bundler's; queueing is neither.
    """
    if role not in ROLES:
        raise InvalidInput("unknown role", role=role, allowed=list(ROLES))
    if kind not in TRIGGER_KINDS:
        raise InvalidInput("unknown trigger kind", kind=kind,
                           allowed=list(TRIGGER_KINDS))
    trigger_id = new_id("trg")
    digest = None
    if payload:
        digest = m.put_json(payload, schema="amoeba.role_trigger_payload/1")
    m.sql("INSERT INTO role_triggers(trigger_id, target_role, kind, source,"
          " source_ref, summary, payload_sha256, correlation_id, operation_id,"
          " causal_parent, expects_answer, lineage, ambient, status,"
          " created_at, state_version)"
          " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',?,?)",
          (trigger_id, role, kind, source, source_ref,
           (summary or "")[:MAX_SUMMARY], digest, correlation_id, operation_id,
           causal_parent, 1 if expects_answer else 0, lineage,
           1 if ambient else 0, time.time(), m.prior_version + 1))
    m.emit(EventKind.ROLE_TRIGGER_QUEUED, {
        "trigger_id": trigger_id, "role": role, "kind": kind, "source": source,
        "source_ref": source_ref, "summary": (summary or "")[:200],
        "payload_sha256": digest, "causal_parent": causal_parent,
        "expects_answer": bool(expects_answer), "lineage": lineage,
        "ambient": bool(ambient),
        "note": "queued is not seen; it becomes cognitive input only in a turn"})
    return {"trigger_id": trigger_id, "role": role, "kind": kind,
            "status": "queued", "expects_answer": bool(expects_answer)}


def pending(conn, role: str) -> list[dict[str, Any]]:
    # Ordered by rowid, which is insertion order, rather than by the clock.
    # `created_at` ties for anything queued inside the same millisecond, and
    # the trigger id does not break that tie usefully -- a ULID's suffix is
    # random, so two messages sent together could be presented to Ego in
    # either order. Conversational ordering has to be exact, and rowid is the
    # only thing here that actually is arrival order.
    return [dict(r) for r in conn.execute(
        "SELECT * FROM role_triggers WHERE target_role = ? AND status = 'queued'"
        " ORDER BY rowid ASC", (role,))]


def pending_count(conn, role: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM role_triggers"
        " WHERE target_role = ? AND status = 'queued'", (role,)).fetchone()
    return int(row["n"])


def open_turn(conn, role: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM role_turns WHERE role = ? AND status = 'running'",
        (role,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# bundling
# ---------------------------------------------------------------------------
def trigger_payload(trigger: dict[str, Any], blobs: Any = None) -> dict[str, Any]:
    """The trigger's stored payload, or an empty one when there is none."""
    digest = trigger.get("payload_sha256")
    if not digest or blobs is None:
        return {}
    try:
        payload = blobs.get_json(digest)
    except Exception:  # noqa: BLE001 - unreadable content is not fatal here
        return {}
    return payload if isinstance(payload, dict) else {}


def output_ceiling_for(triggers: Sequence[dict[str, Any]], blobs: Any = None
                       ) -> int | None:
    """The smallest ceiling every trigger in this bundle agrees to.

    All of them, or none: a quiet heartbeat bundled with a real question must
    not shorten the answer to the question. In practice a heartbeat is only
    ever enqueued when nothing else is waiting, so this is a guard rather
    than a common case.
    """
    ceilings = []
    for t in triggers:
        value = trigger_payload(t, blobs).get("output_ceiling")
        if not isinstance(value, int) or value <= 0:
            return None
        ceilings.append(value)
    return min(ceilings) if ceilings else None


def trigger_body(trigger: dict[str, Any], blobs: Any = None) -> str:
    """What this trigger actually says, not the preview of it.

    The summary is a bounded label for operator listings. The body is the
    request, and a role that only ever saw the label would answer questions it
    was never fully asked -- constraints, in particular, live at the end of a
    message rather than in its first 400 characters.

    Falls back to the summary when there is no stored payload, which is the
    normal case for triggers the Harness synthesises (a heartbeat, a work
    completion) where the summary *is* the whole content.
    """
    digest = trigger.get("payload_sha256")
    if not digest or blobs is None:
        return trigger.get("summary") or ""
    try:
        payload = blobs.get_json(digest)
    except Exception:                      # unreadable content is reportable
        return (trigger.get("summary") or "") + "\n  (full body unavailable)"
    if not isinstance(payload, dict):
        return trigger.get("summary") or ""
    body = trigger.get("summary") or ""
    for field in BODY_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            if len(value) > MAX_BODY_CHARS:
                body = (value[:MAX_BODY_CHARS]
                        + f"\n  [truncated at {MAX_BODY_CHARS} characters; "
                          f"{len(value) - MAX_BODY_CHARS} more in {digest[:12]}]")
            else:
                body = value
            break
    # Files sent with the request. Listed rather than inlined: they may be
    # binary, or large, and a role that was shown the bytes would have no way
    # to decline. It is told what arrived and can read what it needs.
    attachments = payload.get("attachments")
    if isinstance(attachments, list) and attachments:
        lines = ["", f"  {len(attachments)} file(s) sent with this request; "
                     "read one with ego_read_attachment(input_id=...):"]
        for att in attachments[:MAX_ATTACHMENTS_LISTED]:
            if not isinstance(att, dict):
                continue
            lines.append(
                f"    - {att.get('filename')} "
                f"({att.get('media_type') or 'unknown type'}, "
                f"{att.get('bytes')} bytes) input_id={att.get('input_id')}")
        if len(attachments) > MAX_ATTACHMENTS_LISTED:
            lines.append(f"    ... and {len(attachments) - MAX_ATTACHMENTS_LISTED} more")
        body = body + "\n".join(lines)
    return body


def render_bundle(triggers: Sequence[dict[str, Any]], *, role: str,
                  left_behind: int = 0, blobs: Any = None) -> str:
    """The bundle as the text a role actually reads.

    Causal type is preserved per trigger rather than flattened into anonymous
    prose: a mind should know *why* it woke, and "a work item you asked for
    failed" is a different thought from "someone sent you a message". Internal
    identifiers are included because they are what the role's own effectors
    take as arguments.
    """
    lines = ["<turn_input>",
             f"role: {role}   triggers: {len(triggers)}"]
    if left_behind:
        lines.append(f"({left_behind} further trigger(s) queued for a later turn)")
    lines.append("")
    for i, t in enumerate(triggers, start=1):
        ref = f" ref={t['source_ref']}" if t.get("source_ref") else ""
        lines.append(f"{i}. [{t['kind']}] from {t['source']}{ref}")
        body = trigger_body(t, blobs)
        if body:
            lines.extend("   " + line for line in body.splitlines())
    lines.append("</turn_input>")
    return "\n".join(lines)


def claim(m: Mutation, mind: "Mind", *, role: str, incarnation: int | None,
          profile_ref: str | None, profile_sha256: str | None,
          environment_sha256: str | None, environment_blob: str | None,
          model_generation: str = "") -> dict[str, Any] | None:
    """Open a turn and freeze the exact inputs it will see.

    Everything that makes this turn's cognition explicable is fixed here, in
    one transaction: the bound profile, the environment manifest, and the
    trigger bundle. Nothing reopens it. Triggers arriving a microsecond later
    stay queued, which is the whole point -- events wake cognition, they do not
    interrupt it.

    Returns ``None`` when there is nothing to think about. That is the normal
    state of an idle organism, not an error.
    """
    if role not in ROLES:
        raise InvalidInput("unknown role", role=role, allowed=list(ROLES))
    if open_turn(mind.db.conn, role) is not None:
        # The unique index would refuse this anyway; refusing here gives a
        # useful error instead of a constraint violation.
        raise InvalidInput(
            f"{role} already has a turn running; a persistent role runs one "
            "bounded turn at a time", role=role)

    queued = pending(mind.db.conn, role)
    if not queued:
        return None
    # Three independent questions decide what a turn may take.
    #
    #   expects_answer   is this trigger owed a reply?   -> reply routing
    #   lineage          whose information is this?      -> cognitive isolation
    #   ambient          may every turn see it?          -> global policy
    #
    # At most one answer-bearing request per turn, and none at all while a
    # continuation is finishing a thought that already owes one: two requests
    # in one turn would share its single answer, and a caller would receive a
    # reply to somebody else's question.
    #
    # Supporting evidence is scoped to the turn's lineage. Nobody owes a work
    # result a reply, so reply routing alone would happily let one
    # interaction's evidence inform another's answer -- both of the rules
    # above stay satisfied while it happens. "Unrelated" must not become the
    # default merely because nothing is owed a reply, so evidence rides along
    # only when it belongs to this lineage or is explicitly ambient.
    turn_lineage: str | None = None
    taken_request = False

    # -- pass 1: whose thought is this turn? ------------------------------
    for t in queued:
        if t["kind"] == "continuation" and t["causal_parent"]:
            if _chain_owes_an_answer(mind.db.conn, t["causal_parent"]):
                taken_request = True
                turn_lineage = turn_lineage_of(mind.db.conn, t["causal_parent"])
                break
    if not taken_request:
        for t in queued:
            if t["expects_answer"]:
                turn_lineage = t["lineage"]
                break

    # -- pass 2: admit what belongs here ----------------------------------
    admitted, deferred = [], []
    request_taken = taken_request
    for t in queued:
        if len(admitted) >= MAX_BUNDLE:
            deferred.append(t)
            continue
        if t["expects_answer"]:
            if request_taken:
                deferred.append(t)
                continue
            request_taken = True
            admitted.append(t)
            continue
        if t["ambient"] or t["kind"] == "continuation":
            admitted.append(t)
            continue
        if turn_lineage and t["lineage"] != turn_lineage:
            # Another interaction's evidence, or nobody's. Unowned is not
            # ambient: a trigger naming no lineage has no claim on a turn that
            # is already serving someone, and it is admitted by the first turn
            # that is not. Every production path either names a lineage or
            # declares itself ambient -- enforced by
            # `test_every_trigger_producer_declares_ownership` -- so arriving
            # here means a producer forgot, not that work is being dropped.
            deferred.append(t)
            continue
        if t["lineage"] and turn_lineage is None:
            # No request yet, so this evidence decides the turn's lineage
            # rather than mixing with a stranger's.
            turn_lineage = t["lineage"]
        admitted.append(t)
    left_behind = len(deferred)

    turn_id = new_id("turn")
    bundle_id = new_id("bnd")
    text = render_bundle(admitted, role=role, left_behind=left_behind,
                         blobs=mind.blobs)
    members = [{"trigger_id": t["trigger_id"], "kind": t["kind"],
                "source": t["source"], "source_ref": t["source_ref"],
                "summary": t["summary"], "payload_sha256": t["payload_sha256"],
                "operation_id": t["operation_id"],
                "created_at": t["created_at"]} for t in admitted]
    ceiling = output_ceiling_for(admitted, mind.blobs)
    body = {"bundle_id": bundle_id, "role": role, "triggers": members,
            "left_behind": left_behind, "text": text,
            "output_ceiling": ceiling}
    bundle_blob = m.put_json(body, schema="amoeba.trigger_bundle/1")
    bundle_sha = sha256_hex(_canon(body))

    parent = next((t["causal_parent"] for t in admitted
                   if t["kind"] == "continuation" and t["causal_parent"]), None)
    # Resume the cut-off message itself, rather than asking for more in a new
    # one, when the continuation is all this turn carries. Evidence may ride
    # along with a continuation -- that guarantee stands -- and evidence has
    # to be shown, which a resumed generation has no place to do; so a turn
    # with anything else in it asks visibly instead. Decided before this turn
    # exists, so "the parent was the last thing this role did" is still true.
    resume = None
    if parent and len(admitted) == 1:
        resume = resume_point(mind.db.conn, mind.blobs, role=role,
                              parent_turn=parent)
    # The externally visible operation this turn answers, if any. A conclusion
    # recorded during the turn is tied to it, so `audit_dossier` can resolve
    # the claim back to the request that caused it.
    operation = next((t["operation_id"] for t in admitted if t["operation_id"]),
                     None)

    m.sql("INSERT INTO role_turns(turn_id, role, incarnation, profile_ref,"
          " profile_sha256, environment_sha256, environment_blob, bundle_id,"
          " bundle_sha256, bundle_blob, trigger_kinds, trigger_count,"
          " started_at, status, model_generation, parent_turn, operation_id,"
          " lineage, state_version)"
          " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'running',?,?,?,?,?)",
          (turn_id, role, incarnation, profile_ref, profile_sha256,
           environment_sha256, environment_blob, bundle_id, bundle_sha,
           bundle_blob, json.dumps([t["kind"] for t in admitted]),
           len(admitted), time.time(), model_generation, parent, operation,
           turn_lineage, m.prior_version + 1))

    for t in admitted:
        m.sql("UPDATE role_triggers SET status = 'claimed', bundle_id = ?,"
              " turn_id = ?, claimed_at = ?, deliveries = deliveries + 1"
              " WHERE trigger_id = ?",
              (bundle_id, turn_id, time.time(), t["trigger_id"]))

    m.emit(EventKind.ROLE_TRIGGER_CLAIMED, {
        "role": role, "turn_id": turn_id, "bundle_id": bundle_id,
        "trigger_ids": [t["trigger_id"] for t in admitted],
        "kinds": [t["kind"] for t in admitted],
        "left_behind": left_behind, "bundle_sha256": bundle_sha,
        "bundle_blob": bundle_blob,
        "note": ("frozen for this turn; anything arriving now waits for the "
                 "next one")})
    # Which requests this turn owes an answer to, decided by the Harness from
    # the durable record rather than inferred by the role from trigger kinds.
    # A continuation owes whatever the thought it resumes still owes, which is
    # exactly the case kind-sniffing could not see -- and the ids matter as
    # well as the count, because a conclusion has to name what it answers or
    # it cannot be audited against the request that prompted it.
    answering = [r["trigger_id"] for r in awaiting_answer(mind.db.conn, turn_id)]

    return {"turn_id": turn_id, "bundle_id": bundle_id, "role": role,
            "text": text, "triggers": members, "left_behind": left_behind,
            "answering": answering, "owes_answer": bool(answering),
            "bundle_sha256": bundle_sha, "bundle_blob": bundle_blob,
            "parent_turn": parent, "operation_id": operation,
            "lineage": turn_lineage, "resume": resume,
            # A ceiling this turn's inputs carry, when every one of them
            # agrees to it. Narrows the role's own ceiling; never widens it.
            "output_ceiling": ceiling}


# ---------------------------------------------------------------------------
# closing a turn
# ---------------------------------------------------------------------------

def owed_lineages(conn, role: str) -> set[Any]:
    """Interactions this role still owes an answer to.

    "Settled" is a property of the *lineage*, not of the turn. A turn can be
    closed while the thought it belongs to continues in the next one, so
    asking only whether this turn finished would drop half of a live
    interaction. A turn with no lineage -- a heartbeat, a startup review -- is
    settled once it is closed, unless some request with no lineage is still
    unanswered, which is why `None` can be a member.
    """
    return {r["lineage"] for r in conn.execute(
        "SELECT DISTINCT lineage FROM role_triggers"
        " WHERE target_role = ? AND expects_answer = 1"
        "   AND answer_status IS NULL", (role,))}


def active_turns(conn, role: str, *, depth: int = 32) -> set[str]:
    """Turns whose thought is still being continued.

    A heartbeat expects no answer, so nothing is ever *owed* on its lineage --
    but a heartbeat cut off by pressure has a continuation queued, and that
    continuation is told to carry on from exactly where the turn stopped. The
    turn it continues, and every earlier piece of the same thought, are active
    work however little anyone is waiting for them.
    """
    out: set[str] = set()
    for row in conn.execute(
            "SELECT causal_parent FROM role_triggers"
            " WHERE target_role = ? AND kind = 'continuation'"
            "   AND status IN ('queued', 'claimed') AND causal_parent IS NOT NULL",
            (role,)):
        turn = row["causal_parent"]
        for _ in range(depth):
            if not turn or turn in out:
                break
            out.add(turn)
            parent = conn.execute("SELECT parent_turn FROM role_turns WHERE turn_id = ?",
                                  (turn,)).fetchone()
            turn = parent["parent_turn"] if parent else None
    return out


def session_spans(conn, role: str, session_handle: str | None
                  ) -> list[dict[str, Any]]:
    """Every closed turn's position in this session, measured or carried.

    Measured spans are the turn's own coordinates, recorded when it closed.
    Carried spans are where a rebuild put a turn in the session it made; the
    turn's own row keeps the coordinates it was measured under (I94), so the
    new ones are recorded against the new handle instead. Either way a span
    describes only the session named, and a closed session's spans are
    unreachable from its successor.
    """
    if not session_handle:
        return []
    rows = conn.execute(
        "SELECT turn_id, lineage, token_start, token_end FROM role_turns"
        " WHERE role = ? AND session_handle = ? AND status != 'running'"
        "   AND token_start IS NOT NULL AND token_end IS NOT NULL"
        "   AND token_end > token_start"
        " UNION ALL"
        " SELECT s.turn_id, t.lineage, s.token_start, s.token_end"
        "  FROM turn_spans s JOIN role_turns t ON t.turn_id = s.turn_id"
        " WHERE t.role = ? AND s.session_handle = ? AND t.status != 'running'"
        " ORDER BY token_start ASC", (role, session_handle, role, session_handle))
    return [{"turn_id": r["turn_id"], "lineage": r["lineage"],
             "start": int(r["token_start"]), "end": int(r["token_end"])}
            for r in rows]


def continuation_depth(conn, turn_id: str, *, limit: int = 32) -> int:
    """How many continuations in a row led to this turn.

    Walks the recorded parent chain rather than trusting a counter carried in
    a payload, so the depth is a fact about what actually happened. A turn
    reached by ordinary triggers has depth 0 however long the history before
    it.
    """
    depth = 0
    current = turn_id
    seen: set[str] = set()
    while current and depth < limit:
        if current in seen:
            break
        seen.add(current)
        row = conn.execute(
            "SELECT parent_turn FROM role_turns WHERE turn_id = ?",
            (current,)).fetchone()
        if row is None or not row["parent_turn"]:
            break
        depth += 1
        current = row["parent_turn"]
    return depth


def turn_lineage_of(conn, turn_id: str) -> str | None:
    """Whose interaction a turn serves.

    A continuation inherits it, which is what keeps a resumed thought reading
    its own evidence rather than whatever happened to arrive.
    """
    row = conn.execute("SELECT lineage FROM role_turns WHERE turn_id = ?",
                       (turn_id,)).fetchone()
    return row["lineage"] if row else None


def _chain_owes_an_answer(conn, turn_id: str) -> bool:
    """Is some request still waiting on the thought this turn continues?

    Cheap, and it is the difference between a continuation quietly adopting a
    stranger's question and one that only finishes its own.
    """
    return bool(awaiting_answer(conn, turn_id))


def awaiting_answer(conn, turn_id: str) -> list[dict[str, Any]]:
    """The requests this turn owes an answer to.

    Follows the continuation chain back to the turn that admitted the request,
    because a thought split across turns still answers the question that
    started it -- returning only the first turn's text was one of the ways an
    answer stopped belonging to a request.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    current = turn_id
    while current and current not in seen and len(seen) < 32:
        seen.add(current)
        for row in conn.execute(
                "SELECT trigger_id, operation_id, correlation_id, turn_id"
                " FROM role_triggers"
                " WHERE turn_id = ? AND expects_answer = 1"
                " AND answer_status IS NULL", (current,)):
            out.append(dict(row))
        parent = conn.execute(
            "SELECT parent_turn FROM role_turns WHERE turn_id = ?",
            (current,)).fetchone()
        current = parent["parent_turn"] if parent else None
    return out


# ---------------------------------------------------------------------------
# answers belong to interactions, not to turns
# ---------------------------------------------------------------------------
# A continuation may resume the very message its parent was cut off in, but
# only when that is literally what the session holds: the parent stopped at
# its output ceiling, and nothing has been appended since.
RESUMABLE = frozenset({"max_output_tokens"})

_TOOL_OPEN, _TOOL_CLOSE = "<tool_call>", "</tool_call>"


def _segment_of(result: dict[str, Any] | None) -> str:
    """The user-facing text one bounded turn contributed, uncleaned.

    `segment` is recorded raw because a resumed continuation joins its parent
    byte for byte: stripping each piece would weld "self" and " knowledge"
    into "selfknowledge". Older results, from before segments were recorded,
    fall back to what they did store.
    """
    if not isinstance(result, dict):
        return ""
    seg = result.get("segment")
    if seg is None:
        seg = result.get("answer") or result.get("text") or ""
    return str(seg)


def _clean_segment(seg: str, *, resumed: bool) -> str:
    """Remove tool-call machinery from one segment, leaving its prose alone.

    Cleaned per segment rather than after joining, because a tool call can be
    cut in half by the boundary: the parent ends inside `<tool_call>` and the
    resumed turn begins with the rest of it. Cleaning the joined text would
    find that unterminated opener and throw away everything after it --
    including every later segment.
    """
    if resumed:
        close, opened = seg.find(_TOOL_CLOSE), seg.find(_TOOL_OPEN)
        if close >= 0 and (opened < 0 or close < opened):
            seg = seg[close + len(_TOOL_CLOSE):]
    seg = TOOL_CALL_RE.sub("", seg)
    dangling = seg.rfind(_TOOL_OPEN)
    if dangling >= 0:
        seg = seg[:dangling]
    return seg


def _open_tool_call(text: str) -> str:
    """The unterminated tool call a truncated generation ended inside, if any."""
    at = text.rfind(_TOOL_OPEN)
    return text[at:] if at >= 0 and _TOOL_CLOSE not in text[at:] else ""


def _turn_result(conn, blobs, turn_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT result_sha256 FROM role_turns WHERE turn_id = ?",
                       (turn_id,)).fetchone()
    if row is None or not row["result_sha256"]:
        return None
    try:
        out = blobs.get_json(row["result_sha256"])
    except Exception:  # noqa: BLE001 - an unreadable fragment is reported as empty
        return None
    return out if isinstance(out, dict) else None


def resume_point(conn, blobs, *, role: str, parent_turn: str
                 ) -> dict[str, Any] | None:
    """Where a continuation can pick up its parent's message, if anywhere.

    Only when the parent was cut off by its output ceiling and was the last
    thing this role did. The role still checks its own session is exactly
    that long before resuming, because the Harness records positions and
    only the role can see the session itself -- a rejuvenation, or anything
    else appended, and the continuation falls back to asking visibly.
    """
    row = conn.execute(
        "SELECT status, stop_reason, session_handle, token_end"
        " FROM role_turns WHERE turn_id = ?", (parent_turn,)).fetchone()
    if row is None or row["status"] != "completed":
        return None
    if row["stop_reason"] not in RESUMABLE:
        return None
    if not row["session_handle"] or row["token_end"] is None:
        return None
    latest = conn.execute(
        "SELECT turn_id FROM role_turns WHERE role = ?"
        " ORDER BY started_at DESC, rowid DESC LIMIT 1", (role,)).fetchone()
    if latest is None or latest["turn_id"] != parent_turn:
        return None
    parent = _turn_result(conn, blobs, parent_turn)
    return {"parent_turn": parent_turn,
            "session_handle": row["session_handle"],
            "token_end": int(row["token_end"]),
            # A tool call the ceiling cut in half is finished by the resumed
            # generation, so the role parses the two halves together.
            "carry": _open_tool_call(_segment_of(parent))}


def assemble_answer(conn, blobs, *, admitting_turn: str, turn_id: str,
                    result: dict[str, Any] | None
                    ) -> tuple[str, list[dict[str, Any]]]:
    """The whole answer to one request, from every turn that produced it.

    The answer used to be whatever the *last* turn said. A thought cut off
    three times therefore arrived as its final quarter, beginning "Continuing
    from where the previous analysis left off" -- the rest had been generated,
    recorded against each turn, and never delivered.

    Walked from the turn that ends the thought back to the one that admitted
    the request, along recorded parent links, then read forward. Nothing is
    stored twice: each turn's result is its fragment, written once when that
    turn closed and never rewritten, and the assembled answer records which
    turns it came from.

    Model text is never rewritten here. A continuation that opens with a
    preamble despite being told not to keeps it: stripping it would take a
    heuristic guess about what the model meant, applied silently to its reply.
    """
    chain: list[str] = []
    current: str | None = turn_id
    seen: set[str] = set()
    while current and current not in seen and len(seen) < 64:
        seen.add(current)
        chain.append(current)
        if current == admitting_turn:
            break
        row = conn.execute("SELECT parent_turn FROM role_turns WHERE turn_id = ?",
                           (current,)).fetchone()
        current = row["parent_turn"] if row else None
    chain.reverse()

    parts: list[str] = []
    segments: list[dict[str, Any]] = []
    for ordinal, tid in enumerate(chain):
        res = result if tid == turn_id else _turn_result(conn, blobs, tid)
        resumed = bool(isinstance(res, dict) and res.get("resumed"))
        if isinstance(res, dict) and res.get("malformed_call"):
            # A capability request in the wrong form is not part of any reply.
            # Withheld whole rather than trimmed -- trimming would be a guess
            # about where the attempt ends -- and kept intact in the turn's own
            # result, so nothing is lost, only not presented as an answer.
            segments.append({"ordinal": ordinal, "turn_id": tid, "chars": 0,
                             "resumed": resumed,
                             "withheld": f"malformed call: {res['malformed_call']}"})
            continue
        # Joined exactly, with nothing between the pieces that the model did
        # not write. A continuation is told its output is appended directly,
        # so inserting a paragraph break would contradict what it was told --
        # and silently editing a reply is not this function's job. Only
        # tool-call machinery is removed, which is not part of any reply.
        seg = _clean_segment(_segment_of(res), resumed=resumed)
        if seg:
            parts.append(seg)
        segments.append({"ordinal": ordinal, "turn_id": tid,
                         "chars": len(seg), "resumed": resumed})
    return "".join(parts).strip(), segments


def complete(m: Mutation, mind: "Mind", *, turn_id: str, stop_reason: str,
             tool_call_count: int = 0, result: dict[str, Any] | None = None,
             status: str = "completed",
             session_handle: str | None = None,
             token_start: int | None = None, token_end: int | None = None,
             max_continuations: int = 3) -> dict[str, Any]:
    """Close a turn, consume its triggers, and decide whether to continue.

    Consumption happens here rather than at claim time so a role that dies
    mid-turn leaves its triggers recoverable. Being handed to a mind that then
    crashed is not the same as having been thought about.
    """
    row = mind.db.conn.execute(
        "SELECT * FROM role_turns WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        raise NotFound("no such turn", turn_id=turn_id)
    if row["status"] != "running":
        raise InvalidInput("that turn is already closed", turn_id=turn_id,
                           status=row["status"])
    if stop_reason not in STOP_REASONS:
        raise InvalidInput("unknown stop reason", stop_reason=stop_reason,
                           allowed=list(STOP_REASONS))

    result_sha = m.put_json(result, schema="amoeba.role_turn_result/1") if result else None
    # COALESCE so a caller that does not measure its context leaves the span
    # unknown rather than writing NULL over something already recorded. An
    # unknown span is simply never evictable, which is the safe direction.
    m.sql("UPDATE role_turns SET status = ?, stop_reason = ?, finished_at = ?,"
          " tool_call_count = ?, result_sha256 = ?,"
          " session_handle = COALESCE(?, session_handle),"
          " token_start = COALESCE(?, token_start),"
          " token_end = COALESCE(?, token_end)"
          " WHERE turn_id = ?",
          (status, stop_reason, time.time(), int(tool_call_count), result_sha,
           session_handle, token_start, token_end, turn_id))
    m.sql("UPDATE role_triggers SET status = 'consumed', consumed_at = ?"
          " WHERE turn_id = ? AND status = 'claimed'", (time.time(), turn_id))

    consumed = [r["trigger_id"] for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers WHERE turn_id = ?", (turn_id,))]
    m.emit(EventKind.ROLE_TURN_ENDED, {
        "turn_id": turn_id, "role": row["role"], "status": status,
        "stop_reason": stop_reason, "tool_call_count": int(tool_call_count),
        "trigger_ids": consumed, "bundle_id": row["bundle_id"],
        "environment_sha256": row["environment_sha256"],
        "profile_ref": row["profile_ref"], "result_sha256": result_sha})
    m.emit(EventKind.ROLE_TRIGGER_CONSUMED, {
        "turn_id": turn_id, "role": row["role"], "trigger_ids": consumed})
    # A generation that tried to author a role boundary and was stopped before
    # the token entered the session. Recorded so the ledger can tell an
    # attempt that was refused from a boundary the Harness wrote itself: the
    # model can hallucinate the Harness, and the record has to know it was
    # not there.
    for attempt in (result or {}).get("structural_attempts") or []:
        m.emit(EventKind.ROLE_STRUCTURE_REFUSED, {
            "turn_id": turn_id, "role": row["role"],
            "token": attempt.get("token"), "piece": attempt.get("piece"),
            "tool_turn": attempt.get("turn"), "admitted": False,
            "note": ("the model sampled a chat-template control token; "
                     "generation stopped and the token never entered the "
                     "session or the KV")})

    # --- is the thought over? --------------------------------------------
    # A turn closing is not the interaction ending. This turn is finished
    # either way; the question is whether the *request* is, and a turn cut
    # off by its ceiling says nothing about that except "not yet".
    awaiting = awaiting_answer(mind.db.conn, turn_id)
    answered: list[str] = []
    continuation = None
    depth = continuation_depth(mind.db.conn, turn_id)
    exhausted = depth >= max(0, int(max_continuations))
    interrupted = status == "completed" and stop_reason in NON_TERMINAL
    continuing = interrupted and not exhausted

    if awaiting and not continuing:
        # Only the model choosing to stop is a finished answer. Anything else
        # that ends the thought -- the continuation limit, a deadline, a
        # backend failure -- leaves an answer that stopped rather than one
        # that concluded, and reporting it as "answered" would tell the
        # caller, permanently, that the fragment was the reply.
        finished = stop_reason == "model_stop" and not interrupted
        ended_because = "continuation_limit" if interrupted else stop_reason
        outcomes: list[dict[str, Any]] = []
        for req in awaiting:
            text, segments = assemble_answer(
                mind.db.conn, mind.blobs, admitting_turn=req["turn_id"],
                turn_id=turn_id, result=result)
            conclusion_id, conclusion_ids = None, []
            withheld = [s["withheld"] for s in segments if s.get("withheld")]
            if not text:
                # Nothing was said at all. Saying so lets a caller stop
                # waiting instead of hanging on a thought that already ended.
                # When something *was* written but withheld as a malformed
                # call, the record says so, rather than calling it silence.
                state = "unanswerable"
                sha = m.put_json(
                    {"answer": "", "complete": False,
                     "ended_because": "malformed_call" if withheld else ended_because,
                     "withheld": withheld, "turn_id": turn_id,
                     "stop_reason": stop_reason, "role": row["role"],
                     "segments": segments},
                    schema="amoeba.trigger_answer/2") if withheld else None
            else:
                state = "answered" if finished else "incomplete"
                # Not a conclusion. Answering is not concluding: every finished
                # answer used to be recorded as an auditable claim, including a
                # complaint that a tool had refused and a self-report about
                # memory, so the audit queue filled with things that were not
                # claims at all. A conclusion is now something Ego chooses to
                # put into the organism's auditable state, with
                # `record_conclusion`, during the turn. What it chose is found
                # here, under the operation that asked, so the answer still
                # says which claims it carried.
                conclusion_ids = [r["conclusion_id"] for r in mind.db.conn.execute(
                    "SELECT conclusion_id FROM conclusions"
                    " WHERE operation_id = ? AND produced_by = 'ego'"
                    " ORDER BY created_at",
                    (req.get("operation_id") or row["operation_id"],))
                    ] if (req.get("operation_id") or row["operation_id"]) else []
                conclusion_id = conclusion_ids[-1] if conclusion_ids else None
                sha = m.put_json(
                    {"answer": text, "complete": finished,
                     "ended_because": ended_because, "turn_id": turn_id,
                     "stop_reason": stop_reason, "role": row["role"],
                     "conclusion_id": conclusion_id,
                     "conclusion_ids": conclusion_ids, "withheld": withheld,
                     # Which turns the answer came from, in order. The text
                     # lives in each turn's own result; this is the index,
                     # so provenance points back at the producing turns
                     # without storing any fragment twice.
                     "segments": segments},
                    schema="amoeba.trigger_answer/2")
            m.sql("UPDATE role_triggers SET answer_sha256 = ?,"
                  " answer_status = ?, answered_at = ?, answered_by_turn = ?"
                  " WHERE trigger_id = ? AND answer_status IS NULL",
                  (sha, state, time.time(), turn_id, req["trigger_id"]))
            answered.append(req["trigger_id"])
            outcomes.append({"trigger_id": req["trigger_id"],
                             "answer_status": state, "answer_sha256": sha,
                             "segments": len(segments),
                             "conclusion_id": conclusion_id})
        m.emit(EventKind.ROLE_TRIGGER_ANSWERED, {
            "turn_id": turn_id, "role": row["role"], "trigger_ids": answered,
            "answers": outcomes, "stop_reason": stop_reason,
            "ended_because": ended_because,
            "note": ("the answer belongs to the request that asked, not to "
                     "the turn that happened to produce it")})

    if interrupted and exhausted:
        # Stop the chain rather than granting another turn that will almost
        # certainly end the same way. Recorded, because a thought abandoned
        # half-finished is something an operator should be able to find --
        # and the request it owed has just been told so, above.
        m.emit(EventKind.ROLE_CONTINUATION_SCHEDULED, {
            "role": row["role"], "previous_turn": turn_id,
            "stop_reason": stop_reason, "trigger_id": None,
            "continuation_depth": depth, "granted": False,
            "note": (f"continuation limit reached after {depth} consecutive "
                     "continuations; the chain stops here rather than "
                     "continuing to truncate")})
    elif continuing:
        # The Harness decides this, not the model. A thought cut off by an
        # output ceiling cannot be expected to ask for its own continuation:
        # being cut off is what stopped it.
        continuation = enqueue(
            m, role=row["role"], kind="continuation", source="harness",
            source_ref=turn_id, causal_parent=turn_id,
            # What a continuation reads when it cannot simply resume the
            # message in place. Its output is appended directly onto the
            # previous one, and the model has to know that, because nothing
            # afterwards rewrites what it writes: a preamble it adds anyway is
            # kept, and will be read.
            summary=(f"Your previous output stopped before the answer was "
                     f"finished ({stop_reason}). Continue the same answer "
                     "exactly where it stopped: what you write is appended "
                     "directly to it. Do not introduce the continuation, "
                     "recap, restart a section, or repeat earlier text."),
            lineage=row["lineage"],
            # The second half of a thought is accountable to the same
            # externally visible operation as the first. Without this a
            # continuation delegates work under no operation at all, and the
            # result comes back a stranger to the thought that asked for it.
            operation_id=row["operation_id"],
            payload={"stop_reason": stop_reason, "previous_turn": turn_id})
        m.emit(EventKind.ROLE_CONTINUATION_SCHEDULED, {
            "role": row["role"], "previous_turn": turn_id,
            "stop_reason": stop_reason,
            "trigger_id": continuation["trigger_id"],
            "continuation_depth": depth + 1, "granted": True,
            "note": ("a continuation is a new bounded turn, not an invisible "
                     "extension of the last one")})
    return {"turn_id": turn_id, "status": status, "stop_reason": stop_reason,
            "consumed": consumed, "answered": answered,
            "awaiting": [r["trigger_id"] for r in awaiting],
            "continuation": continuation,
            "continuation_depth": depth,
            "continuation_limit_reached": bool(interrupted and exhausted)}


def abandon(m: Mutation, mind: "Mind", *, turn_id: str, reason: str
            ) -> dict[str, Any]:
    """Close a turn that never finished, and make its triggers eligible again.

    At-least-once, deliberately. A trigger that was claimed by a role which
    then died has not been thought about, and pretending otherwise would lose
    the one thing the mailbox exists to preserve. Trigger identity survives, so
    a re-delivery is detectable rather than looking like a new event.
    """
    row = mind.db.conn.execute(
        "SELECT * FROM role_turns WHERE turn_id = ?", (turn_id,)).fetchone()
    if row is None:
        raise NotFound("no such turn", turn_id=turn_id)
    m.sql("UPDATE role_turns SET status = 'abandoned', stop_reason = ?,"
          " finished_at = ? WHERE turn_id = ? AND status = 'running'",
          ("role_failure", time.time(), turn_id))

    requeued, expired = [], []
    for r in mind.db.conn.execute(
            "SELECT trigger_id, deliveries FROM role_triggers"
            " WHERE turn_id = ? AND status = 'claimed'", (turn_id,)):
        if int(r["deliveries"]) >= MAX_DELIVERIES:
            m.sql("UPDATE role_triggers SET status = 'expired' WHERE trigger_id = ?",
                  (r["trigger_id"],))
            expired.append(r["trigger_id"])
        else:
            # Requeued for another turn, so any answer it was owed is still
            # owed: the answer fields are cleared along with the claim.
            m.sql("UPDATE role_triggers SET status = 'queued', bundle_id = NULL,"
                  " turn_id = NULL, claimed_at = NULL, answer_status = NULL,"
                  " answer_sha256 = NULL, answered_by_turn = NULL"
                  " WHERE trigger_id = ?", (r["trigger_id"],))
            requeued.append(r["trigger_id"])

    m.emit(EventKind.ROLE_TURN_ABANDONED, {
        "turn_id": turn_id, "role": row["role"], "reason": reason,
        "requeued": requeued, "expired": expired})
    if requeued or expired:
        m.emit(EventKind.ROLE_TRIGGER_RECOVERED, {
            "turn_id": turn_id, "role": row["role"], "requeued": requeued,
            "expired": expired,
            "note": ("claimed but never consumed; delivery is at-least-once "
                     "and trigger ids are preserved so a replay is visible"
                     + (f"; {len(expired)} expired after {MAX_DELIVERIES} "
                        "deliveries" if expired else ""))})
    return {"turn_id": turn_id, "requeued": requeued, "expired": expired}


def recover(m: Mutation, mind: "Mind", *, role: str | None = None,
            reason: str = "process did not survive the turn") -> dict[str, Any]:
    """Re-open turns left running by a process that is gone.

    Called at supervisor start for every role, and again for a single role
    whenever supervision restarts it. Both matter: one open turn per role is a
    database constraint, so a turn its owner never closed does not merely lose
    that thought -- it blocks *every* future turn for that role. The role goes
    on heartbeating, reports healthy, and never thinks again.

    That was a real outage shape before this took a ``role`` argument: recovery
    ran only when the whole supervisor started, while supervision restarts
    individual roles all the time.
    """
    sql = "SELECT turn_id FROM role_turns WHERE status = 'running'"
    params: tuple[Any, ...] = ()
    if role is not None:
        if role not in ROLES:
            raise InvalidInput("unknown role", role=role, allowed=list(ROLES))
        sql += " AND role = ?"
        params = (role,)
    out = []
    for row in mind.db.conn.execute(sql, params):
        out.append(abandon(m, mind, turn_id=row["turn_id"], reason=reason))
    return {"recovered_turns": out}


def expire_stale_turns(m: Mutation, mind: "Mind", *, max_seconds: float
                       ) -> dict[str, Any]:
    """Re-open turns that have been running impossibly long.

    The backstop for a role that is alive but stuck -- wedged inference, a
    handler that never returns -- where nothing dies and so nothing is
    restarted. A crash is recoverable because the process is visibly gone; a
    hang is not, and it wedges the role exactly the same way.

    ``max_seconds`` is the role's own per-turn deadline plus a grace period,
    so a turn that reaches this has already ignored the bound it enforces on
    itself.
    """
    cutoff = time.time() - max(1.0, float(max_seconds))
    out = []
    for row in mind.db.conn.execute(
            "SELECT turn_id, role, started_at FROM role_turns"
            " WHERE status = 'running' AND started_at < ?", (cutoff,)):
        out.append(abandon(
            m, mind, turn_id=row["turn_id"],
            reason=(f"still running {time.time() - row['started_at']:.0f}s "
                    "after it began; the role never closed it")))
    return {"expired_turns": out}
