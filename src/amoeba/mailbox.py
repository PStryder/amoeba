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
    body = {"bundle_id": bundle_id, "role": role, "triggers": members,
            "left_behind": left_behind, "text": text}
    bundle_blob = m.put_json(body, schema="amoeba.trigger_bundle/1")
    bundle_sha = sha256_hex(_canon(body))

    parent = next((t["causal_parent"] for t in admitted
                   if t["kind"] == "continuation" and t["causal_parent"]), None)
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
            "lineage": turn_lineage}


# ---------------------------------------------------------------------------
# closing a turn
# ---------------------------------------------------------------------------

def settled_spans(conn, role: str, session_handle: str | None
                  ) -> list[dict[str, Any]]:
    """Closed turns, in this session, whose interaction owes nothing.

    "Settled" is a property of the *lineage*, not of the turn. A turn can be
    closed while the thought it belongs to continues in the next one, so
    asking only whether this turn finished would evict half of a live
    interaction.

    A turn with no lineage -- a heartbeat, a startup review -- is settled once
    it is closed, unless some request with no lineage is still unanswered.
    """
    if not session_handle:
        return []
    owed = {r["lineage"] for r in conn.execute(
        "SELECT DISTINCT lineage FROM role_triggers"
        " WHERE target_role = ? AND expects_answer = 1"
        "   AND answer_status IS NULL", (role,))}
    out = []
    for row in conn.execute(
            "SELECT turn_id, lineage, token_start, token_end FROM role_turns"
            " WHERE role = ? AND session_handle = ? AND status != 'running'"
            "   AND token_start IS NOT NULL AND token_end IS NOT NULL"
            "   AND token_end > token_start"
            " ORDER BY token_start ASC", (role, session_handle)):
        if row["lineage"] in owed:
            continue
        out.append({"turn_id": row["turn_id"], "lineage": row["lineage"],
                    "start": int(row["token_start"]),
                    "end": int(row["token_end"])})
    return out

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
                "SELECT trigger_id, operation_id, correlation_id FROM role_triggers"
                " WHERE turn_id = ? AND expects_answer = 1"
                " AND answer_status IS NULL", (current,)):
            out.append(dict(row))
        parent = conn.execute(
            "SELECT parent_turn FROM role_turns WHERE turn_id = ?",
            (current,)).fetchone()
        current = parent["parent_turn"] if parent else None
    return out


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

    # --- answer the request, if this turn finished the thought -----------
    awaiting = awaiting_answer(mind.db.conn, turn_id)
    answered: list[str] = []
    if awaiting and stop_reason not in NON_TERMINAL:
        answer_text = ""
        if isinstance(result, dict):
            answer_text = str(result.get("answer") or result.get("text") or "")
        if answer_text.strip():
            answer_sha = m.put_json(
                {"answer": answer_text, "turn_id": turn_id,
                 "stop_reason": stop_reason, "role": row["role"]},
                schema="amoeba.trigger_answer/1")
            state = "answered"
        else:
            # A terminal stop that produced nothing is not an answer. Saying
            # so lets a caller stop waiting instead of hanging on a thought
            # that already ended.
            answer_sha, state = None, "unanswerable"
        for req in awaiting:
            m.sql("UPDATE role_triggers SET answer_sha256 = ?,"
                  " answer_status = ?, answered_at = ?, answered_by_turn = ?"
                  " WHERE trigger_id = ?",
                  (answer_sha, state, time.time(), turn_id, req["trigger_id"]))
            answered.append(req["trigger_id"])
        m.emit(EventKind.ROLE_TRIGGER_ANSWERED, {
            "turn_id": turn_id, "role": row["role"], "trigger_ids": answered,
            "answer_status": state, "answer_sha256": answer_sha,
            "stop_reason": stop_reason,
            "note": ("the answer belongs to the request that asked, not to "
                     "the turn that happened to produce it")})

    continuation = None
    depth = continuation_depth(mind.db.conn, turn_id)
    exhausted = depth >= max(0, int(max_continuations))
    if status == "completed" and stop_reason in NON_TERMINAL and exhausted:
        # Stop the chain rather than granting another turn that will almost
        # certainly end the same way. Recorded, because a thought abandoned
        # half-finished is something an operator should be able to find.
        m.emit(EventKind.ROLE_CONTINUATION_SCHEDULED, {
            "role": row["role"], "previous_turn": turn_id,
            "stop_reason": stop_reason, "trigger_id": None,
            "continuation_depth": depth, "granted": False,
            "note": (f"continuation limit reached after {depth} consecutive "
                     "continuations; the chain stops here rather than "
                     "continuing to truncate")})
        # Nobody is going to finish this thought, so nobody should keep
        # waiting for it. A partial answer is still an answer; no answer at
        # all is said plainly.
        for req in awaiting:
            partial = ""
            if isinstance(result, dict):
                partial = str(result.get("answer") or result.get("text") or "")
            sha = m.put_json(
                {"answer": partial, "turn_id": turn_id, "partial": True,
                 "stop_reason": stop_reason, "role": row["role"]},
                schema="amoeba.trigger_answer/1") if partial.strip() else None
            m.sql("UPDATE role_triggers SET answer_sha256 = ?,"
                  " answer_status = ?, answered_at = ?, answered_by_turn = ?"
                  " WHERE trigger_id = ?",
                  (sha, "answered" if sha else "unanswerable", time.time(),
                   turn_id, req["trigger_id"]))
            answered.append(req["trigger_id"])
    elif status == "completed" and stop_reason in NON_TERMINAL:
        # The Harness decides this, not the model. A thought cut off by an
        # output ceiling cannot be expected to ask for its own continuation:
        # being cut off is what stopped it.
        continuation = enqueue(
            m, role=row["role"], kind="continuation", source="harness",
            source_ref=turn_id, causal_parent=turn_id,
            summary=(f"your previous turn stopped early ({stop_reason}); "
                     "continue from where you left off"),
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
            "continuation_limit_reached": bool(
                exhausted and stop_reason in NON_TERMINAL)}


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
