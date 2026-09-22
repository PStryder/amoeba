"""Ego's senses and effectors: the outward half's authorised surface.

Ego interprets what the user wants and introduces that objective into the rest
of the organism. It is **not** a mandatory central planner: it may hand over a
one-line objective or a detailed plan, and the work system, the blackboard and
the neuocytes are free to discover decomposition during execution.

The design rule is *bound Ego by capability physics, not by workflow*. Nothing
here dictates how Ego must think. What it fixes is what Ego can reach:

* Ego **asks** for work; the Harness decides whether, when, by whom, with which
  prompt version, and under what budget.
* Ego **reads** durable, published cognition -- results, blackboard posts,
  artifacts and their evidence.
* Ego **proposes** changes to maintained state; it does not author beliefs.

## Running scratch is not a communication channel

Ego may know a neuocyte exists, what work it holds, its board mode and its
status. It may not read its compute sandbox. Half-written scratch is not a
claim anybody made -- reasoning over it would let Ego consume something no
neuocyte ever published, with no provenance and no moment at which the worker
stood behind it.

Anything worth Ego's attention crosses an explicit boundary: a board post, a
work result, an artifact proposal, or durable evidence. Those are the surfaces
with authorship attached.

## Three distinct ways Ego reaches work

Kept separate on purpose, because collapsing them would make the weakest one
the effective semantics of all three:

1. **Admission** (`ego_request_work`) -- intent in, execution conditions
   decided by the Harness.
2. **Blackboard** (`board_post` / `board_read`) -- the ordinary collaborative
   surface, subject to each item's board policy.
3. **Targeted work message** (`ego_work_message`) -- a governed message to a
   *work item*, recorded and delivered by the Harness, refused where the item's
   independence requires it.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Sequence

from .errors import InvalidInput, NotFound
from .ids import new_id
from .store.events import EventKind
from .store.work_repo import BOARD_ACCESS, WORK_CLASSES
from .store.writer import Mutation

REVIEW_SUBJECTS = ("conclusion", "work", "memory", "artifact", "general")

if TYPE_CHECKING:
    from .supervisor import Supervisor

MESSAGE_KINDS = ("clarification", "constraint", "context")
LIVE_STATUSES = ("queued", "leased", "blocked")


def _text(value: str, field: str, *, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInput(f"{field} must be a non-empty string")
    return value.strip()[:limit]


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None

    # ==================================================================
    # Senses
    # ==================================================================
    def ego_work_view(*, limit: int = 50, mine_only: bool = False
                      ) -> dict[str, Any]:
        """A light operational view of productive work.

        Deliberately *not* Id's pulse. Ego needs to know what it asked for and
        how it is going; it does not need failure counters, context occupancy,
        storage pressure or resource digests to do that, and handing it the
        organism's health telemetry would blur the line between the outward
        interface and the inward monitor.
        """
        rows = [dict(r) for r in mind.db.conn.execute(
            "SELECT work_id, objective, work_class, status, origin_actor,"
            " lease_owner, attempt, board_access, sandbox_allowed, created_at,"
            " updated_at FROM work_items"
            " WHERE (? = 0 OR origin_actor = 'ego')"
            " ORDER BY created_at DESC LIMIT ?",
            (1 if mine_only else 0, max(1, min(int(limit), 200))))]
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(row["status"], []).append(row)
        return {
            "by_status": {k: len(v) for k, v in buckets.items()},
            "running": [r["work_id"] for r in buckets.get("leased", [])],
            "queued": [r["work_id"] for r in buckets.get("queued", [])],
            "blocked": [r["work_id"] for r in buckets.get("blocked", [])],
            "originated_by_ego": [r["work_id"] for r in rows
                                  if r["origin_actor"] == "ego"],
            "items": rows,
            "note": ("productive-work state only; organism health telemetry "
                     "belongs to Id"),
        }

    def ego_artifact_evidence(*, artifact_id: str) -> dict[str, Any]:
        """The exact immutable bytes behind a proposal.

        A proposal is content-addressed when it is made, so this is what was
        actually offered -- not what a scratch file happens to hold now, and
        not a summary of it.
        """
        row = mind.db.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise NotFound("unknown artifact", artifact_id=artifact_id)
        digest = row["sha256"]
        available = mind.blobs.exists(digest)
        out: dict[str, Any] = {
            "artifact_id": artifact_id, "status": row["status"],
            "path": row["path"], "proposed_by": row["proposed_by"],
            "work_id": row["work_id"], "rationale": row["rationale"],
            "sha256": digest, "bytes": row["bytes"],
            "evidence_available": available,
        }
        if available and int(row["bytes"]) <= 256 * 1024:
            data = mind.blobs.get(digest)
            text = data.decode("utf-8", "strict") if _is_text(data) else None
            out["content"] = text
            out["content_is_text"] = text is not None
            if text is None:
                out["note"] = ("not UTF-8 text; the digest identifies the exact "
                               "bytes, which are not renderable here")
        return out

    def _is_text(data: bytes) -> bool:
        try:
            data.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def ego_resource_identities() -> dict[str, Any]:
        """Just enough version identity to interpret a result.

        A result produced under a different prompt or tool surface is not
        strictly comparable to one produced under this one. Ego gets the
        identities; it does not get the security policy internals or the
        filespace configuration, which are not its business.
        """
        from .resources import prompt_version, tool_surface_version

        return {
            "model_generation": sup.pulse.capture(
                max_age_seconds=30.0).get("model_generation"),
            "prompt.ego": {
                "configured": prompt_version("ego", sup.cfg, mind).to_dict(),
                "embodied_sha256": sup.role_prompt_digest.get("ego"),
            },
            "tools.neuocyte": tool_surface_version().to_dict(),
            "note": ("identities only; Ego does not receive security or "
                     "filespace configuration"),
        }

    # ==================================================================
    # Effectors: delegation
    # ==================================================================
    def ego_withdraw_conclusion(*, conclusion_id: str, reason: str,
                                operation_id: str | None = None
                                ) -> dict[str, Any]:
        """Stop making a claim you made.

        The act that lets an adverse audit change something. Without it, Id
        could establish that a conclusion was unsupported and the conclusion
        stayed active and unqualified, with the dispute open beside it
        forever.

        Ego's own conclusions only, checked against the stored `produced_by`
        rather than against anything the caller asserts. Withdrawing another
        component's claim would be editing the record rather than changing
        your mind, and no phrasing makes it the former.

        The row survives. What the organism used to assert is a fact about
        it, and the audits and disagreements that name this conclusion keep
        naming something.
        """
        row = mind.memory.get_conclusion(conclusion_id)
        if row is None:
            raise NotFound("no such conclusion", conclusion_id=conclusion_id)
        if row.get("produced_by") != "ego":
            raise InvalidInput(
                "a conclusion may be withdrawn only by whoever made it",
                conclusion_id=conclusion_id, produced_by=row.get("produced_by"),
                hint="dispute it instead; withdrawing another component's "
                     "claim would be editing the record, not changing a mind")
        receipt = mind.memory.withdraw_conclusion(
            conclusion_id=conclusion_id, actor="ego", reason=reason,
            operation_id=operation_id)
        return {"conclusion_id": conclusion_id, "standing": "retracted",
                "receipt_id": receipt.receipt_id}

    def ego_request_work(*, objective: str, work_class: str = "user",
                         replicas: int = 1, independent: bool = False,
                         board_access: str | None = None,
                         sandbox: bool = False, constraints: str = "",
                         evidence: Sequence[dict[str, Any]] = (),
                         budget_tokens: int | None = None,
                         # A leaf name, not a namespace: "research" resolves to
                         # `ego.neuocyte.research`. Ego states intent; the
                         # library decides whether that profile exists.
                         specialisation: str | None = None,
                         operation_id: str | None = None) -> dict[str, Any]:
        """Delegate a bounded objective to one or more disposable neuocytes; use replicas and independent when independent investigation is useful.

        It used to be described as "introduce an objective into the
        productive-work system", which is true and says nothing about what
        it gets you: another mind working on something. Nothing here says
        when to use it -- only what it is.

        Ego states *intent* at whatever level of abstraction fits -- "answer
        this", "investigate that", "have several independent workers look at
        it", or a worked plan when it genuinely has one. It does not have to
        decompose first, and the work system is free to discover structure
        during execution.

        The Harness decides everything about execution: whether to admit,
        when, which neuocyte, which promoted prompt version, what budget and
        which capabilities. Ego cannot instantiate a worker, and there is no
        verb here that would let it.

        ``independent=True`` admits each replica board-naive, so later
        agreement between them is replication rather than an echo.
        """
        if work_class not in WORK_CLASSES:
            raise InvalidInput("unknown work class", work_class=work_class,
                               allowed=list(WORK_CLASSES))
        if not 1 <= int(replicas) <= 8:
            raise InvalidInput("replicas must be between 1 and 8",
                               replicas=replicas)
        if independent and board_access not in (None, "none"):
            raise InvalidInput(
                "independent work is board-naive by definition",
                board_access=board_access,
                hint="omit board_access, or pass 'none'")
        access = "none" if independent else (board_access or "read_write")
        if access not in BOARD_ACCESS:
            raise InvalidInput("unknown board access", board_access=access)

        # A specialisation is a leaf name under this role's neuocyte
        # namespace, not a namespace: Ego says "research", the Harness resolves
        # `ego.neuocyte.research`. Ego cannot reach sideways into `id.*` or
        # upward to a root by naming one, because it never names a namespace.
        wanted = (specialisation or "").strip()
        if wanted:
            if not wanted.replace("_", "").isalnum():
                raise InvalidInput(
                    "a specialisation is a single name, not a path",
                    specialisation=wanted,
                    hint="ask for 'research', not 'ego.neuocyte.research'")
            if len(wanted) > 64:
                raise InvalidInput("specialisation name is too long",
                                   specialisation=wanted[:80])

        text = _text(objective, "objective", limit=4000)
        if constraints:
            text = f"{text}\n\nConstraints: {_text(constraints, 'constraints')}"
        admitted, refused = [], []
        request_id = new_id("ereq")
        for _ in range(int(replicas)):
            out = sup.methods()["admit_work"](
                objective=text, work_class=work_class, origin_actor="ego",
                operation_id=operation_id, board_access=access,
                sandbox_allowed=bool(sandbox), budget_tokens=budget_tokens,
                specialisation=wanted or None)
            (admitted if out.get("admitted") else refused).append(out)

        def body(m: Mutation) -> None:
            m.emit(EventKind.WORK_REQUESTED_BY_EGO, {
                "request_id": request_id, "objective": text[:1000],
                "work_class": work_class, "replicas": int(replicas),
                "independent": bool(independent), "board_access": access,
                "sandbox_allowed": bool(sandbox),
                "evidence": list(evidence)[:20],
                "admitted": [a.get("work_id") for a in admitted],
                "refused": len(refused), "requested_by": "ego"})

        receipt, _ = mind.writer.apply(body, actor="ego",
                                       operation_id=operation_id,
                                       bump_version=False)
        return {"request_id": request_id, "admitted": admitted,
                "refused": refused, "board_access": access,
                "independent": bool(independent),
                "receipt_id": receipt.receipt_id,
                "note": ("Ego stated intent; the Harness decided admission and "
                         "will decide execution conditions")}

    def ego_work_message(*, work_id: str, message: str,
                         kind: str = "clarification",
                         operation_id: str | None = None) -> dict[str, Any]:
        """Send a governed message to a *work item*.

        Not to a worker process and not into its sandbox: the Harness records
        the message and the neuocyte collects it at a turn boundary. That is
        what keeps mid-flight communication from becoming a channel into live
        execution.

        Refused when the item's independence requires it. A board-naive item
        was admitted precisely so that whatever it concludes is its own, and a
        clarification from the executive role mid-flight would destroy exactly
        the property it was admitted for -- quietly, and in a way that still
        looks like independent replication afterwards.

        The original objective is never rewritten. The message is an addition
        to the record, and the work history shows when the worker collected it.
        """
        if kind not in MESSAGE_KINDS:
            raise InvalidInput("unknown message kind", kind=kind,
                               allowed=list(MESSAGE_KINDS))
        row = mind.work.get_work(work_id)
        body_text = _text(message, "message", limit=4000)

        def refuse(reason: str, **detail: Any) -> None:
            def rec(m: Mutation) -> None:
                m.emit(EventKind.WORK_MESSAGE_REFUSED, {
                    "work_id": work_id, "from_role": "ego", "reason": reason,
                    "message_preview": body_text[:200], **detail})

            mind.writer.apply(rec, actor="ego", operation_id=operation_id,
                              bump_version=False)
            raise InvalidInput(reason, work_id=work_id, **detail)

        if row["status"] not in LIVE_STATUSES:
            refuse("work item is not live; a message could not reach it",
                   status=row["status"])
        if row["board_access"] == "none":
            refuse(
                "this work item is board-naive; mid-flight messages would "
                "destroy the independence it was admitted for",
                board_access=row["board_access"])

        message_id = new_id("wmsg")

        def send(m: Mutation) -> None:
            m.sql("INSERT INTO work_messages(message_id, work_id, from_role,"
                  " body, kind, created_at, state_version) VALUES (?,?,?,?,?,?,?)",
                  (message_id, work_id, "ego", body_text, kind, time.time(),
                   m.prior_version + 1))
            m.emit(EventKind.WORK_MESSAGE_SENT, {
                "message_id": message_id, "work_id": work_id,
                "from_role": "ego", "kind": kind,
                "message": body_text[:1000],
                "note": ("an addition to the record; the original objective is "
                         "unchanged")})

        receipt, _ = mind.writer.apply(send, actor="ego",
                                       operation_id=operation_id)
        return {"message_id": message_id, "work_id": work_id, "kind": kind,
                "receipt_id": receipt.receipt_id, "delivered": "pending",
                "note": ("queued for collection at the worker's next turn "
                         "boundary; the objective is unchanged")}

    def ego_request_cancellation(*, work_id: str, reason: str,
                                 operation_id: str | None = None
                                 ) -> dict[str, Any]:
        """Ask for work Ego originated to be stopped.

        Scoped to Ego's own requests on purpose: cancelling someone else's
        work is an organism-level intervention, and that belongs to Id or the
        operator. The Harness performs the transition and remains free to
        refuse.
        """
        row = mind.work.get_work(work_id)
        if row["origin_actor"] != "ego":
            raise InvalidInput(
                "Ego may only request cancellation of work it originated",
                work_id=work_id, origin_actor=row["origin_actor"],
                hint="ask Id or the operator to intervene in other work")
        out = sup.methods()["cancel_work"](
            work_id=work_id, actor="ego",
            reason=f"[ego] {_text(reason, 'reason', limit=1000)}")
        return {**out, "work_id": work_id,
                "note": "the Harness performed the cancellation"}

    # ==================================================================
    # Effectors: proposing and communicating
    # ==================================================================
    def ego_propose_memory(*, claim: str, kind: str = "belief",
                           confidence: float = 0.5,
                           supersedes: str | None = None,
                           evidence: Sequence[dict[str, Any]] = (),
                           rationale: str = "",
                           operation_id: str | None = None) -> dict[str, Any]:
        """Propose a maintained-state change, with evidence.

        Supersession rather than editing: an earlier item keeps its identity
        and its contrary evidence. Ego is the outward interface, so it is the
        component most exposed to a confident user and the most likely to
        acquire a belief that was never checked -- which is exactly why this
        goes through the same governed path as everyone else's.
        """
        if supersedes:
            mind.memory.get_memory(supersedes)
        memory_id, receipt = mind.memory.remember(
            kind=kind, claim=_text(claim, "claim"),
            confidence=max(0.0, min(1.0, float(confidence))),
            created_by="ego", supersedes=supersedes,
            supporting=[*evidence,
                        {"note": _text(rationale or "proposed by Ego",
                                       "rationale", limit=1000)}],
            operation_id=operation_id)
        return {"memory_id": memory_id, "supersedes": supersedes,
                "receipt_id": receipt.receipt_id, "proposed_by": "ego"}

    def ego_message_id(*, kind: str, message: str,
                       payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Backchannel to Id. Transient, attributed, changes nothing."""
        out = sup.methods()["side_channel"](
            to_role="id", kind=_text(kind, "kind", limit=64),
            payload={"message": _text(message, "message", limit=2000),
                     "from": "ego", **(payload or {})},
            from_role="ego")
        return {"delivered": True, "kind": kind, "transport": out,
                "note": "transient: this changed no durable state"}

    def ego_request_id_review(*, subject: str, subject_id: str = "",
                              question: str = "",
                              operation_id: str | None = None) -> dict[str, Any]:
        """Ask Id to look at something.

        Ego cannot audit itself usefully -- a component checking its own output
        is the weakest possible review -- so this is how the outward half asks
        the inward one for a second opinion. Id decides whether to act.
        """
        if subject not in REVIEW_SUBJECTS:
            raise InvalidInput("unknown review subject", subject=subject,
                               allowed=list(REVIEW_SUBJECTS))
        request_id = new_id("rev")

        def body(m: Mutation) -> None:
            m.emit(EventKind.EGO_REVIEW_REQUESTED, {
                "request_id": request_id, "subject": subject,
                "subject_id": subject_id,
                "question": _text(question or "please review", "question",
                                  limit=2000),
                "requested_by": "ego"})

        receipt, _ = mind.writer.apply(body, actor="ego",
                                       operation_id=operation_id,
                                       bump_version=False)
        try:
            sup.methods()["side_channel"](
                to_role="id", kind="review_requested",
                payload={"request_id": request_id, "subject": subject,
                         "subject_id": subject_id, "from": "ego"},
                from_role="ego")
            notified = True
        except Exception:  # noqa: BLE001 - the durable record is what matters
            notified = False
        return {"request_id": request_id, "subject": subject,
                "receipt_id": receipt.receipt_id, "id_notified": notified,
                "note": "Id decides whether and how to review"}

    return {
        "ego_work_view": ego_work_view,
        "ego_artifact_evidence": ego_artifact_evidence,
        "ego_resource_identities": ego_resource_identities,
        "ego_withdraw_conclusion": ego_withdraw_conclusion,
        "ego_request_work": ego_request_work,
        "ego_work_message": ego_work_message,
        "ego_request_cancellation": ego_request_cancellation,
        "ego_propose_memory": ego_propose_memory,
        "ego_message_id": ego_message_id,
        "ego_request_id_review": ego_request_id_review,
    }
