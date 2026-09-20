"""The Operator surface: the human governance boundary.

The Operator is the trusted human running this instance. They see the whole
organism and make the decisions that belong to a person: approving an artifact,
promoting a prompt, cancelling work, consulting Id.

Two things this module is careful about.

**The dashboard is a cockpit, not an authority.** Everything here is a call
into the Harness, which validates and receipts it. Nothing opens the database,
touches the filesystem, or edits state directly. Code does not become
privileged by running on loopback, and a UI that wrote to SQLite would be a
second writer with none of the invariants the first one enforces.

**Operator authority does not leak into cognition.** The Operator can talk to
Ego and consult Id, and those are *inputs* -- auditable, attributed, and
carrying no capability. Ego does not gain the power to promote a prompt because
the Operator asked it a question in the same session. Communication carries
information, not capability.

The verb list here is the operator surface; the HTTP adapter will dispatch
nothing outside it, and it connects with the control credential that the
external adapter does not hold.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .errors import InvalidInput
from .ids import new_id
from .store.events import EventKind
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

# Everything the Operator console may call. Curated rather than "the whole
# method table", so the console's reach is a decision on the record.
OPERATOR_VERBS = (
    # organism view
    "operator_overview", "system_pulse", "id_health", "health", "status",
    "capabilities", "verify_integrity",
    # cognition and history
    "recall", "get_memory", "history", "provenance", "audit_dossier",
    "get_conclusion", "disagreements",
    # work
    "queue_stats", "get_work", "ego_work_view", "cancel_work",
    # blackboard
    "board_read", "board_thread", "board_get_post", "board_stats",
    "board_independence", "board_corroboration", "board_promote_to_memory",
    # artifacts
    "artifact_list", "artifact_promote", "artifact_reject",
    "ego_artifact_evidence",
    # prompts
    "operator_prompt_library", "operator_prompt_decide",
    # filespace and security posture
    "file_roots", "file_list", "file_read", "file_versions",
    "sandbox_capabilities", "sandbox_list",
    # conversation
    "ego_converse", "ego_investigate", "ego_recall", "ego_status",
    "operator_consult_id", "operator_backchannel",
    # homeostasis
    "context_report", "context_assess", "context_rejuvenate",
)


def build(sup: "Supervisor") -> dict[str, Any]:
    mind = sup.mind
    assert mind is not None

    def operator_overview() -> dict[str, Any]:
        """One call behind the dashboard's landing view.

        Assembled from the same authoritative operations everything else uses.
        The console does not get its own query path into the database, because
        a second reader with its own SQL is a second definition of the truth.
        """
        pulse = sup.methods()["system_pulse"](max_age_seconds=2.0)
        return {
            "captured_at": pulse["captured_at"],
            "state_version": pulse["state_version"],
            "harness": pulse["harness"],
            "roles": pulse["roles"],
            "inference": pulse["inference"],
            "neuocytes": pulse["neuocytes"],
            "work": {k: pulse["work"][k] for k in
                     ("by_status", "blocked", "oldest_queued_age_seconds",
                      "retried_items")},
            "scheduler": pulse["scheduler"],
            "context_pressure": pulse["context_pressure"],
            "resources": pulse["resources"],
            "pending_decisions": pulse["pending_decisions"],
            "failures": pulse["failures"],
            "storage": pulse["storage"],
            "board": mind.board.stats(),
            "interactions": _interaction_summary(),
        }

    def _interaction_summary() -> dict[str, Any]:
        rows = mind.db.conn.execute(
            "SELECT status, COUNT(*) AS n FROM interactions GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    # ==================================================================
    # Prompt governance
    # ==================================================================
    def operator_prompt_library(*, role: str | None = None, limit: int = 50
                                ) -> dict[str, Any]:
        """Prompt candidates, their lineage, and what is actually running.

        Reconstructed from the event log rather than a side table, so it cannot
        drift from what happened. Three versions matter and are kept distinct:
        what is configured, what a running role embodied when it started, and
        what has merely been proposed.
        """
        from .resources import prompt_version
        from .store.events import read_events

        proposals = []
        for ev in read_events(mind.db.conn,
                              kinds=[EventKind.PROMPT_PROPOSED,
                                     EventKind.PROMPT_DECIDED], limit=500):
            payload = ev.payload(mind.blobs) or {}
            if role and payload.get("role") != role:
                continue
            proposals.append({"kind": ev.kind, "seq": ev.seq, "ts": ev.ts,
                              "actor": ev.actor_id, **payload})
        decided = {p["proposal_id"] for p in proposals
                   if p["kind"] == EventKind.PROMPT_DECIDED}
        current = {}
        for r in ("ego", "id"):
            if role and r != role:
                continue
            current[r] = {
                "configured": prompt_version(r, sup.cfg).to_dict(),
                "embodied_sha256": sup.role_prompt_digest.get(r),
            }
        return {
            "current": current,
            "proposals": proposals[:limit],
            "pending": [p for p in proposals
                        if p["kind"] == EventKind.PROMPT_PROPOSED
                        and p.get("proposal_id") not in decided][:limit],
            "note": ("a decision is recorded here; installing a prompt is a "
                     "configuration change the operator makes deliberately, "
                     "and a running role keeps the prompt it started with "
                     "until it is reborn"),
        }

    def operator_prompt_decide(*, proposal_id: str, decision: str,
                               rationale: str = "",
                               operation_id: str | None = None) -> dict[str, Any]:
        """Accept or reject a prompt candidate.

        Human authority, recorded. Accepting does not hot-swap anything: it
        records that this candidate is the one to adopt. Changing what a role
        actually runs is a configuration change plus a rebirth, and doing it
        implicitly from a dashboard click would change how the organism thinks
        with no moment at which anyone chose to.
        """
        if decision not in ("accept", "reject"):
            raise InvalidInput("decision must be accept or reject",
                               decision=decision)
        found = None
        from .store.events import read_events

        for ev in read_events(mind.db.conn, kinds=[EventKind.PROMPT_PROPOSED],
                              limit=500):
            payload = ev.payload(mind.blobs) or {}
            if payload.get("proposal_id") == proposal_id:
                found = payload
                break
        if found is None:
            raise InvalidInput("unknown prompt proposal",
                               proposal_id=proposal_id)

        def body(m: Mutation) -> None:
            m.emit(EventKind.PROMPT_DECIDED, {
                "proposal_id": proposal_id, "role": found.get("role"),
                "decision": decision,
                "candidate_sha256": found.get("candidate_sha256"),
                "rationale": rationale[:2000], "decided_by": "operator",
                "note": ("recorded; a running role keeps the prompt it started "
                         "with until it is reborn")})

        receipt, _ = mind.writer.apply(body, actor="operator",
                                       operation_id=operation_id)
        return {"proposal_id": proposal_id, "decision": decision,
                "receipt_id": receipt.receipt_id,
                "candidate_sha256": found.get("candidate_sha256")}

    # ==================================================================
    # Conversation
    # ==================================================================
    def operator_consult_id(*, question: str, scope: str = "all",
                            operation_id: str | None = None) -> dict[str, Any]:
        """Ask Id something directly.

        An auditable *input* into Id's cognition, not maintenance authority.
        The Operator asking a question does not make Id do anything, and Id
        answering does not make the Operator's session privileged inside Id.
        """
        consult_id = new_id("cons")

        def body(m: Mutation) -> None:
            m.emit(EventKind.OPERATOR_CONSULTED_ID, {
                "consult_id": consult_id, "question": question[:2000],
                "scope": scope, "by": "operator",
                "note": "an input into Id's reasoning, carrying no capability"})

        receipt, _ = mind.writer.apply(body, actor="operator",
                                       operation_id=operation_id,
                                       bump_version=False)
        answer = sup.methods()["id_introspect"](question=question, scope=scope,
                                                operation_id=operation_id)
        return {"consult_id": consult_id, "receipt_id": receipt.receipt_id,
                "answer": answer}

    def operator_backchannel(*, limit: int = 50, message: str = "",
                             to_role: str = "", operation_id: str | None = None
                             ) -> dict[str, Any]:
        """Observe, and optionally join, the Ego/Id backchannel.

        Every message carries explicit authorship. The Operator joining the
        room grants nobody anything: Ego does not acquire operator capability
        because the Operator spoke in a thread it can read.
        """
        from .store.events import read_events

        if message:
            if to_role not in ("ego", "id"):
                raise InvalidInput("backchannel messages target ego or id",
                                   to_role=to_role)
            sup.methods()["side_channel"](
                to_role=to_role, kind="operator_message",
                payload={"message": message[:2000], "from": "operator"},
                from_role="operator")

            def body(m: Mutation) -> None:
                m.emit(EventKind.SIDE_CHANNEL_SIGNAL, {
                    "from_role": "operator", "to_role": to_role,
                    "kind": "operator_message", "message": message[:2000],
                    "note": "information, not capability"})

            mind.writer.apply(body, actor="operator", operation_id=operation_id,
                              bump_version=False)

        transcript = []
        for ev in read_events(mind.db.conn,
                              kinds=[EventKind.SIDE_CHANNEL_SIGNAL,
                                     EventKind.EGO_REVIEW_REQUESTED,
                                     EventKind.OPERATOR_CONSULTED_ID],
                              limit=max(1, min(int(limit), 200))):
            payload = ev.payload(mind.blobs) or {}
            transcript.append({"seq": ev.seq, "ts": ev.ts, "kind": ev.kind,
                               "actor": ev.actor_id, **payload})
        return {"transcript": transcript, "count": len(transcript),
                "note": "every entry carries explicit authorship"}

    return {
        "operator_overview": operator_overview,
        "operator_prompt_library": operator_prompt_library,
        "operator_prompt_decide": operator_prompt_decide,
        "operator_consult_id": operator_consult_id,
        "operator_backchannel": operator_backchannel,
    }
