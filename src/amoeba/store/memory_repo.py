"""Maintained memory, conclusions, audits, disagreements and operations.

The distinction this module enforces: *raw history* lives in ``events`` and is
evidence only. *Memory* lives in ``memory_items`` and is an explicitly
maintained interpretation carrying confidence, supporting AND opposing
evidence, a version and a supersession link. Nothing is promoted from history
into memory implicitly.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterable, Sequence

from ..errors import InvalidInput, NotFound
from ..ids import new_id
from .blobs import BlobStore
from .db import Database
from .events import EventKind
from .writer import Mutation, Receipt, StateWriter

MEMORY_KINDS = ("belief", "goal", "constraint", "question", "interpretation", "preference")
MEMORY_STATUSES = ("active", "superseded", "retracted")

_STOPWORDS = frozenset("""
a an and are as at be been but by can could do does did for from had has have how
i if in into is it its may might must no not of on or our shall should so that the
their there these this those to was were what when where which who why will with
would you your about many much
""".split())


def _salient_terms(query: str, *, max_terms: int = 8, min_len: int = 3) -> list[str]:
    """Content words from a query, longest first.

    Substring recall over a whole question almost never matches a stored claim,
    which silently makes memory look empty. Matching on salient terms instead
    keeps recall useful without pretending to be semantic search.
    """
    seen: set[str] = set()
    words: list[str] = []
    for raw in re.split(r"[^0-9A-Za-z_]+", query.lower()):
        if len(raw) < min_len or raw in _STOPWORDS or raw in seen:
            continue
        seen.add(raw)
        words.append(raw)
    words.sort(key=len, reverse=True)
    return words[:max_terms]


class MemoryRepo:
    def __init__(self, db: Database, blobs: BlobStore, writer: StateWriter) -> None:
        self.db = db
        self.conn = db.conn
        self.blobs = blobs
        self.writer = writer

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def open_operation(
        self,
        *,
        kind: str,
        actor: str,
        request: Any,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> tuple[str, Receipt, bool]:
        """Create (or return the existing) operation for an idempotency key."""
        if idempotency_key:
            row = self.conn.execute(
                "SELECT operation_id FROM operations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                op_id = row["operation_id"]
                rec = self.writer.receipt_for(f"op-open:{op_id}")
                assert rec is not None
                return op_id, rec, True

        operation_id = new_id("op")

        def body(m: Mutation) -> None:
            blob = m.put_json(request, schema="operation.request")
            now = time.time()
            m.sql(
                "INSERT INTO operations(operation_id, kind, actor, status, idempotency_key,"
                " request_blob, created_at, updated_at, state_version)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (operation_id, kind, actor, "accepted", idempotency_key, blob,
                 now, now, m.prior_version + 1),
            )
            m.emit(EventKind.INPUT_RECEIVED, {"kind": kind, "request_blob": blob})

        receipt, _ = self.writer.apply(
            body, actor=actor, mutation_id=f"op-open:{operation_id}",
            operation_id=operation_id, correlation_id=correlation_id,
        )
        return operation_id, receipt, False

    def update_operation(
        self,
        *,
        operation_id: str,
        status: str,
        actor: str,
        result: Any = None,
        limitations: Sequence[str] | None = None,
        work_id: str | None = None,
        mutation_id: str | None = None,
    ) -> Receipt:
        def body(m: Mutation) -> None:
            blob = m.put_json(result, schema="operation.result") if result is not None else None
            m.sql(
                "UPDATE operations SET status = ?, result_blob = COALESCE(?, result_blob),"
                " limitations = COALESCE(?, limitations), work_id = COALESCE(?, work_id),"
                " updated_at = ?, state_version = ? WHERE operation_id = ?",
                (status, blob, json.dumps(list(limitations)) if limitations else None,
                 work_id, time.time(), m.prior_version + 1, operation_id),
            )
            m.emit(EventKind.OUTPUT_EMITTED, {"status": status, "result_blob": blob})

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=mutation_id or f"op-update:{operation_id}:{status}",
        )
        return receipt

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise NotFound("unknown operation", operation_id=operation_id)
        out = dict(row)
        if out.get("result_blob"):
            try:
                out["result"] = self.blobs.get_json(out["result_blob"])
            except Exception:
                out["result"] = None
                out["result_unresolved"] = True
        if out.get("limitations"):
            out["limitations"] = json.loads(out["limitations"])
        return out

    # ------------------------------------------------------------------
    # maintained memory
    # ------------------------------------------------------------------
    def remember(
        self,
        *,
        kind: str,
        claim: str,
        confidence: float,
        created_by: str,
        supporting: Iterable[dict[str, Any]] = (),
        opposing: Iterable[dict[str, Any]] = (),
        tags: Sequence[str] | None = None,
        supersedes: str | None = None,
        operation_id: str | None = None,
        mutation_id: str | None = None,
    ) -> tuple[str, Receipt]:
        """Create a maintained interpretation.

        A correction does not edit the old item: it creates a new one that
        ``supersedes`` the old, and the old row is marked superseded while
        keeping all of its recorded contrary evidence.
        """
        if kind not in MEMORY_KINDS:
            raise InvalidInput("unknown memory kind", kind=kind, allowed=list(MEMORY_KINDS))
        if not 0.0 <= float(confidence) <= 1.0:
            raise InvalidInput("confidence must be in [0, 1]", confidence=confidence)
        if not claim.strip():
            raise InvalidInput("claim must be non-empty")

        memory_id = new_id("mem")
        supporting = list(supporting)
        opposing = list(opposing)

        def body(m: Mutation) -> None:
            now = time.time()
            version = 1
            if supersedes:
                prev = m.sql(
                    "SELECT version, status FROM memory_items WHERE memory_id = ?", (supersedes,)
                ).fetchone()
                if prev is None:
                    raise NotFound("memory to supersede does not exist", memory_id=supersedes)
                version = int(prev["version"]) + 1
                m.sql(
                    "UPDATE memory_items SET status = 'superseded', updated_at = ?,"
                    " state_version = ? WHERE memory_id = ?",
                    (now, m.prior_version + 1, supersedes),
                )
                m.emit(EventKind.MEMORY_SUPERSEDED,
                       {"memory_id": supersedes, "superseded_by": memory_id})
            m.sql(
                "INSERT INTO memory_items(memory_id, kind, claim, confidence, status, version,"
                " supersedes, tags, created_by, created_at, updated_at, state_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (memory_id, kind, claim, float(confidence), "active", version, supersedes,
                 json.dumps(list(tags or [])), created_by, now, now, m.prior_version + 1),
            )
            for stance, items in (("supporting", supporting), ("opposing", opposing)):
                for ev in items:
                    m.sql(
                        "INSERT INTO memory_evidence(memory_id, stance, event_seq, event_id,"
                        " blob_sha256, note) VALUES (?,?,?,?,?,?)",
                        (memory_id, stance, ev.get("event_seq"), ev.get("event_id"),
                         ev.get("blob_sha256"), ev.get("note")),
                    )
            m.emit(EventKind.MEMORY_CREATED, {
                "memory_id": memory_id, "kind": kind, "claim": claim,
                "confidence": confidence, "supersedes": supersedes,
                "supporting_count": len(supporting), "opposing_count": len(opposing),
            })

        receipt, _ = self.writer.apply(
            body, actor=created_by, operation_id=operation_id,
            mutation_id=mutation_id or f"mem-create:{memory_id}",
        )
        return memory_id, receipt

    def retract(self, *, memory_id: str, actor: str, reason: str,
                operation_id: str | None = None) -> Receipt:
        def body(m: Mutation) -> None:
            row = m.sql("SELECT status FROM memory_items WHERE memory_id = ?", (memory_id,)).fetchone()
            if row is None:
                raise NotFound("unknown memory", memory_id=memory_id)
            m.sql(
                "UPDATE memory_items SET status = 'retracted', updated_at = ?, state_version = ?"
                " WHERE memory_id = ?",
                (time.time(), m.prior_version + 1, memory_id),
            )
            m.emit(EventKind.MEMORY_RETRACTED, {"memory_id": memory_id, "reason": reason})

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=f"mem-retract:{memory_id}",
        )
        return receipt

    def get_memory(self, memory_id: str, *, with_evidence: bool = True) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise NotFound("unknown memory", memory_id=memory_id)
        item = dict(row)
        item["tags"] = json.loads(item["tags"] or "[]")
        if with_evidence:
            item["evidence"] = self._evidence_for(memory_id)
        return item

    def _evidence_for(self, memory_id: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {"supporting": [], "opposing": []}
        for r in self.conn.execute(
            "SELECT stance, event_seq, event_id, blob_sha256, note FROM memory_evidence"
            " WHERE memory_id = ? ORDER BY id", (memory_id,)
        ):
            out.setdefault(r["stance"], []).append(dict(r))
        return out

    def recall(
        self,
        *,
        query: str | None = None,
        scope: str = "active",
        kinds: Sequence[str] | None = None,
        limit: int = 20,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search maintained memory. Never searches raw history: an audit does
        that explicitly through :mod:`amoeba.store.events`."""
        clauses: list[str] = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if scope == "active":
            clauses.append("status = 'active'")
        elif scope in MEMORY_STATUSES:
            clauses.append("status = ?")
            params.append(scope)
        elif scope != "all":
            raise InvalidInput("unknown scope", scope=scope,
                               allowed=["active", "all", *MEMORY_STATUSES])
        if kinds:
            clauses.append("kind IN (%s)" % ",".join("?" * len(kinds)))
            params.extend(kinds)
        if query:
            # Match on individual salient terms rather than the whole phrase: a
            # question like "how many weight sets are resident?" should still
            # find a belief about weight sets.
            terms = _salient_terms(query)
            if terms:
                ors = []
                for term in terms:
                    ors.append("(claim LIKE ? OR tags LIKE ?)")
                    params.extend([f"%{term}%", f"%{term}%"])
                clauses.append("(" + " OR ".join(ors) + ")")
        params.append(int(limit))
        sql = (
            "SELECT * FROM memory_items WHERE %s ORDER BY confidence DESC, updated_at DESC LIMIT ?"
            % " AND ".join(clauses)
        )
        out = []
        for row in self.conn.execute(sql, params):
            item = dict(row)
            item["tags"] = json.loads(item["tags"] or "[]")
            item["evidence"] = self._evidence_for(item["memory_id"])
            out.append(item)
        return out

    # ------------------------------------------------------------------
    # conclusions
    # ------------------------------------------------------------------
    def record_conclusion(
        self,
        *,
        claim: str,
        produced_by: str,
        evidence: Iterable[dict[str, Any]] = (),
        uncertainty: float | None = None,
        alternatives: Sequence[str] | None = None,
        operation_id: str | None = None,
        model_identity: str | None = None,
        snapshot_id: str | None = None,
        supersedes: str | None = None,
        mutation_id: str | None = None,
    ) -> tuple[str, Receipt]:
        conclusion_id = new_id("concl")
        evidence = list(evidence)

        def body(m: Mutation) -> None:
            m.sql(
                "INSERT INTO conclusions(conclusion_id, claim, uncertainty, alternatives,"
                " operation_id, produced_by, review_status, model_identity, snapshot_id,"
                " created_at, state_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (conclusion_id, claim, uncertainty, json.dumps(list(alternatives or [])),
                 operation_id, produced_by, "unreviewed", model_identity, snapshot_id,
                 time.time(), m.prior_version + 1),
            )
            for ev in evidence:
                m.sql(
                    "INSERT INTO conclusion_evidence(conclusion_id, event_seq, event_id,"
                    " blob_sha256, memory_id, note) VALUES (?,?,?,?,?,?)",
                    (conclusion_id, ev.get("event_seq"), ev.get("event_id"),
                     ev.get("blob_sha256"), ev.get("memory_id"), ev.get("note")),
                )
            if supersedes:
                # A claim is replaced by the claim that replaces it, rather
                # than by a separate act of replacement. The old row keeps
                # its identity: audits and disagreements that named it still
                # name something.
                prior = m.sql("SELECT standing FROM conclusions"
                              " WHERE conclusion_id = ?", (supersedes,)).fetchone()
                if prior is None:
                    raise NotFound("conclusion to supersede does not exist",
                                   conclusion_id=supersedes)
                m.sql("UPDATE conclusions SET standing = 'superseded',"
                      " superseded_by = ?, withdrawn_at = ?, withdrawn_by = ?"
                      " WHERE conclusion_id = ? AND standing = 'active'",
                      (conclusion_id, time.time(), produced_by, supersedes))
                m.emit(EventKind.CONCLUSION_SUPERSEDED, {
                    "conclusion_id": supersedes, "superseded_by": conclusion_id,
                    "actor": produced_by})
                self._close_disputes_in(
                    m, subject_kind="conclusion", subject_id=supersedes,
                    resolution="superseded", actor=produced_by,
                    detail=f"replaced by {conclusion_id}")
            m.emit(EventKind.CONCLUSION_RECORDED, {
                "conclusion_id": conclusion_id, "claim": claim,
                "evidence_count": len(evidence), "model_identity": model_identity,
                "snapshot_id": snapshot_id, "supersedes": supersedes,
            })

        receipt, _ = self.writer.apply(
            body, actor=produced_by, operation_id=operation_id,
            mutation_id=mutation_id or f"concl:{conclusion_id}",
        )
        return conclusion_id, receipt

    def get_conclusion(self, conclusion_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM conclusions WHERE conclusion_id = ?", (conclusion_id,)
        ).fetchone()
        if row is None:
            raise NotFound("unknown conclusion", conclusion_id=conclusion_id)
        item = dict(row)
        item["alternatives"] = json.loads(item["alternatives"] or "[]")
        item["evidence"] = [
            dict(r) for r in self.conn.execute(
                "SELECT event_seq, event_id, blob_sha256, memory_id, note"
                " FROM conclusion_evidence WHERE conclusion_id = ? ORDER BY id",
                (conclusion_id,),
            )
        ]
        return item

    CONCLUSION_STANDINGS = ("active", "retracted", "superseded")

    def withdraw_conclusion(self, *, conclusion_id: str, actor: str,
                            reason: str, operation_id: str | None = None
                            ) -> Receipt:
        """Stop making a claim.

        The author's act. Whether a caller *may* do this is decided by the
        verb that reaches here -- Ego may withdraw an Ego conclusion, and Id
        may not withdraw anybody's, because an auditor that can edit the
        record it audits is not an auditor.

        The row is not deleted. A withdrawn claim is a fact about what the
        organism used to assert, and the audits and disagreements that refer
        to it keep their subject.
        """
        def body(m: Mutation) -> None:
            row = m.sql("SELECT produced_by, standing FROM conclusions"
                        " WHERE conclusion_id = ?", (conclusion_id,)).fetchone()
            if row is None:
                raise NotFound("no such conclusion", conclusion_id=conclusion_id)
            if row["standing"] != "active":
                raise InvalidInput(
                    "that conclusion is no longer being made",
                    conclusion_id=conclusion_id, standing=row["standing"])
            m.sql("UPDATE conclusions SET standing = 'retracted',"
                  " withdrawn_at = ?, withdrawn_by = ?, withdrawn_reason = ?"
                  " WHERE conclusion_id = ?",
                  (time.time(), actor, reason[:500], conclusion_id))
            m.emit(EventKind.CONCLUSION_WITHDRAWN, {
                "conclusion_id": conclusion_id, "actor": actor,
                "reason": reason[:500], "produced_by": row["produced_by"],
                "note": ("the claim is no longer made; the row stays, because "
                         "what the organism used to assert is a fact about it")})
            # A dispute about a claim nobody is making any more has nothing
            # left to be about. This is the subject causing closure by
            # changing the disputed record, which is the one way it may:
            # withdrawing the claim is not dismissing the finding.
            self._close_disputes_in(
                m, subject_kind="conclusion", subject_id=conclusion_id,
                resolution="retracted", actor=actor,
                detail=f"the claim was withdrawn by {actor}")

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=f"concl-withdraw:{conclusion_id}")
        return receipt

    def set_review_status(self, *, conclusion_id: str, status: str, actor: str,
                          operation_id: str | None = None) -> Receipt:
        def body(m: Mutation) -> None:
            m.sql("UPDATE conclusions SET review_status = ? WHERE conclusion_id = ?",
                  (status, conclusion_id))

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=f"concl-review:{conclusion_id}:{status}",
        )
        return receipt

    # ------------------------------------------------------------------
    # audits and disagreements
    # ------------------------------------------------------------------
    def record_audit(
        self,
        *,
        target_kind: str,
        target_id: str,
        verdict: str,
        actor: str = "id",
        focus: str | None = None,
        findings: Sequence[str] | None = None,
        unresolved: Sequence[str] | None = None,
        evidence: Any = None,
        operation_id: str | None = None,
        mutation_id: str | None = None,
    ) -> tuple[str, Receipt]:
        audit_id = new_id("audit")

        def body(m: Mutation) -> None:
            m.sql(
                "INSERT INTO audits(audit_id, target_kind, target_id, focus, verdict,"
                " findings, unresolved, evidence, operation_id, created_at, state_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (audit_id, target_kind, target_id, focus, verdict,
                 json.dumps(list(findings or [])), json.dumps(list(unresolved or [])),
                 json.dumps(evidence) if evidence is not None else None,
                 operation_id, time.time(), m.prior_version + 1),
            )
            if target_kind == "conclusion":
                review = {
                    "supported": "audited_supported",
                    "contested": "audited_contested",
                    "unsupported": "audited_contested",
                }.get(verdict, "audited_inconclusive")
                m.sql("UPDATE conclusions SET review_status = ? WHERE conclusion_id = ?",
                      (review, target_id))
            m.emit(EventKind.AUDIT_RECORDED, {
                "audit_id": audit_id, "target_kind": target_kind,
                "target_id": target_id, "verdict": verdict,
            })

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=mutation_id or f"audit:{audit_id}",
        )
        return audit_id, receipt

    def get_audits(self, *, target_kind: str | None = None, target_id: str | None = None,
                   limit: int = 50) -> list[dict[str, Any]]:
        clauses, params = ["1=1"], []
        if target_kind:
            clauses.append("target_kind = ?")
            params.append(target_kind)
        if target_id:
            clauses.append("target_id = ?")
            params.append(target_id)
        params.append(limit)
        rows = self.conn.execute(
            "SELECT * FROM audits WHERE %s ORDER BY created_at DESC LIMIT ?" % " AND ".join(clauses),
            params,
        )
        out = []
        for row in rows:
            item = dict(row)
            item["findings"] = json.loads(item["findings"] or "[]")
            item["unresolved"] = json.loads(item["unresolved"] or "[]")
            if item.get("evidence"):
                item["evidence"] = json.loads(item["evidence"])
            out.append(item)
        return out

    def open_disagreement(
        self,
        *,
        subject_kind: str,
        subject_id: str,
        claim_a: str,
        actor_a: str,
        claim_b: str,
        actor_b: str,
        evidence_a: Any = None,
        evidence_b: Any = None,
        operation_id: str | None = None,
        evidence_basis_digest: str | None = None,
    ) -> tuple[str, Receipt]:
        disagreement_id = new_id("disag")

        existing: dict[str, Any] = {}

        def body(m: Mutation) -> None:
            # Reaching the same contradiction again is a recurrence, not a
            # second dispute. Without this, auditing one conclusion twenty
            # times turns one unresolved issue into twenty rows and the count
            # stops describing anything. A partial unique index enforces it in
            # the database as well; this is the path that makes the repeat
            # useful rather than an error.
            live = m.sql("SELECT disagreement_id, recurrences FROM disagreements"
                         " WHERE subject_kind = ? AND subject_id = ?"
                         "   AND status = 'open'",
                         (subject_kind, subject_id)).fetchone()
            if live is not None:
                existing["disagreement_id"] = live["disagreement_id"]
                m.sql("UPDATE disagreements SET recurrences = recurrences + 1"
                      " WHERE disagreement_id = ?", (live["disagreement_id"],))
                m.emit(EventKind.DISAGREEMENT_OPENED, {
                    "disagreement_id": live["disagreement_id"],
                    "subject_kind": subject_kind, "subject_id": subject_id,
                    "actor_a": actor_a, "actor_b": actor_b, "recurrence": True,
                    "recurrences": int(live["recurrences"]) + 1,
                    "note": ("the same contradiction was reached again; it is "
                             "recorded against the open dispute rather than "
                             "opening a rival one")})
                return
            m.sql(
                "INSERT INTO disagreements(disagreement_id, subject_kind, subject_id, claim_a,"
                " actor_a, claim_b, actor_b, evidence_a, evidence_b, status,"
                " evidence_basis_digest, created_at,"
                " state_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (disagreement_id, subject_kind, subject_id, claim_a, actor_a, claim_b, actor_b,
                 json.dumps(evidence_a) if evidence_a is not None else None,
                 json.dumps(evidence_b) if evidence_b is not None else None,
                 "open", evidence_basis_digest, time.time(), m.prior_version + 1),
            )
            m.emit(EventKind.DISAGREEMENT_OPENED, {
                "disagreement_id": disagreement_id, "subject_kind": subject_kind,
                "subject_id": subject_id, "actor_a": actor_a, "actor_b": actor_b,
                "evidence_basis_digest": evidence_basis_digest,
            })

        receipt, _ = self.writer.apply(
            body, actor=actor_b, operation_id=operation_id,
            mutation_id=f"disag:{disagreement_id}",
        )
        if existing:
            return existing["disagreement_id"], receipt
        return disagreement_id, receipt

    #: How a dispute may end. Each is a fact the Harness can point at; none of
    #: them is a component deciding it would rather the dispute went away.
    RESOLUTIONS = ("superseded", "retracted", "resolved_supported",
                   "closed_by_operator")

    def _close_disputes_in(self, m: "Mutation", *, subject_kind: str,
                           subject_id: str, resolution: str, actor: str,
                           detail: str = "") -> None:
        """Close the live dispute about a subject, inside the caller's mutation.

        One transaction, deliberately. If the claim changed and the dispute
        survived because a second write failed, the organism would hold an
        open contradiction about a claim nobody is making -- a worse state
        than either outcome on its own.
        """
        row = m.sql("SELECT disagreement_id FROM disagreements"
                    " WHERE subject_kind = ? AND subject_id = ?"
                    "   AND status = 'open'",
                    (subject_kind, subject_id)).fetchone()
        if row is None:
            return
        m.sql("UPDATE disagreements SET status = ?, resolution = ?,"
              " resolved_at = ?, resolved_by = ? WHERE disagreement_id = ?",
              (resolution, detail[:500] or resolution, time.time(), actor,
               row["disagreement_id"]))
        m.emit(EventKind.DISAGREEMENT_RESOLVED, {
            "disagreement_id": row["disagreement_id"], "resolution": resolution,
            "actor": actor, "detail": detail[:500],
            "subject_kind": subject_kind, "subject_id": subject_id,
            "note": ("the disputed record changed, so the dispute ended with "
                     "it; the subject caused this by changing the claim, not "
                     "by dismissing the finding")})

    def resolve_disagreement(self, *, disagreement_id: str, resolution: str,
                             actor: str, detail: str = "",
                             operation_id: str | None = None) -> Receipt:
        """Close a dispute because the record moved.

        `actor` is who the closure is attributed to, not who authorised it:
        authority is decided by the caller, and the callers are the four
        mechanical conditions plus the operator. There is no verb through
        which a component can close a dispute by asserting that it is over.
        """
        if resolution not in self.RESOLUTIONS:
            raise InvalidInput("unknown resolution", resolution=resolution,
                               allowed=list(self.RESOLUTIONS))

        def body(m: Mutation) -> None:
            row = m.sql("SELECT status, subject_kind, subject_id FROM disagreements"
                        " WHERE disagreement_id = ?", (disagreement_id,)).fetchone()
            if row is None:
                raise NotFound("no such disagreement",
                               disagreement_id=disagreement_id)
            if row["status"] != "open":
                raise InvalidInput("that disagreement is already closed",
                                   disagreement_id=disagreement_id,
                                   status=row["status"])
            m.sql("UPDATE disagreements SET status = ?, resolution = ?,"
                  " resolved_at = ?, resolved_by = ? WHERE disagreement_id = ?",
                  (resolution, detail[:500] or resolution, time.time(), actor,
                   disagreement_id))
            m.emit(EventKind.DISAGREEMENT_RESOLVED, {
                "disagreement_id": disagreement_id, "resolution": resolution,
                "actor": actor, "detail": detail[:500],
                "subject_kind": row["subject_kind"],
                "subject_id": row["subject_id"],
                "note": ("closed because the record moved, not because a "
                         "component decided the dispute was over")})

        receipt, _ = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=f"disagree-resolve:{disagreement_id}:{resolution}")
        return receipt

    def open_disagreement_for(self, *, subject_kind: str, subject_id: str
                              ) -> dict[str, Any] | None:
        """The live dispute about this thing, if there is one."""
        row = self.conn.execute(
            "SELECT * FROM disagreements WHERE subject_kind = ? AND subject_id = ?"
            "   AND status = 'open'", (subject_kind, subject_id)).fetchone()
        return dict(row) if row else None

    def close_disagreements_about(self, *, subject_kind: str, subject_id: str,
                                  resolution: str, actor: str, detail: str = ""
                                  ) -> str | None:
        """Close the live dispute about a thing whose record just changed.

        Called by the mechanical conditions -- a conclusion withdrawn or
        superseded -- rather than by anybody deciding. Returns the dispute it
        closed, or None when there was nothing open, which is the ordinary
        case and not an error.
        """
        live = self.open_disagreement_for(subject_kind=subject_kind,
                                          subject_id=subject_id)
        if live is None:
            return None
        self.resolve_disagreement(
            disagreement_id=live["disagreement_id"], resolution=resolution,
            actor=actor, detail=detail)
        return live["disagreement_id"]

    def get_disagreements(self, *, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM disagreements WHERE (? = 'all' OR status = ?)"
            " ORDER BY created_at DESC LIMIT ?",
            (status, status, limit),
        )
        out = []
        for row in rows:
            item = dict(row)
            for key in ("evidence_a", "evidence_b"):
                if item.get(key):
                    item[key] = json.loads(item[key])
            out.append(item)
        return out
