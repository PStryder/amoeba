"""Event kinds and read-side helpers for the append-only history.

Raw history is evidence. It is never rewritten: a correction appends a new
interpretation and a new event, leaving the earlier evidence intact.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable

from ..errors import IntegrityError
from ..ids import GENESIS_HASH, chain_hash
from .blobs import BlobStore, canonical_json


class EventKind:
    # external interface
    INPUT_RECEIVED = "input.received"
    OUTPUT_EMITTED = "output.emitted"
    MCP_CALL = "mcp.call"
    MCP_ERROR = "mcp.error"

    # inference
    INFERENCE_REQUEST = "inference.request"
    INFERENCE_RESULT = "inference.result"
    INFERENCE_ERROR = "inference.error"
    BACKEND_LOADED = "backend.loaded"
    BACKEND_UNLOADED = "backend.unloaded"

    # tools
    TOOL_REQUESTED = "tool.requested"
    TOOL_REJECTED = "tool.rejected"
    TOOL_RESULT = "tool.result"

    # lifecycle
    RUN_STARTED = "run.started"
    AGENT_STARTED = "agent.started"
    AGENT_RETIRED = "agent.retired"
    AGENT_CRASHED = "agent.crashed"
    SUPERVISOR_RECOVERY = "supervisor.recovery"

    # snapshots
    SNAPSHOT_PUBLISHED = "snapshot.published"
    SNAPSHOT_FORKED = "snapshot.forked"
    SNAPSHOT_REF_RELEASED = "snapshot.ref_released"
    SNAPSHOT_RELEASED = "snapshot.released"
    SNAPSHOT_REJECTED = "snapshot.rejected"

    # work
    WORK_ADMITTED = "work.admitted"
    WORK_REJECTED = "work.rejected"
    WORK_LEASED = "work.leased"
    WORK_LEASE_EXPIRED = "work.lease_expired"
    WORK_COMPLETED = "work.completed"
    WORK_FAILED = "work.failed"
    WORK_CANCELLED = "work.cancelled"
    WORK_RESULT_FENCED = "work.result_fenced"

    # state
    MEMORY_CREATED = "memory.created"
    MEMORY_SUPERSEDED = "memory.superseded"
    MEMORY_RETRACTED = "memory.retracted"
    CONCLUSION_RECORDED = "conclusion.recorded"
    AUDIT_RECORDED = "audit.recorded"
    DISAGREEMENT_OPENED = "disagreement.opened"
    DISAGREEMENT_RESOLVED = "disagreement.resolved"

    # cognitive blackboard (communication, not Mind State)
    BOARD_POSTED = "board.posted"
    BOARD_RELATED = "board.related"
    BOARD_STATUS_CHANGED = "board.status_changed"
    BOARD_PROMOTED = "board.promoted_to_memory"

    # sandboxed compute
    SANDBOX_CREATED = "sandbox.created"
    SANDBOX_RUN = "sandbox.run"
    SANDBOX_DESTROYED = "sandbox.destroyed"
    SANDBOX_DENIED = "sandbox.denied"
    ARTIFACT_PROPOSED = "artifact.proposed"
    ARTIFACT_PROMOTED = "artifact.promoted"
    ARTIFACT_REJECTED = "artifact.rejected"
    # RETIRED. Proposals used to lapse when their compute sandbox was
    # destroyed, because the scratch held the only promotable copy. Proposal
    # bytes are now content-addressed when the proposal is made, so sandbox
    # lifetime no longer touches the proposal state machine and nothing emits
    # this any more. The constant stays so historical events still name a
    # known kind; do not reuse it.
    ARTIFACT_LAPSED = "artifact.lapsed"
    # The scratch copy changed after a proposal was made. The reviewed bytes
    # are promoted regardless -- this records the divergence, which is a fact
    # about the neuocyte rather than a reason to refuse.
    ARTIFACT_SCRATCH_DIVERGED = "artifact.scratch_diverged"

    # host filesystem. FILE_SUPERSEDED carries the digest of the content that
    # was there before, which is what makes every write reversible.
    FILE_WRITTEN = "file.written"
    FILE_SUPERSEDED = "file.superseded"
    FILE_DELETED = "file.deleted"
    FILE_RESTORED = "file.restored"
    FILE_ATTACHED = "file.attached"
    FILE_DENIED = "file.denied"

    # context homeostasis
    CONTEXT_MEASURED = "context.measured"
    CONTEXT_PRESSURE = "context.pressure"
    SESSION_RETIRED = "session.retired"
    SESSION_REBORN = "session.reborn"
    REJUVENATION_REQUESTED = "rejuvenation.requested"
    REJUVENATION_PERFORMED = "rejuvenation.performed"
    REJUVENATION_REFUSED = "rejuvenation.refused"

    # cancellation
    OPERATION_CANCELLED = "operation.cancelled"
    GENERATION_CANCELLED = "generation.cancelled"

    # arbiter / side channel
    RESOURCE_DECISION = "resource.decision"
    SIDE_CHANNEL_SIGNAL = "side_channel.signal"
    ERROR = "error"


INLINE_PAYLOAD_LIMIT = 2048
"""Payloads at or below this many canonical-JSON bytes are stored inline in the
event row; larger payloads go to the content-addressed blob store."""


@dataclass(slots=True)
class Event:
    seq: int
    event_id: str
    ts: float
    run_id: str
    actor_id: str
    actor_incarnation: int
    operation_id: str | None
    causation_id: str | None
    correlation_id: str | None
    kind: str
    payload_sha256: str | None
    payload_inline: str | None
    prev_hash: str
    event_hash: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Event":
        return cls(**{k: row[k] for k in cls.__slots__})

    def payload(self, blobs: BlobStore) -> Any:
        if self.payload_inline is not None:
            return json.loads(self.payload_inline)
        if self.payload_sha256 is not None:
            return blobs.get_json(self.payload_sha256)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


def hash_fields(
    *,
    event_id: str,
    ts: float,
    run_id: str,
    actor_id: str,
    actor_incarnation: int,
    operation_id: str | None,
    causation_id: str | None,
    correlation_id: str | None,
    kind: str,
    payload_sha256: str | None,
    payload_inline: str | None,
) -> bytes:
    """Canonical bytes covered by the event hash."""
    return canonical_json(
        {
            "event_id": event_id,
            "ts": ts,
            "run_id": run_id,
            "actor_id": actor_id,
            "actor_incarnation": actor_incarnation,
            "operation_id": operation_id,
            "causation_id": causation_id,
            "correlation_id": correlation_id,
            "kind": kind,
            "payload_sha256": payload_sha256,
            "payload_inline": payload_inline,
        }
    )


def verify_chain(conn: sqlite3.Connection, *, start_seq: int = 0) -> tuple[bool, str | None]:
    """Recompute the hash chain. Returns (ok, first_bad_event_id).

    This detects ordinary mutation and reordering. It is NOT protection against
    an administrator who rewrites rows and recomputes every hash.

    The two checks below are deliberately redundant. ``chain_hash`` already
    folds the predecessor's hash into each event, so the recomputed comparison
    catches excision and reordering on its own; the explicit ``prev_hash``
    comparison catches the same thing one row earlier and names the offending
    event more precisely. Deleting either leaves the other working -- which is
    the point, and is why a mutation of one alone does not make the invariant
    tests fail (see scripts/verify_invariants.py, I5).
    """
    prev = GENESIS_HASH
    if start_seq > 0:
        row = conn.execute(
            "SELECT event_hash FROM events WHERE seq < ? ORDER BY seq DESC LIMIT 1", (start_seq,)
        ).fetchone()
        if row is not None:
            prev = row["event_hash"]
    cur = conn.execute("SELECT * FROM events WHERE seq >= ? ORDER BY seq ASC", (start_seq,))
    for row in cur:
        if row["prev_hash"] != prev:
            return False, row["event_id"]
        payload = hash_fields(
            event_id=row["event_id"],
            ts=row["ts"],
            run_id=row["run_id"],
            actor_id=row["actor_id"],
            actor_incarnation=row["actor_incarnation"],
            operation_id=row["operation_id"],
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
            kind=row["kind"],
            payload_sha256=row["payload_sha256"],
            payload_inline=row["payload_inline"],
        )
        expect = chain_hash(prev, payload)
        if expect != row["event_hash"]:
            return False, row["event_id"]
        prev = row["event_hash"]
    return True, None


def missing_content(conn: sqlite3.Connection, blobs: BlobStore) -> list[dict[str, str]]:
    """Every committed blob reference whose bytes are absent or corrupt.

    A committed reference to missing content is an integrity failure; orphan
    blobs with no reference are merely recoverable garbage.
    """
    missing: list[dict[str, str]] = []
    refs: list[tuple[str, str, str]] = []
    for row in conn.execute(
        "SELECT event_id, payload_sha256 FROM events WHERE payload_sha256 IS NOT NULL"
    ):
        refs.append(("event", row["event_id"], row["payload_sha256"]))
    for table, idcol, col in (
        ("snapshots", "snapshot_id", "tokens_blob"),
        ("snapshots", "snapshot_id", "text_blob"),
        ("operations", "operation_id", "request_blob"),
        ("operations", "operation_id", "result_blob"),
        ("work_items", "work_id", "result_blob"),
    ):
        for row in conn.execute(
            f"SELECT {idcol} AS rid, {col} AS sha FROM {table} WHERE {col} IS NOT NULL"
        ):
            refs.append((table, row["rid"], row["sha"]))
    for kind, rid, sha in refs:
        if not blobs.exists(sha):
            missing.append({"referrer_kind": kind, "referrer_id": rid, "sha256": sha, "reason": "absent"})
        elif not blobs.verify(sha):
            missing.append({"referrer_kind": kind, "referrer_id": rid, "sha256": sha, "reason": "corrupt"})
    return missing


def read_events(
    conn: sqlite3.Connection,
    *,
    operation_id: str | None = None,
    correlation_id: str | None = None,
    kinds: Iterable[str] | None = None,
    since_seq: int = 0,
    limit: int = 200,
) -> list[Event]:
    clauses = ["seq > ?"]
    params: list[Any] = [since_seq]
    if operation_id:
        clauses.append("operation_id = ?")
        params.append(operation_id)
    if correlation_id:
        clauses.append("correlation_id = ?")
        params.append(correlation_id)
    kinds = list(kinds) if kinds else []
    if kinds:
        clauses.append("kind IN (%s)" % ",".join("?" * len(kinds)))
        params.extend(kinds)
    params.append(limit)
    sql = "SELECT * FROM events WHERE %s ORDER BY seq ASC LIMIT ?" % " AND ".join(clauses)
    return [Event.from_row(r) for r in conn.execute(sql, params)]


def resolve_provenance(
    conn: sqlite3.Connection, blobs: BlobStore, *, operation_id: str
) -> dict[str, Any]:
    """Resolve the full input -> work -> inference -> conclusion -> mutation chain
    for one operation, flagging any content that cannot be produced."""
    events = read_events(conn, operation_id=operation_id, limit=1000)
    if not events:
        raise IntegrityError("no events for operation", operation_id=operation_id)
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for ev in events:
        item = ev.to_dict()
        try:
            item["payload"] = ev.payload(blobs)
            resolved.append(item)
        except IntegrityError as exc:
            item["payload"] = None
            item["error"] = exc.to_dict()
            unresolved.append(item)
    receipts = [
        dict(r) for r in conn.execute("SELECT * FROM receipts WHERE operation_id = ?", (operation_id,))
    ]
    conclusions = [
        dict(r) for r in conn.execute("SELECT * FROM conclusions WHERE operation_id = ?", (operation_id,))
    ]
    work = [dict(r) for r in conn.execute("SELECT * FROM work_items WHERE operation_id = ?", (operation_id,))]
    ok, bad = verify_chain(conn, start_seq=max(0, events[0].seq - 1))
    return {
        "operation_id": operation_id,
        "events": resolved,
        "unresolved_content": unresolved,
        "receipts": receipts,
        "conclusions": conclusions,
        "work_items": work,
        "hash_chain_ok": ok,
        "hash_chain_first_bad_event": bad,
    }
