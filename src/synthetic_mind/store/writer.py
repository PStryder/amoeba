"""The single durable state writer.

Every consequential change to the mind goes through :meth:`StateWriter.apply`.
One call = one SQLite transaction containing:

  * the state rows that changed,
  * the append-only events describing the change,
  * a receipt binding the mutation id to (prior_version -> result_version).

Referenced blob bytes are fsynced to the content store *before* the transaction
commits, so a crash can never leave an acknowledged mutation whose content is
missing. A crash before commit rolls the whole thing back and leaves only
orphan blobs.

Mutations are idempotent by ``mutation_id``: replaying one returns the original
receipt and does not re-apply the body.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import StaleVersion
from ..ids import GENESIS_HASH, chain_hash, new_id
from .blobs import BlobStore, canonical_json
from .db import Database
from .events import INLINE_PAYLOAD_LIMIT, hash_fields


@dataclass(slots=True)
class Receipt:
    receipt_id: str
    mutation_id: str
    operation_id: str | None
    prior_version: int
    result_version: int
    event_seq_from: int
    event_seq_to: int
    outcome: str
    detail: str | None = None
    ts: float = 0.0
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class _PendingEvent:
    kind: str
    payload_sha256: str | None
    payload_inline: str | None
    actor_id: str
    actor_incarnation: int
    operation_id: str | None
    causation_id: str | None
    correlation_id: str | None
    event_id: str


@dataclass(slots=True)
class Mutation:
    """Collector handed to the body of :meth:`StateWriter.apply`."""

    writer: "StateWriter"
    actor: str
    incarnation: int
    operation_id: str | None
    correlation_id: str | None
    conn: sqlite3.Connection
    blobs: BlobStore
    events: list[_PendingEvent] = field(default_factory=list)
    prior_version: int = 0
    _last_event_id: str | None = None

    # -- events --------------------------------------------------------
    def emit(
        self,
        kind: str,
        payload: Any = None,
        *,
        actor: str | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Queue an append-only event. Large payloads go to the blob store now."""
        sha: str | None = None
        inline: str | None = None
        if payload is not None:
            data = canonical_json(payload)
            if len(data) <= INLINE_PAYLOAD_LIMIT:
                inline = data.decode("utf-8")
            else:
                sha = self.blobs.put(data)
                self.register_blob(sha, len(data), "application/json")
        event_id = new_id("ev")
        self.events.append(
            _PendingEvent(
                kind=kind,
                payload_sha256=sha,
                payload_inline=inline,
                actor_id=actor or self.actor,
                actor_incarnation=self.incarnation,
                operation_id=self.operation_id,
                causation_id=causation_id if causation_id is not None else self._last_event_id,
                correlation_id=correlation_id or self.correlation_id,
                event_id=event_id,
            )
        )
        self._last_event_id = event_id
        return event_id

    # -- blobs ---------------------------------------------------------
    def put_blob(self, data: bytes, *, encoding: str = "application/octet-stream",
                 schema: str | None = None) -> str:
        sha = self.blobs.put(data)
        self.register_blob(sha, len(data), encoding, schema)
        return sha

    def put_text(self, text: str, *, schema: str | None = None) -> str:
        return self.put_blob(text.encode("utf-8"), encoding="text/plain; charset=utf-8", schema=schema)

    def put_json(self, obj: Any, *, schema: str | None = None) -> str:
        return self.put_blob(canonical_json(obj), encoding="application/json", schema=schema)

    def register_blob(self, sha: str, size: int, encoding: str, schema: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO blobs(sha256, size, encoding, schema, created_at) VALUES (?,?,?,?,?)",
            (sha, size, encoding, schema, time.time()),
        )

    # -- state ---------------------------------------------------------
    def sql(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(query, params)

    def expect_version(self, expected: int | None) -> None:
        if expected is not None and expected != self.prior_version:
            raise StaleVersion(
                "state version changed since the caller last read it",
                expected=expected,
                actual=self.prior_version,
            )


class StateWriter:
    def __init__(self, db: Database, blobs: BlobStore, *, run_id: str) -> None:
        self.db = db
        self.conn = db.conn
        self.blobs = blobs
        self.run_id = run_id
        # One transaction at a time on this connection. See Database.tx_lock.
        self.tx_lock = db.tx_lock

    # -- reads used by the writer itself --------------------------------
    def state_version(self) -> int:
        row = self.conn.execute("SELECT version FROM state_version WHERE id = 1").fetchone()
        return int(row["version"]) if row else 0

    def _tip_hash(self) -> str:
        row = self.conn.execute("SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return row["event_hash"] if row else GENESIS_HASH

    def receipt_for(self, mutation_id: str) -> Receipt | None:
        row = self.conn.execute(
            "SELECT * FROM receipts WHERE mutation_id = ?", (mutation_id,)
        ).fetchone()
        if row is None:
            return None
        return Receipt(
            receipt_id=row["receipt_id"],
            mutation_id=row["mutation_id"],
            operation_id=row["operation_id"],
            prior_version=row["prior_version"],
            result_version=row["result_version"],
            event_seq_from=row["event_seq_from"],
            event_seq_to=row["event_seq_to"],
            outcome=row["outcome"],
            detail=row["detail"],
            ts=row["ts"],
            replayed=True,
        )

    # -- the one write path ---------------------------------------------
    def apply(
        self,
        body: Callable[[Mutation], Any],
        *,
        actor: str,
        mutation_id: str | None = None,
        operation_id: str | None = None,
        correlation_id: str | None = None,
        incarnation: int = 0,
        expect_version: int | None = None,
        bump_version: bool = True,
        detail: str | None = None,
    ) -> tuple[Receipt, Any]:
        """Apply ``body`` atomically and return (receipt, body_return_value).

        On replay of a known ``mutation_id`` the body is not executed and the
        original receipt is returned with ``replayed=True`` and a ``None``
        body value.
        """
        mutation_id = mutation_id or new_id("mut")
        # Everything from the idempotency check through commit happens under
        # one lock: the check, the hash-chain tip read and the insert must see
        # a consistent view, and the connection has only one transaction.
        with self.tx_lock:
            return self._apply_locked(
                body, actor=actor, mutation_id=mutation_id,
                operation_id=operation_id, correlation_id=correlation_id,
                incarnation=incarnation, expect_version=expect_version,
                bump_version=bump_version, detail=detail,
            )

    def _apply_locked(
        self,
        body: Callable[[Mutation], Any],
        *,
        actor: str,
        mutation_id: str,
        operation_id: str | None,
        correlation_id: str | None,
        incarnation: int,
        expect_version: int | None,
        bump_version: bool,
        detail: str | None,
    ) -> tuple[Receipt, Any]:
        existing = self.receipt_for(mutation_id)
        if existing is not None:
            return existing, None

        prior = self.state_version()
        mut = Mutation(
            writer=self,
            actor=actor,
            incarnation=incarnation,
            operation_id=operation_id,
            correlation_id=correlation_id,
            conn=self.conn,
            blobs=self.blobs,
            prior_version=prior,
        )
        mut.expect_version(expect_version)

        # Blob writes happen inside body() and are already fsynced when we
        # reach the transaction below.
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            result = body(mut)

            # Re-read under the transaction: another writer may have advanced it.
            row = self.conn.execute("SELECT version FROM state_version WHERE id = 1").fetchone()
            prior = int(row["version"]) if row else 0
            if expect_version is not None and expect_version != prior:
                raise StaleVersion(
                    "state version changed before commit",
                    expected=expect_version,
                    actual=prior,
                )
            new_version = prior + 1 if bump_version else prior

            ts = time.time()
            prev_hash = self._tip_hash()
            seq_from = -1
            seq_to = -1
            for pending in mut.events:
                fields = hash_fields(
                    event_id=pending.event_id,
                    ts=ts,
                    run_id=self.run_id,
                    actor_id=pending.actor_id,
                    actor_incarnation=pending.actor_incarnation,
                    operation_id=pending.operation_id,
                    causation_id=pending.causation_id,
                    correlation_id=pending.correlation_id,
                    kind=pending.kind,
                    payload_sha256=pending.payload_sha256,
                    payload_inline=pending.payload_inline,
                )
                event_hash = chain_hash(prev_hash, fields)
                cur = self.conn.execute(
                    "INSERT INTO events(event_id, ts, run_id, actor_id, actor_incarnation,"
                    " operation_id, causation_id, correlation_id, kind, payload_sha256,"
                    " payload_inline, prev_hash, event_hash)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        pending.event_id, ts, self.run_id, pending.actor_id,
                        pending.actor_incarnation, pending.operation_id, pending.causation_id,
                        pending.correlation_id, pending.kind, pending.payload_sha256,
                        pending.payload_inline, prev_hash, event_hash,
                    ),
                )
                if seq_from < 0:
                    seq_from = int(cur.lastrowid)
                seq_to = int(cur.lastrowid)
                prev_hash = event_hash

            if bump_version:
                self.conn.execute("UPDATE state_version SET version = ? WHERE id = 1", (new_version,))

            receipt_id = new_id("rcpt")
            self.conn.execute(
                "INSERT INTO receipts(receipt_id, mutation_id, operation_id, prior_version,"
                " result_version, event_seq_from, event_seq_to, outcome, detail, ts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (receipt_id, mutation_id, operation_id, prior, new_version,
                 seq_from, seq_to, "committed", detail, ts),
            )
            self.conn.commit()
        except BaseException:
            try:
                self.conn.rollback()
            except sqlite3.Error:
                pass
            raise

        receipt = Receipt(
            receipt_id=receipt_id,
            mutation_id=mutation_id,
            operation_id=operation_id,
            prior_version=prior,
            result_version=new_version,
            event_seq_from=seq_from,
            event_seq_to=seq_to,
            outcome="committed",
            detail=detail,
            ts=ts,
        )
        return receipt, result

    def record_rejection(
        self, *, actor: str, reason: str, kind: str, payload: Any,
        operation_id: str | None = None, mutation_id: str | None = None,
    ) -> Receipt:
        """Append a rejection to history without changing state."""

        def body(m: Mutation) -> None:
            m.emit(kind, {"reason": reason, **(payload or {})})

        receipt, _ = self.apply(
            body, actor=actor, mutation_id=mutation_id, operation_id=operation_id,
            bump_version=False, detail=reason,
        )
        return receipt
