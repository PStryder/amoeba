"""Leased work queue, agent registry and Ego snapshot registry.

Execution is at-least-once with idempotent commits. Every lease carries a
fencing token; a result whose token is lower than the work item's current token
is rejected as stale rather than committed twice.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from ..errors import Fenced, InvalidInput, NotFound, ResourceExhausted
from ..ids import new_id
from .blobs import BlobStore
from .db import Database
from .events import EventKind
from .writer import Mutation, Receipt, StateWriter

WORK_CLASSES = ("user", "maintenance")
BOARD_ACCESS = ("none", "read", "read_write")
"""'none' produces a board-naive neuocyte. Two naive neuocytes agreeing is
independent replication; agreement after reading is not."""
TERMINAL_WORK = ("done", "failed", "cancelled")


class WorkRepo:
    def __init__(self, db: Database, blobs: BlobStore, writer: StateWriter) -> None:
        self.db = db
        self.conn = db.conn
        self.blobs = blobs
        self.writer = writer

    # ------------------------------------------------------------------
    # work queue
    # ------------------------------------------------------------------
    def admit(
        self,
        *,
        objective: str,
        work_class: str,
        origin_actor: str,
        operation_id: str | None = None,
        priority: int = 0,
        snapshot_id: str | None = None,
        model_generation: str | None = None,
        budget_tokens: int | None = None,
        deadline: float | None = None,
        maintenance_depth: int = 0,
        depends_on: Sequence[str] | None = None,
        board_access: str = "read_write",
        sandbox_allowed: bool = False,
        mutation_id: str | None = None,
    ) -> tuple[str, Receipt]:
        if work_class not in WORK_CLASSES:
            raise InvalidInput("unknown work class", work_class=work_class,
                               allowed=list(WORK_CLASSES))
        if board_access not in BOARD_ACCESS:
            raise InvalidInput("unknown board access mode", board_access=board_access,
                               allowed=list(BOARD_ACCESS))
        work_id = new_id("work")

        def body(m: Mutation) -> None:
            now = time.time()
            m.sql(
                "INSERT INTO work_items(work_id, objective, work_class, origin_actor,"
                " operation_id, priority, depends_on, snapshot_id, model_generation,"
                " pinned_state_ver, status, attempt, fencing_token, budget_tokens, deadline,"
                " maintenance_depth, board_access, sandbox_allowed, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (work_id, objective, work_class, origin_actor, operation_id, priority,
                 json.dumps(list(depends_on or [])), snapshot_id, model_generation,
                 m.prior_version, "queued", 0, 0, budget_tokens, deadline,
                 maintenance_depth, board_access, 1 if sandbox_allowed else 0,
                 now, now),
            )
            m.emit(EventKind.WORK_ADMITTED, {
                "work_id": work_id, "work_class": work_class, "objective": objective,
                "priority": priority, "snapshot_id": snapshot_id,
                "board_access": board_access, "sandbox_allowed": sandbox_allowed,
                "maintenance_depth": maintenance_depth, "budget_tokens": budget_tokens,
            })

        receipt, _ = self.writer.apply(
            body, actor=origin_actor, operation_id=operation_id,
            mutation_id=mutation_id or f"work-admit:{work_id}",
        )
        return work_id, receipt

    def reject(self, *, objective: str, work_class: str, origin_actor: str, reason: str,
               operation_id: str | None = None) -> Receipt:
        return self.writer.record_rejection(
            actor=origin_actor, reason=reason, kind=EventKind.WORK_REJECTED,
            payload={"objective": objective, "work_class": work_class},
            operation_id=operation_id,
        )

    def lease(
        self,
        *,
        neuocyte_id: str,
        work_class: str | None = None,
        work_id: str | None = None,
        lease_seconds: float = 90.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim a ready work item.

        ``work_id`` targets one specific item. The supervisor dispatches a
        neuocyte *for* a particular item, so without this the neuocyte would claim
        whichever item happened to be at the head of the queue, then have to
        hand it back -- burning an attempt on an item nobody was running, until
        it exhausted its retries and failed.

        Bumps ``attempt`` and ``fencing_token``, which invalidates any result
        still in flight from a previous lease of the same item.
        """
        now = now if now is not None else time.time()
        clauses = ["status = 'queued'"]
        params: list[Any] = []
        if work_id:
            clauses.append("work_id = ?")
            params.append(work_id)
        if work_class:
            clauses.append("work_class = ?")
            params.append(work_class)
        sql = (
            "SELECT * FROM work_items WHERE %s"
            " ORDER BY priority DESC, created_at ASC LIMIT 20" % " AND ".join(clauses)
        )
        claimed: dict[str, Any] | None = None

        def body(m: Mutation) -> dict[str, Any] | None:
            for row in m.sql(sql, params).fetchall():
                deps = json.loads(row["depends_on"] or "[]")
                if deps and not self._deps_satisfied(m, deps):
                    continue
                token = int(row["fencing_token"]) + 1
                m.sql(
                    "UPDATE work_items SET status = 'leased', lease_owner = ?, lease_expires = ?,"
                    " attempt = attempt + 1, fencing_token = ?, updated_at = ?"
                    " WHERE work_id = ? AND status = 'queued'",
                    (neuocyte_id, now + lease_seconds, token, now, row["work_id"]),
                )
                m.emit(EventKind.WORK_LEASED, {
                    "work_id": row["work_id"], "neuocyte_id": neuocyte_id,
                    "fencing_token": token, "attempt": int(row["attempt"]) + 1,
                    "lease_expires": now + lease_seconds,
                })
                item = dict(row)
                item.update({
                    "status": "leased", "lease_owner": neuocyte_id,
                    "lease_expires": now + lease_seconds,
                    "fencing_token": token, "attempt": int(row["attempt"]) + 1,
                    "depends_on": deps,
                })
                return item
            return None

        _receipt, claimed = self.writer.apply(body, actor=neuocyte_id, bump_version=False)
        return claimed

    def _deps_satisfied(self, m: Mutation, deps: Sequence[str]) -> bool:
        for dep in deps:
            row = m.sql("SELECT status FROM work_items WHERE work_id = ?", (dep,)).fetchone()
            if row is None or row["status"] != "done":
                return False
        return True

    def renew_lease(self, *, work_id: str, neuocyte_id: str, fencing_token: int,
                    lease_seconds: float = 90.0) -> bool:
        def body(m: Mutation) -> bool:
            cur = m.sql(
                "UPDATE work_items SET lease_expires = ?, updated_at = ?"
                " WHERE work_id = ? AND lease_owner = ? AND fencing_token = ? AND status = 'leased'",
                (time.time() + lease_seconds, time.time(), work_id, neuocyte_id, fencing_token),
            )
            return cur.rowcount > 0

        _r, ok = self.writer.apply(body, actor=neuocyte_id, bump_version=False)
        return bool(ok)

    def complete(
        self,
        *,
        work_id: str,
        neuocyte_id: str,
        fencing_token: int,
        result: Any,
        operation_id: str | None = None,
        pinned_state_ver: int | None = None,
    ) -> Receipt:
        """Commit a neuocyte result. Idempotent by (work_id, fencing_token).

        Rejects a result whose fencing token has been superseded — that neuocyte
        was replaced and its finding is stale.
        """
        mutation_id = f"work-complete:{work_id}:{fencing_token}"
        existing = self.writer.receipt_for(mutation_id)
        if existing is not None:
            return existing

        def body(m: Mutation) -> None:
            row = m.sql("SELECT * FROM work_items WHERE work_id = ?", (work_id,)).fetchone()
            if row is None:
                raise NotFound("unknown work item", work_id=work_id)
            if int(row["fencing_token"]) != int(fencing_token):
                m.emit(EventKind.WORK_RESULT_FENCED, {
                    "work_id": work_id, "neuocyte_id": neuocyte_id,
                    "presented_token": fencing_token, "current_token": row["fencing_token"],
                })
                raise Fenced(
                    "neuocyte result rejected: fencing token superseded",
                    work_id=work_id, presented=fencing_token, current=row["fencing_token"],
                )
            if row["status"] in TERMINAL_WORK:
                raise Fenced("work item already terminal", work_id=work_id, status=row["status"])
            blob = m.put_json(result, schema="work.result")
            # Findings that were produced against a state version older than the
            # current one are still committed, but flagged as staleness for the
            # consumer to validate.
            stale_against = None
            if pinned_state_ver is not None and pinned_state_ver < m.prior_version:
                stale_against = {"pinned": pinned_state_ver, "current": m.prior_version}
            m.sql(
                "UPDATE work_items SET status = 'done', result_blob = ?, lease_owner = NULL,"
                " lease_expires = NULL, updated_at = ? WHERE work_id = ?",
                (blob, time.time(), work_id),
            )
            m.emit(EventKind.WORK_COMPLETED, {
                "work_id": work_id, "neuocyte_id": neuocyte_id, "result_blob": blob,
                "fencing_token": fencing_token, "stale_against": stale_against,
            })

        receipt, _ = self.writer.apply(
            body, actor=neuocyte_id, operation_id=operation_id, mutation_id=mutation_id,
        )
        return receipt

    def fail(self, *, work_id: str, neuocyte_id: str, fencing_token: int, failure: str,
             requeue: bool = True, operation_id: str | None = None) -> Receipt:
        mutation_id = f"work-fail:{work_id}:{fencing_token}"
        existing = self.writer.receipt_for(mutation_id)
        if existing is not None:
            return existing

        def body(m: Mutation) -> None:
            row = m.sql("SELECT * FROM work_items WHERE work_id = ?", (work_id,)).fetchone()
            if row is None:
                raise NotFound("unknown work item", work_id=work_id)
            if int(row["fencing_token"]) != int(fencing_token):
                raise Fenced("stale failure report", work_id=work_id)
            status = "queued" if requeue and int(row["attempt"]) < 3 else "failed"
            m.sql(
                "UPDATE work_items SET status = ?, failure = ?, lease_owner = NULL,"
                " lease_expires = NULL, updated_at = ? WHERE work_id = ?",
                (status, failure, time.time(), work_id),
            )
            m.emit(EventKind.WORK_FAILED, {
                "work_id": work_id, "neuocyte_id": neuocyte_id, "failure": failure,
                "requeued": status == "queued", "attempt": row["attempt"],
            })

        receipt, _ = self.writer.apply(
            body, actor=neuocyte_id, operation_id=operation_id, mutation_id=mutation_id,
        )
        return receipt

    def cancel(self, *, work_id: str, actor: str, reason: str) -> Receipt:
        def body(m: Mutation) -> None:
            m.sql(
                "UPDATE work_items SET status = 'cancelled', failure = ?, lease_owner = NULL,"
                " lease_expires = NULL, updated_at = ? WHERE work_id = ? AND status NOT IN"
                " ('done','failed','cancelled')",
                (reason, time.time(), work_id),
            )
            m.emit(EventKind.WORK_CANCELLED, {"work_id": work_id, "reason": reason})

        receipt, _ = self.writer.apply(body, actor=actor, mutation_id=f"work-cancel:{work_id}")
        return receipt

    def expire_leases(self, *, actor: str = "supervisor", now: float | None = None) -> list[str]:
        """Return expired leases to the queue. Safe to call repeatedly."""
        now = now if now is not None else time.time()

        def body(m: Mutation) -> list[str]:
            rows = m.sql(
                "SELECT work_id, lease_owner, attempt FROM work_items"
                " WHERE status = 'leased' AND lease_expires IS NOT NULL AND lease_expires < ?",
                (now,),
            ).fetchall()
            expired = []
            for row in rows:
                # Bumping the fencing token here is what makes a late result
                # from the dead neuocyte unable to commit.
                m.sql(
                    "UPDATE work_items SET status = 'queued', lease_owner = NULL,"
                    " lease_expires = NULL, fencing_token = fencing_token + 1, updated_at = ?"
                    " WHERE work_id = ?",
                    (now, row["work_id"]),
                )
                m.emit(EventKind.WORK_LEASE_EXPIRED, {
                    "work_id": row["work_id"], "prior_owner": row["lease_owner"],
                    "attempt": row["attempt"],
                })
                expired.append(row["work_id"])
            return expired

        if not self.conn.execute(
            "SELECT 1 FROM work_items WHERE status = 'leased' AND lease_expires < ? LIMIT 1", (now,)
        ).fetchone():
            return []
        _r, expired = self.writer.apply(body, actor=actor, bump_version=False)
        return list(expired or [])

    def get_work(self, work_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM work_items WHERE work_id = ?", (work_id,)).fetchone()
        if row is None:
            raise NotFound("unknown work item", work_id=work_id)
        item = dict(row)
        item["depends_on"] = json.loads(item["depends_on"] or "[]")
        if item.get("result_blob"):
            try:
                item["result"] = self.blobs.get_json(item["result_blob"])
            except Exception:
                item["result"] = None
                item["result_unresolved"] = True
        return item

    def queue_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {"by_status": {}, "by_class": {}, "oldest_queued_age": None}
        for row in self.conn.execute(
            "SELECT status, work_class, COUNT(*) AS n FROM work_items GROUP BY status, work_class"
        ):
            stats["by_status"][row["status"]] = stats["by_status"].get(row["status"], 0) + row["n"]
            key = f'{row["work_class"]}:{row["status"]}'
            stats["by_class"][key] = row["n"]
        row = self.conn.execute(
            "SELECT MIN(created_at) AS oldest FROM work_items WHERE status = 'queued'"
        ).fetchone()
        if row and row["oldest"]:
            stats["oldest_queued_age"] = time.time() - row["oldest"]
        return stats

    def count_recent_maintenance(self, *, window_seconds: float = 3600.0) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM work_items WHERE work_class = 'maintenance' AND created_at > ?",
            (time.time() - window_seconds,),
        ).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # agents
    # ------------------------------------------------------------------
    def register_agent(
        self, *, agent_id: str, role: str, pid: int | None = None,
        session_handle: str | None = None, snapshot_id: str | None = None,
        model_generation: str | None = None, work_id: str | None = None,
        detail: str | None = None,
    ) -> tuple[int, Receipt]:
        """Register or re-register an agent, incrementing its incarnation."""

        def body(m: Mutation) -> int:
            row = m.sql("SELECT incarnation FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
            incarnation = (int(row["incarnation"]) + 1) if row else 1
            now = time.time()
            m.sql(
                "INSERT INTO agents(agent_id, role, incarnation, status, pid, session_handle,"
                " snapshot_id, model_generation, work_id, started_at, heartbeat_at, detail)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(agent_id) DO UPDATE SET role=excluded.role,"
                " incarnation=excluded.incarnation, status=excluded.status, pid=excluded.pid,"
                " session_handle=excluded.session_handle, snapshot_id=excluded.snapshot_id,"
                " model_generation=excluded.model_generation, work_id=excluded.work_id,"
                " started_at=excluded.started_at, heartbeat_at=excluded.heartbeat_at,"
                " retired_at=NULL, detail=excluded.detail",
                (agent_id, role, incarnation, "alive", pid, session_handle, snapshot_id,
                 model_generation, work_id, now, now, detail),
            )
            m.emit(EventKind.AGENT_STARTED, {
                "agent_id": agent_id, "role": role, "incarnation": incarnation,
                "pid": pid, "snapshot_id": snapshot_id, "model_generation": model_generation,
                "work_id": work_id,
            }, actor=agent_id)
            return incarnation

        receipt, incarnation = self.writer.apply(body, actor=agent_id, bump_version=False)
        return int(incarnation or 1), receipt

    def heartbeat(self, agent_id: str) -> None:
        """Liveness only -- no event, no receipt, no version bump.

        It still takes the transaction lock: an unsynchronised ``commit()``
        here would commit whatever transaction another thread had open.
        """
        with self.writer.tx_lock:
            self.conn.execute(
                "UPDATE agents SET heartbeat_at = ? WHERE agent_id = ?",
                (time.time(), agent_id),
            )
            self.conn.commit()

    def retire_agent(self, *, agent_id: str, reason: str, crashed: bool = False) -> Receipt:
        def body(m: Mutation) -> None:
            m.sql(
                "UPDATE agents SET status = ?, retired_at = ?, detail = ? WHERE agent_id = ?",
                ("crashed" if crashed else "retired", time.time(), reason, agent_id),
            )
            m.emit(
                EventKind.AGENT_CRASHED if crashed else EventKind.AGENT_RETIRED,
                {"agent_id": agent_id, "reason": reason},
                actor=agent_id,
            )

        receipt, _ = self.writer.apply(
            body, actor="supervisor", bump_version=False,
            mutation_id=f"agent-retire:{agent_id}:{time.time_ns()}",
        )
        return receipt

    def live_agents(self, *, role: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM agents WHERE status = 'alive'"
        params: list[Any] = []
        if role:
            sql += " AND role = ?"
            params.append(role)
        return [dict(r) for r in self.conn.execute(sql, params)]

    # ------------------------------------------------------------------
    # Ego snapshots (UKV)
    # ------------------------------------------------------------------
    def publish_snapshot(
        self,
        *,
        actor: str,
        model_generation: str,
        token_count: int,
        tokens: Sequence[int],
        text: str | None,
        kv_mode: str,
        backend_handle: str | None,
        operation_id: str | None = None,
    ) -> tuple[str, int, Receipt]:
        """Record a published immutable prefix of the Ego context.

        The exact token prefix is stored in the blob store so the snapshot can
        be reconstructed by recomputation after a restart or a backend change.
        """
        snapshot_id = new_id("snap")

        def body(m: Mutation) -> int:
            row = m.sql(
                "SELECT MAX(version) AS v FROM snapshots WHERE actor = ?", (actor,)
            ).fetchone()
            version = (int(row["v"]) + 1) if row and row["v"] is not None else 1
            tokens_blob = m.put_json(list(tokens), schema="snapshot.tokens")
            text_blob = m.put_text(text, schema="snapshot.text") if text is not None else None
            m.sql(
                "INSERT INTO snapshots(snapshot_id, version, actor, model_generation,"
                " token_count, tokens_blob, text_blob, kv_mode, backend_handle, refcount,"
                " status, created_at, state_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (snapshot_id, version, actor, model_generation, token_count, tokens_blob,
                 text_blob, kv_mode, backend_handle, 0, "published", time.time(),
                 m.prior_version + 1),
            )
            m.emit(EventKind.SNAPSHOT_PUBLISHED, {
                "snapshot_id": snapshot_id, "version": version, "actor": actor,
                "model_generation": model_generation, "token_count": token_count,
                "kv_mode": kv_mode, "tokens_blob": tokens_blob,
                "backend_handle": backend_handle,
            })
            return version

        receipt, version = self.writer.apply(
            body, actor=actor, operation_id=operation_id,
            mutation_id=f"snap-publish:{snapshot_id}",
        )
        return snapshot_id, int(version or 1), receipt

    def latest_snapshot(self, *, actor: str = "ego",
                        model_generation: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM snapshots WHERE actor = ? AND status = 'published'"
        params: list[Any] = [actor]
        if model_generation:
            sql += " AND model_generation = ?"
            params.append(model_generation)
        sql += " ORDER BY version DESC LIMIT 1"
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise NotFound("unknown snapshot", snapshot_id=snapshot_id)
        return dict(row)

    def snapshot_tokens(self, snapshot_id: str) -> list[int]:
        snap = self.get_snapshot(snapshot_id)
        return list(self.blobs.get_json(snap["tokens_blob"]))

    def acquire_snapshot_ref(self, *, snapshot_id: str, holder: str,
                             operation_id: str | None = None) -> tuple[str, Receipt]:
        """Take a reference so the published storage is not reclaimed while a
        neuocyte is still reading from it."""
        ref_id = new_id("sref")

        def body(m: Mutation) -> None:
            row = m.sql("SELECT status FROM snapshots WHERE snapshot_id = ?",
                        (snapshot_id,)).fetchone()
            if row is None:
                raise NotFound("unknown snapshot", snapshot_id=snapshot_id)
            if row["status"] == "released":
                raise ResourceExhausted("snapshot already released", snapshot_id=snapshot_id)
            m.sql(
                "INSERT INTO snapshot_refs(ref_id, snapshot_id, holder, acquired_at)"
                " VALUES (?,?,?,?)",
                (ref_id, snapshot_id, holder, time.time()),
            )
            m.sql("UPDATE snapshots SET refcount = refcount + 1 WHERE snapshot_id = ?",
                  (snapshot_id,))
            m.emit(EventKind.SNAPSHOT_FORKED,
                   {"snapshot_id": snapshot_id, "holder": holder, "ref_id": ref_id})

        receipt, _ = self.writer.apply(
            body, actor=holder, operation_id=operation_id,
            mutation_id=f"snap-ref:{ref_id}", bump_version=False,
        )
        return ref_id, receipt

    def release_snapshot_ref(self, *, ref_id: str, actor: str) -> Receipt:
        def body(m: Mutation) -> None:
            row = m.sql("SELECT snapshot_id, released_at FROM snapshot_refs WHERE ref_id = ?",
                        (ref_id,)).fetchone()
            if row is None:
                raise NotFound("unknown snapshot ref", ref_id=ref_id)
            if row["released_at"] is not None:
                return
            m.sql("UPDATE snapshot_refs SET released_at = ? WHERE ref_id = ?",
                  (time.time(), ref_id))
            m.sql(
                "UPDATE snapshots SET refcount = MAX(refcount - 1, 0) WHERE snapshot_id = ?",
                (row["snapshot_id"],),
            )
            m.emit(EventKind.SNAPSHOT_REF_RELEASED,
                   {"ref_id": ref_id, "snapshot_id": row["snapshot_id"]})

        receipt, _ = self.writer.apply(
            body, actor=actor, mutation_id=f"snap-unref:{ref_id}", bump_version=False,
        )
        return receipt

    def reclaimable_snapshots(self, *, actor: str = "ego", keep_latest: int = 1) -> list[str]:
        """Published snapshots with no live references, excluding the newest
        ``keep_latest`` which stay available for new neuocytes."""
        rows = self.conn.execute(
            "SELECT snapshot_id, version, refcount FROM snapshots"
            " WHERE actor = ? AND status = 'published' ORDER BY version DESC",
            (actor,),
        ).fetchall()
        out = []
        for row in rows[keep_latest:]:
            if int(row["refcount"]) == 0:
                out.append(row["snapshot_id"])
        return out

    def mark_snapshot_released(self, *, snapshot_id: str, actor: str, reason: str) -> Receipt:
        def body(m: Mutation) -> None:
            row = m.sql("SELECT refcount, status FROM snapshots WHERE snapshot_id = ?",
                        (snapshot_id,)).fetchone()
            if row is None:
                raise NotFound("unknown snapshot", snapshot_id=snapshot_id)
            if int(row["refcount"]) > 0:
                raise ResourceExhausted(
                    "snapshot still referenced by live neuocytes",
                    snapshot_id=snapshot_id, refcount=row["refcount"],
                )
            m.sql(
                "UPDATE snapshots SET status = 'released', released_at = ?, backend_handle = NULL"
                " WHERE snapshot_id = ?",
                (time.time(), snapshot_id),
            )
            m.emit(EventKind.SNAPSHOT_RELEASED, {"snapshot_id": snapshot_id, "reason": reason})

        receipt, _ = self.writer.apply(
            body, actor=actor, mutation_id=f"snap-release:{snapshot_id}", bump_version=False,
        )
        return receipt

    def invalidate_backend_handles(self, *, reason: str, actor: str = "supervisor") -> int:
        """After an inference-service restart every KV handle it hosted is gone.

        The snapshot rows survive: their canonical token prefix still allows
        exact recomputation.
        """
        def body(m: Mutation) -> int:
            cur = m.sql(
                "UPDATE snapshots SET backend_handle = NULL, kv_mode = 'recomputed'"
                " WHERE backend_handle IS NOT NULL"
            )
            n = cur.rowcount
            m.sql("UPDATE agents SET session_handle = NULL WHERE session_handle IS NOT NULL")
            if n:
                m.emit(EventKind.SNAPSHOT_REJECTED,
                       {"reason": reason, "invalidated_handles": n})
            return n

        _r, n = self.writer.apply(body, actor=actor, bump_version=False)
        return int(n or 0)
