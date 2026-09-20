"""The durable mind: storage, repositories and the recovery entry point.

One process (the supervisor) owns a writable :class:`Mind`; everything else
reads through it over RPC. This keeps a single state writer, which is what
makes "state change + its events + its receipt" a single transaction.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .config import Config
from .errors import IntegrityError
from .ids import new_id
from .logging_setup import get_logger
from .store.blobs import BlobStore
from .store.db import Database
from .store.events import EventKind, missing_content, read_events, resolve_provenance, verify_chain
from .store.board_repo import BoardRepo
from .store.memory_repo import MemoryRepo
from .store.work_repo import WorkRepo
from .store.writer import Mutation, StateWriter


class Mind:
    def __init__(self, cfg: Config, *, run_id: str | None = None,
                 read_only: bool = False) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.log = get_logger("mind")
        self.db = Database(cfg.db_path, read_only=read_only)
        self.blobs = BlobStore(cfg.blob_dir)
        self.run_id = run_id or new_id("run")
        self.writer = StateWriter(self.db, self.blobs, run_id=self.run_id)
        self.memory = MemoryRepo(self.db, self.blobs, self.writer)
        self.work = WorkRepo(self.db, self.blobs, self.writer)
        self.board = BoardRepo(self.db, self.blobs, self.writer)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.db.close()

    def state_version(self) -> int:
        return self.writer.state_version()

    # ------------------------------------------------------------------
    # integrity and provenance
    # ------------------------------------------------------------------
    def verify_integrity(self, *, deep: bool = False) -> dict[str, Any]:
        """Check the hash chain and every committed content reference.

        A hash chain detects ordinary mutation and reordering. It is NOT
        administrator-proof: someone with write access to this database can
        rewrite rows and recompute every hash. Real immutability needs
        checkpoints exported to independent append-only storage.
        """
        ok, bad = verify_chain(self.db.conn)
        missing = missing_content(self.db.conn, self.blobs) if deep else []
        counts = {
            row["k"]: row["n"]
            for row in self.db.conn.execute(
                "SELECT 'events' AS k, COUNT(*) AS n FROM events"
                " UNION ALL SELECT 'receipts', COUNT(*) FROM receipts"
                " UNION ALL SELECT 'memory_items', COUNT(*) FROM memory_items"
                " UNION ALL SELECT 'conclusions', COUNT(*) FROM conclusions"
                " UNION ALL SELECT 'work_items', COUNT(*) FROM work_items"
                " UNION ALL SELECT 'snapshots', COUNT(*) FROM snapshots"
                " UNION ALL SELECT 'board_posts', COUNT(*) FROM board_posts"
                " UNION ALL SELECT 'artifacts', COUNT(*) FROM artifacts"
            )
        }
        return {
            "hash_chain_ok": ok,
            "first_bad_event": bad,
            "missing_content": missing,
            "missing_content_count": len(missing),
            "counts": counts,
            "state_version": self.state_version(),
            "caveat": ("hash chaining detects ordinary mutation; it is not protection "
                       "against an administrator who rewrites rows and hashes"),
        }

    def provenance(self, *, operation_id: str) -> dict[str, Any]:
        return resolve_provenance(self.db.conn, self.blobs, operation_id=operation_id)

    def history(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [e.to_dict() for e in read_events(self.db.conn, **kwargs)]

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------
    def recover(self, *, actor: str = "supervisor") -> dict[str, Any]:
        """Bring durable state to a consistent, resumable point after a restart.

        Nothing here depends on a disposable worker being alive. Anything a
        worker held is released; anything it half-finished is requeued with a
        fresh fencing token so its late result cannot commit.
        """
        report: dict[str, Any] = {"run_id": self.run_id, "started_at": time.time()}
        integrity = self.verify_integrity(deep=True)
        report["integrity"] = integrity
        if not integrity["hash_chain_ok"]:
            self.log.error("event hash chain failed to verify at %s",
                           integrity["first_bad_event"])
        if integrity["missing_content_count"]:
            self.log.error("%d committed content references are unresolvable",
                           integrity["missing_content_count"])

        # Workers never survive a supervisor restart.
        stale_workers = [
            r["agent_id"] for r in self.db.conn.execute(
                "SELECT agent_id FROM agents WHERE role = 'worker' AND status = 'alive'"
            )
        ]
        # Long-lived roles are marked crashed; they re-register with a new
        # incarnation when they come back.
        stale_roles = [
            r["agent_id"] for r in self.db.conn.execute(
                "SELECT agent_id FROM agents WHERE role IN ('ego','id','inference')"
                " AND status = 'alive'"
            )
        ]

        def body(m: Mutation) -> dict[str, Any]:
            now = time.time()
            for agent_id in stale_workers + stale_roles:
                m.sql("UPDATE agents SET status = 'crashed', retired_at = ?,"
                      " detail = 'not alive after restart', session_handle = NULL"
                      " WHERE agent_id = ?", (now, agent_id))
            # Every backend KV handle died with the previous inference process;
            # the snapshot rows stay valid because they record exact tokens.
            cur = m.sql(
                "UPDATE snapshots SET backend_handle = NULL, kv_mode = 'recomputed'"
                " WHERE backend_handle IS NOT NULL"
            )
            invalidated = cur.rowcount
            # Release references held by agents that are now gone.
            refs = m.sql(
                "SELECT ref_id, snapshot_id FROM snapshot_refs WHERE released_at IS NULL"
            ).fetchall()
            for ref in refs:
                m.sql("UPDATE snapshot_refs SET released_at = ? WHERE ref_id = ?",
                      (now, ref["ref_id"]))
            m.sql("UPDATE snapshots SET refcount = 0 WHERE refcount > 0")
            # Requeue leased work with a bumped fencing token.
            leased = m.sql(
                "SELECT work_id, lease_owner FROM work_items WHERE status = 'leased'"
            ).fetchall()
            for row in leased:
                m.sql(
                    "UPDATE work_items SET status = 'queued', lease_owner = NULL,"
                    " lease_expires = NULL, fencing_token = fencing_token + 1,"
                    " updated_at = ? WHERE work_id = ?",
                    (now, row["work_id"]),
                )
            # Operations that were mid-flight are recorded as interrupted rather
            # than silently reported as complete.
            interrupted = m.sql(
                "SELECT operation_id FROM operations WHERE status IN ('accepted','running')"
            ).fetchall()
            for row in interrupted:
                m.sql(
                    "UPDATE operations SET status = 'interrupted', updated_at = ?"
                    " WHERE operation_id = ?", (now, row["operation_id"])
                )
            summary = {
                "crashed_workers": len(stale_workers),
                "crashed_roles": len(stale_roles),
                "invalidated_backend_handles": invalidated,
                "released_snapshot_refs": len(refs),
                "requeued_work": [r["work_id"] for r in leased],
                "interrupted_operations": [r["operation_id"] for r in interrupted],
                "hash_chain_ok": integrity["hash_chain_ok"],
                "missing_content_count": integrity["missing_content_count"],
            }
            m.emit(EventKind.SUPERVISOR_RECOVERY, summary)
            m.emit(EventKind.RUN_STARTED, {"run_id": self.run_id, "pid": os.getpid()})
            return summary

        receipt, summary = self.writer.apply(body, actor=actor, bump_version=False)
        report["summary"] = summary
        report["receipt_id"] = receipt.receipt_id
        report["state_version"] = self.state_version()
        self.log.info("recovery complete: %s", summary)
        return report

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        qs = self.work.queue_stats()
        agents = self.work.live_agents()
        snap = self.work.latest_snapshot(actor="ego")
        return {
            "run_id": self.run_id,
            "state_version": self.state_version(),
            "queue": qs,
            "live_agents": agents,
            "latest_ego_snapshot": snap,
            "memory_counts": {
                row["status"]: row["n"] for row in self.db.conn.execute(
                    "SELECT status, COUNT(*) AS n FROM memory_items GROUP BY status"
                )
            },
            "open_disagreements": self.db.conn.execute(
                "SELECT COUNT(*) AS n FROM disagreements WHERE status = 'open'"
            ).fetchone()["n"],
            "event_count": self.db.conn.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"],
            "board": self.board.stats(),
        }
