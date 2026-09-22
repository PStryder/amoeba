"""The organism's pulse: a cheap, bounded answer to "what is happening now?"

Id's first sense. One call, structured facts, no judgements.

## It reports observations, never verdicts

There is no `ego_unhealthy` field and there will not be one. The pulse says
what the last heartbeat was, what the context occupancy is, and how many
inference failures happened in the last five minutes; deciding whether that
adds up to "unhealthy" is Id's job. Putting the conclusion here would move
cognition into the Harness and leave Id agreeing with a number it cannot
inspect.

That distinction is enforced by a test, not just a convention: no field name
in the pulse may read as a verdict.

## It is cheap on purpose

`id_health` already existed and is the opposite of this: it shells out to
`icacls` four times, assesses context (which can reach the inference service),
and verifies the hash chain. Fine for an investigation, far too expensive to be
the thing Id polls to notice something is wrong.

So this is cached with a short TTL, the failure counters are maintained
incrementally from the event `seq` watermark rather than rescanned, and the two
RPC round trips (inference, roles) are refreshed on a slower cadence than the
database reads. Id can force freshness with `max_age_seconds=0`.

## Bulky things are deliberately absent

No blackboard contents, no artifact bodies, no logs, no exception text, no
memory claims. The pulse tells Id *where to look*; it already has interfaces
for looking. Keeping it small is what makes it safe to call often.

## Provenance

A pulse carries a `pulse_id` and the `state_version` it observed. When Id forms
a consequential conclusion it cites that id, so the state it was looking at can
be reconstructed -- without turning continuous telemetry into an event stream
nobody reads. Only pulses that are actually *cited* get recorded.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from .ids import new_id
from .resources import all_versions
from .store.events import EventKind

if TYPE_CHECKING:
    from .supervisor import Supervisor

# Kinds that count as "something went wrong". Deliberately a small, explicit
# list: a rolling count is only useful if a change in it means something.
FAILURE_KINDS: dict[str, str] = {
    EventKind.WORK_FAILED: "work_failed",
    EventKind.WORK_LEASE_EXPIRED: "lease_expired",
    EventKind.WORK_RESULT_FENCED: "result_fenced",
    EventKind.TOOL_REJECTED: "tool_rejected",
    EventKind.INFERENCE_ERROR: "inference_error",
    EventKind.SANDBOX_DENIED: "sandbox_denied",
    EventKind.FILE_DENIED: "file_denied",
    EventKind.AGENT_CRASHED: "agent_crashed",
    EventKind.REJUVENATION_REFUSED: "rejuvenation_refused",
    EventKind.ERROR: "error",
}
FAILURE_WINDOW_SECONDS = 3600.0
WINDOWS = (("last_5m", 300.0), ("last_1h", 3600.0))

DB_TTL = 1.0
"""Database-derived facts are cheap; a second of staleness is plenty."""
REMOTE_TTL = 5.0
"""Inference and role round trips are not free, so they age more slowly."""


def _disagreement_pressure(conn) -> dict[str, Any]:
    """Open contradictions, by how long they have been open.

    A bare count only grows, and a number that only grows stops being read.
    Ageing them out would be worse: an unresolved contradiction nobody has
    addressed is a true fact about the organism, and forgetting it would
    falsify the current state rather than improve it. So the accumulation is
    reported in a shape somebody can act on.
    """
    now = time.time()
    ages = sorted((now - float(r["created_at"]) for r in conn.execute(
        "SELECT created_at FROM disagreements WHERE status = 'open'")),
        reverse=True)
    return {
        "open": len(ages),
        "oldest_seconds": round(ages[0], 1) if ages else None,
        "over_a_day": sum(1 for a in ages if a > 86400),
        "over_a_week": sum(1 for a in ages if a > 7 * 86400),
        "note": ("open contradictions are never aged out; an unresolved "
                 "dispute nobody has addressed is a fact about the organism"),
    }


class PulseCollector:
    """Owned by the Harness. Assembles and caches the pulse."""

    def __init__(self, sup: "Supervisor") -> None:
        self.sup = sup
        self._lock = threading.RLock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = -1e9
        self._failures: deque[tuple[float, str]] = deque()
        self._failure_seq = 0
        self._remote: dict[str, Any] = {}
        self._remote_at = -1e9
        self._cited: dict[str, dict[str, Any]] = {}

    # -- failure counters ------------------------------------------------
    def _drain_failures(self) -> None:
        """Advance the failure window from the event log, incrementally.

        Reads only events newer than the last watermark, and only keeps
        failure kinds. That makes the cost proportional to what has happened
        since the previous pulse rather than to the size of history.
        """
        mind = self.sup.mind
        if mind is None:
            return
        rows = mind.db.conn.execute(
            "SELECT seq, ts, kind FROM events WHERE seq > ? ORDER BY seq ASC LIMIT 5000",
            (self._failure_seq,)).fetchall()
        for row in rows:
            self._failure_seq = max(self._failure_seq, int(row["seq"]))
            label = FAILURE_KINDS.get(row["kind"])
            if label is not None:
                self._failures.append((float(row["ts"]), label))
        cutoff = time.time() - FAILURE_WINDOW_SECONDS
        while self._failures and self._failures[0][0] < cutoff:
            self._failures.popleft()

    def _failure_counts(self) -> dict[str, Any]:
        now = time.time()
        out: dict[str, Any] = {"window_seconds": FAILURE_WINDOW_SECONDS}
        for name, span in WINDOWS:
            counts: dict[str, int] = {}
            for ts, label in self._failures:
                if now - ts <= span:
                    counts[label] = counts.get(label, 0) + 1
            out[name] = counts
            out[f"{name}_total"] = sum(counts.values())
        out["watermark_seq"] = self._failure_seq
        return out

    # -- remote facts ----------------------------------------------------
    def _remote_facts(self, *, force: bool) -> dict[str, Any]:
        if not force and (time.monotonic() - self._remote_at) < REMOTE_TTL and self._remote:
            return self._remote
        facts: dict[str, Any] = {"inference": {}, "roles": {}}
        try:
            h = self.sup.client("inference", probe=True).call("health")
            # The interesting fields are nested under capabilities; reading
            # them off the top level silently yields None for everything and
            # would leave Id blind to the backend while looking informed.
            caps = h.get("capabilities") or {}
            facts["inference"] = {
                "reachable": True,
                "status": h.get("status"),
                "incarnation": h.get("incarnation"),
                "uptime_seconds": h.get("uptime_seconds"),
                "active_sessions": h.get("active_sessions"),
                "max_sessions": h.get("max_sessions") or caps.get("max_sessions"),
                "vram_free_bytes": h.get("vram_free_bytes"),
                "model_generation": caps.get("model_generation"),
                "backend_kind": caps.get("backend_kind"),
                "is_simulated": caps.get("is_simulated"),
                "kv_mode": caps.get("kv_mode"),
                "n_ctx": None, "kv_tokens_used": None, "kv_tokens_total": None,
            }
        except Exception as exc:  # noqa: BLE001
            facts["inference"] = {"reachable": False, "error": type(exc).__name__}
        # Occupancy is measured by the inference service, and only there. These
        # fields used to be read off `health` and a role's `health`, neither of
        # which carries them, so the overview and `context_pressure` -- what Id
        # watches -- were null on every pulse while looking informed.
        by_session: dict[str, dict[str, Any]] = {}
        if facts["inference"].get("reachable"):
            try:
                ctx = self.sup.client("inference", probe=True).call("context_report")
                facts["inference"].update({
                    "n_ctx": ctx.get("pool_capacity"),
                    "kv_tokens_used": ctx.get("pool_tokens_used"),
                    "kv_tokens_total": ctx.get("pool_capacity"),
                })
                by_session = {s.get("session_id"): s for s in ctx.get("sessions") or []}
            except Exception:  # noqa: BLE001
                pass
        for role in ("ego", "id"):
            try:
                r = self.sup.client(role, probe=True).call("health")
                # Measured the way the session's allowance is written, which is
                # how homeostasis judges it.
                held = by_session.get(r.get("session_id")) or {}
                facts["roles"][role] = {
                    "reachable": True,
                    "incarnation": r.get("incarnation"),
                    "session_id": r.get("session_id"),
                    "context_tokens": held.get("budgeted_tokens"),
                    "max_context_tokens": held.get("budget_tokens"),
                    "pending_signals": r.get("pending_signals"),
                    "prompt_sha256": self.sup.role_prompt_digest.get(role),
                }
            except Exception as exc:  # noqa: BLE001
                facts["roles"][role] = {"reachable": False,
                                        "error": type(exc).__name__}
        self._remote = facts
        self._remote_at = time.monotonic()
        return facts

    # -- assembly --------------------------------------------------------
    def capture(self, *, max_age_seconds: float = DB_TTL) -> dict[str, Any]:
        with self._lock:
            # Monotonic and strictly less-than, both deliberately. Windows
            # wall-clock resolution is ~15ms, so two calls in quick succession
            # can read the same value: with `<=` an age of exactly 0.0 counted
            # as "fresh enough", and `max_age_seconds=0` -- Id explicitly
            # asking for a new reading -- silently returned the cached one.
            age = time.monotonic() - self._cached_at
            if self._cached is not None and age < max_age_seconds:
                out = dict(self._cached)
                out["cached"] = True
                out["age_seconds"] = round(age, 4)
                return out
            pulse = self._assemble(force_remote=max_age_seconds < REMOTE_TTL)
            self._cached = pulse
            self._cached_at = time.monotonic()
            out = dict(pulse)
            out["cached"] = False
            out["age_seconds"] = 0.0
            return out

    def _assemble(self, *, force_remote: bool) -> dict[str, Any]:  # noqa: C901
        sup = self.sup
        mind = sup.mind
        assert mind is not None
        conn = mind.db.conn
        now = time.time()
        self._drain_failures()
        remote = self._remote_facts(force=force_remote)

        # -- work ---------------------------------------------------------
        by_status: dict[str, int] = {}
        by_class: dict[str, int] = {}
        for row in conn.execute(
                "SELECT status, work_class, COUNT(*) AS n FROM work_items"
                " GROUP BY status, work_class"):
            by_status[row["status"]] = by_status.get(row["status"], 0) + row["n"]
            by_class[f'{row["work_class"]}:{row["status"]}'] = row["n"]
        oldest = conn.execute(
            "SELECT MIN(created_at) AS o FROM work_items WHERE status = 'queued'"
        ).fetchone()["o"]
        attempts = conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN attempt > 1 THEN 1 ELSE 0 END), 0) AS retried,"
            " COALESCE(MAX(attempt), 0) AS max_attempt FROM work_items"
        ).fetchone()
        blocked = conn.execute(
            "SELECT COUNT(*) AS n FROM work_items WHERE status = 'blocked'"
        ).fetchone()["n"]
        in_flight = [dict(r) for r in conn.execute(
            "SELECT work_id, work_class, lease_owner, attempt, fencing_token,"
            " lease_expires, board_access, sandbox_allowed, created_at"
            " FROM work_items WHERE status = 'leased' ORDER BY created_at ASC LIMIT 64")]

        # -- neuocytes ----------------------------------------------------
        live = []
        for wid, info in list(sup.neuocytes.items()):
            alive = info["proc"].poll() is None
            live.append({
                "neuocyte_id": wid,
                "work_id": info.get("work_id"),
                "work_class": info.get("work_class"),
                "alive": alive,
                "age_seconds": round(now - info.get("started", now), 2),
                "hard_deadline_in": round(info.get("hard_deadline", now) - now, 2),
            })
        # Execution mode comes from the work row, not from the process: it is
        # what the item was admitted with, which is what makes a later
        # agreement between two neuocytes interpretable.
        modes = {r["work_id"]: r for r in conn.execute(
            "SELECT work_id, board_access, sandbox_allowed FROM work_items"
            " WHERE status = 'leased'")}
        for entry in live:
            row = modes.get(entry["work_id"])
            if row is not None:
                entry["board_access"] = row["board_access"]
                entry["board_naive"] = row["board_access"] == "none"
                entry["sandbox_allowed"] = bool(row["sandbox_allowed"])

        # -- governance ----------------------------------------------------
        pending = {
            "artifact_proposals": conn.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE status = 'proposed'"
            ).fetchone()["n"],
            "disagreement_pressure": _disagreement_pressure(conn),
            "open_disagreements": conn.execute(
                "SELECT COUNT(*) AS n FROM disagreements WHERE status = 'open'"
            ).fetchone()["n"],
            "unreviewed_conclusions": conn.execute(
                "SELECT COUNT(*) AS n FROM conclusions WHERE review_status = 'unreviewed'"
            ).fetchone()["n"],
            "queued_maintenance": by_class.get("maintenance:queued", 0),
        }

        # -- storage --------------------------------------------------------
        try:
            usage = shutil.disk_usage(str(sup.cfg.state_dir))
            storage = {"free_bytes": usage.free, "total_bytes": usage.total,
                       "used_fraction": round(1.0 - usage.free / usage.total, 4)}
        except OSError as exc:
            storage = {"error": type(exc).__name__}

        # -- resources -------------------------------------------------------
        configured = all_versions(sup.cfg, getattr(sup, "mind", None))
        resources: dict[str, Any] = {"configured": configured, "embodied": {}}
        for role in ("ego", "id"):
            running = sup.role_prompt_digest.get(role)
            cfg_sha = configured[f"prompt.{role}"]["sha256"]
            resources["embodied"][f"prompt.{role}"] = {
                "sha256": running,
                "matches_configured": (running == cfg_sha) if running else None,
                "note": ("a running role embodies the prompt it started with; "
                         "editing configuration changes the next incarnation"),
            }

        arb = sup.cfg.arbiter
        active_count = sum(1 for e in live if e["alive"])
        return {
            "pulse_id": new_id("pls"),
            "captured_at": now,
            "state_version": mind.state_version(),
            "run_id": mind.run_id,
            "harness": {
                "uptime_seconds": round(now - sup.started_at, 2),
                "supervision_passes": sup._supervision_passes,
                "last_supervision_at": sup._supervision_last,
                "schema_version": configured["store.schema"]["detail"]["schema_version"],
            },
            "roles": remote["roles"],
            "inference": remote["inference"],
            "model_generation": remote["inference"].get("model_generation"),
            "neuocytes": {
                "live": live,
                "alive_count": active_count,
                "by_class": {
                    "user": sum(1 for e in live if e["alive"]
                                and e.get("work_class") == "user"),
                    "maintenance": sum(1 for e in live if e["alive"]
                                       and e.get("work_class") == "maintenance"),
                },
                "board_naive_count": sum(1 for e in live if e["alive"]
                                         and e.get("board_naive")),
            },
            "work": {
                "by_status": by_status,
                "by_class": by_class,
                "blocked": blocked,
                "in_flight": in_flight,
                "oldest_queued_age_seconds": (now - oldest) if oldest else None,
                "retried_items": attempts["retried"],
                "max_attempt": attempts["max_attempt"],
            },
            "scheduler": {
                "max_neuocytes": arb.max_neuocytes,
                "in_use": active_count,
                "available": max(0, arb.max_neuocytes - active_count),
                "max_outstanding_work": arb.max_outstanding_work,
                "outstanding": by_status.get("queued", 0) + by_status.get("leased", 0),
                "user_reserved_slots": arb.user_reserved_slots,
                "maintenance_reserved_slots": arb.maintenance_reserved_slots,
            },
            "context_pressure": {
                role: {
                    "context_tokens": f.get("context_tokens"),
                    "max_context_tokens": f.get("max_context_tokens"),
                    "occupancy": (round(f["context_tokens"] / f["max_context_tokens"], 4)
                                  if f.get("context_tokens") is not None
                                  and f.get("max_context_tokens") else None),
                }
                for role, f in remote["roles"].items()
            },
            "resources": resources,
            "pending_decisions": pending,
            "failures": self._failure_counts(),
            "storage": storage,
            "attention": {
                "pending_signals": {
                    role: f.get("pending_signals")
                    for role, f in remote["roles"].items()},
                "artifact_proposals": pending["artifact_proposals"],
                "open_disagreements": pending["open_disagreements"],
            },
            "contract": {
                "reports": "observations only; no health verdicts are made here",
                "excluded": ["blackboard contents", "artifact bodies", "logs",
                             "exception text", "memory claims"],
                "note": ("cheap enough to poll; use id_health, provenance, "
                         "history or audit_dossier to investigate what this "
                         "points at"),
            },
        }

    # -- provenance -------------------------------------------------------
    def cite(self, pulse_id: str, *, state_version: int, captured_at: float,
             by: str, reason: str) -> dict[str, Any]:
        """Record that a pulse was used as reasoning input.

        Continuous telemetry is not an event stream -- recording every pulse
        would bury the log in observations nobody reads. Recording the ones
        that were actually *cited* keeps the property that matters: a
        consequential conclusion can be traced back to the state its author was
        looking at.
        """
        entry = {"pulse_id": pulse_id, "state_version": state_version,
                 "captured_at": captured_at, "cited_by": by, "reason": reason[:500],
                 "cited_at": time.time()}
        with self._lock:
            self._cited[pulse_id] = entry
            while len(self._cited) > 256:
                self._cited.pop(next(iter(self._cited)))
        return entry
