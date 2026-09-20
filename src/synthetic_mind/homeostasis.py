"""Context homeostasis: the Harness keeps the mind's contexts healthy.

A long-running mind fills its KV pool. That is not merely a capacity problem:
with ``kv_unified=True`` the pool is shared, so occupancy taxes *every* decode,
including sessions that have nothing to do with the one hogging space
(measured: 1.94x slowdown at 77% occupancy, fully recovered on retirement --
see ``docs/BENCHMARKS.md`` §2). Left alone, a mind gets slower and then stops.

## Who is allowed to do what

**The model never touches KV.** There is no tool, no verb and no code path by
which Ego, Id or a neuocyte manipulates a cache directly. Id observes pressure
and may *request* rejuvenation; the Harness decides whether to honour it, does
the work, and issues the receipt. A request is a proposal, and the refusal path
is a normal outcome rather than an error.

## What rejuvenation actually does

The honest minimum, which is what is implemented:

1. **Checkpoint.** Publish the session's exact token prefix to the durable
   content store. Nothing is lost from the *record*.
2. **Retire.** Close the backend session. Its cells are reclaimed once no other
   sequence owns them.
3. **Rebirth.** Open a replacement session and reconstitute it from the
   checkpoint.

Reconstitution has three possible modes and they are not interchangeable:

``exact``
    Replay the whole recorded token prefix. Semantically perfect, and useless
    for relieving pressure, because the context ends up the same size.
``trim``
    Replay a **verbatim head and tail** of the recorded tokens with a measured
    span dropped from the middle. Still real tokens -- no paraphrase, no model
    in the loop -- just fewer of them. The dropped span is recorded by count
    and offset, and remains reconstructible from the checkpoint blob.
``summarise``
    Ask a model to compress the context. **Not implemented.** It is a different
    behaviour from trimming, not a better version of it, and calling it
    reconstitution would be a lie about what the mind now contains.

``trim`` is the default, because it is the only one of the three that both
relieves pressure and keeps every surviving token authentic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .errors import CapabilityUnsupported, InvalidInput, NotFound
from .ids import new_id
from .logging_setup import get_logger
from .store.events import EventKind
from .store.writer import Mutation

PRESSURE_LEVELS = ("nominal", "elevated", "high", "critical")
RECONSTITUTION_MODES = ("exact", "trim", "summarise")


@dataclass(slots=True)
class HomeostasisConfig:
    """Thresholds are fractions of the total shared KV pool."""

    elevated: float = 0.55
    high: float = 0.70
    critical: float = 0.85
    # A role is a candidate for rejuvenation once its own context passes this
    # fraction of the per-role budget.
    role_context_high: float = 0.75
    # trim keeps this many tokens verbatim from the head (system prompt and
    # earliest turns) and as many as possible from the tail.
    keep_head_tokens: int = 512
    keep_tail_fraction: float = 0.45
    min_seconds_between_rejuvenations: float = 120.0
    max_rejuvenations_per_hour: int = 12
    auto_rejuvenate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class ContextReport:
    pool_tokens_used: int
    pool_capacity: int
    occupancy: float
    pressure: str
    sessions: list[dict[str, Any]] = field(default_factory=list)
    measured_at: float = 0.0
    backend_available: bool = True
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class ContextHomeostasis:
    """Owned by the supervisor. Deterministic: no model is consulted anywhere."""

    def __init__(self, cfg: HomeostasisConfig, *, mind: Any,
                 inference: Callable[[], Any], log_name: str = "homeostasis") -> None:
        self.cfg = cfg
        self.mind = mind
        self._inference = inference
        self.log = get_logger(log_name)
        self._history: list[float] = []
        self._last: dict[str, float] = {}

    # ------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------
    def measure(self) -> ContextReport:
        """Read occupancy from the inference service. Never estimates."""
        try:
            raw = self._inference().call("context_report")
        except Exception as exc:  # noqa: BLE001
            return ContextReport(
                pool_tokens_used=0, pool_capacity=0, occupancy=0.0,
                pressure="nominal", measured_at=time.time(),
                backend_available=False, detail=f"inference unreachable: {exc!r}")
        used = int(raw.get("pool_tokens_used", 0))
        cap = int(raw.get("pool_capacity", 0)) or 1
        occ = used / cap
        return ContextReport(
            pool_tokens_used=used, pool_capacity=cap, occupancy=occ,
            pressure=self.classify(occ), sessions=raw.get("sessions", []),
            measured_at=time.time(), backend_available=True,
            detail=raw.get("detail", ""))

    def classify(self, occupancy: float) -> str:
        c = self.cfg
        if occupancy >= c.critical:
            return "critical"
        if occupancy >= c.high:
            return "high"
        if occupancy >= c.elevated:
            return "elevated"
        return "nominal"

    def assess(self) -> dict[str, Any]:
        """Measure, then say what the Harness would do about it.

        Assessment never acts. It is safe for Id to call as often as it likes.
        """
        report = self.measure()
        recommendations: list[dict[str, Any]] = []
        if report.backend_available:
            for s in report.sessions:
                role = s.get("role", "")
                budget = max(1, int(s.get("budget_tokens") or report.pool_capacity))
                frac = s.get("n_past", 0) / budget
                if frac >= self.cfg.role_context_high or report.pressure in ("high", "critical"):
                    recommendations.append({
                        "session_id": s.get("session_id"), "role": role,
                        "n_past": s.get("n_past"), "context_fraction": round(frac, 3),
                        "action": "rejuvenate" if role in ("ego", "id") else "retire",
                        "why": ("role context is large" if frac >= self.cfg.role_context_high
                                else f"pool pressure is {report.pressure}"),
                    })
        return {
            "report": report.to_dict(),
            "thresholds": self.cfg.to_dict(),
            "recommendations": recommendations,
            "rejuvenations_last_hour": self._recent_count(),
            "note": ("assessment performs no action; the Harness acts only through "
                     "rejuvenate(), and only it may touch a KV cache"),
        }

    # ------------------------------------------------------------------
    # act (Harness only)
    # ------------------------------------------------------------------
    def _recent_count(self) -> int:
        cutoff = time.time() - 3600
        self._history = [t for t in self._history if t > cutoff]
        return len(self._history)

    def admit_rejuvenation(self, role: str) -> tuple[bool, str]:
        """Rate limits, so a wedged Id cannot thrash the mind's contexts."""
        now = time.time()
        if self._recent_count() >= self.cfg.max_rejuvenations_per_hour:
            return False, (f"rate limit: {self.cfg.max_rejuvenations_per_hour} "
                           "rejuvenations per hour already used")
        last = self._last.get(role, 0.0)
        gap = now - last
        if gap < self.cfg.min_seconds_between_rejuvenations:
            return False, (f"{role} was rejuvenated {gap:.0f}s ago; minimum interval "
                           f"is {self.cfg.min_seconds_between_rejuvenations:.0f}s")
        return True, "admitted"

    def request_rejuvenation(self, *, role: str, reason: str, requested_by: str = "id",
                             mode: str = "trim", operation_id: str | None = None
                             ) -> dict[str, Any]:
        """Id's entry point. A request, not a command.

        Recorded either way: an accepted request and a refused one are both
        facts about how the mind managed itself.
        """
        if role not in ("ego", "id"):
            raise InvalidInput("only ego and id have rejuvenable contexts", role=role)
        ok, detail = self.admit_rejuvenation(role)
        self._emit(EventKind.REJUVENATION_REQUESTED, {
            "role": role, "requested_by": requested_by, "reason": reason,
            "mode": mode, "admitted": ok, "detail": detail,
        }, actor=requested_by, operation_id=operation_id)
        if not ok:
            self._emit(EventKind.REJUVENATION_REFUSED,
                       {"role": role, "reason": detail}, actor="supervisor",
                       operation_id=operation_id)
            return {"performed": False, "role": role, "refused_because": detail,
                    "requested_by": requested_by}
        return self.rejuvenate(role=role, reason=reason, mode=mode,
                               requested_by=requested_by, operation_id=operation_id)

    def rejuvenate(self, *, role: str, reason: str, mode: str = "trim",
                   requested_by: str = "supervisor", operation_id: str | None = None
                   ) -> dict[str, Any]:
        """Checkpoint, retire, reborn. Performed by the Harness, receipted."""
        if mode not in RECONSTITUTION_MODES:
            raise InvalidInput("unknown reconstitution mode", mode=mode,
                               allowed=list(RECONSTITUTION_MODES))
        if mode == "summarise":
            raise CapabilityUnsupported(
                "summarising a context is a different behaviour from reconstituting "
                "it, and is not implemented; use 'trim', which keeps every surviving "
                "token verbatim",
                mode=mode)
        inf = self._inference()
        before = self.measure()

        checkpoint = self.checkpoint(role=role, operation_id=operation_id)
        tokens: list[int] = checkpoint["tokens"]
        plan = self.plan_trim(tokens) if mode == "trim" else {
            "mode": "exact", "keep_head": len(tokens), "keep_tail": 0,
            "dropped_tokens": 0, "kept_tokens": len(tokens),
            "dropped_span": None,
        }

        old_session = checkpoint["session_id"]
        inf.call("close_session", session_id=old_session)
        self._emit(EventKind.SESSION_RETIRED, {
            "role": role, "session_id": old_session, "reason": reason,
            "n_past": len(tokens),
        }, actor="supervisor", operation_id=operation_id)

        new = inf.call("open_session", role=role)
        keep = self.apply_trim(tokens, plan)
        if keep:
            inf.call("restore_prefix", session_id=new["session_id"], tokens=keep,
                     snapshot_id=checkpoint.get("snapshot_id"))

        after = self.measure()
        self._history.append(time.time())
        self._last[role] = time.time()

        result = {
            "performed": True,
            "role": role,
            "requested_by": requested_by,
            "reason": reason,
            "mode": mode,
            "old_session_id": old_session,
            "new_session_id": new["session_id"],
            "checkpoint_snapshot_id": checkpoint.get("snapshot_id"),
            "tokens_before": len(tokens),
            "tokens_after": len(keep),
            "dropped_tokens": plan["dropped_tokens"],
            "dropped_span": plan["dropped_span"],
            "occupancy_before": round(before.occupancy, 4),
            "occupancy_after": round(after.occupancy, 4),
            "pressure_before": before.pressure,
            "pressure_after": after.pressure,
            "reconstitution": (
                "verbatim head and tail of the recorded token prefix; the dropped "
                "span is not summarised and remains reconstructible from the "
                "checkpoint blob"
                if mode == "trim" else
                "the full recorded token prefix, replayed exactly"),
        }
        self._emit(EventKind.REJUVENATION_PERFORMED, result, actor="supervisor",
                   operation_id=operation_id)
        self._emit(EventKind.SESSION_REBORN, {
            "role": role, "session_id": new["session_id"],
            "tokens_restored": len(keep),
        }, actor="supervisor", operation_id=operation_id)
        self.log.info("rejuvenated %s: %d -> %d tokens, occupancy %.1f%% -> %.1f%%",
                      role, len(tokens), len(keep), before.occupancy * 100,
                      after.occupancy * 100)
        return result

    # ------------------------------------------------------------------
    def plan_trim(self, tokens: Sequence[int]) -> dict[str, Any]:
        """Decide what to keep. Deterministic and inspectable before it runs."""
        n = len(tokens)
        head = min(self.cfg.keep_head_tokens, n)
        tail_budget = int(n * self.cfg.keep_tail_fraction)
        tail = max(0, min(tail_budget, n - head))
        dropped = n - head - tail
        if dropped <= 0:
            return {"mode": "trim", "keep_head": n, "keep_tail": 0,
                    "dropped_tokens": 0, "kept_tokens": n, "dropped_span": None,
                    "note": "context already smaller than the trim budget"}
        return {
            "mode": "trim", "keep_head": head, "keep_tail": tail,
            "dropped_tokens": dropped, "kept_tokens": head + tail,
            "dropped_span": {"start": head, "end": head + dropped},
            "note": ("the kept tokens are verbatim; the dropped span is recorded by "
                     "offset and stays in the checkpoint"),
        }

    @staticmethod
    def apply_trim(tokens: Sequence[int], plan: dict[str, Any]) -> list[int]:
        if plan.get("dropped_tokens", 0) <= 0:
            return list(tokens)
        head, tail = plan["keep_head"], plan["keep_tail"]
        return list(tokens[:head]) + (list(tokens[-tail:]) if tail else [])

    def checkpoint(self, *, role: str, operation_id: str | None = None
                   ) -> dict[str, Any]:
        """Persist the role's exact token prefix before anything is discarded."""
        inf = self._inference()
        agents = {a["agent_id"]: a for a in self.mind.work.live_agents()}
        agent = agents.get(role)
        session_id = (agent or {}).get("session_handle")
        if not session_id:
            raise NotFound("role has no live inference session", role=role)
        info = inf.call("session_tokens", session_id=session_id)
        tokens = list(info["tokens"])
        snapshot_id = None
        if role == "ego" and tokens:
            text = inf.call("detokenize", tokens=tokens, special=True)
            caps = inf.call("capabilities")
            snapshot_id, _version, _receipt = self.mind.work.publish_snapshot(
                actor="ego", model_generation=caps.get("model_generation", ""),
                token_count=len(tokens), tokens=tokens, text=text,
                kv_mode=caps.get("kv_mode", "recomputed"), backend_handle=session_id,
                operation_id=operation_id)
        else:
            # Id's private context is never published as a shared snapshot; it
            # is still checkpointed to content-addressed storage so a rebirth
            # can reconstitute it.
            blob = self.mind.blobs.put_json(tokens)
            self._emit(EventKind.CONTEXT_MEASURED,
                       {"role": role, "checkpoint_blob": blob,
                        "token_count": len(tokens)},
                       actor="supervisor", operation_id=operation_id)
        return {"role": role, "session_id": session_id, "tokens": tokens,
                "snapshot_id": snapshot_id, "token_count": len(tokens)}

    def retire_session(self, *, session_id: str, reason: str,
                       operation_id: str | None = None) -> dict[str, Any]:
        """Close one backend session. Safe for worker sessions at any time."""
        self._inference().call("close_session", session_id=session_id)
        self._emit(EventKind.SESSION_RETIRED,
                   {"session_id": session_id, "reason": reason},
                   actor="supervisor", operation_id=operation_id)
        return {"retired": session_id, "reason": reason}

    # ------------------------------------------------------------------
    def tick(self) -> dict[str, Any] | None:
        """Called by the scheduler. Acts only above the critical threshold.

        Deliberately conservative: rejuvenation costs a prefill and loses live
        context, so it happens when the alternative is a mind that is measurably
        degrading, not merely a full-ish pool.
        """
        if not self.cfg.auto_rejuvenate:
            return None
        report = self.measure()
        if not report.backend_available:
            return None
        if report.pressure != "critical":
            return None
        self._emit(EventKind.CONTEXT_PRESSURE, {
            "occupancy": round(report.occupancy, 4), "pressure": report.pressure,
            "pool_tokens_used": report.pool_tokens_used,
            "pool_capacity": report.pool_capacity,
        }, actor="supervisor")
        biggest = None
        for s in report.sessions:
            if s.get("role") in ("ego", "id"):
                if biggest is None or s.get("n_past", 0) > biggest.get("n_past", 0):
                    biggest = s
        if biggest is None:
            return None
        ok, detail = self.admit_rejuvenation(biggest["role"])
        if not ok:
            return {"performed": False, "refused_because": detail}
        return self.rejuvenate(role=biggest["role"],
                               reason=f"pool occupancy {report.occupancy:.0%} is critical",
                               requested_by="supervisor")

    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict[str, Any], *, actor: str,
              operation_id: str | None = None) -> None:
        def body(m: Mutation) -> None:
            m.emit(kind, payload)

        try:
            self.mind.writer.apply(body, actor=actor, operation_id=operation_id,
                                   bump_version=False,
                                   mutation_id=f"homeo:{kind}:{new_id('h')}")
        except Exception:  # noqa: BLE001
            self.log.exception("could not record %s", kind)
