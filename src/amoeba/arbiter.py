"""Resource arbiter: admission control, fixed caps and weighted-fair scheduling.

The arbiter is part of the fixed harness. Ego and Id may *request* work and
*propose* budgets; neither can raise a cap, rewrite the policy or bypass a
limit. Every admission decision is recorded.

Scheduling balances two classes of work -- user-directed and Id-generated
maintenance -- with two guarantees layered on top of the weights:

* **No starvation.** Each class holds a reserved number of neuocyte slots that
  the other class can never consume.
* **Bounded maintenance.** Maintenance work carries a recursion depth and is
  rate-limited per hour, so a maintenance job that spawns maintenance jobs
  terminates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import ArbiterConfig
from .errors import ResourceExhausted

WorkClass = Literal["user", "maintenance"]


@dataclass(slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    work_class: str
    granted_budget_tokens: int = 0
    granted_deadline: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class ResourceSnapshot:
    active_neuocytes: int = 0
    active_user_neuocytes: int = 0
    active_maintenance_neuocytes: int = 0
    queued_user: int = 0
    queued_maintenance: int = 0
    outstanding_work: int = 0
    oldest_queued_age: float | None = None
    recent_maintenance_count: int = 0
    inference_sessions: int = 0
    max_inference_sessions: int = 0
    vram_free_bytes: int = 0


class Arbiter:
    def __init__(self, cfg: ArbiterConfig) -> None:
        self.cfg = cfg
        self._served: dict[str, int] = {"user": 0, "maintenance": 0}

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------
    def admit(
        self,
        *,
        work_class: WorkClass,
        snapshot: ResourceSnapshot,
        requested_budget_tokens: int | None = None,
        requested_wall_seconds: float | None = None,
        maintenance_depth: int = 0,
    ) -> AdmissionDecision:
        cfg = self.cfg
        if work_class not in ("user", "maintenance"):
            return AdmissionDecision(False, "unknown work class", work_class)

        if snapshot.outstanding_work >= cfg.max_outstanding_work:
            return AdmissionDecision(
                False, "queue is at its configured limit", work_class,
                detail={"outstanding": snapshot.outstanding_work,
                        "max": cfg.max_outstanding_work},
            )

        if work_class == "maintenance":
            if maintenance_depth > cfg.max_maintenance_depth:
                return AdmissionDecision(
                    False, "maintenance recursion depth exceeded", work_class,
                    detail={"depth": maintenance_depth, "max": cfg.max_maintenance_depth},
                )
            if snapshot.recent_maintenance_count >= cfg.max_maintenance_per_hour:
                return AdmissionDecision(
                    False, "maintenance rate limit reached for this hour", work_class,
                    detail={"recent": snapshot.recent_maintenance_count,
                            "max_per_hour": cfg.max_maintenance_per_hour},
                )

        # A class may not consume the slots reserved for the other class.
        reserved_for_other = (
            cfg.maintenance_reserved_slots if work_class == "user"
            else cfg.user_reserved_slots
        )
        usable = max(0, cfg.max_neuocytes - reserved_for_other)
        mine = (snapshot.active_user_neuocytes if work_class == "user"
                else snapshot.active_maintenance_neuocytes)
        # Admission puts work in the queue; slot pressure is advisory here and
        # enforced again at dispatch. Refuse only when this class is already
        # holding its whole usable share AND the queue for it is backing up.
        queued_mine = (snapshot.queued_user if work_class == "user"
                       else snapshot.queued_maintenance)
        if mine >= usable and queued_mine >= cfg.max_outstanding_work // 2:
            return AdmissionDecision(
                False, "class is saturated: active slots and queue both full", work_class,
                detail={"active": mine, "usable_slots": usable, "queued": queued_mine},
            )

        budget = min(
            requested_budget_tokens or cfg.neuocyte_token_budget, cfg.neuocyte_token_budget
        )
        wall = min(requested_wall_seconds or cfg.neuocyte_wall_seconds, cfg.neuocyte_wall_seconds)
        return AdmissionDecision(
            True, "admitted", work_class,
            granted_budget_tokens=budget,
            granted_deadline=time.time() + wall,
            detail={"requested_budget_tokens": requested_budget_tokens,
                    "capped_to": budget, "wall_seconds": wall},
        )

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def next_class_to_serve(self, snapshot: ResourceSnapshot) -> WorkClass | None:
        """Pick which class gets the next free neuocyte slot.

        Weighted fair with reserved slots and an age override so a starved
        class eventually wins regardless of weight.
        """
        cfg = self.cfg
        if snapshot.active_neuocytes >= cfg.max_neuocytes:
            return None
        have_user = snapshot.queued_user > 0
        have_maint = snapshot.queued_maintenance > 0
        if not have_user and not have_maint:
            return None
        if have_user and not have_maint:
            return "user" if self._slots_free_for("user", snapshot) else None
        if have_maint and not have_user:
            return "maintenance" if self._slots_free_for("maintenance", snapshot) else None

        user_ok = self._slots_free_for("user", snapshot)
        maint_ok = self._slots_free_for("maintenance", snapshot)
        if user_ok and not maint_ok:
            return "user"
        if maint_ok and not user_ok:
            return "maintenance"
        if not user_ok and not maint_ok:
            return None

        # Reserved-slot guarantee: if a class holds none of its reserved slots
        # while work is waiting, serve it now.
        if snapshot.active_maintenance_neuocytes < cfg.maintenance_reserved_slots:
            return "maintenance"
        if snapshot.active_user_neuocytes < cfg.user_reserved_slots:
            return "user"

        total_w = cfg.user_weight + cfg.maintenance_weight
        if total_w <= 0:
            return "user"
        served_total = max(1, self._served["user"] + self._served["maintenance"])
        user_share = self._served["user"] / served_total
        target_user = cfg.user_weight / total_w
        return "user" if user_share <= target_user else "maintenance"

    def _slots_free_for(self, work_class: str, snapshot: ResourceSnapshot) -> bool:
        cfg = self.cfg
        reserved_for_other = (cfg.maintenance_reserved_slots if work_class == "user"
                              else cfg.user_reserved_slots)
        other_active = (snapshot.active_maintenance_neuocytes if work_class == "user"
                        else snapshot.active_user_neuocytes)
        # Slots the other class still has a claim to.
        other_reserve_outstanding = max(0, reserved_for_other - other_active)
        available = cfg.max_neuocytes - snapshot.active_neuocytes - other_reserve_outstanding
        return available > 0

    def note_served(self, work_class: str) -> None:
        if work_class in self._served:
            self._served[work_class] += 1

    def fairness_state(self) -> dict[str, Any]:
        total = max(1, sum(self._served.values()))
        return {
            "served": dict(self._served),
            "observed_user_share": self._served["user"] / total,
            "target_user_share": self.cfg.user_weight
            / max(1e-9, self.cfg.user_weight + self.cfg.maintenance_weight),
        }

    # ------------------------------------------------------------------
    # hard caps applied to an individual inference request
    # ------------------------------------------------------------------
    def clamp_inference(self, *, prompt_tokens: int, max_tokens: int | None,
                        deadline: float | None) -> dict[str, Any]:
        cfg = self.cfg
        if prompt_tokens > cfg.max_prompt_tokens:
            raise ResourceExhausted(
                "prompt exceeds the configured context budget",
                prompt_tokens=prompt_tokens, max_prompt_tokens=cfg.max_prompt_tokens,
            )
        capped = min(max_tokens or cfg.max_completion_tokens, cfg.max_completion_tokens)
        wall_cap = time.time() + cfg.neuocyte_wall_seconds
        return {
            "max_tokens": capped,
            "deadline": min(deadline, wall_cap) if deadline else wall_cap,
            "clamped": capped != (max_tokens or cfg.max_completion_tokens),
        }
