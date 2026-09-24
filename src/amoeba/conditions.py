"""What the organism wakes Id for, besides the clock.

Id is woken by conclusions, messages, work it owns, board posts on that work,
and the operator. Nothing woke it for a failure, for resource pressure, or
for the other role being wedged -- the heartbeat was the only sense for
unannounced state. That made the heartbeat's interval the detection latency
for everything the inward mind exists to catch, and worse: the heartbeat is
*deferred* while the pool is under pressure, so the organism looked at itself
least often exactly when it was strained.

These are the few conditions worth a turn on their own. Each one measures --
a count, a level, a streak -- and says nothing about what it means, because
what it means is Id's to decide. Each is bounded by a cooldown, so a bad hour
costs a handful of turns rather than a wake storm; and a condition already
waiting in Id's mailbox is not queued twice.

Id is not woken about itself: a mind that cannot complete a turn cannot
complete a turn about being unable to complete turns. That case is the
Harness's own alarm (I126).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRESSURE_ORDER = ("nominal", "elevated", "high", "critical")


@dataclass(slots=True)
class Condition:
    """One thing worth waking a mind for, and the measurement behind it."""

    key: str
    text: str
    measured: dict[str, Any]


def _failure_burst(counts: dict[str, int], threshold: int) -> Condition | None:
    total = sum(v for k, v in counts.items() if isinstance(v, int))
    if threshold <= 0 or total < threshold:
        return None
    detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items(),
                                                     key=lambda kv: -kv[1])
                       if isinstance(v, int) and v)
    return Condition(
        key="failure_burst",
        text=f"{total} failures in the last five minutes: {detail}.",
        measured={"total": total, "by_kind": counts})


def _pressure(level: str | None, age: float, want: str) -> Condition | None:
    if level is None or want not in PRESSURE_ORDER or level not in PRESSURE_ORDER:
        return None
    if PRESSURE_ORDER.index(level) < PRESSURE_ORDER.index(want):
        return None
    return Condition(
        key=f"pressure_{level}",
        text=(f"the shared context pool is at {level} pressure "
              f"(measured {age:.0f}s ago)."),
        measured={"pressure": level, "measured_age_seconds": round(age, 1)})


def _peer(role: str, streaks: dict[str, int], threshold: int) -> Condition | None:
    for other, turns in streaks.items():
        if other == role or threshold <= 0 or turns < threshold:
            continue
        return Condition(
            key=f"not_thinking_{other}",
            text=(f"{other} has failed {turns} turns in a row; the Harness is "
                  "repairing it within its own limits."),
            measured={"role": other, "consecutive_failed_turns": turns})
    return None


def detect(role: str, *, failures: dict[str, int], pressure: tuple[str | None, float],
           streaks: dict[str, int], failure_threshold: int, pressure_level: str,
           failure_turns: int) -> list[Condition]:
    """Everything true right now that this role should be woken for.

    Measurements in, measurements out. A role whose own turns are failing is
    not woken: it could not answer, and the Harness has its own alarm for it.
    """
    if streaks.get(role, 0) >= failure_turns > 0:
        return []
    found = [_failure_burst(failures, failure_threshold),
             _pressure(pressure[0], pressure[1], pressure_level),
             _peer(role, streaks, failure_turns)]
    return [c for c in found if c is not None]


def render(condition: Condition) -> str:
    """The wake as the two lines a role reads."""
    return (f"{condition.text}\n"
            "measured by the Harness; what it means is yours to say. "
            "Check anything you doubt.")
