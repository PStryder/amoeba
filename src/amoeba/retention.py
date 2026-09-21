"""What may be forgotten, and what may never be.

Amoeba's premise is a durable, hash-chained record. That makes retention a
narrower question than it usually is: most of this database is evidence, and
evidence is not a cache. The policy is therefore written as two explicit
lists -- what is never pruned, and what is -- rather than as a rule with
exceptions, because a rule invites the next person to reason their way to an
exception.

**Never pruned.** `events` and `receipts` are the chain; deleting a link does
not free space so much as destroy the ability to verify the rest. Conclusions,
audits and disagreements are what the organism decided and what it disputed.
Memory is what it believes. The prompt library is what it was told to be, and
a turn recorded against a profile version whose row was deleted is a turn
nobody can reconstruct.

**Pruned.** Consumed and expired triggers, and the closed turns that consumed
them. These are an operational index: which cognitive inputs are outstanding,
which turn is open, what the last few turns did. The events remain, and they
carry the bundle digests, so a pruned turn is still reconstructable from the
record -- it simply stops appearing in the working views.

**Not collected.** Blob bytes. Roughly twenty columns reference a digest and
event payloads embed more inside JSON, so a reference scan that missed one
source would delete content a hash-chained event points at, and the damage
would be silent until somebody tried to resolve provenance. `footprint`
measures what such a collector would be reasoning about; building one on
measurements is a better bet than building one on confidence.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .store.events import EventKind

if TYPE_CHECKING:  # pragma: no cover
    from .mind import Mind
    from .store.writer import Mutation


# The record. Listed rather than inferred, so that adding a table to the
# prunable set below is a decision somebody made on purpose.
EVIDENCE_TABLES = (
    "events", "receipts", "blobs",
    "conclusions", "conclusion_evidence",
    "audits", "disagreements",
    "memory_items", "memory_evidence",
    "prompt_versions", "prompt_selections", "prompt_evaluations",
    "prompt_decisions", "incarnation_profiles",
    "operations", "artifacts", "interactions", "interaction_results",
)

# The working set. Everything here is derivable from the event log; these rows
# exist so the running organism does not have to replay it.
PRUNABLE_TABLES = ("role_triggers", "role_turns")


def footprint(conn) -> dict[str, Any]:
    """How much of what there is, and how fast it is arriving.

    Reported, not acted on. The point of measuring before collecting is that a
    collector built on an assumption about what dominates will free the wrong
    thing and take a risk for nothing.
    """
    tables: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        name = row[0]
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        tables[name] = {
            "rows": count,
            "evidence": name in EVIDENCE_TABLES,
            "prunable": name in PRUNABLE_TABLES,
        }

    blob_rows, blob_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM blobs").fetchone()
    return {
        "tables": tables,
        "blobs": {
            "count": blob_rows,
            "bytes": blob_bytes,
            "note": ("blob bytes are never reclaimed: a digest is referenced "
                     "from about twenty columns and from event payloads, so a "
                     "collector that missed one source would delete content "
                     "the chain points at"),
        },
        "policy": {
            "evidence_tables": list(EVIDENCE_TABLES),
            "prunable_tables": list(PRUNABLE_TABLES),
        },
    }


def prunable(conn, *, older_than: float) -> dict[str, list[str]]:
    """What may be forgotten right now, without deleting anything.

    Separated from the deletion so the decision can be inspected, tested and
    reported on its own. A collector whose only interface is "run it and see"
    is one nobody can review.

    Three conditions, and the third is the one that matters: a trigger still
    owed an answer is never old enough. Age is not a reason to stop owing
    somebody a reply, and a request whose row vanished would leave its caller
    waiting on a trigger id that no longer exists.
    """
    triggers = [r["trigger_id"] for r in conn.execute(
        "SELECT trigger_id FROM role_triggers"
        " WHERE status IN ('consumed', 'expired')"
        "   AND COALESCE(consumed_at, created_at) < ?"
        "   AND NOT (expects_answer = 1 AND answer_status IS NULL)",
        (older_than,))]

    # A turn goes only when it is closed, old, and nothing that survives still
    # points at it: no trigger referencing it, and no later turn continuing
    # it. Deleting a parent out from under a continuation would break the
    # chain `awaiting_answer` walks.
    doomed = set(triggers)
    turns = []
    for row in conn.execute(
            "SELECT turn_id FROM role_turns"
            " WHERE status != 'running' AND COALESCE(finished_at, started_at) < ?",
            (older_than,)):
        turn_id = row["turn_id"]
        referencing = [r["trigger_id"] for r in conn.execute(
            "SELECT trigger_id FROM role_triggers WHERE turn_id = ?", (turn_id,))]
        if any(t not in doomed for t in referencing):
            continue
        children = conn.execute(
            "SELECT COUNT(*) FROM role_turns WHERE parent_turn = ?",
            (turn_id,)).fetchone()[0]
        if children:
            continue
        turns.append(turn_id)
    return {"triggers": triggers, "turns": turns}


def prune(m: "Mutation", mind: "Mind", *, older_than_seconds: float,
          now: float | None = None) -> dict[str, Any]:
    """Forget the working set that is past its window. Keep the record.

    Emits an event describing exactly what it forgot, which is not ceremony:
    the one thing an operator will want after noticing a turn is missing is
    evidence that it was pruned rather than lost.
    """
    now = time.time() if now is None else now
    cutoff = now - max(0.0, float(older_than_seconds))
    doomed = prunable(mind.db.conn, older_than=cutoff)

    for trigger_id in doomed["triggers"]:
        m.sql("DELETE FROM role_triggers WHERE trigger_id = ?", (trigger_id,))
    for turn_id in doomed["turns"]:
        m.sql("DELETE FROM role_turns WHERE turn_id = ?", (turn_id,))

    if doomed["triggers"] or doomed["turns"]:
        m.emit(EventKind.STORE_PRUNED, {
            "cutoff": cutoff,
            "older_than_seconds": older_than_seconds,
            "triggers": len(doomed["triggers"]),
            "turns": len(doomed["turns"]),
            "note": ("the operational working set only; the event log, the "
                     "receipts and every evidentiary table are untouched")})
    return {"cutoff": cutoff,
            "triggers_pruned": len(doomed["triggers"]),
            "turns_pruned": len(doomed["turns"])}
