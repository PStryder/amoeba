"""Who gets woken by something that happened, and why.

The relevance rule lives here and only here. It was already implemented once,
for work completion, as a closure inside `supervisor_api`; adding artifact and
board waking meant either importing that closure or writing the rule again in
`harness_api`. Two copies of a relevance rule is two rules, and the second one
drifts silently because nothing fails when they disagree.

The rule is ownership, not similarity. A role is woken by something that
happened to work *it originated* -- an explicit recorded relationship on the
work row, not a judgement about what looks interesting. That is what keeps a
busy neuocyte fleet from becoming a wake storm, and it is why this could be
built at all: a heuristic for "is this post relevant to Ego" would have been a
guess, and the architecture had declined to invent one.

Nothing here is answer-bearing. A neuocyte proposing an artifact is evidence
for a thought already in progress, not a question addressed to Ego, so these
triggers carry `expects_answer=False` and the interaction's own lineage --
they bundle into the turn already serving that thought rather than opening a
rival one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import mailbox

if TYPE_CHECKING:  # pragma: no cover
    from .supervisor import Supervisor


def owner_of_work(mind: Any, work_id: str | None) -> dict[str, Any] | None:
    """The work row, if it exists and a persistent role originated it.

    Work originated by the supervisor, the operator or a neuocyte wakes
    nobody: no persistent role is waiting on it, and waking one would be
    telling it about somebody else's business.
    """
    if not work_id:
        return None
    try:
        row = mind.work.get_work(work_id)
    except Exception:  # noqa: BLE001
        return None
    if not row or row.get("origin_actor") not in mailbox.ROLES:
        return None
    return dict(row)


def wake_owner_of_work(sup: "Supervisor", mind: Any, work_id: str | None, *,
                       kind: str, summary: str,
                       payload: dict[str, Any] | None = None,
                       author: str | None = None) -> str | None:
    """Queue a trigger for the role that asked for this work, if any.

    `author` suppresses self-notification: Ego posting to the blackboard about
    its own work does not need to be told that Ego posted to the blackboard.
    The alternative is a role waking itself in a loop, which is not a wake
    storm so much as a spiral.

    Best effort on purpose, exactly as work completion is. The thing that
    happened happened whether or not anybody was told, and failing the
    operation because a mailbox write failed would lose the event to protect
    the notification about it.
    """
    row = owner_of_work(mind, work_id)
    if row is None:
        return None
    owner = row["origin_actor"]
    if author is not None and author == owner:
        return None

    body = {"work_id": work_id,
            "objective": row.get("objective"),
            "work_class": row.get("work_class"),
            "status": row.get("status")}
    body.update(payload or {})
    try:
        out = sup.methods()["role_enqueue_trigger"](
            role=owner, kind=kind, source="harness", source_ref=work_id,
            summary=summary[:mailbox.MAX_SUMMARY],
            payload=body,
            correlation_id=row.get("operation_id"),
            # The operation that requested the work is its lineage, so this
            # arrives in the turn that is thinking about it and in no other.
            lineage=row.get("operation_id"))
        return out.get("trigger_id")
    except Exception:  # noqa: BLE001
        sup.log.debug("could not queue %s trigger for %s", kind, work_id,
                      exc_info=True)
        return None
