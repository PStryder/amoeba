"""What changed since Id last looked, measured rather than described.

Id is woken by conclusions, messages, work it owns and the operator. Nothing
wakes it for a failure, for resource pressure, or for a contradiction nobody
announced -- the heartbeat is the only sense for unannounced state, which is
why it exists and why it cannot simply be made rarer.

It could be made *cheaper*. Measured on the live organism, one heartbeat cost
about two thousand tokens, of which 1793 were tool results: five calls that
re-read state which had not moved, every half hour, carried forward until the
next rebuild threw them away. The cause was the trigger, which said "nothing
has woken you, check the organism's internal state" and carried no
information, so Id paid five calls to discover a row of zeros.

The digest is built from an event watermark -- every event since the last
heartbeat, counted by kind -- so it cannot quietly omit a kind nobody thought
of. That completeness is the whole point: a mind told "nothing changed" by a
digest that only reports what its author remembered to include is blind in
exactly the way it cannot detect. Where the count is too long to show it says
how much it is not itemising rather than dropping it.

It measures and never interprets. "Two conclusions recorded, one unaudited" is
a fact; "nothing worth your attention" would be the Harness doing Id's job,
and Id keeps every sense it had -- the digest is a starting point it is free
to distrust and check.
"""

from __future__ import annotations

from typing import Any

MAX_KINDS_SHOWN = 20
"""Beyond this, the remainder is summarised as a count -- never dropped."""


def watermark_of(conn, blobs: Any, role: str) -> int | None:
    """Where the previous heartbeat for this role measured up to.

    Read from that trigger's own payload rather than from memory, so a
    restarted supervisor does not silently restart the window -- and when
    there is no previous heartbeat to read, the answer is `None`, which the
    digest reports as a first review rather than as a quiet organism.
    """
    row = conn.execute(
        "SELECT payload_sha256 FROM role_triggers"
        " WHERE target_role = ? AND kind = 'heartbeat'"
        "   AND payload_sha256 IS NOT NULL"
        " ORDER BY created_at DESC LIMIT 1", (role,)).fetchone()
    if row is None or blobs is None:
        return None
    try:
        payload = blobs.get_json(row["payload_sha256"])
    except Exception:  # noqa: BLE001 - unreadable is unknown, not zero
        return None
    seq = payload.get("measured_to_seq") if isinstance(payload, dict) else None
    return int(seq) if isinstance(seq, int) else None


def measure(conn, role: str, *, since: int | None) -> dict[str, Any]:
    """Everything that happened since `since`, by kind, plus what is owed.

    `since` of None means this is the first heartbeat of the process's life:
    the digest says so rather than claiming a quiet organism.
    """
    now_seq = int(conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()[0])
    by_kind: dict[str, int] = {}
    if since is not None:
        by_kind = {r["kind"]: int(r["n"]) for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM events WHERE seq > ?"
            " GROUP BY kind ORDER BY n DESC", (int(since),))}
    total = sum(by_kind.values())
    shown = dict(list(by_kind.items())[:MAX_KINDS_SHOWN])
    return {
        "watermark": since,
        "seq": now_seq,
        "events": total,
        "by_kind": shown,
        # Counted, not dropped: the digest stays complete even when it is
        # not itemised.
        "other_kinds": max(0, len(by_kind) - len(shown)),
        "other_events": total - sum(shown.values()),
        "attention": _attention(conn, role),
        # A count with no route to the thing counted is what made a mind
        # invent identifiers. The oldest one is named, and the rest follow it.
        "oldest_unaudited": _oldest_unaudited(conn),
        "first_review": since is None,
    }


def _attention(conn, role: str) -> dict[str, int]:
    """The few standing questions Id would otherwise spend calls to ask."""
    def count(sql: str, *args: Any) -> int:
        return int(conn.execute(sql, args).fetchone()[0])

    return {
        "open_disagreements": count(
            "SELECT COUNT(*) FROM disagreements WHERE status = 'open'"),
        "unaudited_conclusions": count(
            "SELECT COUNT(*) FROM conclusions c WHERE NOT EXISTS ("
            "SELECT 1 FROM audits a WHERE a.target_kind = 'conclusion'"
            " AND a.target_id = c.conclusion_id)"),
        "work_queued": count(
            "SELECT COUNT(*) FROM work_items WHERE status IN ('queued', 'leased')"),
        "unanswered_requests": count(
            "SELECT COUNT(*) FROM role_triggers WHERE target_role = ?"
            " AND expects_answer = 1 AND answer_status IS NULL", role),
    }


def _oldest_unaudited(conn) -> str | None:
    row = conn.execute(
        "SELECT conclusion_id FROM conclusions c WHERE NOT EXISTS ("
        "SELECT 1 FROM audits a WHERE a.target_kind = 'conclusion'"
        " AND a.target_id = c.conclusion_id) ORDER BY created_at LIMIT 1").fetchone()
    return row["conclusion_id"] if row else None


def quiet(digest: dict[str, Any]) -> bool:
    """Nothing happened and nothing is owed: there is genuinely nothing to say."""
    return (not digest["first_review"] and digest["events"] == 0
            and not any(digest["attention"].values()))


def render(digest: dict[str, Any], *, interval_seconds: float,
           deferred_seconds: float = 0.0) -> str:
    """The digest as the few lines a role reads."""
    lines = []
    if digest["first_review"]:
        lines.append(f"first review of this process; the record holds "
                     f"{digest['seq']} events, none of them read as a change.")
    elif digest["events"] == 0:
        lines.append(f"no events since seq {digest['watermark']} "
                     f"(~{interval_seconds / 60:.0f} min).")
    else:
        kinds = ", ".join(f"{k} {n}" for k, n in digest["by_kind"].items())
        lines.append(f"{digest['events']} events since seq {digest['watermark']} "
                     f"(~{interval_seconds / 60:.0f} min): {kinds}")
        if digest["other_kinds"]:
            lines.append(f"  and {digest['other_events']} more across "
                         f"{digest['other_kinds']} further kinds, not itemised here")
    owed = {k: v for k, v in digest["attention"].items() if v}
    said = [f"{k.replace('_', ' ')} {v}" for k, v in owed.items()]
    if digest.get("oldest_unaudited"):
        said = [s + f" (oldest: {digest['oldest_unaudited']})"
                if s.startswith("unaudited conclusions") else s for s in said]
    lines.append("attention: " + (", ".join(said) if said else "nothing outstanding"))
    if deferred_seconds:
        lines.append(f"this review ran {deferred_seconds:.0f}s late, held back "
                     "while the pool was under pressure.")
    lines.append(f"measured at seq {digest['seq']}; every event since the "
                 "watermark is counted above. Check anything you doubt.")
    return "\n".join(lines)
