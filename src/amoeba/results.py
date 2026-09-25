"""Results a mind was shown only part of: stored exactly, handed over by reference.

A bounded projection is truthful only if what it points at exists and can be
reached by the mind it was shown to. The exact text goes to the content store,
and the reference is recorded as issued to that role, so `result_read` opens
it for that role and for nobody else -- a reference is a capability that was
handed over, not a digest anyone could guess.

Shared by tool delivery and by context rebuilds, which re-render an owed
turn's oversized result the same way rather than deleting evidence a
continuation may need.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from .ids import new_id
from .store.blobs import canonical_json
from .store.writer import Mutation


# Tools that *make* an observation rather than fetch a recorded one. Each call
# is its own acquisition, so two of them are two roots even when they return
# identical bytes -- two computations that both print "0" are two observations.
# Everything else is treated as a retrieval of durable state, where the same
# query by two actors is one source consulted twice. The default is deliberate:
# an unclassified tool must not manufacture independence.
FRESH_ACQUISITION = ("run_code", "read_file", "list_files", "write_file",
                     "current_state_version", "system_pulse", "context_report",
                     "verify_integrity", "vram_free")


def evidence_root(tool: str | None, arguments: Any = None) -> str:
    """The canonical identity of an acquisition's source lineage.

    Not the payload. A retrieval is identified by *what was consulted*, so
    re-reading a board post -- by the same actor or a different one -- is one
    root, which is what makes "the worker went and looked itself" honest
    without making it corroboration. A fresh acquisition is identified by the
    act, because performing a computation is observing something new.
    """
    name = (tool or "unknown").strip()
    if name in FRESH_ACQUISITION:
        return f"acq:{new_id('evr')}"
    canon = canonical_json(arguments if arguments is not None else {})
    return f"src:{name}:{hashlib.sha256(canon).hexdigest()[:16]}"


def issue_result(mind: Any, text: str, *, role: str, tool: str | None,
                 actor: str | None = None, root: str | None = None,
                 arguments: Any = None, acquired: str = "first_hand",
                 inherited_from: str | None = None) -> str:
    """Store `text` exactly and issue its reference to `role`. Returns the digest.

    This is the acquisition receipt as well as the retrieval handle. It used
    to be written only when a result was too large to show whole, which left
    no record of the small ones -- so the system could not answer "what did
    this actor actually observe?", and corroboration fell back to guessing
    from who had read which blackboard post (I138).
    """
    data = text.encode("utf-8")
    digest = mind.blobs.put(data)
    lineage = root or evidence_root(tool, arguments)

    def body(m: Mutation) -> None:
        m.register_blob(digest, len(data), "application/json", "tool_result_full")
        # The retrieval handle: keyed by the payload, because that is what it
        # opens.
        m.conn.execute(
            "INSERT OR IGNORE INTO issued_results(result_ref, issued_to, sha256,"
            " tool, created_at) VALUES (?,?,?,?,?)",
            (digest[:16], role, digest, tool, time.time()))
        # The evidentiary record: keyed by the acquisition, because that is
        # what corroboration counts.
        m.conn.execute(
            "INSERT OR IGNORE INTO acquisitions(actor, evidence_root, sha256,"
            " tool, created_at, acquired, inherited_from) VALUES (?,?,?,?,?,?,?)",
            (role, lineage, digest, tool, time.time(), acquired, inherited_from))

    mind.writer.apply(body, actor=actor or role, bump_version=False)
    return digest


def acquisition_roots(conn: Any, actor: str, *, before: float | None = None,
                      first_hand_only: bool = True) -> set[str]:
    """The source lineages this actor observed for itself.

    Inherited roots are real -- a worker may cite and reason from them -- but
    they are somebody else's observations, so they never add support.
    """
    sql = ["SELECT DISTINCT evidence_root FROM acquisitions WHERE actor = ?"]
    params: list[Any] = [actor]
    if first_hand_only:
        sql.append(" AND acquired = 'first_hand'")
    if before is not None:
        sql.append(" AND created_at <= ?")
        params.append(before)
    return {r["evidence_root"] for r in conn.execute("".join(sql), params)}


def inherit_roots(mind: Any, *, heir: str, source: str,
                  before: float | None = None) -> int:
    """Give an heir the roots its forebear held, marked as inherited.

    A forked worker receives the forker's context, and the forker's
    observations come with it. Recording that is what stops the worker
    looking like a second witness to them.
    """
    sql = ("SELECT evidence_root, sha256, tool FROM acquisitions"
           " WHERE actor = ?")
    params: list[Any] = [source]
    if before is not None:
        sql += " AND created_at <= ?"
        params.append(before)
    rows = list(mind.db.conn.execute(sql, params))
    if not rows:
        return 0

    def body(m: Mutation) -> None:
        for row in rows:
            m.conn.execute(
                "INSERT OR IGNORE INTO acquisitions(actor, evidence_root,"
                " sha256, tool, created_at, acquired, inherited_from)"
                " VALUES (?,?,?,?,?,'inherited',?)",
                (heir, row["evidence_root"], row["sha256"], row["tool"],
                 time.time(), source))

    mind.writer.apply(body, actor="harness", bump_version=False)
    return len(rows)


def issued_digest(mind: Any, result_ref: str, *, role: str) -> str | None:
    row = mind.db.conn.execute(
        "SELECT sha256 FROM issued_results WHERE result_ref = ? AND issued_to = ?",
        (str(result_ref).strip()[:16], role)).fetchone()
    return row["sha256"] if row else None

# What a work result actually *said*, as opposed to how it was produced. A
# neuocyte's result carries its finding beside its telemetry -- raw text, tool
# calls, tools offered, posts seen, token counts -- and the finding is the
# short part. Bounded projection treats the whole dict alike, so the substance
# ends up behind the same reference as the noise.
#
# Live on 2026-09-25 a worker computed 41679167500, recorded it in `finding`,
# and completed. Ego called `get_work`, then `result_read` three times, never
# reached it, and answered the client with a number of its own. The value was
# four hops away inside 2118 bytes of provenance (I142).
SUBSTANCE_KEYS = ("finding", "answer", "claim", "summary", "plan")
MAX_SUBSTANCE = 600


def substance_of(result: Any) -> dict[str, Any]:
    """The reportable core of a work result: what it concluded, and how sure.

    Returned separately so it survives projection whole. Everything else about
    the result stays exactly where it was and is reached the same way.
    """
    if not isinstance(result, dict):
        return {}
    out: dict[str, Any] = {}
    for key in SUBSTANCE_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            out["finding"] = value.strip()[:MAX_SUBSTANCE]
            break
    if result.get("confidence") is not None:
        out["confidence"] = result["confidence"]
    return out

