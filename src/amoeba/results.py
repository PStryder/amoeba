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

import time
from typing import Any

from .store.writer import Mutation


def issue_result(mind: Any, text: str, *, role: str, tool: str | None,
                 actor: str | None = None) -> str:
    """Store `text` exactly and issue its reference to `role`. Returns the digest."""
    data = text.encode("utf-8")
    digest = mind.blobs.put(data)

    def body(m: Mutation) -> None:
        m.register_blob(digest, len(data), "application/json", "tool_result_full")
        m.conn.execute(
            "INSERT OR IGNORE INTO issued_results(result_ref, issued_to, sha256,"
            " tool, created_at) VALUES (?, ?, ?, ?, ?)",
            (digest[:16], role, digest, tool, time.time()))

    mind.writer.apply(body, actor=actor or role, bump_version=False)
    return digest


def issued_digest(mind: Any, result_ref: str, *, role: str) -> str | None:
    row = mind.db.conn.execute(
        "SELECT sha256 FROM issued_results WHERE result_ref = ? AND issued_to = ?",
        (str(result_ref).strip()[:16], role)).fetchone()
    return row["sha256"] if row else None
