"""The backchannel room: what Ego, Id and the Operator are saying right now.

Deliberately ephemeral, and deliberately not a transcript.

Ego and Id have always been able to talk to each other. `ego_message_id` and
`id_message_ego` queue a durable `role_message` trigger in the other role's
mailbox, which becomes cognitive input at a turn boundary -- that part works
and is recorded where causal influence belongs, in the trigger row and in the
turn that consumed it. What was missing was anywhere to *watch* it. Neither
verb emitted an event, so the operator console's "ego <-> id backchannel" page
read three event kinds, none of which either verb produced, and had never in
its life displayed a single Ego-to-Id message.

The fix is not a durable chat log. Reconstructing a conversation from the
event ledger would make the ledger answer a question it is not for, and
storing a second copy of what roles said would create a record that can
disagree with the one that actually shaped cognition. So this is a bounded
in-memory buffer owned by the running supervisor: it shows the current
runtime, it is capped, and it disappears when the process does. A restart
showing an empty room is correct.

The division of labour, stated plainly:

    this buffer        what is being said, right now, for a human to read
    trigger + turn     what was actually delivered and what it influenced
    event ledger       the operator's own acts, which are governance

Only the first is allowed to forget.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

MAX_ENTRIES = 400
"""Enough to scroll through a working session, small enough to never matter.

A cap rather than a policy: the buffer is a viewport, and a viewport that
grows without bound is a memory leak wearing a feature's clothes.
"""

MAX_TEXT = 4000
"""Long enough for a real message, short enough that one cannot flood the room."""

AUTHORS = ("ego", "id", "operator")


class Room:
    """A bounded, in-memory view of the live backchannel.

    Authorship is not a field a caller fills in. Every append goes through the
    Harness, which knows who spoke because it is the thing that carried the
    message; an author outside `AUTHORS` is refused rather than recorded,
    because a room entry nobody can be held to is worse than no entry.
    """

    def __init__(self, maxlen: int = MAX_ENTRIES) -> None:
        self._lock = threading.RLock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0

    def post(self, *, author: str, text: str, kind: str = "") -> dict[str, Any] | None:
        """Record that somebody said something. Returns the entry, or None.

        None when there is nothing to show: the room is a conversation, and a
        structured signal with no prose in it is not part of one.
        """
        text = (text or "").strip()
        if not text:
            return None
        if author not in AUTHORS:
            raise ValueError(f"unknown room author {author!r}")
        with self._lock:
            self._seq += 1
            entry = {"seq": self._seq, "ts": time.time(),
                     "author": author, "text": text[:MAX_TEXT]}
            if kind:
                entry["kind"] = kind
            self._entries.append(entry)
            return dict(entry)

    def since(self, seq: int = 0, limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
        """Entries after `seq`, oldest first.

        A reader polling with the last sequence it saw gets exactly what it
        has not seen. A reader starting at 0 gets the whole live buffer, which
        is what opening the page should show.
        """
        with self._lock:
            out = [dict(e) for e in self._entries if e["seq"] > seq]
        return out[-max(1, int(limit)):] if limit else out

    def state(self) -> dict[str, Any]:
        """What the buffer is, said honestly enough to render."""
        with self._lock:
            return {"latest_seq": self._seq, "held": len(self._entries),
                    "capacity": self._entries.maxlen}
