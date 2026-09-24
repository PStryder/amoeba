"""What an identifier is, and where a real one comes from.

Live, Id was told "unaudited conclusions: 17" and then called
`get_conclusion`, `get_memory` and `audit_dossier` with
`c8f050bd638e10dc` -- the reference from a bounded tool result it had just
been shown. Earlier it had used its own session id with a `con_` prefix
glued on. Each time the answer was "unknown conclusion", which is true and
useless: it says the value is wrong and nothing about where a right one
lives.

The cause was not carelessness. Id's surface can fetch a conclusion by id
and cannot list conclusions at all, so the only ways to obtain one were
digging through raw events or provenance. A mind handed a count with no
route to the things counted will manufacture the route.

Two answers, and neither is fuzzy matching -- guessing which id was meant
would make a mind's mistake into the Harness's claim:

* Say what the value actually is. `c8f050bd638e10dc` is a result reference,
  not a conclusion id, and that is a different mistake from naming a
  conclusion that does not exist.
* Say where a real one comes from, once, in the refusal. The same shape as
  a refusal carrying the values it would have accepted (I117).
"""

from __future__ import annotations

import re

# `prefix_ULID`: the shape everything the organism names is built from.
ID = re.compile(r"^([a-z]+)_[0-9A-HJKMNP-TV-Z]{26}$")
# What a bounded tool result is referenced by: the first sixteen hex digits
# of the stored copy's digest. Shown to a mind constantly, and the nearest
# thing to hand when it needs an identifier it was never given.
RESULT_REF = re.compile(r"^[0-9a-f]{16}$")

KINDS = {
    "concl": "conclusion", "mem": "memory item", "post": "blackboard post",
    "work": "work item", "turn": "turn", "op": "operation",
    "audit": "audit", "disag": "disagreement", "art": "artifact",
    "sess": "inference session", "snap": "snapshot", "ev": "event",
    "trg": "trigger", "bnd": "profile binding", "sbx": "sandbox",
    "ixn": "interaction", "pv": "prompt version", "rcpt": "receipt",
}

# Where each kind of identifier can actually be got, in the words of the
# capabilities that yield one. A count is not an affordance.
SOURCES = {
    "conclusion_id": ("audit_dossier() with no argument resolves the oldest "
                      "conclusion nobody has audited; history(kinds="
                      "['conclusion.recorded']) lists them as they were made"),
    "memory_id": "recall() lists maintained memory, each item with its id",
    "post_id": "board_read(reader=<you>) lists posts, each with its post_id",
    "work_id": "queue_stats() counts work; get_work(work_id) reads one",
    "result_ref": "the reference printed in the bounded result you were shown",
}


def looks_like(value: object) -> str:
    """What this value is, as a phrase -- not what it should have been."""
    text = str(value)
    if RESULT_REF.match(text):
        return "a result reference"
    match = ID.match(text)
    if match:
        kind = KINDS.get(match.group(1))
        if not kind:
            return "an identifier of a kind this organism does not issue"
        return f"{'an' if kind[0] in 'aeiou' else 'a'} {kind} identifier"
    if text.isdigit():
        return "a number"
    return "not an identifier this organism issues"


# What each argument names, by the prefix its identifiers carry.
EXPECTED = {
    "conclusion_id": "concl", "memory_id": "mem", "post_id": "post",
    "work_id": "work", "turn_id": "turn", "operation_id": "op",
    "artifact_id": "art", "session_id": "sess", "audit_id": "audit",
    "post_a": "post", "post_b": "post", "trigger_id": "trg",
}


def misuse(argument: str, value: object) -> str | None:
    """Why this value cannot be what the argument names, when that is plain.

    `None` when the value is a well-formed identifier of the right kind: it
    may simply be absent, and "no such conclusion" is then the whole truth.
    """
    want = EXPECTED.get(argument)
    match = ID.match(str(value))
    if want is None or (match and match.group(1) == want):
        return None
    return f"{value!r} is {looks_like(value)}"


def where_from(argument: str) -> str:
    """How to obtain a real one, if this organism says anywhere."""
    return SOURCES.get(argument, "")


def explain(argument: str, value: object) -> str:
    """The sentence a refusal adds: what it is, and where a real one lives."""
    parts = [p for p in (misuse(argument, value), where_from(argument)) if p]
    return " -- ".join(parts)
