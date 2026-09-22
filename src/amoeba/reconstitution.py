"""Rebuilding a role's context from whole parts, never from token offsets.

Positional trimming kept a head and a tail of the recorded tokens and dropped
whatever lay between. Measured live on Id, it cut its capability declaration
off after 227 tokens, mid-line, kept two later "the declaration given earlier
still applies" references whose referent it had just removed, and resumed the
tail in the middle of a tool result. None of that is a state the conversation
could have reached by itself, which is the test a context transform has to
pass.

The rule this module implements:

    Environment is reconstructed. Cognition is preserved selectively.
    Neither is token-spliced.

- **Messages are found, not guessed.** The checkpoint is split at the chat
  template's message-start token, so every boundary is one the session
  actually has.
- **Substrate is rebuilt.** The governed prompt is rendered fresh from the
  binding. Every environment block -- a full declaration or a reference to
  one -- is removed from the messages that carried it, because the current
  declaration is rendered in full at the next turn of any new session (I118),
  immediately before it is used. Nothing kept can point at text that is gone.
- **Settled work may be removed whole; owed work must stay sufficient to
  continue.** A turn is the unit: its opening, its tool calls with their
  results, its reply. Settled turns go oldest first. A turn whose interaction
  is still owed an answer -- or one the record cannot place -- loses nothing:
  a call whose result vanished is not something a continuation can reason
  from. What may shrink it is re-rendering an oversized result as the same
  bounded projection a live call would get, naming the exact stored copy. If
  even that is not enough the rebuild says so and stops; it never falls back
  to cutting by position.

Pure functions only. Tokenizing, detokenizing and the database belong to the
caller (`ContextHomeostasis.rejuvenate`), so every decision here can be
tested without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

ENVIRONMENT_OPEN = "<role_environment"
TURN_INPUT_OPEN = "<turn_input>"
TOOL_RESULT_OPEN = "<tool_result"
TOOL_CALL_OPEN = "<tool_call>"


@dataclass(slots=True)
class Message:
    """One whole message of the session, exactly as it was held."""

    start: int
    end: int
    tokens: list[int]
    text: str
    role: str                 # system | user | assistant | other
    kind: str                 # system | opening | tool_call | tool_result | reply | other
    terminated: bool
    rewritten: bool = False   # re-rendered: environment removed, or result projected

    @property
    def n(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class Unit:
    """A turn's worth of messages: an opening and everything until the next."""

    messages: list[Message]
    turn_ids: list[str] = field(default_factory=list)
    status: str = "unknown"   # settled | owed | unknown

    @property
    def start(self) -> int:
        return self.messages[0].start

    @property
    def end(self) -> int:
        return self.messages[-1].end

    @property
    def n(self) -> int:
        return sum(m.n for m in self.messages)


def split_messages(tokens: Sequence[int], start_id: int) -> list[tuple[int, int]]:
    """Every [start, end) that begins at a message-start token.

    Anything before the first start token is its own span, so no token is
    unaccounted for.
    """
    cuts = [i for i, t in enumerate(tokens) if t == start_id]
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
    cuts.append(len(tokens))
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def classify(text: str, *, start_text: str, end_text: str) -> tuple[str, str, bool, str]:
    """(role, kind, terminated, content) for one decoded message."""
    body = text[len(start_text):] if text.startswith(start_text) else text
    role, _, content = body.partition("\n")
    role = role.strip()
    if role not in ("system", "user", "assistant"):
        role, content = "other", body
    terminated = content.rstrip().endswith(end_text.strip()) if end_text.strip() else True
    stripped = content.lstrip()
    if role == "system":
        kind = "system"
    elif role == "user" and TURN_INPUT_OPEN in content:
        kind = "opening"
    elif role == "user" and stripped.startswith(TOOL_RESULT_OPEN):
        kind = "tool_result"
    elif role == "assistant" and TOOL_CALL_OPEN in content:
        kind = "tool_call"
    elif role == "assistant":
        kind = "reply"
    else:
        kind = "other"
    return role, kind, terminated, content


def strip_environment(text: str) -> tuple[str, bool]:
    """Remove the environment block that precedes a turn's input.

    The block always sits between the message header and `<turn_input>`: a
    full declaration with its call-form instructions, or the one-line
    reference that stands for it. Everything from the turn input on is kept
    byte for byte.
    """
    at = text.find(ENVIRONMENT_OPEN)
    ti = text.find(TURN_INPUT_OPEN)
    if at < 0 or ti < 0 or at > ti:
        return text, False
    return text[:at] + text[ti:], True


def is_empty_generation_prompt(m: Message) -> bool:
    """An assistant header that was never answered -- a refused generation."""
    if m.role != "assistant" or m.terminated:
        return False
    body = m.text.split("\n", 1)[1] if "\n" in m.text else ""
    return not body.strip()


def group_units(messages: Sequence[Message]) -> list[Unit]:
    """A unit per opening; messages before the first opening form their own."""
    units: list[Unit] = []
    for m in messages:
        if m.kind == "opening" or not units:
            units.append(Unit(messages=[m]))
        else:
            units[-1].messages.append(m)
    return units


def attribute(units: Sequence[Unit], spans: Sequence[dict[str, Any]],
              owed: set[Any]) -> None:
    """Which recorded turns each unit holds, and whether any is still owed.

    A turn's span lies inside exactly one unit when the coordinates are
    current. A span that fits no unit is ignored rather than stretched to fit:
    it describes some other layout. A unit that holds no recorded turn is
    `unknown`, and unknown is kept -- "I do not know what this is" must not
    resolve to "so remove it".
    """
    for u in units:
        held = [s for s in spans
                if u.start <= int(s["start"]) and int(s["end"]) <= u.end
                and int(s["end"]) > int(s["start"])]
        u.turn_ids = [s["turn_id"] for s in held]
        if not held:
            u.status = "unknown"
        elif any(s.get("lineage") in owed or s.get("open") for s in held):
            u.status = "owed"
        else:
            u.status = "settled"


def plan(units: Sequence[Unit], *, system_tokens: int, target: int) -> dict[str, Any]:
    """Choose which whole units survive. Only settled work is ever removed.

    Settled units go oldest first until the target is met. Owed and unknown
    units are never removed and never have a message taken out: a turn still
    owed an answer must stay sufficient to continue, and a call whose result
    vanished is not. What can still shrink them is re-rendering an oversized
    result as a bounded projection (`result_body`), which the caller does.
    """
    kept = [True] * len(units)
    total = system_tokens + sum(u.n for u in units)
    dropped_units: list[dict[str, Any]] = []
    for i, u in enumerate(units):
        if total <= target:
            break
        if u.status != "settled":
            continue
        kept[i] = False
        total -= u.n
        dropped_units.append({"turn_ids": u.turn_ids, "tokens": u.n,
                              "messages": len(u.messages)})
    return {
        "kept_units": [i for i, k in enumerate(kept) if k],
        "dropped_units": dropped_units,
        "kept_tokens": total,
        "target_tokens": target,
        "reached_target": total <= target,
    }


def result_body(text: str) -> tuple[str, str, str, str] | None:
    """(before, tool name, result text, after) of a tool-result message."""
    at = text.find(TOOL_RESULT_OPEN)
    if at < 0:
        return None
    head_end = text.find(">\n", at)
    close = text.find("\n</tool_result>", head_end)
    if head_end < 0 or close < 0:
        return None
    tag = text[at:head_end + 1]
    name = tag.split('name="', 1)[1].split('"', 1)[0] if 'name="' in tag else ""
    return text[:head_end + 2], name, text[head_end + 2:close], text[close:]


def shrinkable(units: Sequence[Unit], kept: Sequence[int]) -> list[tuple[int, int]]:
    """Tool results in kept units, largest first: where a projection can help."""
    out = [(i, j) for i in kept for j, m in enumerate(units[i].messages)
           if m.kind == "tool_result"]
    return sorted(out, key=lambda ij: -units[ij[0]].messages[ij[1]].n)


def assemble(units: Sequence[Unit], chosen: dict[str, Any]
             ) -> tuple[list[int], list[tuple[Unit, int, int]]]:
    """The kept messages in order, and where each kept unit now sits.

    Offsets are relative to the start of the cognition, i.e. after the
    governed prompt; the caller adds the prompt's length.
    """
    out: list[int] = []
    placed: list[tuple[Unit, int, int]] = []
    for i in chosen["kept_units"]:
        u = units[i]
        start = len(out)
        for m in u.messages:
            out.extend(m.tokens)
        placed.append((u, start, len(out)))
    return out, placed
