"""Who owns the structure in a rendered chat turn.

I124 stopped a *generation* from authoring chat-template structure. This is
the same rule facing the other way. Everything a session is told arrives as
content from somewhere that is not the Harness -- a client's question, a
file's bytes, a tool's result, another role's message -- and content is text.
Text may not become the substrate's own boundaries.

The defect this exists to make impossible: render a message with the model's
chat template and tokenize the whole rendered string with specials parsed,
and the tokenizer cannot tell the Harness's `<|im_start|>` from one the
content happened to contain. Both become the same control token, and the
session then holds a message boundary nobody in the record authored.

The mechanism is to never hand the tokenizer a string in which frame and
content are indistinguishable. The frame is rendered around a nonce, split
back off at the nonce, and tokenized with specials parsed. Each message's
content is tokenized beside it with specials *off*, so a marker inside it can
only ever become the characters it is. Structure comes from the template;
content comes from the message; the tokenizer is never asked to guess which
it is looking at.

Pure: no backend, no session. What it returns is a plan someone else runs.
"""
from __future__ import annotations

import secrets
from typing import Any, NamedTuple, Sequence

# A private-use codepoint on both sides: it cannot occur in a chat template,
# and content that contains one still cannot collide, because the nonce in
# the middle is drawn fresh for every render and checked against the content.
MARK = "amoeba-content-{nonce}-{index}"


class Segment(NamedTuple):
    """A run of text, and whether the tokenizer may read structure in it."""

    text: str
    special: bool


class FramingError(RuntimeError):
    """The rendered template did not come back with its content marks intact.

    Raised rather than recovered from. Falling back to tokenizing the whole
    string is precisely the defect, so there is no fallback: a template this
    cannot frame is a template this organism will not ingest through.
    """


def nonce_for(contents: Sequence[str]) -> str:
    """A mark nobody's content already contains.

    Drawn at random and then *checked*, because "unguessable" is an argument
    about an attacker and this also has to hold against an accident.
    """
    for _ in range(8):
        candidate = secrets.token_hex(8)
        if not any(candidate in c for c in contents):
            return candidate
    raise FramingError("could not draw a content mark")


def marks(count: int, nonce: str) -> list[str]:
    return [MARK.format(nonce=nonce, index=i) for i in range(count)]


def framed(messages: Sequence[dict[str, str]], marks_: Sequence[str]
           ) -> list[dict[str, str]]:
    """The messages as the template should see them: roles real, content marked."""
    if len(messages) != len(marks_):
        raise FramingError("a mark per message is the whole point")
    return [{"role": str(m["role"]), "content": mark}
            for m, mark in zip(messages, marks_)]


def split(rendered: str, marks_: Sequence[str]) -> list[str]:
    """The frame around the marks: one more piece than there are marks.

    A mark that is missing, or that the template moved out of order, means the
    rendering is not the one that was planned -- and an unrecognised rendering
    is the case where guessing would put content back inside the frame.
    """
    pieces: list[str] = []
    rest = rendered
    for mark in marks_:
        head, sep, rest = rest.partition(mark)
        if not sep:
            raise FramingError("the template did not render a content mark")
        pieces.append(head)
    for mark in marks_:
        if mark in rest:
            raise FramingError("the template rendered a content mark twice")
    pieces.append(rest)
    return pieces


def segments(rendered: str, marks_: Sequence[str],
             contents: Sequence[str]) -> list[Segment]:
    """Frame and content interleaved, each labelled with who may own structure."""
    if len(marks_) != len(contents):
        raise FramingError("a mark per content is the whole point")
    pieces = split(rendered, marks_)
    out: list[Segment] = []
    for frame, content in zip(pieces, contents):
        out.append(Segment(frame, True))
        out.append(Segment(content, False))
    out.append(Segment(pieces[-1], True))
    return [s for s in out if s.text]


def message_segments(text: str, *, start_text: str, end_text: str
                     ) -> list[Segment]:
    """The same division for a message being rewritten out of a session.

    A rebuild re-tokenizes text it decoded from tokens it already holds, so it
    is a second way into the same defect: content that reached the session as
    literal marker *characters* would become real markers on the way back in.
    The header and the terminator are the template's; everything between them
    is whatever the message said.
    """
    body = text
    head = ""
    if start_text and body.startswith(start_text):
        head, body = start_text, body[len(start_text):]
    line, sep, rest = body.partition("\n")
    head = head + line + sep
    tail = ""
    if end_text:
        at = rest.rfind(end_text)
        if at >= 0:
            tail, rest = rest[at:], rest[:at]
    return [s for s in (Segment(head, True), Segment(rest, False),
                        Segment(tail, True)) if s.text]


def as_payload(segs: Sequence[Segment]) -> list[dict[str, Any]]:
    """Segments in the shape a method call carries them."""
    return [{"text": s.text, "special": s.special} for s in segs]


def from_payload(rows: Sequence[dict[str, Any]]) -> list[Segment]:
    return [Segment(str(r["text"]), bool(r["special"])) for r in rows]
