"""Only the Harness builds the shape of a conversation.

A generation is text the model contributed to the turn it is having. The
chat template's control tokens are something else: they say who is speaking
and where one message ends. If a model can emit them, it can author the
structure of its own context -- open a `user` message, put words in the
Harness's mouth, close it again -- and everything downstream reads that
forgery as substrate.

Measured live on Id, at 11:50 on 2026-09-22: it emitted a real tool call,
then kept generating `<|im_start|>user` followed by a `<tool_result>` block
it copied from a result ten minutes old. The Harness appended the true
result underneath, so the session held a stale reading directly above the
fresh one, in a message the model wrote and the record attributes to nobody.

What made it invisible was the decode side: `token_to_piece(special=False)`
renders a control token as the empty string, so the returned text showed no
marker at all while the token went into the session and the KV. That is why
this is decided on the **token id, before the token is appended** -- never
by looking for markers in decoded text, which is exactly the surface the
defect hides from.
"""

from __future__ import annotations

from typing import Callable, Sequence

# What a generation that tried to author structure is called. It is not an
# error: the model produced text up to that point and that text is kept.
STRUCTURAL_FINISH = "structural_token"


def scan_structural(n_vocab: int, piece: Callable[[int, bool], str]) -> frozenset[int]:
    """Every token that carries structure rather than text.

    A control token is one the vocabulary renders as nothing when asked for
    plain text and as something when asked for the special form: present in
    the stream, absent from the transcript. That is the whole bug class, so
    it is the definition -- not a list of the markers one template happens to
    use, which would miss the next model's.
    """
    out = set()
    for token in range(int(n_vocab)):
        special = piece(token, True)
        if special and not piece(token, False):
            out.add(token)
    return frozenset(out)


def structural_from_text(markers: Sequence[str],
                         tokenize: Callable[[str], Sequence[int]]) -> frozenset[int]:
    """The same rule for a backend whose vocabulary is its own invention."""
    out: set[int] = set()
    for marker in markers:
        out.update(int(t) for t in tokenize(marker))
    return frozenset(out)


def cut_at_structural(tokens: Sequence[int], structural: frozenset[int]
                      ) -> tuple[list[int], int | None]:
    """Everything up to the first structural token, and which token that was."""
    for i, token in enumerate(tokens):
        if token in structural:
            return list(tokens[:i]), token
    return list(tokens), None
