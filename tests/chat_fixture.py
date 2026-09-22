"""A chat-shaped session for tests: a reversible codec and recorded turns.

Rebuilding a context works on message structure, so a test of it needs a
session that has some. The codec is one token per character plus two special
tokens, which makes every boundary exact and every token list readable.
"""

from __future__ import annotations

import json
from typing import Any

START, END = "<|im_start|>", "<|im_end|>"
SPECIAL = {START: 1, END: 2}
_BACK = {v: k for k, v in SPECIAL.items()}
_OFFSET = 16


def tokenize(text: str, parse_special: bool = True) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(text):
        hit = next((s for s in SPECIAL if parse_special and text.startswith(s, i)), None)
        if hit:
            out.append(SPECIAL[hit])
            i += len(hit)
        else:
            out.append(ord(text[i]) + _OFFSET)
            i += 1
    return out


def detokenize(tokens: list[int], special: bool = True) -> str:
    return "".join(_BACK[t] if t in _BACK else chr(t - _OFFSET) for t in tokens)


def render(messages: list[dict[str, str]], add_assistant: bool = False) -> str:
    text = "".join(f"{START}{m['role']}\n{m['content']}{END}\n" for m in messages)
    return text + (f"{START}assistant\n" if add_assistant else "")


ENV_FULL = ("<role_environment>\nrole: ego\ncapabilities you may invoke now (2):\n"
            "  board_read(reader) - Read the blackboard.\n  history() - Raw events.\n"
            "environment: abc123\n</role_environment>\n\nThe declaration above is "
            "authoritative for this turn. To use a capability, emit exactly one "
            "block.\n")
ENV_REF = ("<role_environment unchanged: abc123>\nThe capability declaration given "
           "earlier in this conversation still applies, unchanged, for this turn.\n")


class Conversation:
    """Builds a session message by message and records where each turn sits."""

    def __init__(self, system: str = "You are a test mind. " * 8):
        self.tokens: list[int] = tokenize(render([{"role": "system", "content": system}]))
        self.turns: list[dict[str, Any]] = []
        self.declared = False

    def _add(self, text: str) -> None:
        self.tokens.extend(tokenize(text))

    def turn(self, question: str, *, calls: list[tuple[str, Any]] = (),
             reply: str = "An answer.", pad: int = 0, full_env: bool | None = None,
             open_prompt: bool = False) -> dict[str, Any]:
        start = len(self.tokens)
        env = ENV_FULL if (full_env if full_env is not None else not self.declared) else ENV_REF
        self.declared = True
        opening = (f"{env}<turn_input>\n{question}\n{'.' * pad}</turn_input>")
        self._add(render([{"role": "user", "content": opening}], add_assistant=True))
        for name, result in calls:
            self._add(f'<tool_call>{{"name": "{name}", "arguments": {{}}}}</tool_call>{END}\n')
            body = result if isinstance(result, str) else json.dumps(result)
            self._add(render([{"role": "user", "content":
                               f'<tool_result name="{name}">\n{body}\n</tool_result>\n'
                               "Continue."}], add_assistant=True))
        if open_prompt:
            pass                      # ends on an assistant header nothing was generated into
        else:
            self._add(f"{reply}{END}\n")
        t = {"start": start, "end": len(self.tokens)}
        self.turns.append(t)
        return t


class ChatInference:
    """Enough of the inference service for homeostasis, with real structure."""

    def __init__(self, *, capacity: int = 100000, sessions=None, tokens=None):
        self.capacity = capacity
        self.sessions = sessions or []
        self.tokens = tokens or {}
        self.calls: list[tuple[str, dict]] = []
        self.next_session = 0
        self.unreachable = False

    def call(self, method, **kw):
        self.calls.append((method, kw))
        if self.unreachable:
            raise ConnectionError("inference is down")
        if method == "context_report":
            used = sum(s["n_past"] for s in self.sessions)
            return {"pool_tokens_used": used, "pool_capacity": self.capacity,
                    "occupancy": used / self.capacity,
                    "sessions": [{**s, "budget_tokens": self.capacity}
                                 for s in self.sessions]}
        if method == "session_tokens":
            toks = self.tokens.get(kw["session_id"], [])
            return {"tokens": toks, "n_past": len(toks)}
        if method == "close_session":
            self.sessions = [s for s in self.sessions
                             if s["session_id"] != kw["session_id"]]
            return {"closed": kw["session_id"]}
        if method == "open_session":
            self.next_session += 1
            sid = f"sess_new_{self.next_session}"
            self.sessions.append({"session_id": sid, "role": kw["role"], "n_past": 0,
                                  "prefix_len": 0, "snapshot_id": None})
            return {"session_id": sid, "role": kw["role"], "seq_id": 9}
        if method == "restore_prefix":
            for s in self.sessions:
                if s["session_id"] == kw["session_id"]:
                    s["n_past"] = len(kw["tokens"])
            self.tokens[kw["session_id"]] = list(kw["tokens"])
            return {"n_past": len(kw["tokens"]), "kv_mode": "recomputed"}
        if method == "detokenize":
            return detokenize(list(kw["tokens"]), kw.get("special", False))
        if method == "tokenize":
            return tokenize(kw["text"], kw.get("parse_special", True))
        if method == "apply_chat_template":
            return render(kw["messages"], kw.get("add_assistant", False))
        if method == "capabilities":
            return {"model_generation": "gen_test", "kv_mode": "shared_prefix"}
        raise AssertionError(f"unexpected call {method}")

    def methods_called(self):
        return [m for m, _ in self.calls]


def record_turn(mind, role: str, session: str, span: dict[str, int], *,
                lineage: str, settled: bool = True) -> str:
    """A closed turn in the mailbox occupying `span`. Settled, or still owed."""
    from amoeba import mailbox

    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind="user_input", source="operator",
                                  summary="q", lineage=lineage, expects_answer=True),
        actor="test", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role=role, incarnation=1,
                                profile_ref=f"{role}@1", profile_sha256="p",
                                environment_sha256="e", environment_blob="eb"),
        actor=role, bump_version=False)
    mind.writer.apply(
        lambda m: mailbox.complete(
            m, mind, turn_id=turn["turn_id"],
            stop_reason="model_stop" if settled else "max_output_tokens",
            result={"answer": "done"} if settled else None,
            session_handle=session, token_start=span["start"],
            token_end=span["end"]),
        actor="harness", bump_version=False)
    return turn["turn_id"]


def messages_of(tokens: list[int]) -> list[str]:
    """The session as a list of decoded messages, for assertions."""
    text = detokenize(tokens)
    return [START + part for part in text.split(START) if part]
