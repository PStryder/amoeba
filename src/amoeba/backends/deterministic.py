"""A deterministic, clearly-labelled fake backend.

Its purpose is to exercise recovery, lifecycle, leasing, snapshot refcounting
and the MCP contract without a GPU. It produces reproducible text from a hash
of its input so tests can assert on exact strings.

It is NOT a language model and must never be presented as one:

* ``is_simulated`` is ``True`` and ``capabilities()["backend_kind"]`` is
  ``deterministic``.
* Every generated string begins with ``[SIMULATED]``.
* ``kv_mode`` is ``simulated``; it models fork semantics (isolation,
  refcounting, contamination) but performs no real cache work, and it says so.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..errors import CapabilityUnsupported, InvalidInput, ResourceExhausted
from ..ids import new_id
from .base import SIMULATED_BANNER
from .structure import (STRUCTURAL_FINISH, cut_at_structural,
                        structural_from_text)

# The shape of a rendered conversation here: one marker per message, and the
# roles this simulator ever writes. `structural()` turns these into ids.
CHAT_MARKER = "<|{role}|>"
CHAT_MARKERS = tuple(CHAT_MARKER.format(role=r)
                     for r in ("system", "user", "assistant", "tool"))
MARKER_RE = re.compile(r"(<\|[a-z_]+\|>)")
# Reserved ids, so a hashed word can never land on a marker's id. A real
# vocabulary gives its control tokens their own ids; without the same here,
# "is this token structural?" would be true of ordinary words by collision --
# which is how a 2400-word answer first ran into this.
MARKER_IDS = {m: i + 1 for i, m in enumerate(CHAT_MARKERS)}
FIRST_TEXT_ID = len(MARKER_IDS) + 1
from .llama_engine import GenerationResult, SessionState

SIM_PREFIX = "[SIMULATED]"


class DeterministicBackend:
    """Hash-driven stand-in for an inference service."""

    backend_kind = "deterministic"
    is_simulated = True

    def __init__(self, *, n_seq_max: int = 8, n_ctx: int = 8192,
                 latency_per_token: float = 0.0, vocab_size: int = 4096) -> None:
        self.n_seq_max = n_seq_max
        self.n_ctx = n_ctx
        self.latency_per_token = latency_per_token
        self.vocab_size = vocab_size
        self.model_generation = "gen_deterministic_v1"
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._free_seq: list[int] = []
        self._loaded = False
        self._used_cells = 0
        # The same rule as the real engine: a generation may not author the
        # shape of the conversation. This vocabulary is its own invention, so
        # the markers are tokenized to find their ids.
        self.structural_tokens = structural_from_text(
            CHAT_MARKERS, lambda text: self.tokenize(text))
        self._scripted: list[str] = []

    # -- lifecycle ------------------------------------------------------
    def load(self) -> dict[str, Any]:
        self._free_seq = list(range(self.n_seq_max))
        self._loaded = True
        return {
            "backend_kind": self.backend_kind,
            "is_simulated": True,
            "banner": SIMULATED_BANNER,
            "model_generation": self.model_generation,
            "n_ctx_total": self.n_ctx,
            "n_seq_max": self.n_seq_max,
        }

    def close(self) -> None:
        self._loaded = False
        self._sessions.clear()

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "is_simulated": True,
            "banner": SIMULATED_BANNER,
            "device": "none",
            "device_names": [],
            "weight_ownership": "none_no_weights_loaded",
            "sessions_share_weights": False,
            "kv_mode": "simulated",
            "n_kv_streams": 0,
            "concurrency_mode": "serialized",
            "continuous_batching": False,
            "physical_overlap_verified": False,
            "prefix_reuse_verified": False,
            "tool_calling": "harness_validated",
            "max_sessions": self.n_seq_max,
            "model_generation": self.model_generation,
        }

    # -- tokenisation ---------------------------------------------------
    def tokenize(self, text: str, *, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        # Deterministic, reversible-enough: one token per whitespace word --
        # and one per template marker, whatever it is glued to. A real
        # tokenizer gives a control token its own id; without that here, the
        # rule that generated tokens may not author structure would be
        # checking ids that depend on the text beside the marker.
        words: list[str] = []
        for part in MARKER_RE.split(text):
            if not part:
                continue
            words.extend([part] if MARKER_RE.fullmatch(part) else part.split())
        span = max(1, self.vocab_size - FIRST_TEXT_ID)
        return [MARKER_IDS[w] if w in MARKER_IDS else
                int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) % span + FIRST_TEXT_ID
                for w in words] or [FIRST_TEXT_ID]

    def detokenize(self, tokens: Sequence[int], *, special: bool = False) -> str:
        return " ".join(f"t{t}" for t in tokens)

    def _generated(self, text: str) -> tuple[str, list[int], int | None]:
        """What a generation may keep: text and tokens cut at the same place.

        A generation may not author the shape of the conversation, so this
        stops at the first template marker. Both are cut, because text and
        token stream disagreeing about where messages begin is the very thing
        that hid this in the real engine.
        """
        tokens, forged = cut_at_structural(self.tokenize(text),
                                           self.structural_tokens)
        if forged is not None:
            cuts = [i for i in (text.find(m) for m in CHAT_MARKERS) if i >= 0]
            text = text[:min(cuts)] if cuts else text
        return text, tokens, forged

    def _piece_of(self, token: int | None) -> str:
        if token is None:
            return ""
        return next((m for m in CHAT_MARKERS if self.tokenize(m) == [token]), "")

    def structural(self) -> frozenset[int]:
        """The marker ids, by the same rule the real engine uses."""
        return structural_from_text(CHAT_MARKERS, lambda t: self.tokenize(t))

    def apply_chat_template(self, messages: Sequence[dict[str, str]], *,
                            add_assistant: bool = True) -> str:
        parts = [f"{CHAT_MARKER.format(role=m['role'])}{m['content']}"
                 for m in messages]
        if add_assistant:
            parts.append(CHAT_MARKER.format(role="assistant"))
        return "\n".join(parts)

    def chat_template(self) -> str | None:
        return "deterministic-stub"

    def is_eog(self, token: int) -> bool:
        return False

    # -- sessions -------------------------------------------------------
    def open_session(self, *, role: str, session_id: str | None = None,
                     seq_id: int | None = None,
                     context_budget_tokens: int | None = None,
                     budget_basis: str = "total") -> SessionState:
        with self._lock:
            if seq_id is None:
                if not self._free_seq:
                    raise ResourceExhausted("no free simulated sequence slots",
                                            n_seq_max=self.n_seq_max)
                seq_id = self._free_seq.pop(0)
            elif seq_id in self._free_seq:
                self._free_seq.remove(seq_id)
            else:
                raise ResourceExhausted("sequence slot already in use", seq_id=seq_id)
            sess = SessionState(
                session_id=session_id or new_id("sess"), role=role, seq_id=seq_id,
                context_budget_tokens=context_budget_tokens,
                budget_basis=budget_basis,
                created_at=time.time(), last_used=time.time(),
            )
            self._sessions[sess.session_id] = sess
            return sess

    def close_session(self, session_id: str, *, keep_prefix: bool = False) -> None:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is None:
                return
            # Only the private tail is charged back: prefix cells are modelled
            # as shared and released by the last owner.
            self._used_cells = max(0, self._used_cells - (sess.n_past - sess.prefix_len))
            if sess.seq_id not in self._free_seq:
                self._free_seq.append(sess.seq_id)
                self._free_seq.sort()

    def get_session(self, session_id: str) -> SessionState:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise InvalidInput("unknown inference session", session_id=session_id)
        return sess

    def active_sessions(self) -> list[dict[str, Any]]:
        return [{"session_id": s.session_id, "role": s.role, "seq_id": s.seq_id,
                 "n_past": s.n_past, "prefix_len": s.prefix_len,
                 "shares_prefix": s.shares_prefix,
                 "private_tokens": s.private_tokens,
                 "charged_tokens": s.charged_tokens,
                 "budgeted_tokens": s.budgeted_tokens,
                 "context_budget_tokens": s.context_budget_tokens,
                 "budget_basis": s.budget_basis,
                 "snapshot_id": s.snapshot_id} for s in self._sessions.values()]

    def request_cancel(self, session_id: str) -> bool:
        sess = self._sessions.get(session_id)
        if sess is None:
            return False
        sess.cancel_requested = True
        return True

    def clear_cancel(self, session_id: str) -> None:
        sess = self._sessions.get(session_id)
        if sess is not None:
            sess.cancel_requested = False

    def reset_session(self, session_id: str) -> None:
        sess = self.get_session(session_id)
        sess.tokens.clear()
        sess.prefix_len = 0
        sess.snapshot_id = None

    # -- fork -----------------------------------------------------------
    def fork_prefix(self, *, src_session_id: str, prefix_len: int, role: str,
                    session_id: str | None = None,
                    snapshot_id: str | None = None,
                    context_budget_tokens: int | None = None,
                    budget_basis: str = "private_growth") -> SessionState:
        with self._lock:
            src = self.get_session(src_session_id)
            if prefix_len <= 0 or prefix_len > src.n_past:
                raise InvalidInput("prefix_len must be a valid prefix",
                                   prefix_len=prefix_len, source_n_past=src.n_past)
            dst = self.open_session(
                role=role, session_id=session_id,
                context_budget_tokens=context_budget_tokens,
                budget_basis=budget_basis)
            # A copy of the token list, so a neuocyte mutating its own list can
            # never be mistaken for it mutating the source.
            dst.tokens = list(src.tokens[:prefix_len])
            dst.prefix_len = prefix_len
            dst.shares_prefix = True
            dst.snapshot_id = snapshot_id
            return dst

    def restore_prefix(self, *, session_id: str, tokens: Sequence[int],
                       snapshot_id: str | None = None) -> int:
        sess = self.get_session(session_id)
        sess.tokens = list(tokens)
        sess.prefix_len = len(tokens)
        sess.shares_prefix = False      # recomputed, not shared
        sess.snapshot_id = snapshot_id
        return sess.n_past

    def seq_pos_max(self, session_id: str) -> int:
        return self.get_session(session_id).n_past - 1

    def state_seq_size(self, session_id: str) -> int:
        return self.get_session(session_id).n_past * 1024

    def vram_free(self) -> int:
        return 0

    def top_logits(self, session_id: str, k: int = 5) -> list[tuple[int, float]]:
        sess = self.get_session(session_id)
        digest = hashlib.sha256(repr(sess.tokens).encode()).digest()
        return [(digest[i] * 7 % self.vocab_size, -float(i)) for i in range(k)]

    # -- generation ------------------------------------------------------
    def ingest(self, session_id: str, tokens: Sequence[int], *,
               compute_logits: bool = True) -> int:
        with self._lock:
            sess = self.get_session(session_id)
            tokens = list(tokens)
            if self._used_cells + len(tokens) > self.n_ctx:
                raise ResourceExhausted("simulated context full",
                                        used=self._used_cells, n_ctx=self.n_ctx)
            sess.tokens.extend(tokens)
            self._used_cells += len(tokens)
            sess.last_used = time.time()
            return sess.n_past

    def generate(self, session_id: str, *, max_tokens: int = 64,
                 temperature: float = 0.0, seed: int = 1234,
                 stop_strings: Sequence[str] = (), deadline: float | None = None,
                 on_token: Callable[[str], None] | None = None,
                 **_: Any) -> GenerationResult:
        """Produce reproducible text derived from the session's exact context.

        Two sessions with identical contexts produce identical output; any
        cross-session contamination therefore shows up immediately as a changed
        string, which is what makes this useful for isolation tests.

        A scripted reply may carry `delay_seconds`: a generation that takes a
        while, so a test can hold a turn open past the supervisor's probe
        grace. It sleeps outside the lock, as a real engine decoding would --
        the service stays answerable while one generation takes its time.
        """
        delay = 0.0
        with self._lock:
            role = self._sessions[session_id].role if session_id in self._sessions else None
            at = self._next_scripted(role)
            head = self._scripted[at] if at is not None else None
            if isinstance(head, dict):
                delay = float(head.get("delay_seconds") or 0.0)
        if delay > 0:
            time.sleep(delay)
        with self._lock:
            sess = self.get_session(session_id)
            if sess.cancel_requested:
                sess.cancel_requested = False
                return GenerationResult(
                    session_id=session_id, text="", tokens=[],
                    finish_reason="cancelled", prompt_tokens=sess.n_past,
                    completion_tokens=0, time_to_first_token=0.0,
                    total_seconds=0.0)
            t0 = time.perf_counter()
            at = self._next_scripted(sess.role)
            if at is not None:
                # Still labelled as simulated: a scripted reply is no more
                # model inference than a hashed one, and the label is what
                # stops either being mistaken for it.
                entry = self._scripted.pop(at)
                if isinstance(entry, dict):
                    # A scripted *truncation*: the only way a test can make a
                    # known answer span several bounded turns and then check
                    # that nothing between them went missing. `continues`
                    # marks the tail of a generation that was cut off, which a
                    # real model would not prefix with anything either.
                    finish = str(entry.get("finish_reason") or "scripted")
                    body = str(entry.get("text") or "")
                    text = body if entry.get("continues") else f"{SIM_PREFIX} {body}"
                else:
                    finish, text = "scripted", f"{SIM_PREFIX} {entry}"
                text, out_tokens, forged = self._generated(text)
                if forged is not None:
                    finish = STRUCTURAL_FINISH
                sess.tokens.extend(out_tokens)
                self._used_cells += len(out_tokens)
                sess.tokens_generated += len(out_tokens)
                elapsed = time.perf_counter() - t0
                if on_token is not None:
                    on_token(text)
                return GenerationResult(
                    structural_attempt=forged,
                    structural_piece=self._piece_of(forged),
                    session_id=session_id, text=text, tokens=out_tokens,
                    finish_reason=finish,
                    prompt_tokens=sess.n_past - len(out_tokens),
                    completion_tokens=len(out_tokens),
                    time_to_first_token=elapsed, total_seconds=elapsed)
            seed_material = f"{sess.role}|{seed}|{','.join(map(str, sess.tokens))}"
            digest = hashlib.sha256(seed_material.encode()).hexdigest()
            # The simulated mind has sixteen words to say. It is cut off only
            # when the allowance is smaller than that, and says so only then.
            # This used to report "length" unconditionally -- a false claim of
            # truncation on every turn, harmless while truncation changed
            # nothing and wrong once an interaction that ends on a truncation
            # is reported as incomplete rather than answered.
            natural = 16
            n_words = max(1, min(max_tokens // 4, natural))
            finish = "length" if max_tokens // 4 < natural else "stop"
            words = [digest[i * 4:(i + 1) * 4] for i in range(n_words)]
            text = f"{SIM_PREFIX} {sess.role}:{' '.join(words)}"
            if self.latency_per_token:
                time.sleep(self.latency_per_token * n_words)
            text, out_tokens, forged = self._generated(text)
            if forged is not None:
                finish = STRUCTURAL_FINISH
            sess.tokens.extend(out_tokens)
            self._used_cells += len(out_tokens)
            sess.tokens_generated += len(out_tokens)
            elapsed = time.perf_counter() - t0
            if on_token is not None:
                on_token(text)
            return GenerationResult(
                structural_attempt=forged, structural_piece=self._piece_of(forged),
                session_id=session_id, text=text, tokens=out_tokens,
                finish_reason=finish, prompt_tokens=sess.n_past - len(out_tokens),
                completion_tokens=len(out_tokens),
                time_to_first_token=elapsed, total_seconds=elapsed,
                first_token_logprob_top=self.top_logits(session_id, 3),
            )

    def _next_scripted(self, role: str | None) -> int | None:
        """The next scripted reply this session may take, if any.

        A reply may name the role it is for. Without that, one global queue
        let whichever role generated next take the next reply -- so a test
        in which Ego's action wakes Id raced two minds for one script.
        """
        for i, entry in enumerate(self._scripted):
            wanted = entry.get("role") if isinstance(entry, dict) else None
            if wanted is None or wanted == role:
                return i
        return None

    def script_responses(self, responses: Sequence[str]) -> int:
        """Queue exact replies for the next generations.

        Only the deterministic backend has this, and the inference service
        exposes it only if the backend does -- so it does not exist when a real
        model is loaded. It is here because the tool-execution loop cannot be
        exercised end to end otherwise: hashed text never contains a tool call,
        so the loop would be tested only against inputs that never take its
        interesting branch.
        """
        with self._lock:
            self._scripted = list(responses)
            return len(self._scripted)

    def generate_batched(self, requests: Sequence[dict[str, Any]], **kwargs: Any
                         ) -> dict[str, GenerationResult]:
        """Generate for several sessions, one after another.

        **This models no batching and measures nothing.** There is no fused
        kernel here and no shared decode step; it is a loop, and any timing
        taken from it is the timing of a loop. It must never be cited as
        evidence about batched throughput.

        It exists because the alternative was worse. Refusing meant the
        batching *scheduler* -- how requests are grouped, in what order they
        are answered, what happens when one of them fails, whether the
        fallback works -- could not be exercised anywhere except on a machine
        with a GPU. A truthful stand-in that lets the scheduling logic be
        tested is better than an untested live path defended by a principled
        refusal.
        """
        out: dict[str, GenerationResult] = {}
        for req in requests:
            session_id = req["session_id"]
            params = {**kwargs, **{k: v for k, v in req.items()
                                   if k != "session_id"}}
            out[session_id] = self.generate(session_id, **params)
        return out
