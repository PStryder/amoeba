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
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..errors import CapabilityUnsupported, InvalidInput, ResourceExhausted
from ..ids import new_id
from .base import SIMULATED_BANNER
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
        # Deterministic, reversible-enough: one token per whitespace word.
        words = text.split()
        return [int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) % self.vocab_size
                for w in words] or [0]

    def detokenize(self, tokens: Sequence[int], *, special: bool = False) -> str:
        return " ".join(f"t{t}" for t in tokens)

    def apply_chat_template(self, messages: Sequence[dict[str, str]], *,
                            add_assistant: bool = True) -> str:
        parts = [f"<|{m['role']}|>{m['content']}" for m in messages]
        if add_assistant:
            parts.append("<|assistant|>")
        return "\n".join(parts)

    def chat_template(self) -> str | None:
        return "deterministic-stub"

    def is_eog(self, token: int) -> bool:
        return False

    # -- sessions -------------------------------------------------------
    def open_session(self, *, role: str, session_id: str | None = None,
                     seq_id: int | None = None) -> SessionState:
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
                    snapshot_id: str | None = None) -> SessionState:
        with self._lock:
            src = self.get_session(src_session_id)
            if prefix_len <= 0 or prefix_len > src.n_past:
                raise InvalidInput("prefix_len must be a valid prefix",
                                   prefix_len=prefix_len, source_n_past=src.n_past)
            dst = self.open_session(role=role, session_id=session_id)
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
        """
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
            if self._scripted:
                # Still labelled as simulated: a scripted reply is no more
                # model inference than a hashed one, and the label is what
                # stops either being mistaken for it.
                text = f"{SIM_PREFIX} {self._scripted.pop(0)}"
                out_tokens = self.tokenize(text)
                sess.tokens.extend(out_tokens)
                self._used_cells += len(out_tokens)
                sess.tokens_generated += len(out_tokens)
                elapsed = time.perf_counter() - t0
                if on_token is not None:
                    on_token(text)
                return GenerationResult(
                    session_id=session_id, text=text, tokens=out_tokens,
                    finish_reason="scripted",
                    prompt_tokens=sess.n_past - len(out_tokens),
                    completion_tokens=len(out_tokens),
                    time_to_first_token=elapsed, total_seconds=elapsed)
            seed_material = f"{sess.role}|{seed}|{','.join(map(str, sess.tokens))}"
            digest = hashlib.sha256(seed_material.encode()).hexdigest()
            n_words = max(1, min(max_tokens // 4, 16))
            words = [digest[i * 4:(i + 1) * 4] for i in range(n_words)]
            text = f"{SIM_PREFIX} {sess.role}:{' '.join(words)}"
            if self.latency_per_token:
                time.sleep(self.latency_per_token * n_words)
            out_tokens = self.tokenize(text)
            sess.tokens.extend(out_tokens)
            self._used_cells += len(out_tokens)
            sess.tokens_generated += len(out_tokens)
            elapsed = time.perf_counter() - t0
            if on_token is not None:
                on_token(text)
            return GenerationResult(
                session_id=session_id, text=text, tokens=out_tokens,
                finish_reason="length", prompt_tokens=sess.n_past - len(out_tokens),
                completion_tokens=len(out_tokens),
                time_to_first_token=elapsed, total_seconds=elapsed,
                first_token_logprob_top=self.top_logits(session_id, 3),
            )

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
        raise CapabilityUnsupported(
            "the deterministic backend does not model batching; it would not "
            "measure anything real"
        )
