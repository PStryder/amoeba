"""One GPU owner: a single resident model serving many isolated sequences.

Design notes that matter for honesty of the capability flags:

* **One weight set.** One ``llama_model`` is loaded once. Every session is a
  *sequence id* inside one ``llama_context``. Nothing duplicates weights.
* **Serial vs batched.** ``generate_serial`` runs one session to completion
  before the next; ``generate_batched`` puts one token per active session into
  a single ``llama_decode`` call, which is continuous batching -- a single
  fused kernel over several sequences. Neither is independent kernel overlap,
  and this module never claims it is.
* **Shared vs copied prefix.** With ``kv_unified=True`` the context has one KV
  stream, and ``llama_memory_seq_cp`` only edits per-cell sequence bitsets --
  physically shared storage, reclaimed when the last owning sequence drops the
  cell. With ``kv_unified=False`` the same call copies buffers. The flag is
  configurable so the two can be measured against each other.

The engine is deliberately single-threaded: one lock serialises all calls into
llama.cpp. llama.cpp contexts are not thread safe, and pretending otherwise
would produce exactly the kind of fake concurrency this project is supposed to
avoid.
"""

from __future__ import annotations

import ctypes
import random
import threading
import time
from ctypes import POINTER, c_char, c_float
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..errors import BackendUnavailable, CapabilityUnsupported, InvalidInput, ResourceExhausted
from ..ids import new_id, sha256_hex
from . import llama_ffi as F


# llama_backend_init()/llama_backend_free() act on PROCESS-GLOBAL state, not on
# a single context. A second engine calling free() on close would tear down
# state the first engine is still using, and the first engine's next decode
# would fault. Reference-count instead, so the global teardown happens only
# when the last engine in this process goes away.
_BACKEND_LOCK = threading.Lock()
_BACKEND_REFS = 0


def _backend_acquire(ffi: F.LlamaFFI) -> None:
    global _BACKEND_REFS
    with _BACKEND_LOCK:
        if _BACKEND_REFS == 0:
            ffi.lib.llama_backend_init()
        _BACKEND_REFS += 1


def _backend_release(ffi: F.LlamaFFI) -> None:
    global _BACKEND_REFS
    with _BACKEND_LOCK:
        _BACKEND_REFS = max(0, _BACKEND_REFS - 1)
        if _BACKEND_REFS == 0:
            ffi.lib.llama_backend_free()


@dataclass(slots=True)
class SessionState:
    session_id: str
    role: str
    seq_id: int
    tokens: list[int] = field(default_factory=list)
    prefix_len: int = 0
    snapshot_id: str | None = None
    ref_id: str | None = None
    created_at: float = 0.0
    last_used: float = 0.0
    tokens_generated: int = 0
    # Private copy of the logits row produced by this session's last decode.
    # llama_get_logits_ith() reads a buffer belonging to the CONTEXT, not to a
    # sequence: after any other session decodes, index -1 refers to that other
    # session. Sampling from it would silently mix sessions, so each session
    # keeps its own copy taken immediately after its own decode.
    logits: Any = None

    @property
    def n_past(self) -> int:
        return len(self.tokens)


@dataclass(slots=True)
class GenerationResult:
    session_id: str
    text: str
    tokens: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token: float
    total_seconds: float
    first_token_logprob_top: list[tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class LlamaEngine:
    """Owns the process-wide llama.cpp model and context."""

    backend_kind = "llama_cpp"
    is_simulated = False

    def __init__(
        self,
        *,
        runtime_dir: str | Path,
        model_path: str | Path,
        n_ctx: int = 16384,
        n_seq_max: int = 8,
        n_batch: int = 1024,
        n_ubatch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        kv_unified: bool = True,
        flash_attn: bool = True,
        type_k: str = "f16",
        type_v: str = "f16",
        log_sink: Callable[[int, str], None] | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.model_path = Path(model_path)
        self.n_ctx = int(n_ctx)
        self.n_seq_max = int(n_seq_max)
        self.n_batch = int(n_batch)
        self.n_ubatch = int(n_ubatch)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_threads = int(n_threads)
        self.kv_unified = bool(kv_unified)
        self.flash_attn = bool(flash_attn)
        self.type_k = type_k
        self.type_v = type_v
        self.log_sink = log_sink

        self.ggml: F.GgmlFFI | None = None
        self.ffi: F.LlamaFFI | None = None
        self.model: int | None = None
        self.ctx: int | None = None
        self.vocab: int | None = None
        self.mem: int | None = None
        self.n_vocab = 0
        self.model_generation = ""
        self.abi_info: dict[str, Any] = {}
        self.devices: list[dict[str, Any]] = []
        self.load_report: dict[str, Any] = {}

        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._free_seq: list[int] = []
        self._batch: Any = None
        self._batch_capacity = 0
        self._loaded = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def load(self) -> dict[str, Any]:
        if self._loaded:
            return self.load_report
        if not self.model_path.exists():
            raise BackendUnavailable("model file not found", path=str(self.model_path))

        t0 = time.perf_counter()
        self.ggml = F.GgmlFFI(self.runtime_dir)
        self.ggml.load_backends()
        self.ffi = F.LlamaFFI(self.runtime_dir / "llama.dll")
        # Installing a log sink is process-global; the sink must outlive this
        # engine, so silence_logs() registers it permanently.
        self.ffi.silence_logs(sink=self.log_sink)
        self.abi_info = self.ffi.validate_abi()
        _backend_acquire(self.ffi)
        self.devices = self.ggml.devices()
        gpus = self.ggml.gpu_devices()
        vram_before = self.ggml.device_memory(0)[0] if gpus else 0

        mparams = self.ffi.lib.llama_model_default_params()
        mparams.n_gpu_layers = self.n_gpu_layers
        mparams.use_extra_bufts = True
        model = self.ffi.lib.llama_model_load_from_file(
            str(self.model_path).encode("utf-8"), mparams
        )
        if not model:
            raise BackendUnavailable("llama_model_load_from_file returned NULL",
                                     path=str(self.model_path))
        self.model = model

        cparams = self.ffi.lib.llama_context_default_params()
        cparams.n_ctx = self.n_ctx
        cparams.n_batch = self.n_batch
        cparams.n_ubatch = self.n_ubatch
        cparams.n_seq_max = self.n_seq_max
        cparams.n_threads = self.n_threads
        cparams.n_threads_batch = self.n_threads
        cparams.kv_unified = self.kv_unified
        cparams.flash_attn_type = (
            F.LLAMA_FLASH_ATTN_TYPE_ENABLED if self.flash_attn else F.LLAMA_FLASH_ATTN_TYPE_DISABLED
        )
        cparams.type_k = F.GGML_TYPES[self.type_k]
        cparams.type_v = F.GGML_TYPES[self.type_v]
        cparams.no_perf = True
        # n_outputs_max_per_seq defaults to 1; a batched step asks for one
        # logit row per active sequence, so raise the per-ubatch ceiling.
        cparams.n_outputs_max = self.n_seq_max

        ctx = self.ffi.lib.llama_init_from_model(model, cparams)
        if not ctx:
            self.ffi.lib.llama_model_free(model)
            self.model = None
            raise BackendUnavailable("llama_init_from_model returned NULL")
        self.ctx = ctx
        self.vocab = self.ffi.lib.llama_model_get_vocab(model)
        self.n_vocab = int(self.ffi.lib.llama_vocab_n_tokens(self.vocab))
        self.mem = self.ffi.lib.llama_get_memory(ctx)

        vram_after = self.ggml.device_memory(0)[0] if gpus else 0
        self._free_seq = list(range(self.n_seq_max))
        self.model_generation = self._compute_model_generation()
        self._loaded = True

        self.load_report = {
            "model_path": str(self.model_path),
            "model_generation": self.model_generation,
            "model_desc": self._model_desc(),
            "model_size_bytes": int(self.ffi.lib.llama_model_size(model)),
            "model_n_params": int(self.ffi.lib.llama_model_n_params(model)),
            "n_ctx_total": int(self.ffi.lib.llama_n_ctx(ctx)),
            "n_ctx_per_seq": int(self.ffi.lib.llama_n_ctx_seq(ctx)),
            "n_ctx_train": int(self.ffi.lib.llama_model_n_ctx_train(model)),
            "n_seq_max": self.n_seq_max,
            "n_vocab": self.n_vocab,
            "kv_unified": self.kv_unified,
            "n_kv_streams": 1 if self.kv_unified else self.n_seq_max,
            "gpu_offload_supported": bool(self.ffi.lib.llama_supports_gpu_offload()),
            "devices": self.devices,
            "vram_free_before_load": vram_before,
            "vram_free_after_load": vram_after,
            "vram_consumed_by_load": max(0, vram_before - vram_after),
            "load_seconds": time.perf_counter() - t0,
            "abi": self.abi_info,
        }
        return self.load_report

    def _model_desc(self) -> str:
        buf = ctypes.create_string_buffer(512)
        n = self.ffi.lib.llama_model_desc(self.model, buf, 512)
        return buf.value.decode("utf-8", "replace") if n > 0 else ""

    def _compute_model_generation(self) -> str:
        """A generation identity covering everything that makes cached KV valid.

        Weights, tokenizer (vocab size), positional/context configuration and
        cache precision all participate: if any of them changes, previously
        published KV is not reinterpretable and must be recomputed.
        """
        st = self.model_path.stat()
        parts = [
            F.BUILD_TAG,
            self.model_path.name,
            str(st.st_size),
            str(int(self.ffi.lib.llama_model_n_params(self.model))),
            str(self.n_vocab),
            str(int(self.ffi.lib.llama_model_n_ctx_train(self.model))),
            f"ctx={self.n_ctx}", f"seqmax={self.n_seq_max}",
            f"k={self.type_k}", f"v={self.type_v}",
            f"fa={int(self.flash_attn)}", f"unified={int(self.kv_unified)}",
        ]
        return "gen_" + sha256_hex("|".join(parts).encode())[:24]

    def close(self) -> None:
        with self._lock:
            if self._batch is not None:
                self.ffi.lib.llama_batch_free(self._batch)
                self._batch = None
                self._batch_capacity = 0
            if self.ctx:
                self.ffi.lib.llama_free(self.ctx)
                self.ctx = None
            if self.model:
                self.ffi.lib.llama_model_free(self.model)
                self.model = None
            if self.ffi is not None and self._loaded:
                _backend_release(self.ffi)
            self._loaded = False
            self._sessions.clear()

    def _require(self) -> None:
        if not self._loaded:
            raise BackendUnavailable("inference engine not loaded")

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        """Capability flags. Every value here is either measured at load time
        or a structural property of the configuration -- never aspirational."""
        gpu = [d for d in self.devices if d["type"] == F.GGML_BACKEND_DEVICE_TYPE_GPU]
        return {
            "backend_kind": self.backend_kind,
            "is_simulated": False,
            "build": F.BUILD_TAG,
            "device": "cuda" if gpu and self.n_gpu_layers != 0 else "cpu",
            "device_names": [d["description"] for d in gpu],
            "weight_ownership": "single_resident_set",
            "sessions_share_weights": True,
            "kv_mode": "shared_prefix" if self.kv_unified else "copied_prefix",
            "n_kv_streams": 1 if self.kv_unified else self.n_seq_max,
            "concurrency_mode": "serialized_or_batched",
            "continuous_batching": True,
            # Deliberately false until a GPU timeline profile proves otherwise.
            "physical_overlap_verified": False,
            "prefix_reuse_verified": False,
            "tool_calling": "harness_validated",
            "max_sessions": self.n_seq_max,
            "model_generation": self.model_generation,
        }

    # ------------------------------------------------------------------
    # tokenisation
    # ------------------------------------------------------------------
    def tokenize(self, text: str, *, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        self._require()
        raw = text.encode("utf-8")
        cap = len(raw) + 64
        buf = (F.llama_token * cap)()
        n = self.ffi.lib.llama_tokenize(
            self.vocab, raw, len(raw), buf, cap, add_special, parse_special
        )
        if n < 0:
            cap = -n + 16
            buf = (F.llama_token * cap)()
            n = self.ffi.lib.llama_tokenize(
                self.vocab, raw, len(raw), buf, cap, add_special, parse_special
            )
            if n < 0:
                raise InvalidInput("tokenization failed", needed=-n)
        return [int(buf[i]) for i in range(n)]

    def token_to_piece(self, token: int, *, special: bool = True) -> str:
        buf = ctypes.create_string_buffer(256)
        n = self.ffi.lib.llama_token_to_piece(self.vocab, token, buf, 256, 0, special)
        if n < 0:
            buf = ctypes.create_string_buffer(-n + 1)
            n = self.ffi.lib.llama_token_to_piece(self.vocab, token, buf, -n, 0, special)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    def detokenize(self, tokens: Sequence[int], *, special: bool = False) -> str:
        if not tokens:
            return ""
        arr = (F.llama_token * len(tokens))(*tokens)
        cap = 4 * len(tokens) + 256
        buf = ctypes.create_string_buffer(cap)
        n = self.ffi.lib.llama_detokenize(self.vocab, arr, len(tokens), buf, cap, False, special)
        if n < 0:
            cap = -n + 16
            buf = ctypes.create_string_buffer(cap)
            n = self.ffi.lib.llama_detokenize(self.vocab, arr, len(tokens), buf, cap, False, special)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    def chat_template(self) -> str | None:
        raw = self.ffi.lib.llama_model_chat_template(self.model, None)
        return raw.decode("utf-8", "replace") if raw else None

    def apply_chat_template(self, messages: Sequence[dict[str, str]], *,
                            add_assistant: bool = True) -> str:
        """Render messages with the model's own chat template.

        Role separation has to reach the model: a wrapper that silently drops
        system messages cannot be relied on to isolate Ego from Id.
        """
        self._require()
        n = len(messages)
        arr = (F.llama_chat_message * n)()
        keep: list[bytes] = []
        for i, msg in enumerate(messages):
            r = msg["role"].encode("utf-8")
            c = msg["content"].encode("utf-8")
            keep.extend([r, c])
            arr[i].role = r
            arr[i].content = c
        size = sum(len(m["content"]) + len(m["role"]) for m in messages) * 2 + 1024
        buf = ctypes.create_string_buffer(size)
        got = self.ffi.lib.llama_chat_apply_template(None, arr, n, add_assistant, buf, size)
        if got > size:
            buf = ctypes.create_string_buffer(got + 1)
            got = self.ffi.lib.llama_chat_apply_template(None, arr, n, add_assistant, buf, got + 1)
        if got < 0:
            raise BackendUnavailable("model has no usable built-in chat template")
        return buf.raw[:got].decode("utf-8", "replace")

    def is_eog(self, token: int) -> bool:
        return bool(self.ffi.lib.llama_vocab_is_eog(self.vocab, token))

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------
    def open_session(self, *, role: str, session_id: str | None = None,
                     seq_id: int | None = None) -> SessionState:
        self._require()
        with self._lock:
            if seq_id is None:
                if not self._free_seq:
                    raise ResourceExhausted(
                        "no free inference sequence slots",
                        n_seq_max=self.n_seq_max, active=len(self._sessions),
                    )
                seq_id = self._free_seq.pop(0)
            elif seq_id in self._free_seq:
                self._free_seq.remove(seq_id)
            else:
                raise ResourceExhausted("sequence slot already in use", seq_id=seq_id)
            # Make sure nothing from a previous tenant survives in this slot.
            self.ffi.lib.llama_memory_seq_rm(self.mem, seq_id, -1, -1)
            sess = SessionState(
                session_id=session_id or new_id("sess"),
                role=role, seq_id=seq_id,
                created_at=time.time(), last_used=time.time(),
            )
            self._sessions[sess.session_id] = sess
            return sess

    def close_session(self, session_id: str, *, keep_prefix: bool = False) -> None:
        """Release a session's sequence slot.

        Dropping the sequence from every cell it owns is always the right
        thing, including for a fork: reference counting lives in each cell's
        sequence bitset, so a cell is only reclaimed once its last owner lets
        go. A worker retiring therefore frees exactly its private tail and
        leaves a shared prefix intact for Ego and its siblings.

        ``keep_prefix`` is accepted for call-site clarity and has no effect for
        that reason -- a partial removal would free nothing extra and would
        leave this sequence half-owning cells it can no longer address.
        """
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is None:
                return
            self.ffi.lib.llama_memory_seq_rm(self.mem, sess.seq_id, -1, -1)
            sess.logits = None
            if sess.seq_id not in self._free_seq:
                self._free_seq.append(sess.seq_id)
                self._free_seq.sort()

    def get_session(self, session_id: str) -> SessionState:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise InvalidInput("unknown inference session", session_id=session_id)
        return sess

    def active_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id, "role": s.role, "seq_id": s.seq_id,
                "n_past": s.n_past, "prefix_len": s.prefix_len,
                "snapshot_id": s.snapshot_id, "tokens_generated": s.tokens_generated,
            }
            for s in self._sessions.values()
        ]

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            sess = self.get_session(session_id)
            self.ffi.lib.llama_memory_seq_rm(self.mem, sess.seq_id, -1, -1)
            sess.tokens.clear()
            sess.prefix_len = 0
            sess.snapshot_id = None
            sess.logits = None

    # ------------------------------------------------------------------
    # snapshot fork
    # ------------------------------------------------------------------
    def fork_prefix(
        self, *, src_session_id: str, prefix_len: int, role: str,
        session_id: str | None = None, snapshot_id: str | None = None,
    ) -> SessionState:
        """Give a new session a frozen copy-or-share of a prefix of another.

        With ``kv_unified=True`` this is metadata-only: both sequences now own
        the same KV cells and neither can mutate them, because each appends
        only at its own growing position. The source session keeps writing past
        ``prefix_len`` into cells the fork does not own.
        """
        self._require()
        with self._lock:
            src = self.get_session(src_session_id)
            if prefix_len <= 0 or prefix_len > src.n_past:
                raise InvalidInput(
                    "prefix_len must be a valid prefix of the source session",
                    prefix_len=prefix_len, source_n_past=src.n_past,
                )
            if not self.kv_unified:
                # MEASURED on b11057: with one KV stream per sequence, a
                # partial-prefix seq_cp trips
                #   GGML_ASSERT(is_full && "seq_cp() is only supported for full
                #   KV buffers")
                # which aborts the whole process rather than returning an error.
                # A full-sequence copy does work, but it physically duplicates
                # the prefix into the destination stream's own preallocated
                # capacity. Refuse rather than abort; callers fall back to
                # restore_prefix() recomputation.
                raise CapabilityUnsupported(
                    "partial prefix fork requires kv_unified=True; this context has "
                    "one KV stream per sequence, where a partial seq_cp aborts the "
                    "process and a full seq_cp physically copies the prefix",
                    kv_unified=False, n_kv_streams=self.n_seq_max,
                )
            dst = self.open_session(role=role, session_id=session_id)
            # p1 is exclusive: copy positions [0, prefix_len).
            self.ffi.lib.llama_memory_seq_cp(self.mem, src.seq_id, dst.seq_id, 0, prefix_len)
            dst.tokens = list(src.tokens[:prefix_len])
            dst.prefix_len = prefix_len
            dst.snapshot_id = snapshot_id
            # A fork of a FULL prefix inherits the source's head position, so
            # the source's last logits row is the correct continuation point.
            # A shorter fork ends mid-context and must be re-evaluated instead.
            dst.logits = (src.logits.copy()
                          if src.logits is not None and prefix_len == src.n_past
                          else None)
            return dst

    def restore_prefix(self, *, session_id: str, tokens: Sequence[int],
                       snapshot_id: str | None = None) -> int:
        """Fallback path: rebuild a prefix by *recomputing* its KV from the exact
        token sequence.

        This is semantically equivalent to a fork for the same model generation
        but costs a full prefill. It is what a restart, a backend restart or a
        model change falls back to. Note the distinction the design insists on:
        recomputing the recorded token prefix reproduces the context;
        *summarising* it would be different behaviour entirely.
        """
        with self._lock:
            sess = self.get_session(session_id)
            self.ffi.lib.llama_memory_seq_rm(self.mem, sess.seq_id, -1, -1)
            sess.tokens.clear()
            sess.prefix_len = 0
            self.ingest(session_id, list(tokens), compute_logits=False)
            sess.prefix_len = len(tokens)
            sess.snapshot_id = snapshot_id
            return sess.n_past

    def seq_pos_max(self, session_id: str) -> int:
        sess = self.get_session(session_id)
        return int(self.ffi.lib.llama_memory_seq_pos_max(self.mem, sess.seq_id))

    def state_seq_size(self, session_id: str) -> int:
        """Serialized size of a sequence's KV state.

        Careful: this is a *logical* size. A forked sequence reports the full
        prefix size even when the underlying cells are physically shared, so
        this number is not evidence about physical allocation.
        """
        sess = self.get_session(session_id)
        return int(self.ffi.lib.llama_state_seq_get_size(self.ctx, sess.seq_id))

    def vram_free(self) -> int:
        if self.ggml is None:
            return 0
        gpus = [d for d in self.devices if d["type"] == F.GGML_BACKEND_DEVICE_TYPE_GPU]
        if not gpus:
            return 0
        return self.ggml.device_memory(gpus[0]["index"])[0]

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    def _acquire_batch(self, capacity: int) -> Any:
        """One reusable ``llama_batch`` sized to ``n_batch``.

        NOTE: ``llama_batch_init`` mallocs a per-token ``seq_id`` sub-array and
        ``llama_batch_free`` frees each one by walking to a NULL sentinel.
        Overwriting ``batch.seq_id[i]`` with a caller-owned buffer therefore
        makes the library free memory it does not own -- a heap corruption that
        crashes the process. Always write *into* ``batch.seq_id[i][0]``.
        """
        if self._batch is None or self._batch_capacity < capacity:
            if self._batch is not None:
                self.ffi.lib.llama_batch_free(self._batch)
            cap = max(capacity, self.n_batch)
            self._batch = self.ffi.lib.llama_batch_init(cap, 0, 1)
            self._batch_capacity = cap
        return self._batch

    def _decode_chunk(self, sess: SessionState, tokens: Sequence[int], start_pos: int,
                      want_logits: bool) -> None:
        n = len(tokens)
        batch = self._acquire_batch(n)
        for i, tok in enumerate(tokens):
            batch.token[i] = tok
            batch.pos[i] = start_pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = sess.seq_id
            batch.logits[i] = 0
        batch.n_tokens = n
        if want_logits:
            batch.logits[n - 1] = 1
        rc = self.ffi.lib.llama_decode(self.ctx, batch)
        if rc == 1:
            raise ResourceExhausted(
                "no KV slot available for batch (context full)",
                seq_id=sess.seq_id, start_pos=start_pos, n_tokens=n,
            )
        if rc != 0:
            raise BackendUnavailable("llama_decode failed", rc=rc, seq_id=sess.seq_id)
        if want_logits:
            # llama_get_logits_ith() indexes by position IN THE BATCH, not by
            # output row: only the last token of this chunk requested logits,
            # so n-1 is the row to read.
            sess.logits = self._copy_logits(n - 1)
        else:
            # Anything previously cached is now stale for this session.
            sess.logits = None

    def _copy_logits(self, batch_index: int) -> "np.ndarray":
        """Copy one logits row out of the context-owned buffer.

        The copy is what makes the value safe to keep: the underlying buffer is
        overwritten by the next decode, whichever session that belongs to.
        """
        ptr = self._logits_at(batch_index)
        view = ctypes.cast(ptr, POINTER(c_float * self.n_vocab)).contents
        return np.frombuffer(view, dtype=np.float32, count=self.n_vocab).copy()

    def _session_logits(self, sess: SessionState) -> "np.ndarray":
        if sess.logits is None:
            raise BackendUnavailable(
                "this session has no current logits; ingest at least one token with "
                "compute_logits=True before sampling",
                session_id=sess.session_id,
            )
        return sess.logits

    def ingest(self, session_id: str, tokens: Sequence[int], *,
               compute_logits: bool = True) -> int:
        """Append tokens to a session's context and evaluate them."""
        self._require()
        with self._lock:
            sess = self.get_session(session_id)
            tokens = list(tokens)
            if not tokens:
                return sess.n_past
            pos = sess.n_past
            step = max(1, min(self.n_batch, self.n_ubatch))
            for i in range(0, len(tokens), step):
                chunk = tokens[i:i + step]
                last = (i + step) >= len(tokens)
                self._decode_chunk(sess, chunk, pos + i, compute_logits and last)
            sess.tokens.extend(tokens)
            sess.last_used = time.time()
            return sess.n_past

    def _logits_at(self, idx: int) -> Any:
        ptr = self.ffi.lib.llama_get_logits_ith(self.ctx, idx)
        if not ptr:
            raise BackendUnavailable("llama_get_logits_ith returned NULL", index=idx)
        return ptr

    def _sample(self, logits: "np.ndarray", *, temperature: float, top_p: float,
                top_k: int, rng: random.Random) -> int:
        """Greedy when temperature <= 0, else top-k/top-p nucleus sampling.

        Sampling lives here rather than in a llama.cpp sampler chain so that a
        forked prefix and an exact recomputation can be compared under
        identical, reproducible decisions.
        """
        if temperature <= 0.0:
            return int(np.argmax(logits))
        k = self.n_vocab if top_k <= 0 else min(top_k, self.n_vocab)
        idx = np.argpartition(-logits, k - 1)[:k]
        vals = logits[idx]
        order = np.argsort(-vals)
        idx, vals = idx[order], vals[order]
        probs = np.exp((vals - vals[0]) / temperature)
        probs /= probs.sum()
        if 0.0 < top_p < 1.0:
            cum = np.cumsum(probs)
            cutoff = int(np.searchsorted(cum, top_p) + 1)
            idx, probs = idx[:cutoff], probs[:cutoff]
            probs = probs / probs.sum()
        r = rng.random()
        pick = int(np.searchsorted(np.cumsum(probs), r))
        pick = min(pick, len(idx) - 1)
        return int(idx[pick])

    def top_logits(self, session_id: str, k: int = 5) -> list[tuple[int, float]]:
        """Top-k (token, logit) at the session's current head.

        Used to compare a forked prefix against an exact recomputation: the
        logits must match, not merely the sampled text.
        """
        with self._lock:
            sess = self.get_session(session_id)
            logits = self._session_logits(sess)
            idx = np.argsort(-logits)[:k]
            return [(int(i), float(logits[i])) for i in idx]

    def generate(
        self,
        session_id: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 1234,
        stop_strings: Sequence[str] = (),
        deadline: float | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        """Run one session to completion. Bounded by tokens and wall clock."""
        self._require()
        with self._lock:
            sess = self.get_session(session_id)
            rng = random.Random(seed)
            t0 = time.perf_counter()
            ttft = 0.0
            out_tokens: list[int] = []
            text = ""
            finish = "length"
            prompt_tokens = sess.n_past
            first_top: list[tuple[int, float]] = []

            for step in range(max_tokens):
                logits = self._session_logits(sess)
                if step == 0:
                    order = np.argsort(-logits)[:5]
                    first_top = [(int(i), float(logits[i])) for i in order]
                tok = self._sample(logits, temperature=temperature, top_p=top_p,
                                   top_k=top_k, rng=rng)
                if step == 0:
                    ttft = time.perf_counter() - t0
                if self.is_eog(tok):
                    finish = "stop_token"
                    break
                out_tokens.append(tok)
                piece = self.token_to_piece(tok, special=False)
                text += piece
                if on_token is not None:
                    on_token(piece)
                self._decode_chunk(sess, [tok], sess.n_past, True)
                sess.tokens.append(tok)
                sess.tokens_generated += 1
                if stop_strings and any(s in text for s in stop_strings):
                    finish = "stop_string"
                    for s in stop_strings:
                        idx = text.find(s)
                        if idx >= 0:
                            text = text[:idx]
                    break
                if deadline is not None and time.perf_counter() > deadline:
                    finish = "deadline"
                    break

            sess.last_used = time.time()
            return GenerationResult(
                session_id=session_id, text=text, tokens=out_tokens,
                finish_reason=finish, prompt_tokens=prompt_tokens,
                completion_tokens=len(out_tokens), time_to_first_token=ttft,
                total_seconds=time.perf_counter() - t0,
                first_token_logprob_top=first_top,
            )

    # ------------------------------------------------------------------
    # continuous batching
    # ------------------------------------------------------------------
    def generate_batched(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 1234,
        deadline: float | None = None,
    ) -> dict[str, GenerationResult]:
        """Decode several sessions together: one token per session per
        ``llama_decode``.

        This IS continuous batching -- one fused kernel launch covering several
        independent sequences. It is NOT independent overlapping execution of
        separate inference streams, and must never be reported as such.
        """
        self._require()
        with self._lock:
            t0 = time.perf_counter()
            active = []
            for req in requests:
                sess = self.get_session(req["session_id"])
                active.append({
                    "sess": sess,
                    "rng": random.Random(req.get("seed", seed)),
                    "temperature": req.get("temperature", temperature),
                    "max_tokens": req.get("max_tokens", max_tokens),
                    "stop_strings": req.get("stop_strings", ()),
                    "tokens": [], "text": "", "finish": "length",
                    "ttft": 0.0, "done": False, "prompt_tokens": sess.n_past,
                    "first_top": [],
                })
            step = 0
            while any(not a["done"] for a in active) and step < max_tokens:
                pending = [a for a in active if not a["done"]]
                sampled: list[tuple[dict[str, Any], int]] = []
                for a in pending:
                    logits = self._session_logits(a["sess"])
                    if step == 0:
                        order = np.argsort(-logits)[:5]
                        a["first_top"] = [(int(i), float(logits[i])) for i in order]
                        a["ttft"] = time.perf_counter() - t0
                    tok = self._sample(logits, temperature=a["temperature"], top_p=1.0,
                                       top_k=0, rng=a["rng"])
                    sampled.append((a, tok))

                batch_entries: list[tuple[dict[str, Any], int]] = []
                for a, tok in sampled:
                    if self.is_eog(tok) or len(a["tokens"]) >= a["max_tokens"]:
                        a["finish"] = "stop_token" if self.is_eog(tok) else "length"
                        a["done"] = True
                        continue
                    a["tokens"].append(tok)
                    piece = self.token_to_piece(tok, special=False)
                    a["text"] += piece
                    if a["stop_strings"] and any(s in a["text"] for s in a["stop_strings"]):
                        a["finish"] = "stop_string"
                        a["done"] = True
                        continue
                    batch_entries.append((a, tok))

                if not batch_entries:
                    break
                n = len(batch_entries)
                batch = self._acquire_batch(n)
                for i, (a, tok) in enumerate(batch_entries):
                    sess = a["sess"]
                    batch.token[i] = tok
                    batch.pos[i] = sess.n_past
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = sess.seq_id
                    batch.logits[i] = 1
                    sess.tokens.append(tok)
                    sess.tokens_generated += 1
                batch.n_tokens = n
                rc = self.ffi.lib.llama_decode(self.ctx, batch)
                if rc == 1:
                    raise ResourceExhausted("no KV slot for batched step", n=n)
                if rc != 0:
                    raise BackendUnavailable("llama_decode failed in batch", rc=rc)
                # One fused decode produced n logit rows, one per sequence, in
                # batch order. Each is copied to its own session.
                for i, (a, _tok) in enumerate(batch_entries):
                    a["sess"].logits = self._copy_logits(i)
                step += 1
                if deadline is not None and time.perf_counter() > deadline:
                    for a in active:
                        if not a["done"]:
                            a["finish"] = "deadline"
                            a["done"] = True

            total = time.perf_counter() - t0
            out: dict[str, GenerationResult] = {}
            for a in active:
                sid = a["sess"].session_id
                out[sid] = GenerationResult(
                    session_id=sid, text=a["text"], tokens=a["tokens"],
                    finish_reason=a["finish"], prompt_tokens=a["prompt_tokens"],
                    completion_tokens=len(a["tokens"]), time_to_first_token=a["ttft"],
                    total_seconds=total, first_token_logprob_top=a["first_top"],
                )
            return out
