"""Hand-written ctypes binding for llama.cpp on native Windows.

Written against the exact header of build **b11057** (``include/llama.h`` from
that tag, vendored beside the DLLs). The struct layouts below must match that
build bit-for-bit, so :func:`validate_abi` compares the values returned by
``llama_*_default_params()`` against the constants in that build's source. A
mismatch means the DLL is a different build and the binding must not be used.

Two facts from that build's ``src/llama-kv-cache.cpp`` drive the whole snapshot
design:

* ``n_stream = unified ? 1 : n_seq_max``
* ``llama_kv_cache::seq_cp`` short-circuits when source and destination are in
  the *same stream*: "no data copy is necessary - we just have to update the
  cells meta data". Cross-stream copies instead "require to copy the actual
  buffer data".

So with ``kv_unified = True`` a sequence copy is a *physically shared* prefix
(one set of KV cells, a bitset of owning sequences per cell, storage reclaimed
only when the last owner drops it), and with ``kv_unified = False`` the same
call is a *full physical copy*. Both are exposed so the difference can be
measured rather than asserted.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    CFUNCTYPE, POINTER, Structure, byref, c_bool, c_char, c_char_p, c_float,
    c_int32, c_int8, c_size_t, c_uint32, c_void_p,
)
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# scalar aliases
# ---------------------------------------------------------------------------
llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32
llama_memory_t = c_void_p

BUILD_TAG = "b11057"

# ggml_type
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30
GGML_TYPES = {"f32": GGML_TYPE_F32, "f16": GGML_TYPE_F16,
              "q8_0": GGML_TYPE_Q8_0, "bf16": GGML_TYPE_BF16}

LLAMA_FLASH_ATTN_TYPE_AUTO = -1
LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
LLAMA_FLASH_ATTN_TYPE_ENABLED = 1

LLAMA_SPLIT_MODE_NONE = 0
LLAMA_SPLIT_MODE_LAYER = 1

LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_UNSPECIFIED = -1
LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
LLAMA_CONTEXT_TYPE_DEFAULT = 0
LLAMA_LOAD_MODE_AUTO = -1
LLAMA_LAZY_MODE_OFF = 0

GGML_LOG_LEVEL_NONE = 0
GGML_LOG_LEVEL_DEBUG = 1
GGML_LOG_LEVEL_INFO = 2
GGML_LOG_LEVEL_WARN = 3
GGML_LOG_LEVEL_ERROR = 4


# ---------------------------------------------------------------------------
# structs
# ---------------------------------------------------------------------------
class llama_batch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(llama_pos)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(llama_seq_id))),
        ("logits", POINTER(c_int8)),
    ]


class llama_model_params(Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int32),
        ("load_mode", c_int32),
        ("lazy_mode", c_int32),
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
        ("load_mtp", c_bool),
    ]


class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_outputs_max_per_seq", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_int32),
        ("rope_scaling_type", c_int32),
        ("pooling_type", c_int32),
        ("attention_type", c_int32),
        ("flash_attn_type", c_int32),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int32),
        ("type_v", c_int32),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


class llama_chat_message(Structure):
    _fields_ = [("role", c_char_p), ("content", c_char_p)]


GGML_LOG_CALLBACK = CFUNCTYPE(None, c_int32, c_char_p, c_void_p)

# llama_log_set() stores a raw function pointer with no ownership. If the only
# Python reference to the trampoline is an attribute of an object that later
# goes away, the pointer dangles and the next log line faults the process. Keep
# every installed callback alive for the life of the process.
_LIVE_LOG_CALLBACKS: list[Any] = []

# Values from b11057 ``llama_context_default_params()`` / ``llama_model_default_params()``.
_EXPECTED_CONTEXT_DEFAULTS = {
    "n_ctx": 512, "n_batch": 2048, "n_ubatch": 512, "n_seq_max": 1,
    "n_rs_seq": 0, "n_outputs_max": 0, "n_outputs_max_per_seq": 1,
    "ctx_type": LLAMA_CONTEXT_TYPE_DEFAULT,
    "rope_scaling_type": LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
    "pooling_type": LLAMA_POOLING_TYPE_UNSPECIFIED,
    "attention_type": LLAMA_ATTENTION_TYPE_UNSPECIFIED,
    "flash_attn_type": LLAMA_FLASH_ATTN_TYPE_AUTO,
    "rope_freq_base": 0.0, "rope_freq_scale": 0.0,
    "yarn_ext_factor": -1.0, "yarn_attn_factor": -1.0,
    "yarn_beta_fast": -1.0, "yarn_beta_slow": -1.0,
    "yarn_orig_ctx": 0, "defrag_thold": -1.0,
    "type_k": GGML_TYPE_F16, "type_v": GGML_TYPE_F16,
    "embeddings": False, "offload_kqv": True, "no_perf": True,
    "op_offload": True, "swa_full": True, "kv_unified": False,
    "n_samplers": 0,
}
_EXPECTED_MODEL_DEFAULTS = {
    "split_mode": LLAMA_SPLIT_MODE_LAYER,
    "main_gpu": 0,
    "vocab_only": False,
    "check_tensors": False,
    "load_mtp": False,
}


class AbiMismatch(RuntimeError):
    pass


GGML_BACKEND_DEVICE_TYPE_CPU = 0
GGML_BACKEND_DEVICE_TYPE_GPU = 1
GGML_BACKEND_DEVICE_TYPE_ACCEL = 2


class GgmlFFI:
    """The ggml backend registry.

    ``llama_backend_init()`` does NOT discover the CUDA backend: the dynamic
    backend DLLs are found by ``ggml_backend_load_all_from_path``, which the
    upstream example binaries call from ``common_init()``. Without it the
    library loads, reports ``llama_supports_gpu_offload() == false`` and runs
    entirely on CPU -- a silent and very expensive failure mode. We therefore
    always load backends from the explicit runtime directory and assert that a
    GPU device appeared before claiming a GPU capability.

    ``ggml_backend_dev_memory`` is also the only whole-device VRAM accounting
    available here: under WDDM the per-process memory fields of ``nvidia-smi``
    report N/A.
    """

    def __init__(self, runtime_dir: str | os.PathLike[str]) -> None:
        self.runtime_dir = Path(runtime_dir)
        if sys.platform == "win32":
            os.add_dll_directory(str(self.runtime_dir))
        # Registry entry points live in ggml.dll; device accessors in ggml-base.dll.
        self.base = ctypes.CDLL(str(self.runtime_dir / "ggml-base.dll"))
        self.reg = ctypes.CDLL(str(self.runtime_dir / "ggml.dll"))
        self._bind()
        self._loaded = False

    def _bind(self) -> None:
        r, b = self.reg, self.base
        r.ggml_backend_load_all_from_path.argtypes = [c_char_p]
        r.ggml_backend_load_all_from_path.restype = None
        r.ggml_backend_dev_count.restype = c_size_t
        r.ggml_backend_dev_get.restype = c_void_p
        r.ggml_backend_dev_get.argtypes = [c_size_t]
        b.ggml_backend_dev_name.restype = c_char_p
        b.ggml_backend_dev_name.argtypes = [c_void_p]
        b.ggml_backend_dev_description.restype = c_char_p
        b.ggml_backend_dev_description.argtypes = [c_void_p]
        b.ggml_backend_dev_type.restype = c_int32
        b.ggml_backend_dev_type.argtypes = [c_void_p]
        b.ggml_backend_dev_memory.restype = None
        b.ggml_backend_dev_memory.argtypes = [c_void_p, POINTER(c_size_t), POINTER(c_size_t)]

    def load_backends(self) -> None:
        if self._loaded:
            return
        self.reg.ggml_backend_load_all_from_path(str(self.runtime_dir).encode("utf-8"))
        self._loaded = True

    def devices(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(int(self.reg.ggml_backend_dev_count())):
            dev = self.reg.ggml_backend_dev_get(i)
            free, total = c_size_t(), c_size_t()
            self.base.ggml_backend_dev_memory(dev, byref(free), byref(total))
            out.append({
                "index": i,
                "name": self.base.ggml_backend_dev_name(dev).decode(),
                "description": self.base.ggml_backend_dev_description(dev).decode(),
                "type": int(self.base.ggml_backend_dev_type(dev)),
                "free_bytes": int(free.value),
                "total_bytes": int(total.value),
            })
        return out

    def gpu_devices(self) -> list[dict[str, Any]]:
        return [d for d in self.devices() if d["type"] == GGML_BACKEND_DEVICE_TYPE_GPU]

    def device_memory(self, index: int = 0) -> tuple[int, int]:
        dev = self.reg.ggml_backend_dev_get(index)
        free, total = c_size_t(), c_size_t()
        self.base.ggml_backend_dev_memory(dev, byref(free), byref(total))
        return int(free.value), int(total.value)


class LlamaFFI:
    """Loaded llama.cpp shared library with argtypes applied."""

    def __init__(self, lib_path: str | os.PathLike[str]) -> None:
        self.lib_path = Path(lib_path)
        if not self.lib_path.exists():
            raise FileNotFoundError(f"llama library not found: {self.lib_path}")
        self._dll_dir_cookie = None
        if sys.platform == "win32":
            # llama.dll depends on ggml*.dll / cudart / cublas that sit beside it.
            self._dll_dir_cookie = os.add_dll_directory(str(self.lib_path.parent))
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._bind()
        self._log_cb = None

    # -- binding -------------------------------------------------------
    def _bind(self) -> None:
        L = self.lib

        def sig(name: str, restype: Any, argtypes: list[Any]) -> None:
            fn = getattr(L, name)
            fn.restype = restype
            fn.argtypes = argtypes

        sig("llama_backend_init", None, [])
        sig("llama_backend_free", None, [])
        sig("llama_log_set", None, [GGML_LOG_CALLBACK, c_void_p])
        sig("llama_model_default_params", llama_model_params, [])
        sig("llama_context_default_params", llama_context_params, [])
        sig("llama_model_load_from_file", c_void_p, [c_char_p, llama_model_params])
        sig("llama_model_free", None, [c_void_p])
        sig("llama_init_from_model", c_void_p, [c_void_p, llama_context_params])
        sig("llama_free", None, [c_void_p])

        sig("llama_model_get_vocab", c_void_p, [c_void_p])
        sig("llama_vocab_n_tokens", c_int32, [c_void_p])
        sig("llama_vocab_eos", llama_token, [c_void_p])
        sig("llama_vocab_eot", llama_token, [c_void_p])
        sig("llama_vocab_bos", llama_token, [c_void_p])
        sig("llama_vocab_is_eog", c_bool, [c_void_p, llama_token])
        sig("llama_model_desc", c_int32, [c_void_p, c_char_p, c_size_t])
        sig("llama_model_size", ctypes.c_uint64, [c_void_p])
        sig("llama_model_n_params", ctypes.c_uint64, [c_void_p])
        sig("llama_model_n_ctx_train", c_int32, [c_void_p])
        sig("llama_model_chat_template", c_char_p, [c_void_p, c_char_p])

        sig("llama_n_ctx", c_uint32, [c_void_p])
        sig("llama_n_ctx_seq", c_uint32, [c_void_p])

        sig("llama_tokenize", c_int32,
            [c_void_p, c_char_p, c_int32, POINTER(llama_token), c_int32, c_bool, c_bool])
        sig("llama_token_to_piece", c_int32,
            [c_void_p, llama_token, c_char_p, c_int32, c_int32, c_bool])
        sig("llama_detokenize", c_int32,
            [c_void_p, POINTER(llama_token), c_int32, c_char_p, c_int32, c_bool, c_bool])
        sig("llama_chat_apply_template", c_int32,
            [c_char_p, POINTER(llama_chat_message), c_size_t, c_bool, c_char_p, c_int32])

        sig("llama_batch_init", llama_batch, [c_int32, c_int32, c_int32])
        sig("llama_batch_free", None, [llama_batch])
        sig("llama_decode", c_int32, [c_void_p, llama_batch])
        sig("llama_encode", c_int32, [c_void_p, llama_batch])
        sig("llama_get_logits_ith", POINTER(c_float), [c_void_p, c_int32])
        sig("llama_set_n_threads", None, [c_void_p, c_int32, c_int32])
        sig("llama_synchronize", None, [c_void_p])

        sig("llama_get_memory", llama_memory_t, [c_void_p])
        sig("llama_memory_clear", None, [llama_memory_t, c_bool])
        sig("llama_memory_seq_rm", c_bool, [llama_memory_t, llama_seq_id, llama_pos, llama_pos])
        sig("llama_memory_seq_cp", None,
            [llama_memory_t, llama_seq_id, llama_seq_id, llama_pos, llama_pos])
        sig("llama_memory_seq_keep", None, [llama_memory_t, llama_seq_id])
        sig("llama_memory_seq_pos_min", llama_pos, [llama_memory_t, llama_seq_id])
        sig("llama_memory_seq_pos_max", llama_pos, [llama_memory_t, llama_seq_id])
        sig("llama_memory_can_shift", c_bool, [llama_memory_t])

        sig("llama_state_seq_get_size", c_size_t, [c_void_p, llama_seq_id])

        sig("llama_supports_gpu_offload", c_bool, [])
        sig("llama_max_devices", c_size_t, [])
        sig("llama_print_system_info", c_char_p, [])

    # -- ABI validation -------------------------------------------------
    def validate_abi(self) -> dict[str, Any]:
        """Compare default-parameter structs against the b11057 constants.

        Any mismatch means this DLL is not the build this binding was written
        for. Continuing would silently corrupt memory, so we refuse.
        """
        cparams = self.lib.llama_context_default_params()
        mparams = self.lib.llama_model_default_params()
        problems: list[str] = []
        for key, expect in _EXPECTED_CONTEXT_DEFAULTS.items():
            got = getattr(cparams, key)
            if isinstance(expect, float):
                ok = abs(float(got) - expect) < 1e-6
            else:
                ok = got == expect
            if not ok:
                problems.append(f"context_params.{key}: expected {expect!r}, got {got!r}")
        for key, expect in _EXPECTED_MODEL_DEFAULTS.items():
            got = getattr(mparams, key)
            if got != expect:
                problems.append(f"model_params.{key}: expected {expect!r}, got {got!r}")
        if mparams.n_gpu_layers not in (0, -1, 999):
            problems.append(f"model_params.n_gpu_layers implausible: {mparams.n_gpu_layers}")
        if problems:
            raise AbiMismatch(
                f"llama.dll at {self.lib_path} does not match the {BUILD_TAG} ABI this "
                f"binding was written for:\n  " + "\n  ".join(problems)
            )
        return {
            "build_tag": BUILD_TAG,
            "lib_path": str(self.lib_path),
            "context_params_size": ctypes.sizeof(llama_context_params),
            "model_params_size": ctypes.sizeof(llama_model_params),
            "supports_gpu_offload": bool(self.lib.llama_supports_gpu_offload()),
            "max_devices": int(self.lib.llama_max_devices()),
        }

    # -- logging --------------------------------------------------------
    def set_log_callback(self, fn) -> None:
        """Route llama.cpp diagnostics away from stdout.

        The MCP stdio transport owns stdout; anything the runtime prints there
        would corrupt the protocol stream.
        """
        cb = GGML_LOG_CALLBACK(fn)
        self._log_cb = cb
        _LIVE_LOG_CALLBACKS.append(cb)
        self.lib.llama_log_set(cb, None)

    def silence_logs(self, sink=None) -> None:
        def _cb(level: int, text: bytes, _user: Any) -> None:
            if sink is not None and level >= GGML_LOG_LEVEL_WARN:
                try:
                    sink(level, text.decode("utf-8", "replace"))
                except Exception:
                    pass

        self.set_log_callback(_cb)

    def system_info(self) -> str:
        raw = self.lib.llama_print_system_info()
        return raw.decode("utf-8", "replace") if raw else ""


def default_lib_path(runtime_dir: str | os.PathLike[str]) -> Path:
    return Path(runtime_dir) / "llama.dll"
