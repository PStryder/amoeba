# Runtime investigation — what was actually measured

Everything here was measured on this machine. Where something was not
established, it says so.

**Host.** Windows 11 Pro for Workstations, build 26200. AMD Ryzen 7 7800X3D
(16 logical CPUs), 63.15 GiB RAM. NVIDIA GeForce RTX 4080, 15.99 GiB, WDDM,
driver 596.49. CUDA toolkit 12.8.93 installed (not used — see below).

**Python.** `C:\Python311\python.exe` 3.11.8, in a project-local `.venv`.
The system PyTorch is 2.10.0+cpu; the unrelated `cathedral` conda environment
has 2.5.1+cu121. **Neither is used.** This project does not depend on PyTorch.

**Runtime.** llama.cpp **b11057**, official prebuilt
`llama-b11057-bin-win-cuda-12.4-x64.zip` plus
`cudart-llama-bin-win-cuda-12.4-x64.zip`. Nothing was compiled: the CUDA 12.4
build runs against the installed 596.49 driver. The matching `include/llama.h`
from tag b11057 is vendored beside the DLLs as the ABI reference.

**Model.** Qwen3-4B-Instruct-2507, Q5_K_M GGUF (2.69 GiB on disk,
4,022,468,096 parameters, 151,936 vocab, 262,144 trained context). One model
was downloaded; nothing was fetched speculatively.

**BitNet was not used.** `F:\hexylab\bitnet` and `F:\hexylab\pcdc` are
untouched. BitNet's GPU path is Linux/Docker-oriented (`libbitnet.so` via
ctypes, a Bash `compile.sh`, `readline`/xformers imports) and its Windows CPU
build is `GGML_CUDA=OFF`. A conventional small transformer on a current
llama.cpp reached every milestone target without that port.

---

## 1. Binding llama.cpp from native Python

A hand-written `ctypes` binding (`backends/llama_ffi.py`) against the exact
b11057 header. `validate_abi()` compares every field of
`llama_context_default_params()` and `llama_model_default_params()` against the
constants in that build's source and refuses to run on a mismatch — a wrong
struct layout would silently corrupt memory rather than fail loudly.

Measured: `sizeof(llama_context_params) == 160`,
`sizeof(llama_model_params) == 80`.

### Four native-interop defects found and fixed

These were all found by tests, and each would have been an intermittent crash
or a silent correctness bug in production.

1. **`llama_batch_init` owns its `seq_id` sub-arrays.**
   It mallocs one `llama_seq_id*` per token and `llama_batch_free` frees each
   by walking to a NULL sentinel. Assigning `batch.seq_id[i] = <ctypes array>`
   makes the library `free()` memory Python owns — heap corruption, immediate
   process death. Correct usage writes *into* `batch.seq_id[i][0]`.

2. **`llama_get_logits_ith` reads a CONTEXT-owned buffer.**
   Index `-1` is "the last decode", regardless of which sequence produced it.
   Interleaving two sessions and then sampling made one session continue from
   the other's distribution — genuine cross-session contamination through an
   API that looks per-sequence. Each session now copies its own logits row
   immediately after its own decode.
   Also: the index is a position **in the batch**, not an output-row ordinal.

3. **`llama_backend_free()` is process-global.**
   A second engine closing tore down state the first was still using; the
   first engine's next decode faulted. Now reference-counted.

4. **`llama_log_set()` stores a raw pointer with no ownership.**
   Keeping the ctypes trampoline only as an attribute of an object that later
   goes away leaves a dangling pointer, and the next log line faults the
   process. Callbacks are now kept alive for the life of the process.

### The CUDA backend is not loaded by `llama_backend_init()`

`llama_backend_init()` does **not** discover CUDA. The dynamic backend DLLs are
found by `ggml_backend_load_all_from_path()`, which upstream example binaries
call from `common_init()`. Without it, everything loads, `llama_supports_gpu_offload()`
returns **false**, and the model runs entirely on CPU — a silent and very
expensive failure. `doctor` asserts a GPU device actually appeared before any
GPU capability is claimed.

`ggml_backend_dev_memory()` is also the only whole-device VRAM accounting
available here: under WDDM the per-process memory fields of `nvidia-smi` report
N/A.

---

## 2. Ego snapshots: shared prefix vs copied prefix

The decisive question: does `llama_memory_seq_cp` **share** physical KV cells
or **copy** them?

From b11057 `src/llama-kv-cache.cpp`:

- `n_stream = unified ? 1 : n_seq_max` (line 84).
- `seq_cp` when source and destination are in the **same stream**: *"since both
  sequences are in the same stream, no data copy is necessary — we just have to
  update the cells meta data"*; it calls `cells.seq_add(i, seq_id_dst)` per cell.
- Different streams: *"cross-stream sequence copies require to copy the actual
  buffer data"*.
- `seq_rm` frees a cell only when `cells.seq_rm(i, seq_id)` reports the cell's
  sequence set became empty — reference counting in the cell bitset.

So with `kv_unified = True` a fork is metadata-only and reference-counted. This
project forces `kv_unified = True`.

### Measurement, not inspection

VRAM deltas cannot answer this — KV is allocated when the context is created,
so a fork never changes VRAM either way. `llama_state_seq_get_size` cannot
either: it reports the **logical** serialized size and is identical for a
shared and a copied prefix (measured: 132,722,088 bytes for a forked 900-token
sequence, exactly the same as its source).

Capacity is the evidence. `bench/prefix_sharing.py`:

| | `kv_unified=True` |
|---|---|
| KV capacity | 2048 cells |
| prefix | 900 tokens in Ego |
| forks | 3 workers, each holding the same 900-token prefix |
| tail tokens accommodated before exhaustion | **1136** |
| cells occupied if shared | 900 + 1136 = **2036** (of 2048) |
| cells that would be occupied if copied | 4 × 900 + 1136 = **4736** |
| **verdict** | **PHYSICALLY_SHARED** |

4736 cells do not fit in 2048. 2036 do, and 2036/2048 = 99.4% of the pool —
exactly what metadata-only sharing predicts.

### The non-unified contrast

| | `kv_unified=False` |
|---|---|
| streams | `n_seq_max` (4) |
| context split | `n_ctx_per_seq = n_ctx / n_seq_max` — **not** a shared pool |
| partial-prefix `seq_cp` | **aborts the process**: `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`, exit 127 |
| full-sequence `seq_cp(-1,-1)` | succeeds, copies buffer data |
| worker tail capacity after a 900-token copy into a 1024-cell stream | **120 tokens** (900 + 4 + 120 = 1024) |
| **verdict** | **PHYSICALLY COPIED** |

The worker consumed its own 900 cells. That is the difference between the two
modes, measured rather than asserted.

The engine therefore **refuses** `fork_prefix` when `kv_unified=False` with
`capability_unsupported`, rather than letting a partial `seq_cp` abort the
process. Callers fall back to exact recomputation.

**Consequence for the architecture:** `kv_unified=True` is not an optimisation,
it is structural. It is the only mode in which `n_ctx` is a shared pool that one
large Ego context can occupy and share.

---

## 3. Fork vs exact recomputation

`bench/fork_vs_recompute.py`. A fork reuses the source's physical cells; a
recomputation re-evaluates the same tokens into *different* cells.

| prompt | 0 (47 tokens) | 1 (30 tokens) |
|---|---|---|
| top-1 token identical | **yes** | **yes** |
| logits cosine | 0.99935 | 0.99965 |
| KL(fork ‖ recompute) | 0.0062 nats | 0.00055 nats |
| max abs logit diff (over 151,936 logits) | 0.998 | 0.968 |
| bit-identical logits | no | no |
| greedy decoding identical for | 36 tokens | 45 tokens |

Greedy output diverges after ~30–45 tokens into semantically equivalent
continuations. **A control experiment settles the cause:** recomputing the same
prefix a *second* time into different cells is also not bit-identical
(max abs diff 1.02 / 0.88) and diverges at **the same step (36)** on prompt 0.

So the divergence is **not caused by forking**. It is a property of this CUDA
backend: identical tokens evaluated into different cache positions produce
slightly different logits, because attention reduces over cells and the
reduction order depends on cache position.

The acceptance criterion is therefore the defensible one: *a fork must agree
with an exact recomputation to within the backend's own run-to-run
nondeterminism*. `test_fork_matches_exact_recomputation` asserts same top-1,
KL < 0.05, and fork↔recompute divergence no larger than recompute↔recompute
divergence.

**This is the single most important caveat in this document.** Anyone expecting
bit-reproducible greedy decoding from this stack will not get it, from forking
or from anything else.

---

## 4. Concurrency

See `bench/out/concurrency.json` for the full matrix and
[BENCHMARKS.md](BENCHMARKS.md) for the numbers.

What is established:

- **One resident weight set serving many sessions.** One `llama_model`, one
  `llama_context`, sessions are sequence ids. Opening four extra sessions
  consumes no measurable additional VRAM.
- **Serialized execution.** One lock serialises every call into llama.cpp.
  llama.cpp contexts are not thread safe.
- **Continuous batching.** `generate_batched` places one token per active
  session into a single `llama_decode` — one fused kernel launch over several
  independent sequences. Measured: 160 tok/s flat under serial execution
  regardless of session count, versus 723 tok/s at 8 batched sessions, at the
  cost of time-to-first-token rising 0.007 s to 0.031 s.

What is **not** established:

- **Independent overlapping GPU execution.** **Not attempted**, not merely
  unmeasured. One lock serialises every call into llama.cpp, so there is no
  concurrent submission for a profiler to find. `physical_overlap_verified` is
  `false`.

  On tooling, precisely: **Nsight Systems (`nsys`) is not installed**, and it
  is the only tool here that produces a GPU timeline capable of showing
  concurrent kernels. **Nsight Compute 2025.1.1 (`ncu`) is present** but
  serialises kernel launches by design to collect per-kernel metrics, so it can
  never demonstrate overlap. `nvprof` is on PATH but does not support compute
  capability 8.9 and produces no output.

  The 13x aggregate throughput gain at 64 sessions comes from continuous
  batching — more sequences per fused kernel launch — not from more kernels
  running at once. The marginal return per added session collapses at
  **n ≈ 32** (0.058 ms/session below it, 0.310 ms/session above — a 5.4x jump);
  see [BENCHMARKS §1](BENCHMARKS.md#1-the-scaling-curve-where-batching-stops-paying). Throughput improvement, asynchronous Python calls, multiple
  processes and a shared model pointer are none of them evidence of
  simultaneous device execution.
- **Cross-process weight sharing.** Not attempted. One GPU-owner process is the
  design, per the brief.

### The unified-KV occupancy tax

`kv_unified=True` is required for physical prefix sharing, and it has a
measured cost. One KV stream means attention for a ubatch is computed over the
pool's *used extent*, so occupancy is a global tax, not a per-session one.

Measured (`bench/kv_occupancy_tax.py`): a probe session decodes at 162.2 tok/s
against an empty pool and **83.5 tok/s with 63 idle sessions resident** — a
**1.94x slowdown caused entirely by sessions doing nothing** — recovering
*exactly* (100.03%) once they retire. Roughly 1 tok/s lost per 1% of pool
occupied.

Two consequences. Worker retirement is a **throughput mechanism**, not
hygiene: a worker that finishes without releasing its session slows the whole
mind. And this explains the otherwise puzzling decline in *serial* aggregate
throughput as session count rises (163 -> 133 tok/s, i.e. 6.13 -> 7.52
ms/token), where per-sequence work is unchanged by construction.

---

## 5. Model quality

Qwen3-4B-Instruct-2507 at Q5_K_M answers coherently and follows the
three-line structured output format the workers and Id require. It was **not**
evaluated for whether a 4B model is good enough for genuine Ego synthesis or
genuine Id audit — that remains an open question, and Id sharing the same model
as Ego is a real independence concern (see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)).

The model emits one benign load-time warning:
`control-looking token: 128247 '</s>' was not control-type; this is probably a
bug in the model. its type will be overridden`.
