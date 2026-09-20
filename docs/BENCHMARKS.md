# Benchmarks

All numbers measured on this machine. Raw data in `bench/out/*.json`.

**Setup.** RTX 4080 (15.99 GiB, WDDM, driver 596.49), Ryzen 7 7800X3D, Windows
11 build 26200. llama.cpp b11057 win-cuda-12.4. Qwen3-4B-Instruct-2507 Q5_K_M,
all layers offloaded. `kv_unified = true`, f16 KV, flash attention on, greedy
decoding, 64 completion tokens per request.

**Protocol.** Warm-up excluded (the first decode pays graph build and allocator
costs). 3 repeats per configuration. Run order randomised across the whole
matrix with a fixed seed, so drift cannot favour one mode. Prompts of
deliberately unequal length — short questions and ~90-token contexts — each
session carrying a unique marker so cross-session leakage would appear in the
output. Session counts are always multiples of 8 so the prompt mix stays
identical across configurations.

---

## 1. The scaling curve: where batching stops paying

Swept 1 → 64 sessions geometrically (`n_ctx = 32768`, `n_seq_max = 64`;
`bench/out/concurrency_geo64.json`).

| n | serial agg tok/s | batched agg tok/s | ×vs n=1 | batched step (ms) | TTFT (ms) | per-session tok/s |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 163.1 | 163.8 | 1.00 | 6.11 | 7.1 | 163.8 |
| 2 | 164.6 | 290.8 | 1.78 | 6.61 | 11.1 | 145.4 |
| 4 | 156.2 | 434.0 | 2.65 | 7.56 | 20.5 | 108.5 |
| 8 | 158.7 | 714.3 | 4.36 | 9.08 | 31.5 | 89.3 |
| 16 | 155.1 | 1037.6 | 6.34 | 12.60 | 59.7 | 64.9 |
| 32 | 142.8 | **1660.9** | 10.14 | 15.80 | 115.5 | 51.9 |
| 64 | 133.0 | 2132.4 | 13.02 | 24.61 | 232.7 | 33.3 |

Aggregate throughput never flattens outright — it is still climbing at 64. But
the *marginal* return collapses, and geometric spacing is too coarse to say
where. Zooming in at uniform +8 steps (`concurrency_zoom.json`):

| n | batched agg tok/s | Δagg per added session | step (ms) | **Δstep per added session (ms)** | TTFT (ms) |
|--:|--:|--:|--:|--:|--:|
| 24 | 1256.4 | — | 15.64 | — | 94.5 |
| 32 | 1629.7 | **+46.7** | 16.10 | **+0.058** | 118.9 |
| 40 | 1727.2 | +12.2 | 18.98 | +0.360 | 151.1 |
| 48 | 1836.3 | +13.6 | 21.50 | +0.315 | 183.0 |
| 56 | 1819.9 | −2.1 | 25.39 | +0.487 | 225.9 |
| 64 | 2021.8 | +25.2 | 26.03 | +0.081 | 238.0 |

### The knee is at n ≈ 32

Below 32, an added session costs **0.058 ms** of decode step time — essentially
free, and worth **+47 tok/s**. Above 32, an added session costs **0.310 ms**
— a **5.4× jump** — and is worth only **~+12 tok/s**, a 4× drop.

That is the bend. `bench/analyze_scaling.py` locates it mechanically by looking
for the largest jump in marginal step cost rather than by eye.

Why it is there: at small batch a decode step is dominated by streaming all
2.69 GiB of weights, which happens once per step regardless of batch size, so
extra sequences ride along nearly free. Past ~32 the per-sequence work — the
GEMM's compute and each sequence's attention over its own KV — stops being
negligible, and every further session costs real time.

### Operational reading

- **Schedule up to ~32 concurrent batched sessions.** Past that you pay close
  to linear time for roughly quarter returns.
- **TTFT is the real budget, not throughput.** It rises essentially linearly
  with n: 7 ms at n=1, 119 ms at n=32, 238 ms at n=64. For an interactive
  `ego_converse` turn that is the number that matters, and it argues for a
  small batch on the interactive path and a large one for background neuocytes.
- The default `arbiter.max_neuocytes = 3` sits far below the knee, which is the
  right place for a desktop GPU shared with a compositor.

---

## 2. Resident idle sessions tax every other decode

Serial aggregate throughput *falls* as session count rises — 163 → 133 tok/s
(6.13 → 7.52 ms/token) — even though serial decode advances exactly one
sequence at a time and per-sequence work is unchanged. That needed explaining
rather than hand-waving, so it was tested directly (`bench/kv_occupancy_tax.py`):
measure one probe session's decode rate, then add *idle* sessions that do
nothing but occupy the shared KV pool.

| pool occupancy | idle sessions resident | probe tok/s |
|--:|--:|--:|
| 0.2% | 0 | **162.2** |
| 19.8% | 16 | 129.0 |
| 39.3% | 32 | 109.7 |
| 58.8% | 48 | 93.9 |
| 77.1% | 63 | **83.5** |
| 0.2% | 0 (after retiring them) | **162.2** |

**A 1.94× slowdown from sessions that are doing nothing at all** — and it
recovers *exactly* (100.03%) once they retire. Roughly 1 tok/s lost per 1% of
pool occupied.

This is a genuine cost of `kv_unified = true`, the mode required for physical
prefix sharing (§4). One KV stream means attention for a ubatch is computed
over the pool's used extent, so occupancy is a global tax rather than a
per-session one.

Three consequences worth stating plainly:

1. **Neuocyte retirement is a throughput mechanism, not hygiene.** A neuocyte that
   finishes but does not release its session silently slows the whole mind. The
   full recovery on retirement confirms reclamation works — and that forgetting
   to retire would be expensive.
2. It explains part of the batching knee: at n=64 the pool is fuller than at
   n=32, so some of that 5.4× marginal jump is occupancy tax rather than
   per-sequence kernel cost. The two effects compound, and this experiment
   separates them only partially.
3. Sizing `n_ctx` generously is not free. A larger pool is not itself a cost,
   but leaving it *occupied* is.

---

## 3. Serial vs batched at the original scale

| config | aggregate tok/s | makespan (s) | TTFT (s) | p95 latency (s) | leakage |
|---|---:|---:|---:|---:|:--:|
| serial n=1 | 165.1 | 0.388 | 0.007 | 0.389 | no |
| serial n=8 | 160.2 | 2.591 | 0.007 | 0.402 | no |
| batched n=8 | 723.5 | 0.574 | 0.031 | 0.577 | no |

- **Serial aggregate throughput is flat** from 1 to 8 sessions. That is the
  definition of serialised: adding sessions adds makespan, not throughput.
- **No contamination at any level, up to 64 sessions.** Every session's unique
  marker stayed in its own output across every configuration and repeat.
- **Additional VRAM for extra sessions: ~0.** The KV pool is allocated at
  context creation; sessions are sequence ids within it.
- Run-to-run stdev is small for serial (1–6 tok/s) and larger for batched at
  high n, so treat individual high-n figures as approximate — which is why the
  knee was located from the trend, not a single pair of points.

### What the gain is and is not

It is **continuous batching**: one `llama_decode` per step carrying one token
for each active sequence. More work per launch, same number of launches.

It is **not** independent kernel overlap. No two kernels from distinct sessions
run simultaneously; there is no concurrent submission at all. The engine holds
one lock around every call into llama.cpp because llama.cpp contexts are not
thread safe.

`physical_overlap_verified` is `false`, and independent overlap is recorded as
**not attempted** — not merely "unmeasured". There is nothing concurrent for a
profiler to find in this design.

On tooling, precisely: **Nsight Systems (`nsys`) is not installed**, and it is
the only one of the three that produces a GPU timeline capable of showing
concurrent kernels. Nsight Compute 2025.1.1 (`ncu`) *is* present but serialises
kernel launches by design, so it can never demonstrate overlap. `nvprof` is on
PATH but does not support this GPU's compute capability (8.9).

### Conditions not run

The brief lists five comparison conditions. (A) serial and (E) batching are
above. (B) an existing serialised wrapper and (C) independent processes with
duplicated weights were **not run**: the design commits to one GPU owner, and
measuring a duplicated-weights variant would not change that decision — it
would double VRAM for the weights and provide no prefix sharing between
processes. (D) one owner with isolated sessions *and separate streams* is not
implemented, for the thread-safety reason above.

---

## 4. Prefix sharing: cost and savings

441-token Ego prefix, 4 neuocytes:

| metric | value |
|---|---:|
| prefill of the prefix | 42.1 ms |
| **fork** (`llama_memory_seq_cp`) | **0.090–0.203 ms** |
| **exact recomputation** of the same prefix | **7.7–8.3 ms** |
| speedup of fork over recomputation | **38–88×** |
| KV per token (measured) | 147,470 bytes |
| **KV bytes avoided by sharing** | **248.1 MiB** |

A fork is essentially free — it edits per-cell sequence bitsets.

## 5. Shared vs copied prefix — the discriminating measurement

VRAM deltas cannot answer this (KV is preallocated) and
`llama_state_seq_get_size` cannot either (it reports *logical* size —
132,722,088 bytes for both a forked and a source sequence). Capacity is the
evidence.

| | `kv_unified=True` | `kv_unified=False` |
|---|---|---|
| KV streams | 1 (shared pool) | 4 (one per sequence) |
| `n_ctx_per_seq` | 2048 = full pool | 1024 = pool ÷ 4 |
| partial-prefix `seq_cp` | works, metadata only | **aborts the process** (`GGML_ASSERT(is_full …)`) |
| 900-token prefix + 3 forks, tail tokens accommodated | **1136** | **120** (after one full copy) |
| cells occupied if shared | 2036 / 2048 (99.4%) | — |
| cells if copied | 4736 — does not fit | 1024 exactly consumed per sequence |
| **verdict** | **PHYSICALLY SHARED** | **PHYSICALLY COPIED** |

## 6. Fork vs exact recomputation

| | prompt 0 (47 tok) | prompt 1 (30 tok) |
|---|---:|---:|
| top-1 identical | yes | yes |
| logits cosine | 0.99935 | 0.99965 |
| KL(fork ‖ recompute) | 0.0062 nats | 0.00055 nats |
| bit-identical | no | no |
| greedy identical for | 36 tokens | 45 tokens |
| **control:** recompute vs recompute, max abs diff | 1.02 | 0.88 |
| **control:** recompute vs recompute, first divergence | **step 36** | none in 48 |

The control is the point. Two *recomputations* of the same tokens into
different cache cells are also not bit-identical, and diverge at the same step.
The divergence is a property of this CUDA backend's position-dependent
reduction order, **not** of forking. See [RUNTIME §3](RUNTIME.md#3-fork-vs-exact-recomputation).

---

## Reproducing

```powershell
.\.venv\Scripts\python.exe bench\prefix_sharing.py
.\.venv\Scripts\python.exe bench\fork_vs_recompute.py
.\.venv\Scripts\python.exe bench\concurrency.py --repeats 3 --tag geo64
.\.venv\Scripts\python.exe bench\concurrency.py --repeats 3 --counts 24,32,40,48,56,64 --tag zoom
.\.venv\Scripts\python.exe bench\analyze_scaling.py
.\.venv\Scripts\python.exe bench\kv_occupancy_tax.py
```

Close other GPU consumers first; the desktop compositor alone holds ~3.5 GiB
and whole-device VRAM is the only accounting available under WDDM.
