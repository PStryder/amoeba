# Amoeba

A persistent local cognitive system exposed through MCP. Native Windows, native
Python, one resident set of model weights on the GPU. No Docker, no WSL.

Two long-lived halves — **Ego** (outward: conversation, investigation,
synthesis) and **Id** (inward: homeostasis, introspection, audit,
contradictions) — each with its own process and its own private inference
context. Disposable neuocytes fork from published snapshots of Ego's *actual*
context and retire. A fixed supervisor owns lifecycle, admission control,
scheduling, hard resource limits and the single durable writer. An external
frontier model is a client and a cognitive peer, not part of the mind.

Documents: [ARCHITECTURE](ARCHITECTURE.md) ·
[MCP contract](MCP_CONTRACT.md) ·
[Runtime findings](RUNTIME.md) ·
[Benchmarks](BENCHMARKS.md) ·
[Open questions](OPEN_QUESTIONS.md)

---

## Capability status

Read this before trusting anything below it.

| Capability | Status | Evidence |
|---|---|---|
| Durable state: events, receipts, blobs, transactional mutation | **Implemented, tested** | `tests/test_durable_foundation.py` |
| Recovery: restart, lease expiry, fencing, requeue | **Implemented, tested** | `test_supervisor_restart_recovers_state` |
| Maintained memory separate from raw history | **Implemented, tested** | `test_contradictory_history_does_not_become_belief` |
| Leased work queue, at-least-once, idempotent commits | **Implemented, tested** | `test_duplicate_commit_is_idempotent` |
| Ego/Id/neuocyte/supervisor/inference as separate processes | **Implemented, tested** | `test_role_restarts_independently` |
| MCP facade: 10 cognitive verbs, typed in/out schemas, stdio | **Implemented, tested** | `test_real_mcp_client_calls_both_halves` |
| MCP cancellation (explicit + on client abort) | **Implemented, tested** | `tests/test_cancellation.py` |
| MCP resources / prompts / sampling / streaming | **Not implemented** | tools-only surface — see [MCP contract](MCP_CONTRACT.md#what-is-not-implemented) |
| Id audits an Ego conclusion through the record | **Implemented, tested** | `test_id_audits_ego_conclusion_without_asking_ego` |
| One real local model, one resident weight set | **Implemented, tested** | `test_single_resident_weight_set` |
| Ego snapshot fork with **physically shared** prefix KV | **Implemented, measured** | `bench/prefix_sharing.py` → 2036/2048 cells vs 4736 if copied |
| Reference-counted snapshot release and reclamation | **Implemented, tested** | `test_reclaim_only_unreferenced_and_superseded` |
| Fork agrees with exact recomputation | **Implemented, measured with a caveat** | same top-1, KL < 0.007; **not** bit-identical — see [RUNTIME §3](RUNTIME.md#3-fork-vs-exact-recomputation) |
| Cross-neuocyte / Ego-neuocyte cache isolation | **Implemented, tested** | `test_worker_tails_are_private` |
| Tool-call schema + permission validation | **Implemented, tested** | `tests/test_tools.py` |
| Tool *execution* loop (model requests -> harness runs -> result returned) | **Wired** | multi-turn loop in the neuocyte; execution happens in the Harness via `tool_invoke`, gated on the work row's `sandbox_allowed`, bounded by turns/budget/deadline, every call receipted |
| Host filesystem: allowlisted roots, versioned writes, explicit attach | **Implemented, tested** | `tests/test_filespace.py`, `tests/test_filespace_harness.py`; no root configured by default, so Amoeba has no host access until one is |
| Four separated stores (filespace / blobs / compute sandbox / accepted artifact) | **Implemented, tested** | `tests/test_store_boundaries.py`; sandbox reach measured from inside the container, destruction verified to preserve input, evidence and accepted work product |
| Id sensory surface: `system_pulse` + Harness-mediated investigation | **Implemented, tested** | `tests/test_id_senses_and_effectors.py`; cached, bounded, facts-only, tracks work/failure/resource change |
| Id effectors: request/propose/challenge/escalate, all receipted | **Implemented, tested** | nine verbs, each attributed to `id` and carrying the `pulse_id` it was formed from |
| Id-only capability isolation (scoped RPC method tables) | **Implemented, tested** | scope is the presented credential; no role field to forge, no enumeration, no dispatcher bypass; Ego checked separately |
| Cognitive blackboard: posts, threads, relations, receipts | **Implemented, tested** | `tests/test_blackboard.py` |
| Independent replication vs socially propagated agreement | **Implemented, tested** | every read recorded; `board_corroboration` splits the two |
| Board-naive neuocytes (`board_access="none"`) | **Implemented, tested** | `test_a_naive_worker_posts_without_having_read_the_board` |
| Sandboxed compute (OS-enforced AppContainer) | **Implemented, tested** | 29 boundary tests; network + host FS blocked |
| Artifact promotion by Harness decision | **Implemented, tested** | proposal -> re-hash -> receipt |
| Context homeostasis: measure, retire, checkpoint, rebirth | **Implemented, tested** | `tests/test_homeostasis.py` |
| Context trim (verbatim head+tail) | **Implemented, tested** | dropped span recorded, reconstructible from checkpoint |
| Context summarisation | **Refused by design** | a different behaviour from reconstitution; raises `capability_unsupported` |
| Context compaction | **Not implemented** | `trim` drops a span; nothing merges or rewrites |
| Continuous batching (several sequences, one fused kernel) | **Implemented, measured** | 13x aggregate at 64 sessions; knee at n≈32 |
| Unified-KV occupancy tax (idle sessions slow others) | **Measured, unmitigated** | 1.94x slowdown at 77% pool, fully reversible |
| **Independent overlapping GPU execution** | **NOT attempted, NOT claimed** | engine serialises by design; Nsight Systems absent, Nsight Compute serialises kernels |
| Cross-process GPU weight sharing | **Not attempted** | design commits to one GPU owner |
| Remote MCP transport | **Not implemented** | localhost is not a remote deployment |
| Streaming output over MCP | **Not implemented** | |
| Context trimming / eviction | **Not implemented** | a long-running Ego will eventually overflow `n_ctx` |
| Model quality for genuine Ego synthesis / Id audit | **Unevaluated** | plumbing is proven; cognition is not |

The deterministic backend (`config.test.toml`) is **not** a model. Every result
it produces is labelled `[SIMULATED]` and every response carries a limitation
saying so.

---

## Layout

Third-party runtime, weights, source and runtime data are separate trees:

```
F:\hexylab\amoeba\        application source (this repo)
F:\hexylab\amoeba-runtime\    llama.cpp b11057 win-cuda-12.4 + vendored llama.h
F:\hexylab\amoeba-models\     Qwen3-4B-Instruct-2507-Q5_K_M.gguf
F:\hexylab\amoeba-state\      SQLite WAL, content-addressed blobs, logs
```

`F:\hexylab\pcdc` and `F:\hexylab\bitnet` are untouched.

---

## Setup

```powershell
Set-Location F:\hexylab\amoeba

# 3.11 only; do not use the global or the cathedral environment
C:\Python311\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
    mcp==1.26.0 pydantic==2.12.4 anyio==4.12.0 numpy==2.2.6 `
    pytest==8.4.2 pytest-timeout==2.4.0
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The llama.cpp runtime and the model are downloaded separately into the trees
above; `doctor` tells you if either is missing.

---

## Running

```powershell
# 1. Check what is actually present and working. Exits non-zero on a fatal gap.
.\.venv\Scripts\python.exe -m amoeba doctor --config config.toml

# 2. Start the mind. This process owns everything and stays running.
.\.venv\Scripts\python.exe -m amoeba supervise --config config.toml

# 3. From another shell: check on it.
.\.venv\Scripts\python.exe -m amoeba status --config config.toml

# 4. A client starts a separate stdio facade that connects to the supervisor.
.\.venv\Scripts\python.exe -m amoeba mcp --config config.toml --transport stdio

# 5. Stop cleanly.
.\.venv\Scripts\python.exe -m amoeba shutdown --config config.toml
```

Killing the facade in step 4 does not touch the mind.

Drive a running mind directly over the control plane:

```powershell
.\.venv\Scripts\python.exe scripts\exercise.py --config config.toml
```

---

## Tests

```powershell
# everything (spawns real process stacks and loads the real model)
.\.venv\Scripts\python.exe -m pytest tests -q -p no:randomly

# no GPU required
.\.venv\Scripts\python.exe -m pytest tests -q -k "not gpu"

# GPU only
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_engine.py -q
```

The GPU tests skip automatically when the runtime or model is absent.

## Benchmarks

```powershell
.\.venv\Scripts\python.exe bench\prefix_sharing.py      # shared vs copied prefix KV
.\.venv\Scripts\python.exe bench\fork_vs_recompute.py   # fork vs exact recomputation
.\.venv\Scripts\python.exe bench\concurrency.py         # 1/2/4/8 sessions, serial vs batched
```

Results land in `bench/out/*.json`.

---

## Configuration

`config.toml` — real backend. `config.test.toml` — deterministic stub.

Notable settings:

- `backend.n_ctx` is the **total shared KV pool**, not per-session.
  `kv_unified` is forced on; it is the only mode in which an Ego prefix can be
  physically shared (see [RUNTIME §2](RUNTIME.md#2-ego-snapshots-shared-prefix-vs-copied-prefix)).
- `arbiter.*` are hard caps. A model may request a larger budget; it gets the
  capped one and the response says so.
- `arbiter.user_reserved_slots` / `maintenance_reserved_slots` are what stop
  either class of work from starving the other.
- All ports bind to `127.0.0.1` and are authenticated with a token written to
  `state/control.token`.
