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
| MCP facade: typed in/out schemas, stdio | **Implemented, tested** | `test_a_real_mcp_client_gets_the_io_surface_only`. The surface is `scopes.EXTERNAL_IO` — see the demotion row below; it is not a second list maintained here |
| Cancellation (control plane only) | **Implemented, tested** | `tests/test_cancellation.py`. An external client has no cancellation verb and disconnecting cancels nothing — `test_an_mcp_client_has_no_cancellation_verb`, `test_disconnecting_does_not_cancel_anything`, invariant I48. An earlier row claimed cancel-on-client-abort, which described a surface that no longer exists |
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
| Ego sensory surface: productive-work view, artifact evidence, resource identity | **Implemented, tested** | `tests/test_ego_senses_and_effectors.py`; no Id telemetry, no sandbox scratch |
| Ego effectors: request work, governed work message, cancel own work, propose memory, reach Id | **Implemented, tested** | Ego states intent; the Harness owns admission, worker choice, prompt version and budget |
| Ego/Id/neuocyte scopes mutually disjoint | **Implemented, tested** | role authority is the presented credential, never a request field; board-naive work refuses mid-flight messages |
| External I/O surface (MCP + JSON-RPC API): input in, output out | **Implemented, tested** | `tests/test_external_interfaces.py`; eight io_* verbs, exact-byte input provenance, SSE scoped to the client |
| MCP demoted from control token to `external_io` scope | **Implemented, tested** | was 23 tools incl. file write/delete, artifact promotion, Id maintenance; now 8 I/O tools |
| Operator console + governance surface on a separate table | **Implemented, tested** | same listener, different credential and method table; console reaches state only through the Harness |
| Prompt Library: namespaced, versioned, parent-pinned cognitive profiles | **Implemented, tested** | `tests/test_prompt_library.py`; 37 tests, 9 mutation-verified invariants (I49-I57) |
| Roots (`ego`, `id`) authorable only by bootstrap | **Implemented, tested** | `create_runtime_version` has no parameter that permits one; the private flag is dropped and a root is refused outright -- both must be removed for the test to pass |
| Prompt inheritance, composition modes, lineage vectors (`ego.neuocyte.research@3.7.5`) | **Implemented, tested** | nearest-ancestor properties, four composition modes, self-checking references, `explain_profile` attributes every line and setting |
| Governed change: Id evaluates and proposes, the Operator decides | **Implemented, tested** | approving/selecting/cascading appear in no scope table; an edited prompt file becomes a candidate, never an override |
| Cascade: none / queue / approve, definitions copied unchanged | **Implemented, tested** | asserted on the local digest, which excludes the parent binding; descends level by level and reports what it skipped |
| Neuocyte cognitive profiles | **Implemented, tested** | neuocytes previously had no profile at all; they now descend from `ego.neuocyte` / `id.neuocyte`, making specialisations expressible |
| Incarnation binding: resolved bytes frozen at birth | **Implemented, tested** | digests and lineage stored, not a pointer; a forked neuocyte's injected bytes and inherited prefix are recorded separately |
| Prompt-library A/B evaluation | **Not implemented** | `experimental_approved` and the experimental selection purpose exist and are honoured, but nothing measures whether a profile performs better -- see [PROMPTLIB §14](PROMPTLIB.md#14-residual-limits) |
| Root namespace creation reserved to bootstrap; root *versions* governed | **Implemented, tested** | `establish_root` is reachable from no scope table; two independent defences against a new top-level name, and negating either leaves the other defended |
| Role environment manifest (profile / environment / turn input) | **Implemented, tested** | `tests/test_role_environment.py`; Harness-built per turn, frozen within one, capabilities derived from the live dispatch table |
| Ego/Id bounded Harness-mediated tool loop | **Implemented, tested** | roles previously parsed tool calls without executing them; one call per turn, validated and run by the Harness, result fed back, bounded by turns and deadline |
| Environment provenance: exact bytes reconstructable | **Implemented, tested** | manifest content-addressed before it is handed over; `role.turn_began` records profile, environment digest, blob and trigger |
| `cfg.<role>.system_prompt` ungoverned doctrine path | **Removed** | the field is gone and a non-empty value is refused at load with migration guidance; the live prompt comes only from the Prompt Library |
| Persistent roles run bounded turns, one at a time | **Implemented, tested** | `tests/test_persistent_turns.py`; one turn thread per role plus a partial unique index over open turns — previously two callers produced two concurrent turns against one inference session |
| Durable Harness-owned role mailbox | **Implemented, tested** | queued / claimed / consumed / expired; consumption at commit so a crash mid-turn returns the inputs, at-least-once with preserved identity |
| Ego event-driven, Id event + heartbeat | **Implemented, tested** | Ego has no autonomous turn; Id gets a startup turn and a backing-off heartbeat that is explicit input rather than a fake user message |
| Stop reasons and bounded deterministic continuation | **Implemented, tested** | ten distinct reasons; non-terminal stops earn a continuation with `parent_turn` recorded, bounded at `max_continuations` |
| Turn provenance: profile + environment + trigger bundle | **Implemented, tested** | bundle and environment content-addressed before the turn runs and read back rather than recomputed |
| Artifact and blackboard wake triggers | **Implemented, tested** | ownership, not similarity: a role is woken by what happens to work it originated, and never by its own posts. The heuristic the architecture declined to invent turned out not to be needed — see I92 |
| Role turns render the full request, not a preview | **Implemented, tested** | the body is read from the content store with a stated budget; truncation names the digest holding the rest |
| Audit verdicts parsed as whole words | **Implemented, tested** | `unsupported` was read as `supported`, inverting an adverse audit and suppressing the disagreement it should have opened |
| Role-targeting effectors callable through the tool loop | **Implemented, tested** | `role` names the asker, so a target parameter is `target_role` |
| An interaction is complete only when answered | **Implemented, tested** | the async worker waits across continuation turns and fails truthfully rather than reporting an empty answer; a thought stopped by the continuation limit, a deadline or a failure is `incomplete`, not complete — see I108 |
| An answer is the whole interaction's | **Implemented, tested** | every bounded turn's piece, in order, exactly once, concatenated exactly along parent links; a continuation resumes the message it continues where the session allows; an answer records no conclusion; Ego records one on purpose — see I107, I109, I112 |
| Id audits what Ego concludes | **Implemented, tested** | a recorded conclusion wakes Id; the operator can request an audit; a verdict nobody waits for is committed when Id answers — see I116 |
| Capability vocabularies are discoverable | **Implemented, tested** | allowed values and argument kinds are declared from the constants the checks use; a refusal carries its allowed values back — see I117 |
| Environment declared once per session | **Implemented, tested** | an unchanged declaration is referenced, not re-ingested, within a session — see I118 |
| Role arguments validated before dispatch | **Implemented, tested** | a wrong type is refused in words and the verb is never called — see I119 |
| RPC calls cannot read each other's replies | **Implemented, tested** | a failed call drops its connection; a mismatched reply id is refused — see I120 |
| Operator views measured and attributed | **Implemented, tested** | tool events carry their operation; overview context/KV read from `context_report`; bindings show harness constraints — see I121 |
| Structural context rebuild | **Implemented, tested** | governed prompt and environment reconstructed, settled turns dropped whole, owed turns kept with oversized results projected; no positional trim — see I122 |
| Bounded truthful tool-result delivery | **Implemented, tested** | token-budgeted per role; whole-JSON projection with a reference `result_read` opens for the issuing role; compact prompt views — see I123 |
| Only the Harness authors chat structure | **Implemented, tested** | control tokens are refused on the token id before append, in both decode paths; refused attempts are recorded — see I124 |
| Roles can invoke what they are offered | **Implemented, tested** | the environment block states the call syntax the loop parses; a malformed attempt is refused and retried, never delivered as an answer — see I113, I115 |
| A busy role is not restarted as dead | **Implemented, tested** | health probes use their own inference connection, so a long generation cannot fail liveness — see I114 |
| Per-role output ceilings | **Implemented, tested** | Ego 3072, Id 1024, `ego.neuocyte` 512, `id.neuocyte` 384, one canonical value per bound profile; the platform cap refuses rather than clamps, and startup fails on a contradiction — see I110 |
| One answer per *request* rather than per turn | **Implemented, tested** | the answer is written onto the request by trigger id; a turn admits at most one answer-bearing request, and a continued thought answers the request that started it — see [TURNS §16](TURNS.md) and invariants I80–I83 |
| Interaction lineage scoping evidence | **Implemented, tested** | assigned by the producer, enforced at bundling, and *not* a confidentiality boundary — see [ARCHITECTURE](ARCHITECTURE.md), "One Amoeba is one cognitive trust domain" |
| Specialist neuocyte profiles dispatched | **Implemented, tested** | Ego asks for a leaf (`research`), the library governs whether `ego.neuocyte.research` is approved, and an unavailable one falls back to the base *and says so on the work item* — see I91 |
| Batched inference on the live path | **Implemented, tested** | one dispatcher groups requests that were already waiting; never waits for company, so the idle path is unchanged. A backend that cannot batch is served serially — see I90. The deterministic backend batches as a *loop* and says so: no timing from it describes batching |
| External attachments and surfaced results wired end to end | **Implemented, tested** | the manifest is named in the bundle and read through `ego_read_attachment`; `ego_surface_result` is the production caller `surface_result` never had. Both are scoped to the turn's interaction, so Ego chooses what crosses the boundary and never whose — see I89 |
| Tool results say when they were cut | **Implemented, tested** | one budget in the Harness, which stores what does not fit and names the digest; the notice says how much was withheld and to narrow the call — see I86 |
| Data retention | **Partly implemented** | consumed triggers and closed turns are pruned past a window; evidence never is, and is listed explicitly — see I87, I88 |
| Blob garbage collection | **Not implemented** | a digest is referenced from ~20 columns and from event payloads; a collector that missed one source would delete content the chain points at. `store_footprint` measures what one would be reasoning about |
| Cognitive blackboard: posts, threads, relations, receipts | **Implemented, tested** | `tests/test_blackboard.py` |
| Independent replication vs socially propagated agreement | **Implemented, tested** | every read recorded; `board_corroboration` splits the two |
| Board-naive neuocytes (`board_access="none"`) | **Implemented, tested** | `test_a_naive_worker_posts_without_having_read_the_board` |
| Sandboxed compute (OS-enforced AppContainer) | **Implemented, tested** | 29 boundary tests; network + host FS blocked |
| Artifact promotion by Harness decision | **Implemented, tested** | proposal -> re-hash -> receipt |
| Context homeostasis: measure, retire, checkpoint, rebirth | **Implemented, tested** | `tests/test_homeostasis.py` |
| Pressure gating for discretionary turns | **Implemented, tested** | the heartbeat defers while the pool is strained and runs regardless after a bounded wait; event-driven turns are never gated — see I99 |
| Withdrawing a claim or a belief | **Implemented, tested** | a conclusion can be retracted by its author or superseded by its successor; `operator_retract_memory` makes belief retraction reachable. Neither existed: an adverse audit could not lead to the claim changing — see I100 |
| Disagreement resolution | **Implemented, tested** | closed by the record moving — superseded, retracted, a supporting audit on a *changed* evidence basis, or an operator decision. One open dispute per subject; a reversal on unchanged evidence is recorded, not accepted — see I101, I102 |
| Open contradictions reported as pressure | **Implemented** | `system_pulse` gives open count, oldest age, and counts over a day and a week. Disputes are never aged out: an unresolved contradiction nobody addressed is a true fact about the organism |
| Per-actor context budgets | **Implemented, tested** | a session carries `context_budget_tokens` and `budget_basis` from creation; Ego 16384, Id 8192, `ego.neuocyte` 6144 private-growth, `id.neuocyte` 4096 total — see I96, I97 |
| Physical KV admission | **Implemented, tested** | measured occupancy for live sessions, work-class budget as a conservative estimate for prospective ones, with a 15% reserve held back from new work for decode — see I98 |
| Proactive rejuvenation before the hard ceiling | **Implemented, tested** | a generation is admitted only if prompt plus its allowance fits, so a role is rejuvenated before a turn that could not finish (Ego at 13312 of 16384); `role_context_high` is an advisory recommendation — see I96, I111 |
| Context trim, positional (verbatim head+tail) | **Implemented, tested** | the fallback when there is not enough settled work to evict; dropped span recorded and reconstructible from the checkpoint |
| Context summarisation | **Refused by design** | a different behaviour from reconstitution; raises `capability_unsupported` |
| Context compaction | **Not implemented** | eviction removes whole finished turns; nothing merges or rewrites, and summarisation stays refused by design |
| Continuous batching (several sequences, one fused kernel) | **Implemented, measured** | 13x aggregate at 64 sessions; knee at n≈32 |
| Unified-KV occupancy tax (idle sessions slow others) | **Measured, unmitigated** | 1.94x slowdown at 77% pool, fully reversible |
| **Independent overlapping GPU execution** | **NOT attempted, NOT claimed** | engine serialises by design; Nsight Systems absent, Nsight Compute serialises kernels |
| Cross-process GPU weight sharing | **Not attempted** | design commits to one GPU owner |
| Remote MCP transport | **Not implemented** | localhost is not a remote deployment |
| Streaming output over MCP | **Not implemented** | |
| Context eviction, by finished interaction | **Implemented, tested** | preferred over positional trim: rejuvenation drops settled interactions at turn boundaries, oldest-first and only as far as the budget needs; a lineage still owed an answer is never dropped — see I93, I94 |
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
