# MCP contract v1

`schema_version: "1.0.0"` · transport: local stdio · server name `amoeba`

The facade exposes **cognitive verbs**. It exposes no neuocyte topology, no
sequence ids, no KV handles and no snapshot controls. A client is a cognitive
peer, not an operator of the machinery.

A test asserts that no tool name contains `neuocyte`, `snapshot`, `kv`, `fork`,
`lease` or `session`.

---

## Response envelope

Every tool returns:

```jsonc
{
  "schema_version": "1.0.0",
  "operation_id":   "op_...",      // durable handle, null for pure reads
  "status":         "completed",   // accepted | completed | failed | interrupted
  "receipt_id":     "rcpt_...",    // null when nothing was committed
  "state_version":  42,            // durable state version after the call
  "result":         { ... },
  "limitations":    [ "..." ],     // ALWAYS read this
  "error":          { "code": "...", "message": "...", "details": {} }  // on failure
}
```

`limitations` is not decoration. It carries, among others:

- `"SIMULATED BACKEND: this text was produced by a deterministic stub, not by
  model inference"` — whenever the backend is not a real model.
- `"work not admitted: <reason>"` — when the arbiter refused or reduced scope.
- `"some referenced content could not be resolved; the audit is incomplete"`.
- `"recall searches MAINTAINED memory only; raw history is evidence and is
  reached through id_audit"`.

### Error codes

`invalid_input`, `not_found`, `stale_version`, `backend_unavailable`,
`resource_exhausted`, `deadline_exceeded`, `integrity_error`, `fenced`,
`capability_unsupported`, `internal_error`.

A failed call still carries provenance: the operation is recorded with status
`failed` and its limitations.

### Idempotency

Mutating verbs accept `idempotency_key`. Replaying a key returns the stored
result with `"replayed": true` and does **not** re-run the cognition.

### Long operations

`ego_investigate` returns promptly with a durable `operation_id` and the scope
the arbiter actually accepted — which may be smaller than requested. Poll with
`ego_status(operation_id=...)`.

---

## Tools

| Tool | Principal inputs | Result |
|---|---|---|
| `ego_converse` | `message`, `conversation_id?`, `idempotency_key?`, `max_tokens?` | `answer`, `conclusion_id`, `recalled[]`, `cited_memory_ids[]`, `tool_requests[]`, `finish_reason`, `model_generation` |
| `ego_investigate` | `question`, `constraints?`, `budget_tokens?` | `plan`, `claim`, `snapshot_id`, `work{work_id, granted_budget_tokens, deadline}`, `accepted_scope` |
| `ego_recall` | `query?`, `scope` (`active`\|`all`\|`superseded`\|`retracted`), `limit` | `memories[]` each with `claim`, `confidence`, `status`, `version`, `supersedes`, `evidence{supporting[], opposing[]}` |
| `ego_status` | `operation_id?` | `ego{...}`, `operation`, `work`, `queue`, `state_version` |
| `id_introspect` | `question`, `scope?` | `measured` (read from durable state) **and** `interpretation` (model output), kept separate |
| `id_health` | `scope?` | children, resources, limits, fairness, capability flags, integrity |
| `id_audit` | `conclusion_id?` \| `operation_id?`, `focus?` | `verdict`, `findings[]`, `unresolved[]`, `evidence_reviewed`, `audit_id`, `disagreement_id?`, `asked_ego_to_defend_itself: false` |
| `id_disagreements` | `scope`, `limit` | competing claims with evidence on both sides |
| `id_maintenance` | `objective`, `scope?`, `budget_tokens?` | accepted operation, or the arbiter's reason for refusing |
| `mind_provenance` | `operation_id` | full event chain, receipts, conclusions, `hash_chain_ok`, `unresolved_content[]` |
| `mind_cancel` | `operation_id`, `reason?` | what was stopped: cancelled work, killed neuocytes, generations halted |
| `board_read` | `query?`, `post_types?`, `since_seq?`, `limit` | the swarm's discussion; **your reads are recorded** |
| `board_post` | `body`, `post_type`, `replies_to?`, `relation?` | contribute as a peer; changes no belief |
| `board_corroboration` | `post_id` | independent replication vs socially propagated agreement |

---

## Capability flags

Published by `id_health` under `capabilities`. These describe distinct things
that must never be conflated:

| Flag | Meaning |
|---|---|
| `device` | `cuda` \| `cpu` \| `none` |
| `weight_ownership` | `single_resident_set` — one loaded copy serving every session — vs `per_process` vs `none_no_weights_loaded` |
| `sessions_share_weights` | sessions address one `llama_model` |
| `kv_mode` | `shared_prefix` (fork shares physical cells) \| `copied_prefix` (fork duplicates bytes) \| `recomputed` (no cache transfer; prefix rebuilt from exact tokens) \| `simulated` |
| `n_kv_streams` | 1 = unified pool; >1 = one stream per sequence |
| `concurrency_mode` | `serialized` \| `serialized_or_batched` \| `overlapped` |
| `continuous_batching` | several sequences advance inside one fused `llama_decode` |
| `physical_overlap_verified` | **false** unless a GPU timeline profile proves simultaneous device execution |
| `prefix_reuse_verified` | **false** unless a fork has been compared against exact recomputation |
| `is_simulated` | true for the deterministic test backend |
| `model_generation` | identity covering weights, tokenizer, positional config, cache precision and runtime build |

`physical_overlap_verified` is deliberately hard to turn on. Overlapping Python
calls, several OS processes, batching and raw throughput are **not** evidence.

---

## What is NOT implemented

The nine cognitive verbs from the brief are all present and the stdio facade
works. But "the MCP contract is complete" and "MCP is fully implemented" are
different claims, and only the first is true. Verified against a live client on
protocol `2025-11-25`:

| MCP feature | Status | Why |
|---|---|---|
| Tools (10, typed input **and** output schemas) | **implemented** | the whole cognitive surface |
| `initialize` / capability handshake | **implemented** | |
| stdio transport | **implemented** | the brief's first transport |
| Resources | **not implemented** (0 served) | memory items, snapshots and the event log would map well onto resources; not required by the brief |
| Prompts | **not implemented** (0 served) | |
| Progress notifications | **not implemented** | deliberate: the brief specifies durable handles polled via `ego_status`, not streamed progress |
| Streaming / partial results | **not implemented** | nothing streams; an answer is delivered once it is terminal, whole across however many bounded turns it took (Ego's per-turn ceiling is 3072 tokens) |
| Request cancellation (`notifications/cancelled`) | **implemented** | `mind_cancel`, plus automatic cancellation when a client aborts an in-flight call |
| Sampling (server asks the client's model) | **not implemented** | see below — this is the most interesting gap |
| Elicitation | **not implemented** | |
| Logging notifications / `setLevel` | **not implemented** | diagnostics go to `state/logs/*.log` |
| Completions (argument autocomplete) | **not implemented** | |
| Resource subscriptions | **not implemented** | |
| Streamable HTTP / SSE transport | **not implemented** | explicitly a separate concern per the brief; localhost is not a remote deployment |
| Auth (OAuth / token verification) | **not implemented** | loopback + shared-secret token on the control plane only |

### Three of these matter more than the rest

**Cancellation** is the most defensible omission to close. The machinery exists
(`cancel_work`, cooperative neuocyte shutdown, wall-clock budgets); only the
protocol wiring is missing. Today a client that walks away from a long
investigation leaves it to run to its budget.

**Streaming** is user-visible. A conversational turn returns all at once.

**Sampling** is architecturally interesting rather than merely missing. The
design says an external frontier model is a *cognitive peer*, not a component.
MCP sampling would make that literal: Id could ask the connected client's model
for a second opinion on an audit, which is the only route to genuinely
independent judgement identified in
[OPEN_QUESTIONS](OPEN_QUESTIONS.md) — Id currently shares Ego's weights, so its
disagreement is not independent. Nothing in the current design uses it.

### One honest note on the capability handshake

The server declares `prompts` and `resources` capabilities while serving zero of
each. That is the Python SDK's doing: `FastMCP` registers `ListPrompts` and
`ListResources` handlers unconditionally, and the low-level server derives
declared capabilities from which handlers exist.

This is left as-is deliberately. Stripping those handlers would make a client
that calls `list_prompts` receive a *method-not-found error* instead of a clean
empty list — worse behaviour, not better. An empty list is the honest answer to
"what prompts do you have"; the capability flag is the SDK's convention for
"you may ask".

---

## Cancellation

Two routes, because a client can either change its mind or simply vanish.

**Explicit.** `mind_cancel(operation_id)` stops an operation and everything
downstream: queued work is cancelled so no neuocyte picks it up, a neuocyte
already running is killed, and the in-flight generation is asked to stop.

**Automatic.** `ego_converse`, `id_introspect` and `id_audit` are async and
cancellable. If the client aborts the call or disconnects, the facade catches
the cancellation and — under a shielded scope, so the cleanup survives the
cancellation that triggered it — tells the supervisor to cancel the operation.
The operation is addressed by idempotency key, because a cancellation can
arrive before the operation id has reached the client.

Three properties worth knowing:

- **Granularity is one decode step (~6 ms).** The generation loop checks a flag
  between tokens. A long *prefill* is not interruptible this way; that would
  need llama.cpp's `abort_callback` on the compute path, which is not wired.
- **Cancelling is not undoing.** Durable state already committed stays
  committed. Cancellation stops future work, it does not roll back the past.
- **Cancelling a finished operation is not an error.** You get
  `already_terminal: true` and its final status, because a client that cancels
  just as the work lands deserves the truth rather than a failure.

Cancellation is receipted like any other consequential act, and emits
`operation.cancelled` plus a `generation.cancelled` per stopped session.

---

## Why the facade is disposable

The mind lives in the supervisor. The facade holds no state and forwards over a
loopback control plane authenticated with a token in the state directory. When
the client disconnects, the stdio process dies and every long-lived process
keeps running — `test_mcp_client_disconnect_does_not_kill_the_mind` asserts the
child pids are unchanged.

Diagnostics never touch stdout; logging goes to `state/logs/*.log` and stderr.

Remote connectivity is a separate transport concern and is not implemented.
Localhost is not a remote deployment.

---

## Registering with a client

```jsonc
{
  "mcpServers": {
    "amoeba": {
      "command": "F:\\hexylab\\amoeba\\.venv\\Scripts\\python.exe",
      "args": ["-m", "amoeba", "mcp",
               "--config", "F:\\hexylab\\amoeba\\config.toml"],
      "cwd": "F:\\hexylab\\amoeba"
    }
  }
}
```

The supervisor must already be running; the facade will report
`backend_unavailable` until it is.
