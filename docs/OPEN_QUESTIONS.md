# Open questions and known failure modes

Nothing here is rhetorical. Each item is either unresolved or a real way this
system can fail.

## Unresolved design questions

**Is a 4B model good enough for both roles?**
Qwen3-4B-Instruct-2507 follows the structured formats Ego, Id and workers need,
and produces coherent short answers. Whether it can do *genuine* synthesis or a
*genuine* audit was not evaluated. No quality benchmark was run. The honest
position is that the plumbing is proven and the cognition is not.

**How independent is Id's judgement?**
Id has a separate process, a separate private context and no access to Ego's
snapshot, and it audits the record rather than asking Ego to defend itself.
But it shares the same weights and often the same selected memories. Separate
context is not independent judgement. Two genuinely independent judgements
would need different models — which conflicts with the one-resident-weight-set
policy. This is unresolved.

**What authority lets Id revise a belief?**
Today: none directly. Id can record an audit, open a disagreement and propose
maintenance. It cannot supersede an Ego belief on its own. That is deliberate
but it also means contradictions can accumulate unresolved, and there is no
policy yet for who breaks a tie between a user goal and a maintenance goal.

**Does an Ego-derived shared projection actually help?**
Not measured. Forking is cheap and correct, but whether a worker inheriting
Ego's context produces better findings than a worker given a narrow brief is an
open empirical question — and a shared prefix spreads Ego's mistakes as
efficiently as its context.

## Runtime failure modes

**Greedy decoding is not reproducible.** Identical tokens in different cache
positions give slightly different logits (see RUNTIME.md §3). This affects
forks and recomputations equally. Do not build anything on bit-exact replay.

**Thread safety.** llama.cpp contexts are not thread safe; one lock serialises
every call. A long generation therefore blocks every other session, including
Id's health introspection path that needs inference. Health itself stays
answerable because it does not require inference.

**Unified KV taxes decode in proportion to occupancy.** Measured 1.94x
slowdown with 63 idle sessions resident, fully reversible on retirement. This
is the price of the shared pool that makes prefix sharing possible. There is no
mitigation in place beyond retiring promptly; a future option would be to trim
or evict idle sessions on a timer, which is not implemented.

**Context overflow.** `n_ctx` is a shared pool. A long Ego context plus several
forked workers can exhaust it; `llama_decode` returns 1 and the engine raises
`resource_exhausted`. There is no automatic eviction, compaction or context
trimming. A mind left running long enough will hit this.

**Snapshot growth.** Ego's context grows monotonically within a run. Nothing
trims it and nothing summarises it — deliberately, since summarising is a
different behaviour from snapshotting. Long-running Ego will eventually
overflow.

**Positional consistency.** A fork copies positions `[0, prefix_len)` and the
worker continues from `prefix_len`. Nothing currently shifts positions, so
`llama_memory_seq_add` / `seq_div` are unused and untested here. Any future
context trimming would have to deal with RoPE position shifts, and
`llama_memory_can_shift` would need checking first.

**Model change mid-flight.** A rolling upgrade would briefly need two weight
sets. Under the one-resident-set policy the old generation must be drained —
including Ego and Id sessions — before the next model loads. The refusal path
(`capability_unsupported` on a cross-generation snapshot) is implemented and
tested; the *drain* sequence is not.

**Disk full / invalid checkpoint / tokenizer mismatch.** Blob writes are
fsynced before commit, so a full disk fails the mutation cleanly rather than
committing a dangling reference. An invalid GGUF fails at load with
`backend_unavailable`. A tokenizer change alters `model_generation`, which
invalidates snapshot reuse. None of these three paths has a dedicated test.

**Hash chaining is not administrator-proof.** Anyone with write access to the
SQLite file can rewrite rows and recompute every hash. Real immutability would
need checkpoints exported to independent append-only storage. Every integrity
report states this.

**Windows process identity.** The venv `python.exe` is a trampoline, so
`Popen.pid` is not the pid of the interpreter serving RPC. Supervision uses
reachability with a grace period rather than trusting `poll()`. A child that is
alive but wedged for longer than the grace period will be killed and restarted.

**Loopback security.** The control plane is loopback TCP with a shared token in
the state directory. That stops another local process without filesystem access
to the token; it is not a security boundary against a user who can read the
state directory. There is no TLS and no remote transport.

## Not implemented

- Remote MCP transport (a separate, separately secured concern).
- Independent overlapping GPU execution. Not attempted: llama.cpp contexts are
  not thread safe, so the engine serialises every call. Nsight Systems, the one
  tool that could verify overlap if it were implemented, is not installed;
  Nsight Compute is installed but serialises kernels by design.
- Duplicated-weights multi-process comparison (condition C in the brief's
  concurrency matrix): not run, because the design commits to one GPU owner and
  the measurement would not change that decision.
- Streaming token output over MCP.
- Automatic context trimming, eviction or compaction.
- Any tool with side effects outside the state directory.
