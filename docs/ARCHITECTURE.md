# Amoeba — architecture, invariants and data model

Version 0.1.0 · schema_version `1.0.0` · Windows-native, no Docker, no WSL.

This supersedes the conflicting assumptions in the earlier design brief, in
particular everything about UKV. See [RUNTIME.md](RUNTIME.md) for the measured
runtime behaviour this design rests on.

---

## 1. Process topology

```
   external frontier client / human client   (a peer, NOT part of the mind)
                       |  MCP stdio
        +--------------v--------------+
        |   MCP facade  (disposable)  |   holds no state; dies with the client
        +--------------+--------------+
                       |  loopback JSON-lines RPC + shared token
        +--------------v--------------+
        |  Supervisor  (fixed harness)|   single state writer, arbiter,
        |  - SQLite WAL + blobs       |   admission control, lifecycle,
        |  - work queue + leases      |   process supervision, scheduling
        |  - snapshot registry        |
        +---+-------+-------+---------+
            |       |       |  spawns and owns every process below
   +--------v-+  +--v----+  +v-----------------+
   |   Ego    |  |  Id   |  | disposable       |
   | process  |<>| proc  |  | neuocytes (N)      |
   +--------+-+  +--+----+  +--+---------------+
            |       |          |   all inference goes through one service
        +---v-------v----------v---+
        |   Inference service       |  ONE resident weight set
        |   one llama_model         |  one llama_context, n_seq_max sequences
        |   one unified KV pool     |  Ego = seq 0, Id = seq 1, neuocytes 2..N
        +---------------------------+
```

Five long-lived processes (supervisor, inference, Ego, Id) plus short-lived
neuocytes and a short-lived MCP facade. Each is restartable independently.

`Ego <-> Id` signals are **relayed by the supervisor** (`side_channel` ->
each role's `signal` method), not sent peer-to-peer. One authority sees every
signal, at the cost of a hop. Either design is defensible; this is the one that
is built and tested.

---

## 2. Invariants

These are the properties the tests exist to defend. Each names the test that
pins it.

### State and provenance

**I1. One writer.** Only the supervisor holds a writable `Mind`. Every
consequential change is one `StateWriter.apply` call = one SQLite transaction
containing the state rows, the events, and the receipt.
→ `test_crash_during_commit_leaves_no_half_applied_mutation`

**I2. Raw history is append-only.** `events` rows are never updated or deleted.
A correction appends a superseding interpretation; the earlier evidence stays.
→ `test_correction_supersedes_and_preserves_contrary_evidence`

**I3. History is not memory.** Nothing is promoted from `events` into
`memory_items` implicitly. `ego_recall` searches maintained interpretations;
raw events are reached only through `id_audit` / `mind_provenance`.
→ `test_contradictory_history_does_not_become_belief`

**I4. Content before reference.** Blob bytes are fsynced before any event
referencing them commits. An orphan blob is recoverable garbage; a committed
reference to missing content is an integrity failure and is reported as one.
→ `test_missing_blob_is_detected_not_glossed_over`

**I5. Hash chaining detects mutation, not administrators.** Two distinct
attacks, so two tests: rewriting an event's payload (caught by the recomputed
hash) and excising or reordering events while leaving each internally
consistent (caught only by the `prev_hash` linkage). Every integrity report
states the administrator caveat explicitly.
→ `test_hash_chain_detects_tampering`,
`test_a_broken_chain_link_is_detected_not_just_a_tampered_payload`,
`test_hash_chain_caveat_is_stated`

**I6. Acknowledged means durable.** A receipt is returned only after commit.
Two separate properties, tested separately because one test cannot show both:
the data survives a clean close and reopen, and commits are configured to fsync
(`synchronous=FULL`). Crash durability itself is not demonstrated here — that
needs a power cut, not a test.
→ `test_acknowledged_mutation_survives_restart`,
`test_durability_pragma_is_set_where_durability_is_configured`

### Work and neuocytes

**I7. At-least-once with idempotent commits.** Replaying a `mutation_id`
returns the original receipt and does not re-apply.
→ `test_writer_itself_refuses_to_reapply_a_mutation_id` (the writer, which owns
the guarantee), `test_duplicate_commit_is_idempotent` (through the work queue)

**I8. Fencing.** Every lease bumps a fencing token. A result presenting a
superseded token is rejected, not committed.
→ `test_each_lease_advances_the_fencing_token` (the lease, which owns the
bump), `test_stale_worker_result_is_fenced` (rejection after expiry)

**I9. Neuocyte death is always safe.** Killing every neuocyte loses no state and no
work; leases expire, tokens advance, items requeue.
→ `test_killing_all_workers_preserves_state_and_resumes_work`

**I10. Retirement never destroys authoritative state.** A retired or crashed
neuocyte releases inference and snapshot resources only.
→ `test_retiring_a_worker_does_not_destroy_work_state`

**I11. Stale findings are flagged, not silently trusted.** A result pinned to an
older `state_version` commits with a `stale_against` marker for the consumer to
validate.
→ `test_findings_pinned_to_an_older_state_version_are_flagged`

### Ego snapshots (what "UKV" means here)

**I12. Only Ego publishes.** A snapshot is a versioned, immutable snapshot of a
valid prefix of **Ego's actual inference context**. It is not a merge of Ego and
Id, not a cross-model tensor format, and not a curated summary. Id's PKV is
never published.
→ `test_maintenance_workers_get_no_ego_snapshot`

**I13. Ego continues after publishing.** Publishing freezes nothing for Ego; it
keeps appending past the published prefix. Neuocytes see the frozen prefix only.
→ `test_ego_continues_independently_after_publishing`

**I14. Neuocyte tails are private.** A neuocyte's continuation is invisible to Ego
and to sibling neuocytes. No live UKV updates reach a running neuocyte; there is no
cache merging, no live prefix replacement, no shared writable KV.
→ `test_worker_tails_are_private`

**I15. Pinned for life.** A neuocyte stays on its snapshot and model generation
until retirement. Replacements fork from the newest published snapshot.
→ `neuocyte.py::_execute_ego_derived`, `test_old_snapshot_survives_while_referenced`

**I16. Referenced storage is never recycled.** Reference-counted; the newest
snapshot is always retained; releasing a referenced snapshot is refused.
→ `test_reclaim_only_unreferenced_and_superseded`

**I17. Incompatible cached tensors are never reinterpreted.** A snapshot from a
different model generation is refused; the recorded token prefix allows exact
recomputation instead.
→ `test_worker_refuses_cross_generation_snapshot`,
`test_backend_restart_invalidates_handles_but_keeps_tokens`

**I18. KV is replaceable acceleration, not memory.** Losing the inference
process loses every KV handle and no durable state.
→ `test_inference_restart_invalidates_handles_but_keeps_snapshots`

### Roles and authority

**I19. Neither half is the harness.** Ego and Id request and propose; the
arbiter decides and the writer commits.
→ `test_maintenance_recursion_is_bounded`, `test_requested_budget_is_capped_not_honoured_blindly`

**I20. Id audits the record, not Ego.** An audit resolves a conclusion through
recorded evidence without asking Ego to defend itself.
→ `test_id_audits_ego_conclusion_without_asking_ego`

**I21. Disagreement, not overwrite.** A contested audit opens a recorded
disagreement; Ego's claim is not rewritten.
→ `test_contested_audit_opens_a_disagreement_rather_than_overwriting`

**I22. The side channel changes nothing.** Signals are transient, bounded and
receipt-free; consequential changes go through the writer.
→ `test_side_channel_signal_changes_no_state`

**I23. A model can request a tool call; it cannot perform one.** Requests are
parsed out of generated text, validated against a declared schema, checked
against role permissions, executed **by the Harness**, and receipted. The
neuocyte process that holds the model's output never executes anything: it
sends the request to `tool_invoke` and receives a result.
→ `test_role_permissions_are_enforced`,
`test_no_registered_tool_can_reach_a_shell_or_the_network`,
`test_the_neuocyte_process_never_executes_a_tool_itself`

**I23b. A neuocyte cannot widen its own permissions.** `sandbox_allowed` is
read from the work row at the moment of the call, never from the request. When
it is false the sandbox tools are not registered at all, so there is no handler
to reach. Asking is not a way to be granted.
→ `test_a_neuocyte_cannot_grant_itself_the_sandbox`,
`test_sandbox_tools_are_absent_when_the_work_item_did_not_allow_them`

**I23c. A model cannot name a sandbox.** No tool takes a sandbox id: the
sandbox is resolved from `work_id` server-side and created on first use, so
there is no argument in which to put another neuocyte's sandbox.
→ `test_no_neuocyte_tool_accepts_a_sandbox_id`,
`test_two_work_items_get_different_sandboxes`

**I23d. A fenced neuocyte cannot still run code.** The lease, owner and fencing
token are checked on the work row before any tool runs, so a neuocyte that was
killed, expired or superseded is refused rather than left executing.
→ `test_a_fenced_neuocyte_cannot_invoke_a_tool`

**I23e. The tool loop is bounded three ways.** Turns, token budget and
wall-clock deadline each terminate it independently, and the binding reason is
reported rather than swallowed. A model that keeps calling tools is an expected
outcome, not a malfunction.
→ `test_the_tool_loop_stops_at_the_turn_limit`,
`test_the_tool_loop_stops_when_the_token_budget_is_exhausted`,
`test_the_tool_loop_stops_at_the_deadline`

### Reporting

**I24. No overclaiming.** `physical_overlap_verified` and
`prefix_reuse_verified` are false until evidence exists. Batching is never
reported as overlap. A simulated backend labels every result it produces.
→ `test_capability_flags_do_not_overclaim` (real backend),
`test_health_reports_capabilities_without_conflating_them` (simulated backend),
`test_simulated_backend_is_labelled_on_every_cognitive_result`

**I25. Health stays answerable.** Status and health respond while inference is
down or saturated.
→ `test_health_stays_answerable_when_inference_is_down`

**I26. A client is not the mind.** An MCP client disconnecting does not touch
any long-lived process.
→ `test_mcp_client_disconnect_does_not_kill_the_mind`

**I27. Cancelling stops future work; it does not undo the past.** Cancellation
drops queued work, kills a running neuocyte and asks the in-flight generation
to stop between tokens. Durable state already committed stays committed, and
cancelling an already-finished operation reports `already_terminal` rather than
failing.
→ `test_cancellation_does_not_undo_committed_state`,
`test_cancel_is_idempotent_and_truthful_about_finished_work`

**I28. Supervision must keep making passes.** A stalled supervision loop is the
failure that hides every other one, so its pass counter is exposed in `health`.
→ `test_supervision_keeps_making_passes`

**I29. Amoeba's directories are not reachable from outside Amoeba.** Every
state directory has inheritance removed and an explicit DACL naming only the
Amoeba account, `SYSTEM` and `Administrators`. Removing the ACEs without
severing inheritance is not enough: any later grant on the parent would flow
straight back down.
→ `test_hardening_removes_every_ace_for_everyone`,
`test_hardening_removes_inheritance_so_the_parent_cannot_regrant`

**I30. Hardening fails safe, never locked.** It is one atomic `icacls`
invocation, so no failure can leave a directory with a stripped DACL, and the
result is verified afterwards — if the Amoeba account can no longer use the
directory, inheritance is restored rather than the state being left unopenable.
→ `test_hardening_is_one_atomic_icacls_call`,
`test_hardening_rolls_back_rather_than_locking_the_account_out`

**I31. Every state directory is hardened, not just the root.** `blob_dir` and
its siblings are separately configurable, so relying on inheritance from
`state_dir` would silently leave a relocated blob store world-writable.
→ `test_every_state_directory_is_hardened_including_ones_outside_the_root`

**I32. Sandboxed code cannot modify the interpreter it runs on.** Scratch dies
with its sandbox; the runtime is shared across all of them, so a write there
would execute in every future sandbox. The container gets read+execute, and
that grant is revoked when the container is destroyed.
→ `test_sandboxed_code_cannot_modify_its_own_runtime`,
`test_destroying_a_sandbox_revokes_its_grant_on_the_shared_runtime`

**I33. Filesystem hardening is reported, not assumed.** `audit_paths` states
the residual exposure — a process running as the Amoeba account owns these
directories and can rewrite their DACLs — so the boundary is never described
as stronger than it is.
→ `test_the_audit_does_not_claim_protection_it_does_not_have`,
`test_audit_reports_exposure_instead_of_asserting_safety`

**I34. The Harness promotes the bytes it reviewed, by construction.**
Promotion materialises the *proposal blob* — content-addressed and immutable —
so the reviewed digest names the bytes that land. Substitution is not detected,
it is impossible: there is nothing mutable in the path from decision to file. A
later change to the scratch copy is recorded, because a neuocyte rewriting a
file after proposing it is a fact worth having, but it cannot influence the
result.

An earlier version re-read the scratch file and refused on mismatch. That was
fail-closed detection, and it was the best available while the scratch held the
only promotable copy; it is strictly weaker than making the substitution
impossible.
→ `test_promotion_promotes_the_reviewed_bytes_not_whatever_scratch_holds`,
`test_promotion_materialises_the_reviewed_bytes_not_whatever_scratch_holds`

**I35. Concurrent sandboxes share no handles.** Process creation names exactly
the two handles a container may inherit. Without that, `bInheritHandles=True`
means every inheritable handle in the process, so a container spawned while
another was running inherited its writable output handles. An open handle
carries the access it was granted and Windows checks the DACL at open time, so
no ACL closes this — it is a separate guarantee from I29-I33.
→ `test_a_sandbox_does_not_inherit_another_sandboxs_handles`

**I36. Sandboxes run concurrently, and that is measured as overlap.** N
containers are live simultaneously with N distinct OS processes. Asserted by
counting runs in flight, never by a speedup ratio, which a run simply getting
faster can produce.
→ `test_sandboxes_run_concurrently_rather_than_serialised`

### Host files

**I37. Nothing outside a configured filespace root is reachable.** Roots are an
allowlist in config. A path that does not resolve inside one is refused — not
sanitised, not clamped into the root, refused — and there is no default or
fallback destination, so losing the root name produces an error rather than a
write somewhere arbitrary.
→ `test_paths_that_leave_the_root_or_name_a_device_are_refused`,
`test_a_refused_path_is_never_silently_clamped`,
`test_an_absolute_host_path_outside_every_root_is_refused`

**I37b. Containment survives links.** A junction or symlink inside a root is an
ordinary-looking name that resolves elsewhere, which no string check can catch.
Paths are resolved fully and re-checked for containment, listings do not walk
through links, and writes refuse to go through one at all.
→ `test_a_junction_pointing_out_of_the_root_is_refused`,
`test_listing_does_not_walk_through_a_junction`

**I37c. A read-only root is read-only.** The mode is checked when the path is
resolved, before any handler sees it.
→ `test_a_read_only_root_refuses_writes`,
`test_a_read_only_root_refuses_writes_through_the_harness`,
`test_promotion_cannot_target_a_read_only_root`

**I37d. Containment is about files, not only paths.** An NTFS hard link is a
second directory entry for the same file record, so nothing about the path is
unusual: `resolve()` has nothing to resolve and `is_symlink()` is false, and
containment says "inside the root" while being wrong about the file. Measured
before it was fixed — a planted hard link read content from outside the root.
Files with more than one name are refused, and listings mark them.
→ `test_a_hard_link_into_the_root_cannot_be_used_to_read_outside_it`,
`test_a_hard_linked_file_is_listed_but_marked_inaccessible`,
`test_a_write_replaces_the_directory_entry_rather_than_the_file_record`

**I37e. One file has one identity, whatever the caller called it.** NTFS is
case-insensitive and keeps 8.3 aliases, so `report.md`, `REPORT.MD` and
`REPORT~1.MD` are one file. The identity key is taken from the resolved
on-disk path rather than the caller's spelling, because otherwise each
spelling keeps its own version history and a supersession made under one is
invisible from another — the prior bytes stay in the blob store but stop being
*findable*, which is the half of I38 that matters when someone is trying to
undo something.
→ `test_case_variants_resolve_to_one_identity`,
`test_an_8_3_short_name_resolves_to_the_long_name`,
`test_version_history_is_not_split_by_how_the_path_was_spelled`

**I38. No write destroys.** Prior content is content-addressed into the blob
store before any overwrite, delete or promotion-over, and the digest goes in
the event log, so every version is recoverable. Restoring is itself a write, so
undo does not lose the version it replaces.
→ `test_an_overwrite_supersedes_and_the_prior_version_is_restorable`,
`test_a_delete_keeps_the_content_recoverable`,
`test_restoring_does_not_lose_the_version_it_replaces`

**I39. A neuocyte has no verb that reaches the host filesystem.** Files leave a
sandbox only as a proposal that the Harness decides on, and no neuocyte tool
names a filespace root. The decision to put bytes on disk, and where, belongs
to whoever promotes.
→ `test_a_neuocyte_has_no_tool_that_reaches_the_host_filesystem`,
`test_promotion_can_target_a_filespace_root`

**I40. Nothing is read that was not named.** A host file enters Amoeba only via
`file_attach`, which takes an explicit path, requires it to be inside a root,
and content-addresses it on the way in so a finding about a file can later be
checked against the exact bytes that produced it.
→ `test_attaching_a_file_puts_it_in_the_work_items_sandbox`,
`test_attaching_a_file_outside_every_root_is_refused`

### The four stores

Bytes live in exactly four places, with four different lifetimes. Conflating
any two is how a "safe to destroy" claim quietly becomes false.

| Name | Where | Lifetime | Written by |
|---|---|---|---|
| **Filespace** | configured host roots | yours; outlives Amoeba | the Harness only |
| **Blob store** | `state_dir/blobs` | durable, content-addressed | the Harness only |
| **Compute sandbox** | `state_dir/sandbox/<id>` | one work item, then destroyed | code running inside it |
| **Artifact proposal** | blob store, by digest | durable evidence; never authoritative | the Harness, at propose time |
| **Accepted artifact** | a filespace root, or `state_dir/artifacts` | durable and authoritative | the Harness, on promotion |

**I42. Destroying a compute sandbox cannot destroy authoritative input,
durable evidence, or accepted work product.** The sandbox is a disposable
laboratory: everything that matters already lives outside it, or was promoted
out before it died. Input stays in Filespace and is content-addressed on the
way in; evidence is the hash chain and the blob store; accepted work product
was copied out by the Harness at promotion.
→ `test_destroying_a_sandbox_preserves_input_evidence_and_work_product`,
`test_the_four_stores_are_in_different_places`

**I42b. Sandbox lifetime is absent from the proposal state machine.**
Destroying a compute sandbox removes a copy, not *the* copy: a proposal's bytes
are content-addressed when it is made, so it stays pending and promotable by
its digest afterwards.

| | after destruction |
|---|---|
| scratch copy | gone |
| proposal record | still pending, still promotable |
| proposal bytes | preserved, and are what promotion uses |
| accepted artifact | only if someone promotes it |

Proposals used to *lapse* on destruction. That was a workaround for a
constraint that stopped existing the moment proposals became durable evidence,
and keeping it would have coupled a decision to an unrelated lifetime.
Destruction decides nothing; only `artifact_promote` and `artifact_reject` do.
→ `test_a_proposal_stays_promotable_after_its_sandbox_is_destroyed`,
`test_destroying_a_sandbox_decides_nothing`,
`test_an_undecided_proposal_keeps_its_evidence_and_its_pending_status`,
`test_rejecting_a_proposal_still_closes_it`

**I43. Code inside a compute sandbox cannot directly mutate Filespace, the
blob store, or another work item's state.** Movement across that boundary is
performed only by the Harness — inputs materialised in, artifacts promoted out.
This is what makes I42 true: if code inside could reach out, destroying the
sandbox would not bound what it had already changed. Measured from *inside* the
container, since a check from outside tests the Harness's opinion of the
boundary rather than the boundary.
→ `test_sandboxed_code_cannot_reach_filespace_blobs_or_state`,
`test_sandboxed_code_cannot_reach_another_work_items_scratch`,
`test_movement_across_the_boundary_is_only_ever_the_harness`

**I41. A receipt's digest is ground truth, verifiable from inside.** If a
receipt claims a neuocyte received bytes with digest D, then hashing the bytes
actually available to that neuocyte must produce D. This holds for attachments,
for what a neuocyte is told it wrote, and for what a proposal claims — in the
return value *and* in the durable event, because a return value nobody rereads
is not provenance.

The environment has to be trustworthy enough that a neuocyte can reason from it
as ground truth; "this is what you were given" cannot be approximate. The tests
hash from *inside the container* rather than from the test process, because a
check the Harness performs on itself proves only that the Harness is
self-consistent. The first `file_attach` passed every Harness-side check while
handing the sandbox different bytes.

There is no lossy surface and no exception. `read_file` returns text, so it
returns *exact* text or refuses: if the bytes are not UTF-8 there is no correct
string to hand back, and a flagged rendering is still something a model will
reason about as though it were the file. The refusal carries the size, the
digest and the verb that does work, so nothing is hidden except the bytes.
Arbitrary bytes are read through `run_code`, which sees them exactly.
→ `test_an_attached_files_digest_is_what_the_neuocyte_can_hash`,
`test_the_durable_event_carries_the_same_digest`,
`test_a_digest_a_neuocyte_was_told_it_wrote_is_what_is_on_disk`,
`test_a_proposed_artifacts_digest_matches_what_the_sandbox_holds`,
`test_a_non_text_read_is_refused_not_rendered`,
`test_a_host_file_read_refuses_non_text_too`,
`test_exact_text_reads_still_work_including_non_ascii`,
`test_truncation_does_not_make_a_text_file_look_like_binary`

---

## 3. Data model

SQLite in WAL mode with `synchronous=FULL`, plus a content-addressed blob store
on disk (`blobs/ab/cd/<sha256>.blob`).

| Table | Purpose | Key fields |
|---|---|---|
| `events` | append-only raw history | `seq` (monotonic), `event_id`, `run_id`, `actor_id`, `actor_incarnation`, `operation_id`, `causation_id`, `correlation_id`, `kind`, `payload_sha256` \| `payload_inline`, `prev_hash`, `event_hash` |
| `receipts` | durable acknowledgement | `receipt_id`, `mutation_id` (unique = idempotency), `prior_version`, `result_version`, `event_seq_from/to`, `outcome` |
| `blobs` | content index | `sha256`, `size`, `encoding`, `schema` |
| `state_version` | single monotonic counter | `version` |
| `memory_items` | maintained interpretations | `memory_id`, `kind`, `claim`, `confidence`, `status`, `version`, `supersedes`, `created_by` |
| `memory_evidence` | supporting **and** opposing | `memory_id`, `stance`, `event_id`, `blob_sha256`, `note` |
| `conclusions` | auditable Ego outputs | `conclusion_id`, `claim`, `uncertainty`, `alternatives`, `operation_id`, `produced_by`, `review_status`, `model_identity`, `snapshot_id` |
| `conclusion_evidence` | what a conclusion rests on | `event_id`, `blob_sha256`, `memory_id` |
| `work_items` | leased queue | `work_id`, `objective`, `work_class`, `origin_actor`, `snapshot_id`, `model_generation`, `pinned_state_ver`, `status`, `lease_owner`, `lease_expires`, `attempt`, `fencing_token`, `budget_tokens`, `deadline`, `maintenance_depth` |
| `operations` | externally visible units | `operation_id`, `kind`, `actor`, `status`, `idempotency_key`, `request_blob`, `result_blob`, `limitations` |
| `agents` | identity and incarnation | `agent_id`, `role`, `incarnation`, `status`, `pid`, `session_handle`, `snapshot_id`, `model_generation` |
| `snapshots` | published Ego prefixes | `snapshot_id`, `version`, `actor`, `model_generation`, `token_count`, `tokens_blob`, `text_blob`, `kv_mode`, `backend_handle`, `refcount`, `status` |
| `snapshot_refs` | reference counting | `ref_id`, `snapshot_id`, `holder`, `acquired_at`, `released_at` |
| `audits` | Id verdicts | `audit_id`, `target_kind`, `target_id`, `verdict`, `findings`, `unresolved`, `evidence` |
| `disagreements` | competing claims | `claim_a`/`actor_a`, `claim_b`/`actor_b`, `evidence_a`/`evidence_b`, `status` |

`tokens_blob` is what makes a snapshot survive everything: the exact token
prefix is durable content, so the context can be rebuilt by recomputation after
a restart, a backend change or a model change.

---

## 4. Neuocyte lifecycle

1. The supervisor publishes (or reuses, if fresh) an Ego snapshot at an
   inference boundary — `ensure_snapshot`.
2. The neuocyte acquires a **reference** on that snapshot (refcount +1).
3. It instantiates a session from it:
   - `fork_prefix` when `kv_mode == "shared_prefix"` — a physically shared
     prefix, reported as `forked_shared_prefix`;
   - otherwise `restore_prefix` — exact recomputation of the recorded tokens,
     reported as `recomputed_exact_prefix`.
   The two are reported distinctly and never conflated.
4. It appends its private instruction tail and generates.
5. It commits its finding with its fencing token and its pinned state version.
6. It releases the reference, closes its session, and retires.

Maintenance neuocytes skip steps 1–3 entirely: they receive a narrow task plus
state references, never a snapshot of Id's private context.

---

## 5. Scheduling

Weighted fair between `user` and `maintenance` work, with two hard guarantees
layered on top:

- **Reserved slots.** Each class holds slots the other can never take, so
  neither starves.
- **Bounded maintenance.** `maintenance_depth` caps recursion;
  `max_maintenance_per_hour` caps rate. A maintenance job that spawns
  maintenance jobs terminates.

Retirement triggers: task completion, wall-clock budget, token budget,
age, staleness, failure, or supervisor shutdown.

**Retirement is a throughput mechanism, not hygiene.** With `kv_unified=True`
the KV pool is shared, and attention is computed over its used extent, so a
session that merely *exists* taxes every other decode. Measured: 63 idle
sessions slow an unrelated probe session by 1.94x, recovering exactly on
retirement ([BENCHMARKS §2](BENCHMARKS.md#2-resident-idle-sessions-tax-every-other-decode)).
A neuocyte that finishes but does not release its session slows the whole mind.

---

## 6. Recovery

On supervisor start:

1. Verify the hash chain and every committed content reference.
2. Acquire the single-supervisor lock (stale-PID aware).
3. Mark all `neuocyte` agents crashed; mark `ego`/`id`/`inference` crashed so they
   re-register with a new incarnation.
4. Null every `backend_handle` and set those snapshots to `kv_mode=recomputed`.
5. Release every outstanding snapshot reference; zero refcounts.
6. Requeue every leased work item with `fencing_token + 1`.
7. Mark in-flight operations `interrupted` rather than reporting them complete.
8. Emit `supervisor.recovery` and `run.started`.

No step depends on a neuocyte being alive.

Child processes are supervised by **reachability**, not only by process exit:
on Windows the venv `python.exe` is a trampoline, so `Popen.pid` is not the pid
of the interpreter that serves RPC, and a killed child can leave the trampoline
behind with `poll()` still returning `None`.

Three properties that took a real bug each to arrive at:

- **Supervision runs on its own thread and starts before startup finishes.**
  `start()` used to block up to 180 s per role waiting for its port, with
  supervision beginning only afterwards — so a role that died inside that
  window went unrestarted for three minutes while the supervisor sat blind.
  Role readiness is now advisory logging; supervision brings up whatever is
  not answering. → `test_a_role_that_dies_during_startup_is_still_restarted`
- **Liveness probes and work calls use separate connection pools.** A probe
  must fail fast (2 s, one attempt) or one dead child makes every `health`
  call block on the patient reconnect window — including the supervision loop
  trying to restart it. They must not share a pool either: a 2 s probe socket
  reused for a minute-long call looks exactly like a dead child.
  → `test_health_stays_fast_while_a_child_is_down`
- **Termination actively obtains a connection to say `shutdown`.** Relying on
  a cached work client meant that once health moved to the probe pool, no
  shutdown was ever sent and every child had to be force-killed after a
  timeout. → visible as a 3x slower test suite before the fix.

---

## How these invariants are verified

**Every important architectural claim needs at least one test whose failure
condition directly expresses that claim, at the layer where the guarantee
lives.** A passing test is weak evidence; a test that *fails when the guarantee
is removed* is the real thing.

`scripts/verify_invariants.py` applies one targeted mutation per invariant —
a minimal edit that negates precisely that claim, at the layer that owns it —
runs only the tests named for it, and **requires them to fail**. Source is
always restored.

```powershell
.\.venv\Scripts\python.exe scripts\verify_invariants.py          # all
.\.venv\Scripts\python.exe scripts\verify_invariants.py --only I7,I8
```

Current result: **21 of 21 defended, 0 weak.**

### What the first run found

Six of twenty-one tests passed with their guarantee removed. Every one was
passing for a different reason than its name claimed:

| Claim | Why the test did not express it |
|---|---|
| I7 idempotency | The test went through `WorkRepo.complete`, which carries its *own* `receipt_for` short-circuit, so the writer's check could be deleted entirely. Now asserted against `StateWriter` directly. |
| I8 fencing | The test expired the lease first, and `expire_leases` *also* bumps the token — so the bump in `lease()` was never exercised. Now two consecutive leases with no expiry between them. |
| I16 refcount | The test only checked `reclaimable_snapshots()`, which filters by refcount separately. The guard inside `mark_snapshot_released` was untouched. Now asserted. |
| I6 durability | A clean close and reopen cannot observe `fsync`; the test stayed green with `synchronous=OFF`. The claim was overstated and is now split: logical persistence *and* the pragma, asserted where it is configured. Crash durability needs a power cut, not a test. |
| I24 no-overclaiming | The named test asserts the *real* backend; the simulated one was covered by a differently-named test that was not listed. |
| I5 hash chain | Not a weak test. The claim is defended **twice** — `chain_hash` folds the predecessor into every event hash, *and* `prev_hash` is compared directly — so removing either leaves the other catching excision. Negating it requires both, which the harness now does. The redundancy is deliberate and is noted in `events.py` so nobody "simplifies" one away. |

### Guarding the guard

`tests/test_invariants_are_defended.py` enforces the bookkeeping in CI:

- every invariant names at least one test that **exists** (a renamed test
  silently orphans its claim);
- every mutation anchor still **matches its source** — a drifted anchor turns
  into a `SKIP`, and a skipped mutation proves nothing while looking fine;
- every invariant is either mutation-verified or **explicitly exempt** with a
  recorded reason, so an omission cannot masquerade as coverage.

Invariants needing a GPU, a live process stack, or a power cut are out of scope
for source mutation and are listed as exemptions by name.
