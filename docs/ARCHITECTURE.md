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
→ `test_a_transient_signal_changes_no_state`,
`test_a_message_that_reaches_cognition_is_attributable`

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

### Id's senses and effectors

Full reference: `ID.md` (sensory surface, the `system_pulse` contract, the effector set and its authority boundaries, and the capability-isolation model).

**I44. Id has one cheap, bounded sense of the whole organism.** `system_pulse`
answers "what is happening right now" in a single call: work, neuocytes,
scheduler capacity, inference, context pressure, resource versions, pending
decisions, rolling failure counters, storage and attention counters. It is
cached, incrementally maintained, and small enough to poll — a sense Id cannot
afford to use is not a sense.
→ `test_id_can_obtain_the_complete_bounded_pulse`,
`test_the_pulse_excludes_bulky_content`, `test_the_pulse_is_cheap_and_cached`

**I44b. The pulse reports observations, never verdicts.** There is no
`ego_unhealthy` field. It reports the heartbeat, the occupancy and the failure
counts; deciding what they add up to is Id's cognition, and moving that
conclusion into the Harness would leave Id agreeing with a number it cannot
inspect.
→ `test_the_pulse_reports_observations_not_verdicts`

**I44c. The pulse tracks reality.** It moves when work is admitted or leased,
when failures occur, and when a versioned resource changes — and reports the
execution mode of running work, because board-naive or informed is what makes
later agreement interpretable.
→ `test_the_pulse_moves_when_work_moves`,
`test_the_pulse_moves_when_failures_happen`,
`test_a_resource_version_changes_when_the_resource_does`,
`test_the_pulse_reports_execution_mode_of_running_work`

**I44d. Configured is distinguished from embodied.** A running role primed its
context with the prompt that existed when it started; editing configuration
changes the next incarnation, not the live one. Reporting only the configured
version would assert the running mind is something it is not.
→ `test_id_can_obtain_the_complete_bounded_pulse`,
`test_id_proposals_do_not_install_themselves`

**I45. Id requests; the Harness decides.** Every Id effector goes through the
Harness, is validated there, and leaves a receipt attributed to `id` carrying
the `pulse_id` it was formed from. Id may raise findings, propose corrections
and prompts, request investigation, rejuvenation and cancellation, message Ego
and escalate to the operator — and cannot install a prompt, change scheduler
policy, or edit a belief in place.
→ `test_id_can_invoke_every_authorised_effector`,
`test_consequential_id_actions_are_receipted_and_attributed`,
`test_id_proposals_do_not_install_themselves`,
`test_id_cannot_change_scheduler_policy_or_requeue`

**I46. A neuocyte has no path to an Id-only verb.** Architectural absence, not
a permission check: Id's effectors are in no other scope's method table, so for
a neuocyte connection they do not exist. The scope is decided by the secret
presented, so there is no role, actor or caller field to forge; the generic
tool dispatcher builds only a neuocyte registry; and an unknown-method error no
longer enumerates the table, so names cannot be discovered.
→ `test_a_neuocyte_cannot_invoke_an_id_only_verb_by_name`,
`test_a_neuocyte_cannot_enumerate_the_methods_it_lacks`,
`test_a_neuocyte_cannot_spoof_its_way_into_id_scope`,
`test_the_generic_tool_dispatcher_cannot_reach_an_id_verb`,
`test_a_neuocytes_model_facing_tool_list_contains_no_id_verb`

**I46b. Ego is checked separately.** Ego and Id are both long-lived role
processes and share plumbing; they do not share authority. Ego's scope contains
no Id effector and not the pulse.
→ `test_ego_cannot_reach_ids_effectors`,
`test_the_neuocyte_scope_contains_no_id_only_verb`

### Ego's senses and effectors

Full reference: `EGO.md` (sensory surface, effector set, the three Ego->neuocyte communication mechanisms, and the prohibition on using compute scratch as a channel).

**I47. Ego states intent; the Harness owns execution.** Ego requests work at
whatever level of abstraction fits — a one-line objective or a worked plan —
and cannot instantiate a worker, choose one, set a budget, or pick a prompt
version. The architecture does not require Ego to decompose first: the work
system, the blackboard and the neuocytes may discover structure during
execution.
→ `test_ego_requests_work_and_cannot_instantiate_a_worker`,
`test_ego_can_request_independent_replication`

**I47b. Running compute scratch is not a communication channel.** Ego may know
a neuocyte exists, what work it holds, its board mode and its status. It cannot
read the sandbox. Half-written scratch is not a claim anybody made; reasoning
over it would let Ego consume something no neuocyte ever published, with no
authorship and no moment at which the worker stood behind it. Anything worth
Ego's attention crosses an explicit boundary: a board post, a work result, an
artifact proposal, or durable evidence.
→ `test_ego_cannot_inspect_compute_sandbox_scratch`,
`test_ego_sees_proposal_evidence_not_scratch`,
`test_ego_can_follow_a_result_from_work_to_answer`

**I47c. A mid-flight message goes to a work item, not a worker.** The Harness
records it and the neuocyte collects it at a turn boundary. The original
objective is never rewritten, collection is recorded so a later finding is
marked as possibly influenced, and a board-naive item refuses the message
outright — a clarification from the executive role would destroy exactly the
independence it was admitted for, quietly, in a way that still looks like
replication afterwards.
→ `test_ego_can_message_eligible_work_and_the_worker_collects_it`,
`test_board_naive_work_refuses_mid_flight_messages`,
`test_a_message_to_finished_work_is_refused`,
`test_a_neuocyte_cannot_collect_another_work_items_messages`

**I47d. Ego proposes; it does not author or execute.** Maintained state changes
by supersession through the governed path, cancellation is scoped to work Ego
originated and is still performed by the Harness, and Ego holds no Id
telemetry, no scheduler policy, no security or filespace configuration, no
promotion authority, and no verb that widens its own scope.
→ `test_ego_proposes_memory_rather_than_authoring_it`,
`test_ego_can_cancel_its_own_work_but_not_anyone_elses`,
`test_ego_cannot_reach_a_prohibited_power`,
`test_ego_cannot_widen_its_own_capabilities`,
`test_ego_sees_work_state_without_id_telemetry`

**I47e. Role authority is the credential, never a request field.** Ego-only,
Id-only and neuocyte tables are mutually disjoint where it matters, and no
caller can reach another role's verbs by naming them, by passing
identity-shaped arguments, or by re-presenting a different token mid-connection.
→ `test_a_neuocyte_cannot_invoke_an_ego_only_verb_by_name`,
`test_a_neuocyte_cannot_spoof_ego_identity`,
`test_id_cannot_invoke_an_ego_only_verb`,
`test_the_three_scopes_are_disjoint_where_it_matters`

### One Amoeba is one cognitive trust domain

> **Interaction lineage provides routing, provenance, and reduction of
> accidental cross-talk. It is not a confidentiality boundary.**
>
> **If two workloads require true zero-trust cognitive isolation, run them in
> separate Amoeba instances.**

Ego and Id are *persistent identities*. They hold a context across turns and
maintained state across interactions — that is the whole point of them, and it
is what makes the organism able to notice a contradiction between something it
was told on Tuesday and something it concluded on Friday.

That same persistence is why lineage cannot be a security boundary. Lineage
decides what enters a **turn**: a continuation reads its own delegated work
rather than a stranger's, and one interaction's evidence does not arrive in
the same bundle as another's question. It does nothing about what Ego already
saw three turns ago, because a persistent mind does not forget between
requests.

So lineage buys three real things, and one it does not:

| | |
|---|---|
| routing | an answer reaches the request that asked for it |
| provenance | what a turn was given is recorded and reconstructable |
| reduced cross-talk | unrelated evidence does not pollute an unrelated answer |
| **confidentiality** | **no** |

The alternative — per-client amnesia compartments inside one Ego — would mean
a persistent identity that is required to forget selectively, which is both
far harder to get right and a worse thing to depend on. A boundary you can
point at is worth more than a maze you have to trust.

Multiple API and MCP clients sharing one Amoeba share a mind. That is a
deployment decision, and the honest way to make it is knowingly.

### External interfaces

Full reference: `INTERFACES.md` (authority classes, the JSON-RPC protocol and endpoints, the MCP adapter, the operator console, and why an I/O client has no route to protected state).

**I48. External input is not external control.** MCP and API clients submit
input and collect output. Input may cause Amoeba to do a great deal — request
workers, run tools, fill the blackboard, propose artifacts, change maintained
cognition — and none of that makes the caller a control-plane actor, because
none of the verbs that did it are reachable from the external surface.
→ `test_an_api_client_can_submit_input_and_collect_output`,
`test_an_external_client_cannot_reach_a_control_verb`,
`test_disconnecting_does_not_cancel_anything`

**I48b. The route is absent, not refused.** An external client that authenticates,
reads discovery, and then posts the exact spelled-out name of an operator, Ego
or Id verb receives `unknown method` — never an authorization decision. A
refusal would mean the operation exists here and something decided against it,
which is one refactor away from deciding differently. Checked twice: at the
adapter, and against the credential the adapter itself holds.
→ `test_discovery_then_calling_the_exact_operator_verb_anyway`,
`test_discovery_does_not_reveal_privileged_methods`,
`test_the_operator_surface_and_the_external_surface_are_separate_tables`

**I48c. Identity is the credential.** `client_id` comes from the authenticated
key; fields named `client_id`, `role`, `actor`, `caller` or `scope` in a request
are discarded rather than honoured. "My interactions" is a fact about who asked,
not a filter that could be widened.
→ `test_spoofed_identity_fields_buy_nothing`,
`test_mcp_and_api_reach_the_same_semantic_operations`

**I48d. Loopback is not authentication.** Every request needs a credential,
cross-origin browser requests are refused before dispatch, and the operator
session travels in a header rather than a cookie.
→ `test_a_credential_is_required_even_on_loopback`,
`test_a_cross_origin_browser_request_is_refused`,
`test_an_api_key_cannot_reach_the_operator_surface`

**I48e. External bytes are admitted input, never a path.** Attachments are
content-addressed with exact-byte provenance and an attachment name is a label:
no host path is accepted, no filespace is written, and knowing an artifact id or
digest is not authority to fetch anything.
→ `test_attached_bytes_enter_as_admitted_input_with_exact_provenance`,
`test_an_attachment_name_is_a_label_not_a_path`

**I48f. The console is a cockpit, not an authority.** Operator actions go
through the Harness and are receipted; dashboard code opens no database and
touches no filesystem. Running on loopback grants nothing.
→ `test_the_operator_can_govern_through_the_harness`,
`test_operator_governance_actions_are_receipted`,
`test_the_dashboard_never_touches_the_database_or_filesystem`,
`test_accepting_a_prompt_does_not_silently_change_cognition`

### The prompt library

Full reference: `PROMPTLIB.md` (namespaces and lineage vectors, inheritance and prompt composition, the bootstrap comparison, the governance state machine, the three cascade modes, and incarnation binding).

**I49. Runtime cannot establish a new top-level namespace.** `godmode`,
`operator`, `supervisor` and any other new root are not requests that get
refused — they are sentences the runtime cannot say. Two independent defences:
`validate_namespace` rejects a name whose root is not `ego` or `id`, and
`create_version` rejects a top-level namespace that has no versions yet, which
is what stops a *known* root being conjured on an empty library. Establishing
one lives in `establish_root`, which no scope table names and no RPC verb
calls — the permission is which function you can reach, not a flag you decline
to pass.
→ `test_runtime_cannot_invent_a_new_root`,
`test_runtime_cannot_establish_even_a_known_root`,
`test_establishing_a_root_is_not_reachable_from_any_scope`,
`test_establish_root_refuses_an_existing_namespace`

**I49b. An existing root may receive governed new versions.** Ego's and Id's
doctrine has to be able to change. A new version of `ego` is a candidate like
any other and goes through validation, evaluation, Operator approval and
selection; Id can propose one and cannot approve it. The first implementation
forbade this along with new roots — it conflated "no new top-level namespace"
with "no new version of a root", and made a doctrine change require editing a
shipped file. Negating either of I49 and I49b leaves the other defended, which
is checked rather than assumed.
→ `test_runtime_can_propose_a_new_version_of_an_existing_root`,
`test_id_can_propose_root_doctrine_through_governance`,
`test_a_root_candidate_cannot_be_selected_before_approval`,
`test_a_root_version_completes_the_normal_governance_path`

**I50. An edited prompt file is a candidate, never an override.** A shipped
file is authoritative exactly once, when its namespace does not yet exist.
After that the database is authoritative and the file is a proposal: editing a
prompt and restarting creates a governed candidate and changes nothing that is
running. "Restart and the organism thinks differently" is a change nobody
chose to make.
→ `test_edited_prompt_file_becomes_a_candidate_not_an_override`,
`test_bootstrap_establishes_roots_and_is_idempotent`

**I51. A child pins an exact parent version.** Approving a new `ego` changes
no existing descendant. Ancestry is resolved by walking the *stored* parent
bindings, never by consulting what is selected now — which is the whole reason
the binding is stored.
→ `test_child_pins_an_exact_parent_version`,
`test_pinning_a_nonexistent_parent_version_is_refused`

**I52. A lineage reference resolves exactly, or not at all.**
`ego.neuocyte.research@3.7.5` means that ancestry. Component count must equal
namespace depth, so an incomplete reference is malformed rather than resolved
against today's parents; and a complete one that does not match the real
ancestry is refused, naming what the actual lineage is. Guessing which level
was omitted is how an explicit request for a historical profile quietly
becomes a current one.
→ `test_a_lineage_reference_resolves_to_one_thing_or_nothing`,
`test_lineage_reference_component_count_must_match_depth`,
`test_historical_lineage_still_resolves_after_the_tree_moves`

**I53. Selection changes what is born next, not what is alive.** Approving and
selecting a version affects new incarnations only. A running Ego, Id or
neuocyte keeps the profile it was bound to, because its context was primed
with those bytes; pretending otherwise would make the incarnation binding a
lie. Resolution of an explicit lineage reads no selection table at all.
→ `test_selection_does_not_change_a_running_mind`,
`test_resolution_is_independent_of_the_selection_table`

**I54. A cascade moves the pin and copies definitions unchanged.** Propagating
a parent version rebases each descendant onto it while its local definition is
copied byte-for-byte — asserted on the local digest, which deliberately
excludes the parent binding. Cascade descends level by level, and reports what
it skipped, so an empty cascade never looks like a complete one.
→ `test_cascade_copies_local_definitions_unchanged`,
`test_cascade_descends_level_by_level`,
`test_cascade_queue_creates_candidates_without_selecting`,
`test_cascade_reports_what_it_skipped`, `test_cascade_none_moves_nothing`

**I55. Id evaluates and proposes; the Operator decides.** Id can read the
whole family tree, compare lineages, record a verdict and author a candidate.
Approving, selecting and cascading appear in **no** scope table, so there is
no secret Id could present that resolves to them. An endorsement that promoted
would make Id the approver by a longer route.
→ `test_id_may_propose_but_the_approval_verbs_are_absent`,
`test_only_an_approved_version_may_be_selected`,
`test_state_machine_refuses_illegal_jumps`

**I56. The prompt library is absent from the external surface.** No MCP or API
client can read the organism's cognitive configuration, let alone propose to
it. Defended in depth: the adapter allowlist and the credential scope are
independent lists, and both would have to be widened.
→ `test_the_prompt_library_is_absent_from_the_external_surface`,
`test_neuocytes_cannot_read_or_govern_the_library`,
`test_ego_cannot_govern_its_own_prompt`

**I57. An incarnation binding freezes bytes, not a pointer.** At birth, a mind
records the resolved prompt digest, the config digest and the full lineage
vector. Cognition that happened stays explicable from what the organism held
at the time; re-resolving against a library that has since moved would quietly
rewrite history. Where a neuocyte inherits its ancestors' text physically from
a forked context, the binding records the injected bytes and the inherited
prefix **separately**, rather than claiming the whole profile was handed over.
→ `test_binding_freezes_resolved_bytes_not_a_pointer`,
`test_role_system_text_prefers_the_library_over_the_constant`,
`test_suffix_after_reproduces_the_resolved_profile`

**Model variables are only the ones the backend applies.** Six:
`temperature`, `top_p`, `top_k`, `max_output_tokens`, `seed`,
`stop_sequences`. An unknown name is refused rather than dropped — a silently
discarded `repetition_penalty` would be a profile claiming to have shaped
cognition that it did not. Harness constraints narrow a profile and never
widen it, and are not a parameter any caller can supply.
→ `test_unsupported_model_variables_are_refused_not_dropped`,
`test_every_model_variable_reaches_the_backend`,
`test_harness_constraints_narrow_and_never_widen`

### Persistent roles and bounded turns

Full reference: `TURNS.md` (the mailbox, trigger bundles, Ego and Id wake
semantics, stop reasons, continuation, context homeostasis, crash recovery and
turn provenance).

Ego and Id are persistent **identities** whose cognition happens in bounded
turns. The Harness owns when a turn begins, what triggered it, what inputs are
admitted and what happens when it ends; the model owns only the reasoning
inside one.

**I64. A persistent role runs one bounded turn at a time.** Defended twice: the
role process runs exactly one turn thread, so a second concurrent turn has
nowhere to execute, and `role_turns` carries a partial unique index over open
turns so a second claim is a constraint violation. Before this, `ego_converse`
called into the Ego process synchronously over a threaded RPC server, and two
callers produced two concurrent turns against one inference session.
Serialization is per role: Ego and Id still run concurrently, and neuocytes are
untouched.
→ `test_a_role_cannot_have_two_turns_at_once`,
`test_ego_and_id_run_concurrently`,
`test_input_arriving_during_an_ego_turn_is_queued_not_injected`

**I65. Input arriving during a turn waits for the next boundary.** The profile,
the environment manifest and the trigger bundle are frozen in one transaction
before generation starts, and nothing reopens them. Events wake cognition; they
do not interrupt it. Queued is also not *seen*: a trigger becomes a cognitive
input only when a turn bundles it, and being claimed by a role that then died
is not consumption either.
→ `test_input_arriving_during_a_turn_waits_for_the_next_one`,
`test_queued_is_not_seen`, `test_trigger_order_is_deterministic_and_bundled`,
`test_a_burst_larger_than_the_bundle_leaves_the_rest_queued`,
`test_the_bundle_preserves_every_member_identity`,
`test_inputs_arriving_together_are_bundled_into_one_turn`

**I66. Turn-end reasons are first class.** `model_stop`, `max_output_tokens`,
`token_budget_exhausted`, `tool_turn_limit_reached`, `deadline_reached`,
`cancelled`, `context_pressure`, `backend_error`, `role_failure`,
`no_environment` — not collapsed into "the turn ended", because they drive
different continuation behaviour. Detected from the substrate rather than from
the model cooperating.
→ `test_stop_reasons_are_recorded_distinctly`

**I67. The Harness continues an interrupted thought.** A non-terminal stop
earns another turn, decided by the Harness. A thought cut off by an output
ceiling cannot be relied on to ask for its own continuation, because being cut
off is what stopped it. A continuation is a new bounded turn with `parent_turn`
recorded, not an invisible extension.
→ `test_a_non_terminal_stop_schedules_a_continuation`,
`test_a_terminal_stop_leaves_the_role_idle`

**I68. The continuation chain is bounded.** Found by running it: with a backend
that always truncated, every turn scheduled a successor that also truncated,
and the organism burned its context until inference refused the prompt. An
unbounded continuation policy is a token furnace. The chain stops at
`max_continuations` and that is recorded rather than silent.
→ `test_the_continuation_chain_is_bounded`,
`test_continuation_depth_counts_only_the_chain`

**I69. A role that dies mid-turn does not swallow its inputs.** Triggers are
consumed at commit, not at claim, so a turn left running by a process that is
gone returns its inputs to the queue with `deliveries` incremented — a replay
is visible rather than looking like a new event. At-least-once with preserved
identity, stated rather than claimed away as exactly-once. A trigger that
repeatedly outlives the role reading it expires instead of becoming an undying
poison message.
→ `test_a_role_that_dies_mid_turn_does_not_swallow_its_inputs`,
`test_a_trigger_that_keeps_killing_the_role_expires`,
`test_a_completed_turn_does_not_reconsume_its_triggers`

**I76. A role reads the request, not a preview of it.** The trigger summary
is a bounded label for operator listings; the body is what was actually said.
Rendering only the summary meant Ego answered questions it was never fully
asked — and a request's constraints sit at its end far more often than in its
first 400 characters. The body is rendered from the content store with a
stated budget, and when that budget truncates, the rendering says so and names
the digest holding the rest.
→ `test_a_turn_shows_the_request_not_a_preview`,
`test_an_oversized_body_says_that_it_was_truncated`,
`test_a_trigger_with_no_payload_still_renders`

**I77. An adverse audit is never recorded as a favourable one.** Verdicts are
matched as whole words. Substring matching read `unsupported` as `supported` —
the word contains it — so an adverse audit became a favourable durable verdict
*and* suppressed the disagreement it should have opened, because that branch
tests for `unsupported`. An answer naming more than one verdict is treated as
unstated rather than resolved by ordering: a model echoing the menu back has
not reached a judgement.
→ `test_an_adverse_audit_is_not_recorded_as_a_favourable_one`

**I78. An effector whose target is a role stays callable.** `role` means "who
is asking" everywhere here, so the tool loop strips it — which made two Id
effectors permanently uncallable, because for them `role` named the *target*.
The collision was the defect; the parameter is `target_role`, and the
authority strip is unchanged.
→ `test_id_can_actually_call_the_effectors_that_name_a_target`

**I79. "Complete" means answered.** External input is queued and answered at a
turn boundary. An interaction whose wait elapsed used to be marked complete
with an empty answer — telling the client, permanently, that nothing was the
organism's reply. The worker is asynchronous by construction, so it waits for
the answer across however many continuation turns the thought needs, and fails
truthfully if it never arrives rather than reporting silence as a result.
→ `test_an_unanswered_interaction_is_not_reported_complete`

**I76. A role reads the request, not a preview of it.** The trigger summary
is a bounded label for operator listings; the body is what was actually said.
Rendering only the summary meant Ego answered questions it was never fully
asked — and a request's constraints sit at its end far more often than in its
first 400 characters. The body is rendered from the content store with a
stated budget, and when that budget truncates, the rendering says so and names
the digest holding the rest.
→ `test_a_turn_shows_the_request_not_a_preview`,
`test_an_oversized_body_says_that_it_was_truncated`,
`test_a_trigger_with_no_payload_still_renders`

**I77. An adverse audit is never recorded as a favourable one.** Verdicts are
matched as whole words. Substring matching read `unsupported` as `supported` —
the word contains it — so an adverse audit became a favourable durable verdict
*and* suppressed the disagreement it should have opened, because that branch
tests for `unsupported`. An answer naming more than one verdict is treated as
unstated rather than resolved by ordering: a model echoing the menu back has
not reached a judgement.
→ `test_an_adverse_audit_is_not_recorded_as_a_favourable_one`

**I78. An effector whose target is a role stays callable.** `role` means "who
is asking" everywhere here, so the tool loop strips it — which made two Id
effectors permanently uncallable, because for them `role` named the *target*.
The collision was the defect; the parameter is `target_role`, and the
authority strip is unchanged.
→ `test_id_can_actually_call_the_effectors_that_name_a_target`

**I79. "Complete" means answered.** External input is queued and answered at a
turn boundary. An interaction whose wait elapsed used to be marked complete
with an empty answer — telling the client, permanently, that nothing was the
organism's reply. The worker is asynchronous by construction, so it waits for
the answer across however many continuation turns the thought needs, and fails
truthfully if it never arrives rather than reporting silence as a result.
→ `test_an_unanswered_interaction_is_not_reported_complete`

**I73. A role is never wedged by a turn it did not close.** One open turn per
role is a database constraint, so a turn left running blocks every future turn
for that role — the role heartbeats, reports healthy, and never thinks again
while its mailbox fills. Recovery therefore runs whenever supervision restarts
a role, not only at supervisor start, and a turn still open long past the
role's own deadline is swept as abandoned.
→ `test_recovery_can_target_one_role`,
`test_a_turn_nobody_closed_does_not_wedge_the_role`

**I75. A turn that is no longer running cannot act.** Being refused at
commit is not enough. A turn that hung, was swept as stale and had its inputs
handed to a replacement could still *act* — requesting work, posting findings,
recording conclusions into an organism that had moved on — because a role's
effector call carried no turn identity at all. Every capability now goes
through `role_tool_invoke` carrying the turn it belongs to, exactly as a
neuocyte carries its fencing token. The turn id is the capability: minted by
the Harness, given only to the role that claimed that turn, and readable
nowhere a role can reach, so there is no `role` argument to forge. A turn that
is not `running` buys nothing, and the verb must still be one that role is
offered — this narrows what a role may do and widens nothing.
→ `test_a_turn_that_is_no_longer_running_cannot_act`,
`test_a_role_holding_no_turn_cannot_act`,
`test_capabilities_reach_the_harness_through_the_fence`

**I74. A role cannot forge attribution in a mailbox.** `role_enqueue_trigger`
takes `source` as an argument, so it is absent from every role scope: Ego
holding it could queue an operator-attributed instruction into Id's cognition.
Identity is the credential here as everywhere else, and roles message each
other through effectors that attribute the sender themselves.
→ `test_a_role_cannot_forge_attribution_in_a_mailbox`

**I70. A turn records exactly what caused it.** Role, incarnation, profile
reference and digest, environment digest and blob, the bundle digest and blob,
trigger kinds, stop reason, model generation and `parent_turn`. The bundle and
environment are content-addressed *before* the turn runs, so a past turn's
inputs are read back rather than recomputed from state that has moved. Later
changes never rewrite a historical turn.
→ `test_a_turn_records_what_caused_it`,
`test_later_changes_do_not_rewrite_a_historical_turn`,
`test_queued_at_and_consumed_by_remain_distinguishable`,
`test_a_role_turn_records_profile_environment_and_triggers`

**I71. The scheduler is substrate, never a cognitive component.** No scheduler
neuocyte, no supervisor neuocyte, no arbiter agent: the scheduling modules
contain no inference calls at all, and a test asserts it. `role_claim_turn`,
`role_complete_turn` and `role_enqueue_trigger` sit in the role *process* scope
and are not model-facing — claiming your own next turn is not a cognitive act,
and a mind that could would be scheduling itself. Cognition may decide what it
wants; Harness physics decides when execution happens.
→ `test_the_scheduler_is_substrate_not_a_cognitive_component`,
`test_the_turn_verbs_are_not_offered_to_the_model`,
`test_the_mailbox_is_absent_from_the_external_surface`

**I72. One ingestion path: cognition happens only in claimed turns.** A verb
that calls into a role process to make it think bypasses the mailbox and runs
a generation against the session a claimed turn may already be using --
`converse`, `investigate`, `introspect` and `audit` all did. There is
deliberately no fast lane for an idle role: that would make conversational
ordering a race between whoever called while the role happened to be free.
Status, health and snapshot calls into a role are allowed, because they read
rather than generate.
→ `test_no_verb_generates_cognition_outside_the_mailbox`

**Cognitive results never lose their labelling.** A simulated backend is
declared as a limitation on every cognitive verb, and `is_simulated` is carried
out through the turn result. This was briefly lost when conversation moved onto
the turn model, which is why it is asserted rather than assumed.
→ `test_simulated_backend_is_labelled_on_every_cognitive_result`,
`test_a_simulated_answer_is_always_labelled`

**Rejuvenation stays Harness-initiated.** `context_rejuvenate` is in no role
scope. A role reports `context_pressure` and the Harness reclaims context
between turns, then hands over the replacement session; identity, incarnation,
profile binding and mailbox all survive.
→ `test_ego_cannot_reach_a_prohibited_power`

**Ego wakes because something relevant happened.** Never because its process
exists. Relevance comes from a recorded relationship — `origin_actor` on the
work row — not a heuristic, so work Ego did not originate does not wake it.
Id additionally gets a startup turn and a deterministic heartbeat that backs
off while the organism is quiet; the heartbeat is explicit input with its own
kind, not a fake user message.
→ `test_id_wakes_at_startup_and_ego_stays_quiet`,
`test_id_receives_a_deterministic_heartbeat`,
`test_an_idle_organism_does_not_spin`,
`test_the_operator_can_see_the_mailbox`

### Profile, environment, turn input

Three things reach a mind, and conflating any two is how a system ends up
rewriting its constitution because a tool was added:

| Layer | Question | Source | Lifetime |
|---|---|---|---|
| **Profile** | who am I, how should I think | Prompt Library | bound at incarnation, immutable |
| **Environment** | what exists, what can I do now | Harness | rebuilt per turn |
| **Turn input** | what should I think about | the trigger | one turn |

Neuocytes always worked this way. Ego and Id had a profile and a turn input and
nothing in between, so a newly approved `ego.neuocyte.research` was invisible
to Ego unless somebody rewrote Ego's root prompt — constitutional doctrine was
the only channel for environmental fact.

**I58. A role is never offered a capability it cannot invoke.** The manifest's
verb list *is* `scopes.model_facing_verbs(role)`, every entry is resolved
against the supervisor's live dispatch table, and each description and argument
schema is read off the real function. There is no second hand-maintained list
to drift. The inverse is also checked: everything in `EGO_ONLY` and `ID_ONLY`
is discoverable, so an effector cannot exist in dispatch while being invisible
to the mind it was built for. Lifecycle plumbing — `register_agent`,
`heartbeat`, `bind_profile`, `role_environment` — is deliberately not offered:
a mind is not invited to operate its own life support.
→ `test_model_facing_verbs_are_a_subset_of_the_roles_scope`,
`test_every_deliberate_role_effector_is_discoverable`,
`test_role_environments_do_not_leak_across_roles`,
`test_lifecycle_plumbing_is_not_offered_to_the_model`,
`test_an_undeclared_model_facing_verb_is_caught_at_build`

**I59. A role can actually execute what its environment offers.** Ego and Id
have a bounded Harness-mediated tool loop: the model emits one call, the
Harness validates and runs it, the result is appended, generation resumes.
Before this, `roles.py` parsed tool requests and reported them without running
them — injecting a manifest on top of that would have advertised capabilities
to a mind that could not use any of them. The loop lives in the role process;
the authority does not.
→ `test_ego_can_invoke_an_advertised_sense`,
`test_ego_can_invoke_an_advertised_effector`,
`test_id_can_invoke_an_advertised_sense_and_effector`,
`test_a_verb_outside_the_environment_is_refused_and_never_dispatched`,
`test_a_failing_tool_is_reported_to_the_model_not_fatal`,
`test_the_tool_loop_is_bounded_by_turns`,
`test_the_tool_loop_is_bounded_by_the_deadline`,
`test_a_role_credential_cannot_reach_another_roles_effectors`

**I60. Authority-shaped arguments cannot widen authority.** `role`, `actor`,
`scope`, `client_id` and friends are discarded, and identity arguments the verb
genuinely takes (`reader`, `author`, `produced_by`) are overwritten with the
role's own name. Arguments the manifest never advertised are dropped, so a
model cannot smuggle a parameter the verb was not offered as taking. None of
this is what makes the boundary hold — the scope table is — but a request
should not reach the Harness pretending to be someone else.
→ `test_authority_shaped_arguments_cannot_widen_authority`,
`test_undeclared_arguments_are_dropped`

**I61. One turn sees one environment.** Built once at turn start, used for the
whole turn however many tools it calls, rebuilt for the next. A manifest that
shifted mid-generation would make the transcript unexplainable: the model would
have reasoned against a world that no longer matches what provenance recorded.
If a tool call changes the world, the tool *result* is what says so. When the
Harness cannot answer, the turn has no capabilities rather than unchecked ones.
→ `test_the_environment_is_built_once_per_turn`,
`test_the_environment_is_rebuilt_for_the_next_turn`,
`test_the_environment_reaches_the_context_before_the_turn_input`,
`test_a_turn_without_an_environment_offers_nothing`,
`test_a_role_turn_carries_a_frozen_environment`

**I62. A changing environment does not require rewriting doctrine.** Approving
`ego.neuocyte.do_thing` makes it discoverable to Ego on its next turn with
`ego`'s own prompt untouched and still at the same version. Only
approved-and-selected lineages appear: an approved version nobody selected is
not something Ego can be given, and advertising it would describe the library's
possibilities rather than the organism's capability.
→ `test_a_new_profile_becomes_visible_without_editing_doctrine`,
`test_an_unselected_profile_disappears_from_the_environment`,
`test_id_environment_carries_id_profiles_not_egos`,
`test_the_environment_digest_tracks_the_authoritative_state`

**Environment provenance is reconstructable, not merely digested.** The exact
manifest bytes are content-addressed before they are handed over, and
`role.turn_began` records role, incarnation, profile reference, prompt digest,
environment digest, environment blob, trigger and model generation. A digest
whose content cannot be recovered is not provenance: profile + exact
environment + turn input is what produced a piece of cognition, and all three
are recoverable afterwards. Identical environments across turns resolve to one
blob.
→ `test_the_exact_environment_a_turn_saw_is_reconstructable`,
`test_the_manifest_is_deterministic_for_unchanged_state`

**I63. Configuration cannot silently rewrite constitutional doctrine.**
`cfg.<role>.system_prompt` used to be appended to whatever the Prompt Library
resolved — an ungoverned second constitution with no version, candidate,
evaluation or approval. The field is gone and a non-empty value is refused at
config load with migration guidance, because silently dropping it would restart
somebody into different cognition with no signal at all.
→ `test_a_configured_system_prompt_is_refused_not_honoured`,
`test_the_role_prompt_has_no_source_but_the_library`,
`test_the_resource_digest_reports_the_library_as_its_source`

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
| `role_triggers` | the durable role mailbox | `trigger_id`, `target_role`, `kind`, `source`, `source_ref`, `summary`, `payload_sha256`, `status`, `turn_id`, `deliveries` |
| `role_turns` | one bounded turn of a persistent role | `turn_id`, `role`, `incarnation`, `profile_ref`, `environment_sha256`/`_blob`, `bundle_sha256`/`_blob`, `stop_reason`, `parent_turn` |
| `prompt_versions` | immutable nodes of the cognitive family tree | `version_id`, `namespace`, `local_version`, `parent_namespace`, `parent_version` (the pin), `prompt_mode`, `prompt_text`, `model_vars`, `local_sha256`, `state`, `origin` |
| `prompt_selections` | which approved version new incarnations get | `namespace`, `purpose`, `version_id`, `selected_by` |
| `prompt_evaluations` | Id's advisory verdicts | `evaluation_id`, `version_id`, `evaluator`, `verdict`, `evidence` |
| `prompt_decisions` | the Operator's recorded governance | `decision_id`, `version_id`, `decision`, `decided_by`, `rationale` |
| `incarnation_profiles` | what a mind was actually born with | `binding_id`, `actor_id`, `actor_kind`, `incarnation`, `profile_ref`, `lineage`, `prompt_sha256`, `config_sha256`, `effective_settings` |
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
