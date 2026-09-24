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
`test_a_burst_of_notices_is_still_bundled_into_one_turn`

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

**I80. An answer belongs to the request that asked for it.** A turn is a unit
of work, not a unit of accountability. Marking the *turn* answered let one
reply discharge every request that happened to share it, so the answer is
written onto the request, by trigger id, and a caller waits for its own. A
thought that takes four continuation turns answers the request that started
it, because the answer walks the continuation chain back to the question.
→ `test_each_request_gets_its_own_answer`,
`test_a_continued_thought_answers_the_request_that_started_it`,
`test_a_requeued_request_is_owed_its_answer_again`

**I81. No result from one interaction satisfies another merely because their
triggers shared a turn.** A turn may consume many triggers; an interaction
becomes complete only when a terminal result explicitly addressed to it has
been durably produced. Two answer-bearing requests therefore never occupy one
turn, and a turn continuing an unanswered thought admits no further request at
all -- otherwise the newcomer is satisfied by the older thought's reply
through the back door of the continuation chain.
→ `test_two_requests_never_share_one_turn`,
`test_a_continuation_does_not_adopt_a_new_request`,
`test_a_burst_of_requests_is_not_bundled_into_one_turn`

**I82. Reply routing and information ownership are different questions.**
`expects_answer` says who is owed a reply. It says nothing about whose
information a trigger carries -- so a work result belonging to one
interaction could ride into another's turn, informing a stranger's answer
while both rules above stayed satisfied. Supporting evidence is scoped to the
turn's lineage, and only an explicitly ambient trigger is seen by every
lineage. Unowned is not ambient: "unrelated" must not become the default
merely because nothing is owed a reply. What a continuation *is* given is its
own evidence, because withholding that would make the rule about keeping a
turn uninformed rather than about who is owed a reply.
→ `test_evidence_from_another_interaction_does_not_ride_along`,
`test_unowned_evidence_does_not_default_to_everyone`,
`test_unowned_evidence_is_admitted_by_a_turn_that_owns_nothing`,
`test_an_ambient_trigger_is_seen_by_any_lineage`,
`test_a_continuation_still_receives_its_supporting_evidence`,
`test_every_trigger_producer_declares_ownership`

**I83. A thought that ends without an answer says so.** A request whose turn
stopped for any reason that is not an answer -- a tool budget, a context
ceiling, an exhausted continuation chain, a crash -- is recorded as
unanswerable rather than left forever pending or quietly marked done. The
waiter is released with the reason. Silence reported as a result is the
failure mode this whole tier exists to remove.
→ `test_a_thought_that_ends_without_an_answer_says_so`,
`test_an_exhausted_continuation_chain_releases_its_waiter`,
`test_an_event_expects_no_answer`

**I84. An existing database gains the columns a release adds.** The schema is
applied with `CREATE TABLE IF NOT EXISTS`, which does nothing to a table that
already exists -- so a release that adds a column ships an organism that
starts, heartbeats, and fails on its first mailbox read. The migration
inspects the live table rather than trusting a version number, runs before the
schema script because the new indexes name the new columns, and defaults an
old request to "owed nothing" rather than inventing an answer for it.
→ `test_a_database_from_the_previous_release_gains_the_ownership_columns`,
`test_the_upgrade_does_not_invent_answers_for_old_requests`

**I85. Work delegated during a turn is accountable to that turn.** Scoping
evidence by lineage only helps if evidence carries the right one. A role's
tool call cannot supply its own operation -- `operation_id` is stripped from
model-supplied arguments along with every other authority argument -- so the
Harness supplies it from the turn, and a continuation keeps the operation it
continues. Without either link, work delegated inside a thought comes back
tagged with nothing and is held out of the very continuation that asked for
it, which would have made lineage look like a bug rather than a boundary.
→ `test_a_turns_operation_reaches_what_the_role_does`,
`test_a_continuation_keeps_the_operation_it_continues`,
`test_a_delegated_result_returns_to_the_thought_that_asked`,
`test_the_harness_does_not_override_an_operation_the_caller_gave`

**I86. A result the model cannot see whole says so.** A tool result was
serialized and cut at a fixed length with nothing said, so a model that asked
for twenty work items and was shown eight had no way to know the other twelve
existed -- and would reason confidently from a truncated world. The same
defect as I76, in the opposite direction. The Harness bounds the result,
because it is the side that can store what does not fit and therefore name a
copy something actually holds, and what it shows says how much was left out
and how to reach it. The cognitive processes render what they are given and
cut nothing. How the bound is applied -- never by cutting -- is I123.
→ `test_an_oversized_result_is_projected_whole_and_says_what_it_left_out`,
`test_the_copy_a_projection_names_is_really_there`,
`test_a_projection_claims_no_copy_it_was_not_given`,
`test_the_neuocyte_feeds_back_the_bounded_text_it_was_given`,
`test_a_tool_result_that_fits_is_shown_whole`

**I87. Evidence is never pruned; the working set is.** Retention here is a
narrower question than usual, because most of this database is evidence and
evidence is not a cache. The event chain, the receipts, conclusions, audits,
disagreements, memory and the prompt library are listed explicitly as never
pruned -- a list rather than a rule, because a rule invites the next person to
reason their way to an exception. What is forgotten is `role_triggers` and
`role_turns` past a window: an operational index into the record, not the
record. Blob bytes are never reclaimed at all, because a digest is referenced
from around twenty columns and from event payloads, and a collector that
missed one source would delete content the chain points at. The footprint
measures what such a collector would be reasoning about; that measurement can
argue for building one later, which is a better basis than confidence.
→ `test_pruning_touches_no_evidence_table`,
`test_the_event_chain_still_verifies_after_pruning`,
`test_the_footprint_reports_without_collecting`,
`test_no_scope_can_prune`

**I88. Nothing still in use is forgotten.** Age is not a reason to stop owing
somebody a reply, so a request still awaiting an answer is never old enough --
its row vanishing would leave the caller waiting on a trigger id that no
longer exists, and be reported to them as a question that never existed. A
running turn is the present whatever its timestamp says, and a turn wedged
open for a week is exactly when somebody is investigating it. A turn a
continuation still points at is kept, because `awaiting_answer` walks
`parent_turn` to find the request that started a thought: prune the parent and
the chain ends early, reintroducing by housekeeping the precise failure I80
exists to remove.
→ `test_a_request_still_owed_an_answer_is_never_old_enough`,
`test_an_open_turn_is_never_pruned_however_old`,
`test_a_turn_a_continuation_still_points_at_is_kept`,
`test_an_old_consumed_trigger_and_its_closed_turn_are_forgotten`

**I89. Ego chooses what crosses the external boundary, never whose.** A client
can send files and can be sent results, and both routes are scoped to the
interaction the current turn is answering -- resolved by the Harness from the
turn, never named by Ego. There is no interaction parameter on either
effector, so reaching another client's attachment or surfacing into another
client's results is not refused so much as unsayable. That rests on I81: a
turn holds at most one answer-bearing request, so "the interaction this turn
is answering" is a single well-defined thing rather than a guess among
several. Attachments are *named* in the bundle rather than inlined, because a
role handed the bytes has no way to decline them, and a result becomes
fetchable only by this deliberate act -- knowing a digest has never been
authority to fetch it.
→ `test_ego_cannot_read_an_attachment_from_another_request`,
`test_an_attachment_reaches_the_turn_and_can_be_read`,
`test_ego_surfaces_a_result_to_the_client_that_asked`,
`test_a_request_tells_ego_what_was_sent_with_it`,
`test_a_request_with_no_attachments_says_nothing_about_them`

**I90. Batching changes throughput, never outcomes.** A grouped generation
returns what a serial one would: the same text to the same caller, and a
failure to the caller that caused it rather than to everyone decoding
alongside them. A backend that cannot batch is not an error but a backend, so
the fallback runs the group one at a time and everybody is still served --
which matters more than the fast path, because it is the path every engine
that is not llama.cpp takes. Batches form only from requests that were already
waiting: the dispatcher blocks for the first one and then takes whatever is
queued, so contention produces batches and an idle organism pays nothing for
the machinery. The deterministic backend implements batching as a loop and
says so in its docstring, because a truthful stand-in that lets the scheduling
be tested beats a principled refusal that leaves the live path exercised only
on a GPU.
→ `test_one_broken_session_does_not_fail_the_others`,
`test_a_backend_that_cannot_batch_still_serves_everyone`,
`test_concurrent_generations_are_decoded_together`,
`test_a_lone_generation_is_not_delayed_waiting_for_company`,
`test_the_deterministic_backend_batches_without_claiming_to_measure`

**I91. A specialisation nothing can be born into does not exist.** A specialist
profile could be authored, versioned, approved, selected and advertised, and
neuocyte birth bound the base namespace regardless, because it was derived
from `work_class` alone. Ego now asks for a specialisation when it requests
the work; the library governs whether that profile exists and is approved; the
Harness records which one was actually bound. Ego names a *leaf*, never a
namespace, so a specialisation cannot reach sideways into the other role's
family tree or upward to a root -- that is unsayable rather than refused. An
unavailable specialisation falls back to the base, because refusing a job
because a prompt was missing is a worse failure than doing it on the baseline,
and the fallback is recorded on the work item: one that quietly stopped
applying otherwise looks exactly like one nobody asked for.
→ `test_a_requested_specialisation_is_bound`,
`test_an_unapproved_specialisation_falls_back_and_says_so`,
`test_maintenance_work_specialises_under_id`,
`test_ego_cannot_name_a_namespace_only_a_leaf`,
`test_a_specialisation_reaches_the_work_item`,
`test_work_with_no_specialisation_binds_the_base_directly`

**I92. A role is woken by what happens to work it originated, and by nothing
else.** `artifact_event` and `board_event` were declared trigger kinds that
nothing ever emitted, so a neuocyte proposing an artifact -- a decision only
the requesting role can make -- or posting a finding left that role asleep.
"Ego is event-driven" was true of a narrower set of events than the
architecture claimed. What had been missing was a relevance rule that was not
a guess, and ownership supplies one: the relationship is recorded on the work
row rather than inferred from what looks interesting. A role posting about its
own work does not wake itself, because that notification carries nothing its
recipient did not just create. The rule lives in one module, because two
copies of a relevance rule are two rules and the second drifts silently.

These triggers are evidence, not questions: they carry the work's lineage and
expect no answer, so a burst joins the turn already thinking about that work
rather than opening rival interactions. Bundling absorbs volume, which is why
there is no throttle -- a second mechanism for a problem the first already
solves would eventually disagree with it.
→ `test_an_artifact_proposal_wakes_the_role_that_asked_for_the_work`,
`test_a_board_post_about_owned_work_wakes_the_owner`,
`test_a_board_post_about_nobody_s_work_wakes_nobody`,
`test_a_role_posting_about_its_own_work_does_not_wake_itself`,
`test_a_wake_trigger_is_evidence_not_a_question`

**I93. Context is reclaimed by dropping finished work, not by cutting the
middle out.** Trimming was positional -- keep the head, keep a tail fraction,
drop whatever lay between -- because nothing recorded what a span of context
*was*. It could and did cut through the middle of a message. A role runs
exactly one turn at a time, enforced by a partial unique index and a single
turn thread, so the tokens appended between the session length when a turn
started and the length when it ended are that turn's, all of them and nothing
else. That makes the mapping exact rather than estimated, and it is what lets
a finished interaction be removed at a boundary the chat format already has.

Removal is demand-driven and refuses more than it accepts. Settled
interactions go oldest-first and only as far as the target requires, so a
quiet role loses nothing; a lineage still owed an answer is never removed
however old, because losing a thought mid-flight is the failure I80 exists to
prevent and arriving there through housekeeping is no better; a turn that was
never measured, or was measured in a previous session, is kept, because "I do
not know what this is" must not resolve to "so remove it". Positional trim no
longer exists even as a fallback: see I122 for what a rebuild does instead.
→ `test_a_lineage_still_owed_an_answer_is_never_removable`,
`test_a_turn_with_no_measured_span_is_never_removable`,
`test_spans_from_another_session_are_never_removable`,
`test_a_settled_interaction_is_removable`,
`test_a_span_that_fits_no_unit_places_nothing`,
`test_settled_turns_go_oldest_first_and_only_as_far_as_needed`,
`test_a_turn_still_owed_an_answer_is_never_removed`,
`test_a_turn_the_record_cannot_place_is_kept`

**I94. Everyone holding a role's session handle learns when it changes.**
Rejuvenation closes a role's session and opens a fresh one. Two parties hold
that handle -- the durable record the Harness rebuilds from, and the role
process itself -- and both have failed to learn it, in the same way and a
year apart.

The second failure was worse, because it was the mind asking for help. Live,
Id used `id_request_rejuvenation` on its own context at 77%, correctly. The
Harness rebuilt the session, updated the record, and never told Id: handing
over was done by the *caller*, and only the turn-boundary caller did it. Id
went on addressing a session that had been closed, and every turn for the
next day and a half failed with "unknown inference session" -- a mind's own
self-care was the one way it could wedge itself permanently. The handover now
happens inside the rejuvenation, where the session is actually replaced, so
every path does it; whether it landed is recorded on the receipt
(`role_told`) rather than assumed.

Because a handover can still fail to land -- the role restarting at that
instant, the Harness briefly unreachable -- a role's liveness beat carries the
handle the record holds, and a role holding a different one adopts it, unless
it is mid-turn, in which case the running generation keeps its session until
the turn closes. Recovery in seconds, without an operator.

`hand_over_session` tells the role so. That updated the handle the role holds in memory and not
`agents.session_handle`, which is the durable record the Harness itself reads
to find the session to checkpoint. The first rejuvenation worked anyway. The
*second* checkpointed the session the first one had closed, and restored the
whole pre-rejuvenation context into a new one -- everything the first pass
dropped came back, and the organism went round again. Positional trim had the
same bug and it merely looked like rejuvenation not sticking.

Recorded on its own rather than through `register_agent`, which increments the
incarnation: a replacement session is not a new incarnation. Identity survives
rejuvenation -- the profile binding, the mailbox and the turn history all
continue -- which is the reason the session is handed over instead of the role
being restarted.

This is also what makes token coordinates safe to keep forever. A span is
recorded against the handle it was measured under, and after a rebuild the
surviving messages sit at different offsets, so every earlier span is stale.
They are not remapped and they are not deleted: they stay attached to a
handle that is now closed, and a rebuild reads only the current one. Where
it placed each kept turn is recorded separately, against the new handle
(I122), so the historical record stays truthful about the session it
described and the next rebuild can still place what it carried.
→ `test_a_second_rejuvenation_does_not_resurrect_what_the_first_dropped`,
`test_a_rejuvenated_role_keeps_its_identity_and_gains_a_new_handle`,
`test_every_path_that_replaces_a_session_tells_the_role`,
`test_a_handover_that_does_not_land_is_recorded_not_swallowed`,
`test_a_role_adopts_the_recorded_session_when_it_missed_a_handover`,
`test_a_role_mid_turn_is_not_moved_under_its_own_feet`,
`test_a_beat_carries_the_session_the_record_holds`,
`test_spans_of_a_closed_session_never_place_a_turn_in_its_successor`,
`test_spans_from_another_session_are_never_removable`

**I95. What became of an attempt travels with what it said.** A neuocyte's
window onto other work is almost entirely the blackboard -- no `get_work`, no
history, no artifact list -- so what the board fails to show, nothing shows. A
read filtered on `status != 'retracted'` and nothing else, which made a
finding from a neuocyte that then died indistinguishable from one whose work
completed, and nothing could correct it afterwards: `fail_work` does not touch
the board, a dead neuocyte cannot retract its own post, and `board_set_status`
is in no cognitive scope.

**The fate reported is the attempt's, not the work item's.** A work item can
fail attempt 1, requeue, and complete on attempt 2; joining the post to
`work_items.status` renders the fenced attempt's finding as `done`, laundering
a dead attempt's post through a later attempt's success. The author is
identified by the fencing token that was live when it posted -- read from the
work row by the Harness, never supplied by the author, for the same reason the
token works as a fence. A token behind the current one means the author did
not survive, whatever became of the work afterwards.

**Facts, never verdicts.** Nothing is retracted, hidden, discounted or
excluded: a neuocyte can find something true and then die of a deadline, and
"the author's process crashed" is not "the finding was wrong". The recorded
outcome travels verbatim rather than collapsed to "failed", because a
deadline, a fencing, a tool error and an exhausted budget mean different
things to a reader deciding whether to try again. An attempt that failed and
*will be retried* reads as still running, since a retry may yet corroborate
it.

**What a reader was shown is frozen**, alongside the state version, exactly as
`informed_by` is: a fate changes after the read, so without recording what was
rendered, a reader influenced by `running` is indistinguishable later from one
influenced by `fenced`. The fate travels through corroboration and into
promotion, because a memory item outlives the post and what is missing at
promotion is missing from the belief.

**Silence is reported too.** An attempt that died before posting leaves
nothing to annotate and would read as though it never happened, so attempts on
the same **recorded lineage** -- a shared `operation_id`, never a similarity
of objective -- that ended without posting are returned as a bounded count
with their recorded outcomes. Scoping by lineage rather than likeness is what
keeps the Harness from deciding what counts as "the same ground", and nothing
is posted in anybody's name.
→ `test_a_later_attempt_s_success_does_not_launder_a_fenced_attempt_s_post`,
`test_a_finding_from_failed_work_does_not_look_like_one_that_succeeded`,
`test_a_finding_from_failed_work_is_still_readable`,
`test_a_failure_that_will_be_retried_is_not_reported_as_unfinished`,
`test_the_fate_a_reader_was_shown_is_recorded`,
`test_corroboration_reports_a_dead_supporter_without_discounting_it`,
`test_attempts_that_died_before_posting_are_reported`,
`test_silence_on_another_lineage_is_not_reported`,
`test_an_attempt_that_posted_is_not_reported_as_silent`,
`test_work_still_running_is_reported_as_such`,
`test_cancelled_work_is_reported_like_failed_work`,
`test_a_post_belonging_to_no_work_reports_no_fate`,
`test_reading_a_thread_says_the_same_as_reading_a_query`

**I96. A session carries its own allowance, and carries it unchanged.** One
global scalar on the Arbiter did all the enforcing, while `max_context_tokens`
looked like a role's ceiling and reached only the dashboard -- two
configuration concepts, one of which governed reality, and no way for a reader
to tell which. The ceiling is now the session's own, declared where the policy
is known: a role from its configuration, a worker from its work class. Nothing
downstream infers a budget from circumstance, and in particular nothing infers
it from whether a fork succeeded, because that would make a worker's
capability a function of a performance optimisation. A session created without
a policy falls back to the global default rather than becoming unlimited --
unbudgeted is a bug, and granting it the pool would hide the bug behind good
behaviour. A rejuvenated role keeps the ceiling it had, or it would silently
fall back the first time it was reborn.
→ `test_a_role_session_is_bounded_by_its_own_ceiling_not_a_global_one`,
`test_ids_ceiling_is_independent_of_egos`,
`test_an_unbudgeted_session_falls_back_rather_than_becoming_unlimited`,
`test_the_budget_a_role_reports_is_the_budget_it_is_held_to`

**I97. What a session may think and what it costs are different questions.**
`budget_basis` decides which tokens count against the allowance; the physical
charge is derived separately from backend state. The case that forces them
apart is the fork fallback: an Ego-derived worker forks a large prefix and is
allowed its own growth past it, and when forking fails and the prefix is
recomputed its *cognitive* allowance is unchanged -- it is the same worker
doing the same job -- while the recomputed prefix now occupies its own cells
and is charged in full. Answering both with one flag would have to either
demote the worker for an optimisation it did not control, or pretend
recomputed KV were free. Judging a forked worker's *total* against a
private-growth allowance is the specific failure this prevents: it would be
refused on its first generate, before thinking anything at all.
→ `test_a_forked_worker_is_not_charged_for_the_prefix_it_inherited`,
`test_a_shared_prefix_is_charged_once_but_a_recomputed_one_in_full`,
`test_a_cold_worker_is_judged_on_everything_it_holds`,
`test_the_service_judges_a_forked_worker_on_its_growth_not_its_total`

**I98. Admission spends a pool it has measured.** `Arbiter.admit` weighed
queue depth, maintenance depth and slot reservations and read no token or
memory figure at all -- `vram_free_bytes` and the session counts were
collected, displayed, and never consulted. Aggregate overcommit was prevented
only by the numbers happening to be small, which stopped being true the moment
the role ceilings were sized properly. Admission now measures what exists,
with shared prefixes counted once and recomputed ones charged in full, and
estimates what does not from the work class's own budget -- stated as an
estimate rather than dressed up as measurement, because it cannot know about a
prefix the prospective worker may have to recompute. A reserve is held back
from *new* work only: running sessions grow into it freely, and it exists
because a prompt that fits the pool exactly has left nowhere for its own
answer to go. When the pool cannot be measured admission does not guess; the
queue is durable, and refusing on an absent number would stall work for a
reason that is not about resources.
→ `test_admission_refuses_when_the_pool_has_no_room`,
`test_the_reserve_is_held_back_from_new_work_only`,
`test_admission_does_not_guess_when_the_pool_cannot_be_measured`,
`test_maintenance_demand_is_estimated_from_its_own_budget`,
`test_the_shipped_policy_fits_the_pool_it_shares`

**I99. Only the discretionary turn yields to pressure.** A heartbeat is the
one turn the organism gives itself when nothing has happened, so it is the one
worth not taking while the pool is strained: it costs a prefill and grows Id's
context at exactly the wrong moment, and nobody is waiting on it. An
event-driven turn is never held back at any pressure -- a role that something
happened to must be able to think about it -- which is why the gate lives in
the heartbeat scheduler and is reachable from nowhere else.

The tension this has to survive is that **Id's heartbeat is the homeostatic
review**, so deferring it risks suppressing the thing that notices pressure.
That is tolerable only because relief does not depend on Id: the Harness
rejuvenates on its own authority at critical. It is still bounded, because a
review that never happens is worse than a turn that costs a prefill, so a
deferral has a hard ceiling and the heartbeat that finally runs is told how
long it waited. The clock measures one continuous run of strain and resets
when pressure clears, so a busy hour last week cannot spend the protection
owed to a real episode now.

Unknown pressure proceeds. An absent measurement is not evidence of pressure,
and holding cognition back because a monitor was down would stop the organism
thinking for a reason that has nothing to do with its resources. Deferrals are
recorded rather than merely logged: an operator asking why Id went quiet
deserves an entry in the same record as everything else, because a silence
with nothing beside it is indistinguishable from a scheduler that stopped
working.
→ `test_a_heartbeat_is_held_back_under_pressure`,
`test_a_deferred_heartbeat_eventually_runs_anyway`,
`test_the_deferral_clock_resets_when_pressure_clears`,
`test_an_unknown_pressure_never_defers`,
`test_only_the_heartbeat_consults_the_gate`,
`test_a_heartbeat_is_not_held_back_below_the_threshold`,
`test_the_gate_can_be_turned_off`

**I100. A claim can stop being made.** Id audits conclusions, and until this a
conclusion could be recorded and never changed: no retraction, no
supersession, only a `review_status` saying what an audit *found*. So an
adverse audit could establish that a claim was unsupported while the claim
stayed active and unqualified, and the dispute stayed open beside it forever.
The thing an audit is about was the one thing with no way to change.

`standing` is separate from `review_status` because they answer different
questions: what an audit found, and whether the claim is still being made. A
claim may be audited `contested` and still stand -- its author may disagree
with the audit -- and may be withdrawn with no audit having happened.

The author may change its own claim and nobody may change another's. Ego
withdrawing an Ego conclusion is Ego changing its mind; Id doing it would be
the auditor editing the record it audits, so there is no Id verb for it. The
check is on the stored `produced_by`, not on anything the caller asserts. The
row survives withdrawal: what the organism used to assert is a fact about it,
and the audits and disputes that name a conclusion keep naming something.
→ `test_only_the_author_may_withdraw_a_claim`,
`test_a_superseded_claim_cannot_then_be_withdrawn`,
`test_withdrawing_twice_is_a_no_op_that_keeps_the_first_reason`

**I101. A disagreement ends because the record moved, not because somebody
said so.** It could be opened and never closed -- one INSERT, no UPDATE -- so
`system_pulse` counted a number that only grew. Closure is now mechanical, and
each condition is a fact the Harness already holds: the disputed conclusion
superseded, the disputed conclusion retracted, a later audit supporting it on
a changed evidence basis, or an operator deciding.

The rule about the subject is narrower than "the subject may not close it". If
Ego retracts or supersedes its own disputed conclusion, that *should* end the
dispute -- Ego has not dismissed the finding, it has changed the thing being
disputed. What the subject may not do is end a dispute by assertion, and there
is no verb through which it could: closure is reached only by changing the
record, or by an operator, whose decision is recorded as a decision rather
than dressed up as evidence.

One open dispute per disputed subject, enforced by a partial unique index
rather than a check. Every adverse audit opened a disagreement
unconditionally, so auditing one conclusion twenty times produced twenty rows
and one unresolved issue looked like twenty. A repeat now attaches to the live
dispute as a recurrence.
→ `test_withdrawing_the_claim_ends_the_dispute_about_it`,
`test_superseding_the_claim_ends_the_dispute_about_it`,
`test_a_repeated_contradiction_does_not_open_a_second_dispute`,
`test_a_closed_dispute_does_not_block_a_later_one`,
`test_a_dispute_cannot_be_closed_twice`

**I102. An adjudicator changing its mind is not the ground moving.** A later
audit returning `supported` does not settle a dispute on its own: that is the
same adjudicator reversing itself about the same evidence, and closing on it
would let Id open a dispute and quietly self-certify it away -- exactly what
separating Ego from Id exists to prevent. Resolution by audit therefore
requires the evidence basis to have changed, measured as a digest of the
Harness's own dossier rather than of anything Id reports having looked at.

A verdict that flips on an unchanged basis is preserved rather than erased.
Two opposite verdicts against identical evidence is a fact about the
organism's own reasoning, and it is recorded as one; smoothing it over would
be the single kind of forgetting this system refuses. Open disputes are never
aged out either, and the pulse reports them by age instead of as a bare count,
because an unresolved contradiction nobody has addressed is true and a number
that only grows stops being read.
→ `test_the_opening_evidence_basis_is_recorded`,
`test_the_evidence_basis_digest_reflects_the_measured_dossier`,
`test_a_supporting_audit_on_a_changed_basis_settles_the_dispute`

**I103. A refused request does not poison the next one.** A response that
never read the request body leaves it in the socket, and because the operator
console speaks over a keep-alive connection the *next* request is then parsed
starting from the middle of it. The failure surfaces one request late, on an
innocent one, as a nonsense method name assembled from the previous body and
the new verb -- which reads like a server that has lost its mind rather than
like an unread body. Every response therefore drains what the handler did not
read, in `_send` rather than in each handler, because every error path returns
through there including ones nobody has written yet.

Two things this deliberately does not do. An oversized body is not drained:
reading an attacker-chosen number of bytes in order to discard them is exactly
what the size limit exists to prevent, so the connection is closed instead. And
the drain state is cleared per request, not per handler -- one handler instance
serves the whole connection, so state left over from the previous request made
the *second* refusal skip its drain and desynchronise anyway.
→ `test_a_refused_request_does_not_poison_the_next_one`,
`test_several_refusals_in_a_row_leave_the_connection_usable`

**I104. The operator console is valid JavaScript.** The dashboard script lives
inside a Python string, where `"a"` on one line and `"b"` on the next
concatenate silently. In JavaScript that is a syntax error, and a syntax error
anywhere in the script means none of it runs: no nav, no panels, no request
ever attempted, and a header left saying "connecting..." forever. The page
presents a compile error as a connection problem, so the operator debugs the
wrong thing. The embedded script is checked for that specific mistake without
needing an engine present, and parsed outright when one is.
→ `test_the_dashboard_script_has_no_python_string_concatenation`,
`test_the_dashboard_script_parses_as_javascript`,
`test_the_dashboard_page_is_served_whole`

**I105. The Operator contributes to the room and cannot speak as either
mind in it.** The backchannel is one room with three participants, and an
Operator message addresses the room rather than a role: the Harness delivers a
separately attributed `role_message` to Ego and to Id, so which mind heard
what is a fact on the record and not an inference. Two deliveries rather than
one hidden broadcast.

Authorship is structural rather than validated. `operator_backchannel` has no
`from_role`, `actor` or `author` parameter, so posting as Ego is not refused —
it is unsayable. Every room entry is stamped by the Harness with the author it
actually carried, and an entry nobody can be held to is refused rather than
recorded. The Operator speaking in a room Ego reads grants Ego nothing; this
is information, not capability.
→ `test_the_operator_speaks_to_the_room_and_both_minds_hear_it`,
`test_the_operator_cannot_post_as_ego_or_id`,
`test_an_author_nobody_can_be_held_to_is_refused`,
`test_each_direction_of_the_backchannel_lands_in_the_other_mind`

**I106. The live view may forget; the record may not.** The room buffer is
bounded, in memory, and owned by the running supervisor: it shows this
runtime's traffic and is empty after a restart. That is correct for a
viewport, and it is only acceptable because it is not the record. What a
backchannel message *influenced* lives where influence always lives here — in
the durable trigger it was queued as, and in the turn that consumed it — and
the Operator's own act is in the event ledger, because that is governance.

Reconstructing the conversation from the event ledger was the alternative, and
it is worse: it makes the ledger answer a question it is not for, and a second
stored copy of what a role said is a record that can disagree with the one
that actually shaped cognition. The division is the point. Only the view
forgets.
→ `test_the_room_does_not_survive_a_restart_but_the_record_does`,
`test_a_room_message_is_durably_recorded_even_though_the_view_is_not`,
`test_the_room_is_bounded`, `test_a_reader_gets_only_what_it_has_not_seen`

**I107. An answer belongs to its interaction, whole.** One request may take
several bounded turns; the turns are how the organism thinks, not how the
answer is cut up. The answer is every turn's piece of it, in order, exactly
once -- assembled by walking the recorded parent links from the turn that
ended the thought back to the turn that admitted the request.

It used to be whichever turn ended the thought. Live, a reply cut off three
times arrived as its last quarter, opening "Continuing from where the previous
analysis left off" and ending mid-list; the first three quarters had been
generated, recorded against their turns, and never delivered. Nothing is
stored twice to fix it: each turn's result is its fragment, written once when
that turn closed, and the assembled answer records which turns it came from.
Tool-call machinery is removed per piece, because the boundary can cut a call
in half and cleaning the joined text would then discard everything after it.

Otherwise the model's text is never rewritten. Pieces are concatenated
exactly, with nothing inserted between them. A continuation that has to be
asked for visibly is told that its output is appended directly to the
previous output, and not to introduce, recap, restart or repeat -- and a
preamble it writes anyway is kept, because stripping it would be a heuristic
guess applied silently to somebody's reply.
→ `test_an_answer_spanning_three_turns_arrives_whole_and_in_order`,
`test_each_piece_appears_exactly_once`,
`test_fragments_are_attached_to_their_own_interaction`,
`test_retrying_an_abandoned_continuation_does_not_duplicate_a_piece`,
`test_a_rival_request_cannot_enter_the_chain_or_its_answer`,
`test_a_three_turn_answer_reaches_the_operator_whole`,
`test_the_continuation_instruction_says_its_output_is_appended`,
`test_a_preamble_the_model_writes_anyway_is_kept`

**I108. Only a concluded thought is a finished answer.** A turn completing is
not an interaction completing. A turn cut off by its output ceiling is
finished as a turn and says nothing about the reply except "not yet", so
nothing is delivered until the thought ends. When it ends, only the model
choosing to stop is `answered`. Running out of continuations, a deadline or a
failure ends it too, and leaves an answer that stopped rather than one that
concluded: that is `incomplete`, carrying everything said so far and why it
stopped. Reporting it as answered -- which the continuation limit used to do,
keeping only the last fragment -- tells the caller, permanently, that the
fragment was the reply.
→ `test_running_out_of_continuations_is_incomplete_not_complete`,
`test_a_stop_that_is_not_the_model_finishing_is_not_complete`,
`test_an_exhausted_continuation_chain_releases_its_waiter`,
`test_running_out_of_continuations_reaches_the_client_as_incomplete`

**I109. A continuation resumes the message it continues -- only where that is
literally what the session holds.** A continuation used to reach the model as
a fresh "please continue" message with the environment re-rendered in front
of it. The model closed its unfinished reply and began another, so every
piece of a long answer opened with its own preamble, and each cost about nine
hundred tokens before it said a word. Now a continuation whose parent was cut
off by its ceiling generates on from the exact token where it stopped:
nothing appended, the pieces joining byte for byte.

Offered by the Harness only when the parent stopped at its ceiling, was the
role's last turn, and the continuation is the whole bundle -- evidence may
still ride with a continuation, and evidence has to be shown, which a resumed
generation has no place to do. The role then confirms its session is still
exactly the recorded length, because only it can see the session: after a
rejuvenation or anything else appended, it asks visibly instead.
→ `test_a_bare_continuation_is_offered_its_parents_exact_position`,
`test_a_continuation_carrying_evidence_asks_visibly_instead`,
`test_a_turn_that_ran_in_between_forbids_resuming`,
`test_a_resumed_piece_joins_exactly_where_it_was_cut`,
`test_a_tool_call_cut_in_half_leaves_no_machinery_in_the_answer`,
`test_a_three_turn_answer_reaches_the_operator_whole`

**I110. Each mind has its own output ceiling, and nothing silently lowers
it.** Ego 3072, Id 1024, `ego.neuocyte` 512, `id.neuocyte` 384 -- ceilings,
not target lengths, stated in each governed profile and inherited as model
variables are, so a silent specialist gets its worker's ceiling rather than
Ego's. The Ego root shipped with none, so a hardcoded 384 decided how much Ego
could say to anyone; and a global 512 backstop would have clamped a real one.
There is one canonical ceiling: the bound profile's. Nothing narrows it. An
approved profile that states none -- a root that predates the setting -- is
bound with the shipped value for its namespace, supplied through the
binding's `harness_constraints` so the record says the number was supplied
rather than chosen, and startup says so aloud with what to approve. That
value is a per-namespace code constant held equal to the shipped headers by
a test, not a read of the shipped file: editing a shipped file makes a
candidate, and must not reach a running mind unapproved.

`[arbiter] max_completion_tokens` is a hard platform cap, and it refuses
rather than clamps. Startup refuses to run with any selected profile above
it; binding refuses such a profile; a request above it is an error. A silent
clamp is how a governed 3072 became 512 with nobody told. And nothing
between the model and the operator cuts an answer by characters.
→ `test_each_mind_is_granted_its_own_ceiling_by_its_governed_profile`,
`test_changing_one_ceiling_does_not_move_another`,
`test_the_fallback_ceilings_match_the_shipped_headers`,
`test_the_backstop_clamps_no_governed_ceiling`,
`test_the_shipped_config_backstop_clamps_no_governed_ceiling`,
`test_a_long_answer_is_not_cut_by_characters_anywhere`,
`test_the_platform_cap_refuses_rather_than_clamps`,
`test_startup_refuses_ceilings_that_contradict_the_platform_cap`,
`test_binding_refuses_a_profile_above_the_platform_cap`,
`test_a_silent_profile_is_bound_with_the_shipped_ceiling_on_the_record`,
`test_a_stated_ceiling_is_never_narrowed_by_the_harness`

**I111. A generation is admitted only if its whole allowance fits.** The
arbiter used to check the prompt alone, so a session a few hundred tokens
short of its budget was admitted and generated straight through it -- and a
large ceiling was fictional near the wall, because the room it promised was
never reserved. Admission now requires prompt plus allowance to fit, and the
refusal is worded as context pressure, so a role is rejuvenated *before* the
turn rather than after the overrun. For Ego that places the wall at 13312 of
16384; for Id at 7168 of 8192.
→ `test_a_generation_is_admitted_only_if_its_allowance_fits`

**I112. A conclusion is a claim Ego chooses to make -- never a side effect of
answering.** Ego used to record a conclusion for every turn that owed an
answer, so each bounded piece of a long reply became its own auditable claim;
live, one question left four, three of them cut off mid-word. Moving the
recording to the whole answer fixed the fragments and kept the deeper
mistake: every finished reply was still a "conclusion", including a
complaint that a tool had refused and a self-report about memory, and one run
left seven, none reviewed and the wrong ones never superseded.

Answering is not concluding. A conclusion is something Ego puts into the
organism's auditable state on purpose, with `record_conclusion`, bound by the
Harness to the operation that asked. The answer names the conclusions Ego
recorded while giving it, and nothing else.
→ `test_answering_is_not_concluding`,
`test_neither_a_fragment_nor_a_whole_answer_is_a_conclusion`,
`test_an_incomplete_answer_is_not_a_conclusion`,
`test_the_answer_reports_the_conclusion_ego_chose_to_record`,
`test_a_live_three_turn_answer_records_no_conclusion`

**I116. An audit can actually happen, and its verdict lands.** The audit and
disagreement machinery (I100-I102) was unreachable in operation: `id_audit`
sat behind the control token, so neither Id nor the operator could start one,
and even then a verdict was committed only by a caller that waited for it --
an unwaited audit produced its verdict and dropped it. Now Ego recording a
conclusion wakes Id with a `conclusion_recorded` trigger from the Harness,
which says it is the Harness; the operator can ask for an audit; and a
verdict nobody waits for is committed when Id answers, through the same
function the waiting path uses, so there is one definition of what an audit
does. A target the record cannot resolve is refused before anyone is woken.
→ `test_a_conclusion_ego_records_wakes_id_and_the_verdict_lands`,
`test_the_operator_can_ask_for_an_audit_and_its_verdict_is_recorded`,
`test_an_unauditable_target_is_refused_without_waking_id`,
`test_a_contested_verdict_opens_a_disagreement`,
`test_an_unstated_verdict_is_recorded_as_unparsed_not_judged`

**I113. A role is taught the call syntax its parser accepts.** The role tool
loop parses exactly `<tool_call>{"name": ..., "arguments": ...}</tool_call>`,
and until this it told no role so: neuocytes were shown the form, Ego and Id
were shown verbs and arguments and nothing about how to invoke one. Across
the organism's recorded life not one Ego or Id turn executed a tool -- every
affordance was unreachable, and every observation about what Ego "chose" to
do with them would have been an observation about that. The environment
block now states the form, and a test fills in the taught template and hands
it to the loop's own parser, so the two cannot drift apart.
→ `test_a_role_is_taught_the_syntax_its_parser_accepts`,
`test_the_role_is_told_to_wait_for_the_result_not_to_stop`

**I114. A role is not restarted for thinking.** `RpcClient` holds one lock for
a whole request, and a role's health handler asked inference through the
client its turn thread holds for the length of a generation. A busy role
therefore failed its liveness probe, and after the grace period the
supervisor restarted it mid-turn. At 384 output tokens a generation never
outlasted the grace; at Ego's 3072 most long answers did. Health now uses a
connection of its own, closed after any failure so a late reply cannot be
read by the next probe.
→ `test_health_answers_while_a_turn_holds_the_inference_client`,
`test_a_long_generation_does_not_get_its_role_restarted`,
`test_a_failed_probe_leaves_no_reply_behind_for_the_next`

**I115. A malformed capability call is never delivered as an answer.** Teaching
the syntax is the fix; this is what keeps one typo from turning the
operator's answer into a service hatch. A reply whose line begins with an
offered verb applied like a function, or with a JSON object naming one, or
that carries tool-call markup the parser could not read, is a request in the
wrong form -- recognised structurally, not guessed at, and inline code quoted
in prose is left alone. It is refused like any other request the Harness
will not run: the model is told it was neither executed nor delivered, and
shown the form. If a thought ends still malformed, that piece is withheld
from the answer, kept intact in its turn's record, and the answer says so --
Converse shows that plainly rather than the raw call.
→ `test_a_malformed_attempt_is_refused_and_the_model_may_try_again`,
`test_a_turn_that_ends_still_malformed_is_flagged_not_answered`,
`test_a_malformed_piece_is_withheld_from_the_answer_and_recorded`,
`test_an_answer_that_was_only_a_malformed_call_says_so`,
`test_converse_never_shows_the_raw_call`,
`test_a_live_malformed_attempt_is_corrected_not_delivered`

**I117. What a capability accepts is declared, and a refusal says what it
would have accepted.** A verb that takes `post_type` or `kind` without saying
what the valid values are is present and undiscoverable. Live, Ego tried
`incident_synthesis` for both and was refused each time with nothing more
than "unknown post type" -- the checks carried the allowed values and the
dispatcher dropped them. Each vocabulary is now one constant, read by the
check and by the declaration alike, so the two cannot drift; the declaration
marks non-string arguments with their kind (`evidence:list`,
`confidence:number`), because those were the other things Ego guessed wrong;
and a refusal carries its allowed values back to the model.
→ `test_every_declared_vocabulary_is_the_one_enforced`,
`test_the_roles_are_told_the_values_they_are_checked_against`,
`test_a_refusal_says_what_would_have_been_accepted`,
`test_no_vocabulary_is_still_spelled_inline_where_it_is_checked`

**I118. A session reads an unchanged declaration once.** Each role's environment
digest was identical on every turn measured, and the full declaration was
ingested every turn anyway -- 1071 tokens for Id, accumulating as duplicates
in a single session. That, not the number of verbs, is what kept Id in
context pressure: two of its five turns in the first pressure run ended
there, and its one heartbeat was spent on its own context. The whole
declaration is now rendered once per session; a later turn with the same
digest gets a one-line reference and the call form. A changed digest, or a
new session -- which every rejuvenation makes -- gets it in full, and the
turn records which it received.
→ `test_a_session_reads_an_unchanged_declaration_once`,
`test_a_new_session_or_a_changed_declaration_gets_the_whole_thing`,
`test_a_second_turn_does_not_pay_for_the_declaration_again`

**I119. A role's arguments are checked before the verb sees them.** Live, Ego
passed `evidence` to `record_conclusion` as a string, and the Harness replied
`AttributeError: 'str' object has no attribute 'get'` -- an internal failure,
told to the model as the reason, and harmless only because the crash came
before anything was written. Arguments are now held to the verb's own
annotations, the same ones its declaration renders kinds from, and a mismatch
is refused in words ("argument 'evidence' must be a list of objects, got a
string") with the verb never called. What the Harness binds itself
(`operation_id`, `turn_id`) is not checked, and an annotation the checker
cannot read is let through: refusing a correct call because the checker was
unsure would be worse than the crash it prevents, so every argument of every
role-facing verb is shown to accept a plainly correct value.
→ `test_the_live_mistakes_are_refused_in_words`,
`test_a_wrong_type_is_refused_before_the_verb_runs`,
`test_no_correct_call_on_the_role_surface_is_refused`,
`test_an_annotation_the_checker_cannot_read_is_let_through`

**I120. A call that fails leaves nothing behind for the next one.** An
`RpcClient` kept its connection after a call failed mid-flight. A timed-out
call's reply still arrives, and the next call on that connection read it as
its own -- so a probe of a wedged child could come back "reachable" off a
stale answer, and health lied. A call that fails while writing or reading
now drops its connection, and a reply whose id is not the request's is
refused and drops it too.
→ `test_a_late_reply_is_never_read_as_the_next_calls`,
`test_a_reply_to_another_request_is_refused`

**I121. What the operator is shown is measured, and attributed.** Three views
from the second pressure run looked informed and were not. Every
`role.tool_invoked` event carried no operation, so an operation's history
never showed what its roles reached for; it now carries the turn's. The
overview's role context and inference KV were null on every pulse, read off
`health` replies that do not carry them -- and the same nulls fed
`context_pressure`, which Id watches. They are now read from the inference
service's own `context_report`, the one place occupancy is measured, with
each role's session matched by id and judged the way its allowance is
written. And `prompt_incarnations` omitted `harness_constraints`, so a
ceiling the Harness supplied looked like one the profile stated.
→ `test_a_role_tool_call_is_on_its_operations_history`,
`test_the_overview_shows_measured_context`,
`test_a_binding_shows_what_the_harness_supplied`

**I122. A rebuilt context is one the conversation could have reached by
itself.** Measured on Id's live session, rejuvenation kept a 512-token head
that cut the capability declaration off after 227 tokens, mid-line; kept two
"the declaration given earlier still applies" references to text it had just
removed; resumed the tail in the middle of a tool result; and then sent the
whole declaration again -- a cycle that retired Id's session every few
minutes from seq 292 on. None of those states could have arisen naturally.
The rule is: environment is reconstructed, cognition is preserved
selectively, neither is token-spliced. The checkpoint is split at the
template's own message-start token. The governed prompt is rendered fresh
from the binding and held to its frozen digest. Every environment block,
declaration or reference, is removed, because a new session's next turn is
given the current declaration in full (I118) immediately before it is used,
so nothing kept can point at text that is gone. Settled turns are removed
whole; owed turns lose nothing -- owed meaning an interaction still waiting
for its answer, or a thought still being continued: a heartbeat awaits no
answer, but one cut off by pressure has a continuation told to carry on from
where it stopped -- and an oversized result in one is re-rendered
as the bounded projection a live call would get, from its exact stored copy --
"I called X, here is what it returned, the rest is retrievable" -- rather than
deleted. Where a turn sits in the new session is recorded against the new
handle, so the next rebuild can still tell settled work from owed work; the
turn's own row keeps the coordinates it was measured under (I94). A rebuild
that cannot reach its target says so and cuts nothing.
→ `test_every_message_of_a_rebuilt_session_is_whole`,
`test_no_environment_block_or_reference_survives_a_rebuild`,
`test_the_governed_prompt_is_rendered_fresh`,
`test_an_owed_result_is_projected_never_deleted_and_its_exact_copy_is_there`,
`test_a_turn_being_continued_is_active_even_when_nobody_awaits_an_answer`,
`test_a_second_rebuild_can_still_remove_carried_settled_turns`,
`test_spans_of_a_closed_session_never_place_a_turn_in_its_successor`,
`test_an_empty_generation_prompt_is_not_carried`,
`test_a_rebuild_that_cannot_reach_its_target_says_so_and_cuts_nothing`,
`test_positional_trim_and_eviction_are_gone`,
`test_the_turn_boundary_asks_for_a_rebuild`

**I123. A tool result too large for its delivery budget is never chopped.**
Two defects made Id's session fill itself. The Harness computed a bounded
text for every role result and `RoleProcess._invoke` dropped it, so every
role result was delivered whole -- one `prompt_incarnations(limit=5)` cost
Id 1827 tokens. And where the bound did apply it cut the serialized JSON at
2000 characters: half a value in the context, priced in the unit the context
does not spend. A result now fits the role's budget in model tokens, counted
by the model's tokenizer, or is shown as a projection that is whole JSON:
whole list items with how many of how many, and anything else too large
replaced by a marker naming its path and size. The exact text is stored and
its reference issued to that role, and `result_read` opens it -- or any part
of it -- for that role only. The two views Id reached for are compact by
default: `prompt_incarnations` answers "what is each mind bound to" in a line
per binding, and `prompt_resolve` gives the doctrine only when asked.
→ `test_a_role_is_shown_the_bounded_result_not_the_whole_one`,
`test_the_harness_bounds_a_role_result_by_that_roles_budget_and_issues_it`,
`test_an_oversized_result_is_projected_whole_and_says_what_it_left_out`,
`test_a_value_too_large_to_show_is_named_not_cut`,
`test_result_read_returns_the_exact_stored_result_and_any_part_of_it`,
`test_a_reference_opens_only_for_the_role_it_was_issued_to`,
`test_the_compact_views_answer_the_usual_question`

**I124. Generated tokens may never introduce chat-template structural tokens
into session state. Only Harness-controlled context construction may append
role and control tokens.** Live on 2026-09-22 at 11:50, Id emitted a real tool
call and kept generating: `<|im_start|>user` followed by a `<tool_result>`
block copied from a result ten minutes old. The Harness appended the true
result underneath, so the session held a stale reading above the fresh one, in
a message the model had authored and the record attributes to nobody. It was
invisible downstream because `token_to_piece(special=False)` renders a control
token as the empty string: the decoded text carried no marker while the token
entered the session and the KV. What exposed it was the ledger disagreeing
with the context -- five recorded tool calls, seven results.

Three classes of sampled token, kept apart, and decided **on the token id
before the token is appended** -- never by looking for markers in decoded
text, which is the surface the defect hides from:

* ordinary content: appended to the session, rendered to the caller;
* a legitimate terminal (`is_eog`): generation stops, nothing is appended,
  and it is not an attempt at anything -- a model ending its message is
  ordinary;
* a chat-template control token: generation stops, nothing is appended, and
  the attempt is recorded.

Which tokens those are is read from the vocabulary rather than assumed from
documentation: a control token is one the vocabulary renders as nothing in
plain text and as something in the special form. On the running model (Qwen3
4B, 151936 tokens) that is 21 tokens, scanned in 0.37s at load; the active
template writes exactly two of them, `<|im_start|>` and `<|im_end|>`; six are
end-of-generation, including `<|im_end|>`, which is why the terminal case is
checked first. The rest -- vision, box, quad and FIM markers -- are structure
this model can emit and this organism never writes.

The refusal is recorded as `role.structure_refused` carrying the token and
its rendered form, so the causal ledger can prove who authored a boundary:
an attempt that was refused is distinguishable from the thousands of
boundaries the Harness writes legitimately.
→ `test_a_generated_structural_token_never_enters_the_session`,
`test_the_vocabulary_says_which_tokens_carry_structure`,
`test_end_of_generation_is_a_terminal_not_a_forgery`,
`test_a_generation_that_opens_a_message_is_stopped_before_the_token_lands`,
`test_the_forged_message_never_becomes_context`,
`test_text_and_token_stream_agree_about_boundaries`,
`test_a_checkpointed_session_carries_no_model_authored_boundary`,
`test_an_ordinary_tool_call_is_untouched`,
`test_a_refused_attempt_is_recorded_and_a_harness_boundary_is_not`,
`test_structure_is_what_the_transcript_cannot_show`

**I125. A review that reports no change is only as true as the watermark it
was built from.** Id is woken by conclusions, messages, work it owns and the
operator. Nothing wakes it for a failure, for resource pressure, or for a
contradiction nobody announced, so the heartbeat is the only sense for
unannounced state -- which is why it exists, why it cannot simply be made
rarer, and why making it cheap is delicate. Measured live, one review cost
2315 tokens: five tool calls, 1793 tokens of results, all re-reading state
that had not moved, carried until the next rebuild discarded them. The cause
was a trigger that said "nothing has woken you, check the organism's internal
state" and carried no information, so Id paid five calls to find a row of
zeros.

The review now arrives knowing what changed. The digest is built from an
event watermark -- every event since the previous review, counted by kind --
and that completeness is load-bearing: a digest that reports only what its
author remembered to include would blind a mind in precisely the way it
cannot detect. A tail too long to itemise is reported as a count, never
dropped; no watermark is a *first review*, not a quiet organism; and standing
obligations are reported even when nothing happened, so an organism where
nothing moved but something is owed is not called quiet.

It measures and never interprets: "two conclusions recorded, one unaudited"
is a fact, and "nothing worth your attention" would be the Harness doing Id's
job. Id keeps every sense it had and is told to check anything it doubts. A
review with nothing to report carries an output ceiling
(`heartbeat_quiet_ceiling_tokens`), and only when every trigger in the bundle
agrees to it, so a real question sharing the turn is never shortened.
→ `test_every_kind_since_the_watermark_is_counted`,
`test_what_is_not_itemised_is_still_counted`,
`test_a_first_review_does_not_claim_a_quiet_organism`,
`test_what_is_owed_is_reported_even_when_nothing_happened`,
`test_the_watermark_is_read_from_the_previous_review`,
`test_the_digest_measures_and_does_not_interpret`,
`test_a_quiet_review_carries_a_ceiling_and_a_busy_one_does_not`,
`test_a_claimed_turn_carries_the_ceiling_its_inputs_agreed`,
`test_a_live_quiet_review_is_cheap`,
`test_context_telemetry_is_compact_for_a_role_by_default`

**I126. Answering is not thinking, and the Harness is the one who knows the
difference.** Id lost its inference session to a rejuvenation it had
requested itself (I94) and failed every turn for thirty-seven hours --
seventy-five of them, not one success. Nothing noticed. The process answered
every health probe, so supervision saw a healthy child; a role's turns
failing were not counted as a failure anywhere, so the pulse reported none;
and the component whose job is to watch the organism was the one that could
not think.

Health is therefore measured as whether a role's turns are *working*, from
the Harness's own record rather than from the role's account of itself -- a
role asked whether it is well is the worst available witness, and this one
was cheerful throughout. A run of consecutive turns ending in `role_failure`
or `backend_error` is the signal. A turn stopped by its output ceiling, by
the tool-turn limit, by a deadline or by context pressure is not: those are a
mind meeting a known limit, which is the organism working.

What follows is bounded and receipted. The streak is recorded once as
`role.not_thinking`, which counts as a failure in the pulse and marks the
role `thinking: false` for the operator. The repair is a restart of the role
-- the remedy for a process that is alive and useless -- rate limited to
`role_repairs_per_hour`. A role still failing after its repairs are spent is
left running and visibly unwell rather than restarted in a loop, because an
organism thrashing itself is worse than one waiting for a human.
→ `test_consecutive_failures_are_counted_from_the_record`,
`test_one_good_turn_ends_the_streak`,
`test_a_mind_meeting_a_known_limit_is_not_failing`,
`test_a_role_that_cannot_think_is_recorded_and_repaired`,
`test_one_streak_is_reported_once_however_often_it_is_checked`,
`test_repairs_are_bounded_and_the_alarm_outlasts_them`,
`test_the_pulse_reports_a_role_that_cannot_think`,
`test_a_role_that_cannot_think_counts_as_a_failure`

**I127. A count comes with a way to reach the thing counted, and a refusal
says where a real identifier lives.** Told "unaudited conclusions: 17", Id
called `get_conclusion`, `get_memory` and `audit_dossier` with
`c8f050bd638e10dc` -- the reference from a bounded tool result it had just
been shown -- and earlier with its own session id under a `con_` prefix.
Each answer was "unknown conclusion": true, and useless.

The surface caused it. Id can fetch a conclusion by id and cannot list
conclusions at all, so the only routes to one were raw events or provenance;
and `audit_dossier()` with no argument -- the question Id actually had --
refused with "no operation to resolve". A mind handed a count with no route
to the things counted will manufacture the route.

Three things, and none of them guesses which identifier was meant: guessing
would turn a mind's mistake into the Harness's claim.

* `audit_dossier()` with no argument resolves the oldest conclusion nobody
  has audited, and says plainly when there is none. A conclusion recorded
  outside any operation still yields a dossier of its own evidence rather
  than a refusal, because a claim Id was told to audit must be reachable.
* The digest names what it counts: `unaudited conclusions 17 (oldest: ...)`.
* A refusal says what the value *is* -- a result reference, a session
  identifier, a number -- when that is plain, and where a real one comes
  from. A well-formed identifier of the right kind is not called a misuse:
  it may simply be absent, and then "no such conclusion" is the whole truth.
→ `test_a_value_is_described_as_what_it_is`,
`test_a_well_formed_identifier_of_the_right_kind_is_not_a_misuse`,
`test_nothing_guesses_which_identifier_was_meant`,
`test_a_refusal_says_what_the_value_was_and_where_a_real_one_lives`,
`test_a_missing_but_well_formed_identifier_is_not_called_a_misuse`,
`test_audit_dossier_with_no_argument_resolves_the_oldest_unaudited`,
`test_when_nothing_is_waiting_it_says_so_plainly`,
`test_the_digest_names_the_conclusion_it_is_counting`

**I128. The organism is not blind when it is strained.** Id was woken by
conclusions, messages, work it owns, board posts on that work, and the
operator. Nothing woke it for a failure, for resource pressure, or for the
other role being wedged, so the heartbeat's interval was the detection
latency for everything the inward mind exists to catch -- and the heartbeat
is *deferred* while the pool is under pressure, so the organism looked at
itself least often exactly when it was most strained.

A few measured conditions now earn a turn of their own: a burst of failures
in the last five minutes, pool pressure at or above a configured level, and
the other role failing turns in a row (I126). Each wake carries the
measurement and says nothing about what it means, because that is Id's to
decide, and each says so: *what it means is yours to say*.

Bounded so a bad hour costs a handful of turns rather than a wake storm: a
cooldown per condition, and nothing is queued twice while the same condition
still sits unread in the mailbox -- matched on the trigger's `source_ref`,
not by looking for a marker inside prose. An absent pressure measurement
wakes nobody, because unknown is not evidence. And a role is never woken
about its own inability to think: it could not answer, and that case is the
Harness's own alarm.

These are event-driven, so the pressure gate never holds them back: that
gate exists to stop a *discretionary* review adding load while the pool is
tight, and pressure is the reason to wake rather than a reason to stay
quiet. Because the clock is no longer the only sense, the idle review may
now back off to an hour rather than half of one.
→ `test_a_burst_of_failures_wakes_the_inward_mind`,
`test_a_failure_or_two_does_not`,
`test_pressure_wakes_it_at_the_configured_level`,
`test_unknown_pressure_is_not_pressure`,
`test_the_other_role_being_wedged_wakes_it`,
`test_a_role_is_never_woken_about_its_own_wedging`,
`test_a_wake_measures_and_does_not_interpret`,
`test_the_wake_is_queued_once_and_then_held_by_the_cooldown`,
`test_news_already_waiting_is_not_said_twice_even_once_the_cooldown_lapses`,
`test_a_condition_wake_is_never_deferred_by_pressure`

**I129. An answer that exists is delivered, whatever became of whoever was
waiting.** An external request is answered by a turn, and that answer lands
on the trigger -- durably. Publishing it onto the interaction was done by the
daemon thread that submitted it, and a thread does not survive a restart: a
recovered thought completed and `io_output` reported `output: null` forever,
while the record held the answer the whole time.

The association is now durable -- `interactions.trigger_id`, written by the
Harness at the moment the trigger is enqueued -- and delivery is a function
of the record rather than of a live thread. A reconciler runs at every
supervision pass, finds interactions still waiting whose trigger has been
answered, and publishes; a trigger reported unanswerable fails the
interaction rather than leaving it open. The thread remains, because it
delivers sooner, but nothing depends on it surviving.

Exactly once, two ways: the reconciler skips an interaction somebody is
still waiting on, and the write itself requires a status that is still
waiting -- which is what makes the race safe when the thread publishes
between the reconciler's read and its write.
→ `test_the_interaction_records_which_trigger_answers_it`,
`test_an_investigation_records_its_trigger_too`,
`test_an_answer_reaches_a_client_whose_thread_is_gone`,
`test_delivering_twice_does_not_move_a_finished_interaction`,
`test_a_thread_publishing_first_is_not_overwritten_by_the_reconciler`,
`test_an_interaction_nobody_answered_is_left_alone`,
`test_reconciling_is_not_an_external_capability`

**I130. What a client sent arrives whole, and every kind of request carries
the same context.** Two ways the path between a client and cognition lost
part of the request.

The door accepts 32,000 characters. A conversation then stored 16,000 of
them and an investigation 8,000, silently -- so trailing instructions, which
is exactly where constraints live, were dropped from otherwise valid
requests. Storage keeps the whole text now; *rendering* is what bounds it,
and rendering says how much it left out and which blob holds the rest
(`mailbox.trigger_body`, I86). More than the door allows is refused at the
door, not trimmed behind it.

And an investigation was handed neither `interaction_id` nor `attachments`,
though a conversation was given both: its turn could not resolve the files
its own request arrived with, and could not return one through
`ego_surface_result`. A request is a request whichever verb serves it.
→ `test_a_long_request_is_not_quietly_shortened`,
`test_an_investigation_keeps_its_constraints`,
`test_what_is_rendered_says_where_the_rest_is`,
`test_more_than_the_door_allows_is_refused_not_trimmed`,
`test_an_investigation_carries_its_request_context`,
`test_the_investigation_branch_passes_what_it_was_given`

**I131. Starting a new organism does not erase the old one.** Everything in
the state directory is the record of a mind that ran: what it concluded,
what it was told, what it did and when. `amoeba reset` therefore moves it
aside intact, under a timestamped name beside it, and says where it went.
`--delete` exists and has to be meant -- it refuses without `--yes`, because
an irreversible instruction that takes one keystroke to give is not an
instruction anyone gave deliberately. This is I87's stance at the scale of a
whole organism: evidence is not a cache.

It refuses to run at all while a supervisor owns the directory, judged by
the supervisor's own lock and its own staleness rule rather than a second
opinion that could disagree; deleting a database under a live writer leaves
a half-state nobody can reason about afterwards. It touches nothing outside
the state directory -- the configured file roots are the operator's own
directories. And it keeps credentials by default: a reset is about the mind,
not about the doors it is reached through, and reissuing keys is a separate
decision (`--rotate-credentials`). A directory holding only the empty
scaffolding a configuration creates is not an organism, and resetting it
announces nothing.
→ `test_the_mind_is_archived_and_the_plumbing_is_kept`,
`test_rotating_credentials_takes_them_too`,
`test_a_dry_run_moves_nothing`,
`test_deleting_really_deletes`,
`test_deleting_needs_to_be_meant`,
`test_a_running_organism_is_not_reset`,
`test_a_lock_whose_holder_is_gone_does_not_stop_a_reset`,
`test_nothing_outside_the_state_directory_is_touched`,
`test_an_empty_directory_is_not_an_error`,
`test_the_command_says_what_it_did_and_where_it_went`

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
