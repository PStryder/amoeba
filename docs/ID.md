# Id: senses, effectors, and the capability boundary

Id is the inward half. Its job is to watch the organism, reason about what it
sees, notice problems, and ask for them to be fixed. That requires two things
to be true for every responsibility eventually put in Id's prompt:

* a **sense** that lets it detect the condition, and
* an **authorised effector** that lets it respond usefully,

without Id becoming a second authority over durable state.

## The sensory surface

| Sense | Verb | Cost | For |
|---|---|---|---|
| **Live telemetry** | `system_pulse` | cheap, cached | "what is happening right now" |
| Deep health | `id_health` | expensive | investigating what the pulse pointed at |
| Maintained state | `recall`, `get_memory` | cheap | what the organism believes |
| Raw history | `history`, `provenance`, `audit_dossier` | medium | what actually happened |
| Integrity | `verify_integrity` | expensive | hash chain and missing content |
| Blackboard | `board_read`, `board_thread`, `board_independence`, `board_corroboration` | cheap | what neuocytes said, and whether agreement was independent |
| Artifacts | `artifact_list` | cheap | pending proposals and their digests |
| Work | `get_work`, `queue_stats` | cheap | task state |
| Context | `context_report`, `context_assess` | medium | homeostasis |
| Side channel | the role's own `_signals`, surfaced as `pending_signals` | free | messages addressed to Id |
| Resources | in the pulse | cheap | prompt/tool/policy/schema versions |

Id needs no SQL, no filesystem access and no process inspection. The Harness is
the observational surface.

## `system_pulse`

One call, bounded, cached, and made of facts.

```
pulse_id, captured_at, state_version, run_id, cached, age_seconds
harness        uptime_seconds, supervision_passes, last_supervision_at, schema_version
roles          ego|id: reachable, incarnation, session_id, context_tokens,
               max_context_tokens, pending_signals, prompt_sha256
inference      reachable, status, incarnation, active_sessions, max_sessions,
               vram_free_bytes, model_generation, backend_kind, is_simulated,
               kv_mode, n_ctx, kv_cells_used, kv_cells_total
model_generation
neuocytes      live[], alive_count, by_class{user,maintenance}, board_naive_count
               live[i]: neuocyte_id, work_id, work_class, alive, age_seconds,
                        hard_deadline_in, board_access, board_naive, sandbox_allowed
work           by_status{}, by_class{}, blocked, in_flight[],
               oldest_queued_age_seconds, retried_items, max_attempt
scheduler      max_neuocytes, in_use, available, max_outstanding_work,
               outstanding, user_reserved_slots, maintenance_reserved_slots
context_pressure  per role: context_tokens, max_context_tokens, occupancy
resources      configured{prompt.ego, prompt.id, tools.neuocyte, security.policy,
                          filespace.config, store.schema}  -> sha256 + detail
               embodied{prompt.ego, prompt.id} -> sha256, matches_configured
pending_decisions  artifact_proposals, open_disagreements,
                   unreviewed_conclusions, queued_maintenance
failures       last_5m{}, last_5m_total, last_1h{}, last_1h_total, watermark_seq
storage        free_bytes, total_bytes, used_fraction
attention      pending_signals{ego,id}, artifact_proposals, open_disagreements
contract       reports, excluded, note
```

**Facts, not verdicts.** There is no `ego_unhealthy`. The pulse reports the
occupancy and the failure counts; deciding what they add up to is Id's
cognition. A field name that reads as a judgement fails a test.

**Configured vs embodied.** A running role primed its context with the prompt
that existed when it started. Editing configuration changes what the *next*
incarnation runs. Both are reported and compared, because reporting only the
configured version would assert the running mind is something it is not.

**Bounded.** No blackboard contents, artifact bodies, logs, exception text or
memory claims. The pulse says where to look; the senses above do the looking.
Under 64 KB, enforced.

**Cheap.** Cached (1s for database facts, 5s for the inference and role round
trips), with failure counters maintained incrementally from an event `seq`
watermark rather than rescanned. `max_age_seconds=0` forces a fresh reading.

**Provenance.** Each pulse has a `pulse_id`. Telemetry that influences
cognition is identifiable later: Id cites the pulse it reasoned from, and that
citation is recorded — rather than recording every pulse, which would bury the
log in observations nobody reads.

## Effectors

Every one is `sense → reason → request → Harness validates, executes, receipts`.
All are attributed to `id` and carry the `pulse_id` they were formed from.

| Effector | What Id is asking for | What it cannot do |
|---|---|---|
| `id_cite_pulse` | record the telemetry behind a conclusion | — |
| `id_raise_finding` | put an anomaly/contradiction/risk on the record | it lands as an **interpretation**, not a belief |
| `id_propose_memory_correction` | supersede a maintained belief | cannot edit one in place |
| `id_request_investigation` | bounded work by Id neuocytes, optionally board-naive | cannot schedule; the Arbiter decides |
| `id_request_work_intervention` | cancel a specific work item | no requeue, no policy change |
| `id_request_rejuvenation` | context relief for a role | never touches KV itself |
| `id_propose_prompt` | a candidate prompt with rationale | cannot install it |
| `id_message_ego` | a provenance-bearing nudge | transient; changes nothing |
| `id_escalate_to_operator` | surface what Id may not remediate | — |

Deliberate absences: **requeue** (returning a leased item to the queue is lease
expiry, which the supervision loop owns; a second actor forcing it would race
that loop), **prompt installation**, **scheduler policy**, and anything that
edits state in place. Detecting a problem is exactly when a mind is most likely
to be confidently wrong, so the moment Id is most certain is the moment its
conclusion should pass through something that can refuse.

## Id-only capability isolation

The requirement is that a neuocyte has **no logical path** to an Id-only verb.
This is satisfied by absence, not by a check.

**Scoped method tables.** `scopes.py` declares, as data, exactly which verbs
each caller sees. The RPC server holds one table per scope. A verb outside a
caller's table does not exist for that caller: the dispatcher never finds it,
and there is no shared implementation containing an `if caller != id` to be
refactored away.

**The scope is the credential.** A connection's scope is decided by the secret
it presents — `scope.<name>.token` — and there is no role, actor, caller,
work-class or process field anywhere in the handshake. There is nothing to
forge, because nothing is claimed.

**No enumeration.** An unknown-method error used to return `details.known`
listing every method on the server, which made it a discovery oracle: ask for
something fake, learn the name of everything you are not allowed to call. It
now returns only the caller's own scope name.

**No generic dispatcher bypass.** `tool_invoke` is the only generic dispatcher
a model can drive. It builds a neuocyte registry and nothing else, so an Id
verb name arriving there finds no tool.

**Not in the model's tool surface.** Id's effectors appear in no tool schema, so
a model cannot enumerate or name them.

**Ego is checked separately.** Ego and Id are both long-lived role processes and
share plumbing. They do not share authority: Ego's scope contains no Id effector
and not the pulse.

If a neuocyte needs something in this category it says so — through a work
result, a board post, or a finding. It does not borrow Id's hands.

### Residual limit, stated

Scope tokens are files in the state directory, so a process already running as
the Amoeba account can read any of them. This is the same boundary documented
in `SANDBOX.md` for ACLs: it removes accidental and model-driven capability,
not a determined same-account process. The genuinely untrusted execution
environment is the compute sandbox, and that provably cannot reach the state
directory or the network at all (see `STORES.md`).

The stronger form — a dedicated account per role — is a deployment decision,
the same one the ACL work reaches.

## What Id is given each turn

Id has two distinct views of the organism and they are deliberately separate:

* **`system_pulse`** — live physiological telemetry. How the organism is doing.
* **`role_environment`** — the cognitive operating environment. What the
  organism is currently made of and what Id can reach: available `id.*`
  profiles, the capabilities Id may invoke, resource identities.

Plus the governed profile `id@N` bound at incarnation, and the turn's trigger.

Id's capabilities come from the same scope table that gates dispatch (I58), and
Id executes them through the same bounded Harness-mediated tool loop Ego uses
(I59). Id's environment never contains Ego-only effectors or Operator verbs.

Id may propose a new version of the `id` or `ego` root through the Prompt
Library and cannot approve one (I49b).
