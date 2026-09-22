# Context homeostasis

A long-running mind fills its KV pool. With `kv_unified=True` that is not just
a capacity problem: occupancy taxes **every** decode, including sessions with
nothing to do with the one hogging space. Measured at 1.94x slowdown at 77%
occupancy, fully recovered on retirement
([BENCHMARKS §2](BENCHMARKS.md#2-resident-idle-sessions-tax-every-other-decode)).
Left alone, a mind gets slower and then stops.

---

## The authority boundary

**The model never touches KV.** There is no tool, no MCP verb and no code path
by which Ego, Id or a neuocyte manipulates a cache. `test_no_kv_verb_is_reachable_from_a_model_facing_tool`
greps the MCP surface and the tool registry for `seq_cp`, `memory_seq`,
`fork_prefix`, `restore_prefix`, `close_session` and fails if any appears.

The division is:

| Who | May |
|---|---|
| **Id** | observe pressure, and **request** rejuvenation |
| **Harness** | decide, perform, and issue the receipt |

A request is a proposal. Refusal is a normal outcome, and a refused request is
recorded — it is a fact about how the mind governed itself, not an error.

Rejuvenation is also **deterministic**: no model is consulted anywhere in the
path. `test_rejuvenation_never_consults_a_model` asserts no `generate` call
occurs.

---

## Measuring

`context_report` reads occupancy from the inference service and never
estimates. Forked prefixes are shared, so counting them against every owner
would overstate the pool: a shared prefix is charged once, and a fork's private
tail is charged to the fork.

When inference is unreachable the report says so and reports `nominal` rather
than inventing pressure it cannot see.

| Pressure | Default threshold |
|---|---|
| nominal | < 55% |
| elevated | >= 55% |
| high | >= 70% |
| critical | >= 85% |

`context_assess` measures and says what the Harness *would* do. It performs no
action and is safe for Id to call as often as it likes.

---

## Rejuvenation

Three steps, receipted:

1. **Checkpoint.** Publish the session's exact token prefix to durable content
   storage. Nothing is lost from the *record*. For Ego this becomes a published
   snapshot; for Id it becomes a content-addressed blob, because **Id's private
   context is never published as a shared snapshot**
   (`test_id_context_is_checkpointed_but_never_published_as_a_snapshot`).
2. **Retire.** Close the backend session; cells are reclaimed once no other
   sequence owns them.
3. **Rebirth.** Open a replacement session and reconstitute from the
   checkpoint.

### Reconstitution modes are not interchangeable

| Mode | What it does | Status |
|---|---|---|
| `exact` | Replay the whole recorded token prefix | Implemented — and useless for relieving pressure, because the context ends up the same size |
| `trim` | Replay a **verbatim head and tail**, dropping a measured span from the middle | **Default.** Still real tokens: no paraphrase, no model in the loop |
| `summarise` | Ask a model to compress the context | **Refused.** It is a *different behaviour*, not a better version of trimming, and calling it reconstitution would misdescribe what the mind now contains |

`trim` keeps `keep_head_tokens` verbatim from the head (system prompt and
earliest turns) and `keep_tail_fraction` of the context from the tail. The
dropped span is recorded by offset and count, and **remains reconstructible
from the checkpoint blob** — only the live context shrinks, not the record.

Attempting `summarise` raises `capability_unsupported` with that explanation.

---

## Rate limiting

A wedged Id must not be able to thrash the mind's contexts:

- `min_seconds_between_rejuvenations` (default 120) per role
- `max_rejuvenations_per_hour` (default 12) overall

---

## The automatic path

The scheduler calls `tick()` every 15 s. It acts **only at critical** pressure,
and then only on the largest role context. This is deliberately conservative:
rejuvenation costs a prefill and loses live context, so it happens when the
alternative is a mind that is measurably degrading — not merely a full-ish
pool. Set `auto_rejuvenate = false` to disable it entirely.

---

## Configuration

```toml
[homeostasis]
elevated = 0.55
high = 0.70
critical = 0.85
role_context_high = 0.75
keep_head_tokens = 512
keep_tail_fraction = 0.45
min_seconds_between_rejuvenations = 120.0
max_rejuvenations_per_hour = 12
auto_rejuvenate = true
```

## Budgets: what a session may think, and what it costs

Two questions, kept apart because they have different answers.

**Cognitive allowance** is policy. A session is given `context_budget_tokens`
and a `budget_basis` when it is created, and they never move afterwards:

| actor | allowance | basis |
|---|---|---|
| Ego | 16384 | total |
| Id | 8192 | total |
| `ego.neuocyte` | 6144 | private growth |
| `id.neuocyte` | 4096 | total |

A role inherits nothing, so *total* is what its allowance means. An
Ego-derived worker forks Ego's prefix, so its allowance is written against
what it adds past what it inherited -- judging its total would refuse it on
its first generate, before it had thought anything. A maintenance worker
starts cold and owns everything it holds.

These are logical ceilings **inside** the shared pool, not reservations of it.
Nothing is partitioned; the physical KV pool stays shared, and a role at its
ceiling has not taken anything from anyone.

**Physical cost** is measurement, derived from what the backend actually did:
a shared prefix is counted once and a recomputed one is charged in full.

The two come apart on the fork fallback. If forking fails and the prefix is
recomputed, the worker's allowance is unchanged -- it is the same worker doing
the same job, and a performance fallback must not decide what it may think --
while its physical cost rises by the whole recomputed prefix. One flag could
not have said both.

## Pressure is measured against the allowance

`role_context_high` is a fraction of the session's own ceiling, so the
proactive threshold and the hard refusal are in the same units:

| | hard | proactive at 0.75 |
|---|---|---|
| Ego | 16384 | 12288 |
| Id | 8192 | 6144 |

It used to be a fraction of the whole KV pool while the hard refusal was one
global 6144, so the proactive threshold sat six times higher than the wall and
rejuvenation could only ever happen by collision -- the role discovered the
limit by hitting it and losing a turn.

`role_context_high` is **advisory**: it produces a recommendation in
`context_assess` and `id_health`, and nothing acts on it automatically. What
the Harness enforces is the generation's own headroom. A generation is
admitted only if prompt *plus its output allowance* fits the budget (I111), so
a role is rejuvenated at the boundary before a turn that could not finish,
rather than after overrunning. Generated tokens grow the same quantity the
budget measures, and so do the environment and trigger text ingested at the
start of a turn.

| | budget | allowance | admitted while prompt ≤ |
|---|---|---|---|
| Ego | 16384 | 3072 | 13312 |
| Id | 8192 | 1024 | 7168 |

A continuation that resumes its message ingests nothing, so a long answer
spends its runway on the answer rather than on re-rendering the environment
for every piece (I109).

## Admission

Starting new work consults the pool. Measured for sessions that exist;
estimated from the work class's budget for one that does not, which is stated
as an estimate because it cannot account for a prefix that worker may have to
recompute.

`kv_admission_reserve_fraction` (0.15) is held back from *new* work only. It
is not unavailable KV -- running sessions grow into it freely. It exists
because a prompt that fits the pool exactly has left nowhere for its own
answer to go.

When the pool cannot be measured, admission does not guess. The queue is
durable, and refusing on an absent number would stall work for a reason that
has nothing to do with resources. See invariants I96, I97 and I98.

## Not implemented

- **Compaction.** Nothing merges or rewrites context. Eviction removes whole
  finished turns and `trim` drops a positional span; neither rephrases
  anything, which is the point.
- **Summarisation.** Refused by design, as above.
- **Neuocyte-session rejuvenation.** Neuocytes are mortal by design — the answer
  for a neuocyte with a stale context is retirement, not repair.
