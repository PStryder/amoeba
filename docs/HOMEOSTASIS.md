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

## Not implemented

- **Compaction.** Nothing merges or rewrites context; `trim` only drops a span.
- **Summarisation.** Refused by design, as above.
- **Worker-session rejuvenation.** Neuocytes are mortal by design — the answer
  for a worker with a stale context is retirement, not repair.
