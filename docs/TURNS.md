# Persistent roles, bounded turns

```
                        HARNESS
                           │
                 deterministic triggers
                           │
             ┌─────────────┴─────────────┐
             │                           │
            EGO                          ID
     persistent identity          persistent identity
     event-triggered              event + continuation + heartbeat
             │                           │
        bounded turn                bounded turn
             │                           │
             └────────  requests  ───────┘
                           │
                        HARNESS
                           │
                       NEUOCYTES
                 disposable work cognition
```

Ego and Id are **identities**, not infinite generations. Their cognition
happens in bounded turns, and the Harness owns everything around those turns:
when one begins, what triggered it, what inputs are admitted, and what happens
when it ends. The model owns only the reasoning inside one.

---

## 1. What this replaced

`ego_converse` used to call into the Ego process synchronously:
`sup.client("ego").call("converse", ...)`. The role's RPC server is a
`ThreadingTCPServer`, so **two callers produced two concurrent turns against a
single inference session** — two thoughts interleaved into one context. The
role's lock guarded a signal list, not cognition. `io_api` spawns a thread per
external interaction, so two submits reproduced it.

There was no mailbox, no queue, no ordering, and no record of why a role
thought anything.

## 2. One turn at a time, per role

Enforced twice, independently:

* the role process runs **exactly one turn thread**, so a second concurrent
  turn has nowhere to execute;
* `role_turns` carries a partial unique index over open turns, so a second
  claim is a database constraint violation.

Ego and Id still run concurrently with each other; neuocytes are untouched.
Serialization is per persistent role, never global — waiting for Ego blocks
nothing else.

## 3. The mailbox

```
something happens  ->  queued      durable, and NOT yet seen
a turn begins      ->  claimed     bundled and frozen for that turn
the turn commits   ->  consumed
the role dies      ->  queued      eligible again, identity preserved
```

**Queued is not seen.** A trigger becomes a cognitive input only when the
Harness puts it in a specific turn's bundle. Submitting something is not the
same as a mind having considered it, and the two states are separately visible
so nobody has to guess which one they are looking at.

Consumption happens at *commit*, not at claim. A role that was handed inputs
and then died has not thought about them.

| Column | Meaning |
|---|---|
| `trigger_id` | stable identity, preserved across redelivery |
| `kind` | why the role woke; see below |
| `source` / `source_ref` | who caused it, and what it points at |
| `summary` | the bounded text the model reads |
| `payload_sha256` | the full immutable payload |
| `status` | `queued` → `claimed` → `consumed`, or `expired` |
| `turn_id` | the turn that consumed it |
| `deliveries` | how many times it has been handed out |

## 4. Trigger kinds

`user_input` · `work_completed` · `work_failed` · `work_cancelled` ·
`artifact_event` · `board_event` · `role_message` · `operator_message` ·
`continuation` · `heartbeat` · `startup`

The kind survives into what the model reads. A mind should know *why* it woke:
"a work item you asked for failed" is a different thought from "someone sent
you a message", and flattening both into anonymous prose loses that.

**Summary is not body.** `summary` is a bounded label for operator listings;
the **body** is what was actually said, stored in full and rendered into the
turn up to `MAX_BODY_CHARS`. Rendering only the summary meant Ego answered
questions it had only seen the first 400 characters of — and a request's
constraints sit at its end far more often than in its opening. When the budget
does truncate, the rendering says so and names the digest holding the rest
(I76).

## 5. Bundling

Multiple things accumulate while a role is busy. At the next boundary they
become **one** bundle rather than one turn each — Ego waking once for four
things that happened while it was working is the point; waking four times is
the thrash it exists to avoid.

Bounded at `MAX_BUNDLE` (16), ordered by arrival, and the bundle *says what it
left behind* rather than dropping it. Every member's identity is preserved: a
bundle is not permission to erase individual causal provenance.

Ordering is by insertion, not by the clock. `created_at` ties for anything
queued inside the same millisecond, and a ULID's random suffix does not break
that tie usefully — two messages sent together could otherwise reach Ego in
either order.

## 6. Frozen at the boundary

At turn start the Harness fixes three things in **one transaction**:

```
PROFILE       the already-bound incarnation profile
ENVIRONMENT   a freshly built authoritative manifest
BUNDLE        the exact triggers admitted to this turn
```

Nothing reopens them. A trigger arriving a microsecond later waits.

> **Events wake cognition; they do not interrupt it.**

If a tool call during the turn changes the world, the *tool result* is what
tells the model — not a manifest or a bundle that shifted underneath it.

## 7. Ego: event-driven, never autonomous

Ego is quiescent when nothing requires outward cognition. It gets a turn
because something deterministic happened:

| Source | Relevance rule |
|---|---|
| user / operator input | addressed to Ego |
| work completed / failed / cancelled | `origin_actor == "ego"` on the work row |
| message from Id | explicitly targeted |
| continuation | its own previous turn stopped non-terminally |

Relevance comes from a **recorded relationship** — `origin_actor` — not a
heuristic. Work Ego did not originate does not wake Ego, which is what stops a
busy neuocyte fleet becoming a wake storm. A requeued failure is an attempt
rather than an outcome, so it does not wake anyone.

Ego has **no heartbeat**. A persistent identity that generates cognition
because its process exists is not the same as one that responds.

**Limitations, stated rather than invented.** Artifact and blackboard events do
not yet create triggers. The relevance rule for them would have to be
"blackboard posts on work Ego originated", and rather than ship a heuristic
that wakes Ego on unrelated board traffic, the trigger kinds exist and nothing
emits them yet. Work-derived wakes cover the case that matters today.

## 8. Id: continuously responsible

Id keeps responsibility for internal state even when nobody sends it anything.

| When | What |
|---|---|
| startup | one turn, to form an initial view of the organism it woke in |
| relevant event or message | wakes promptly |
| its previous turn stopped non-terminally | a continuation |
| quiet organism | a deterministic heartbeat |

The heartbeat is **explicit input** with `kind="heartbeat"` and a stated
reason, not a fake user message. It does not dump telemetry into context: Id
has `system_pulse` and its other senses and can look if it wants to.

A heartbeat that finds nothing backs off (`×2`, to a ceiling), so a quiet
organism gets quieter rather than paying full price to keep discovering that
nothing happened. Any real trigger resets the interval. `id_heartbeat_seconds
= 0` disables it, leaving Id purely event-driven.

Id is logically always on. It is not a token furnace.

## 9. External input queues; there is no fast lane

`ego_converse` enqueues and then *waits* for the turn that consumes its
trigger. Waiting is a convenience for the caller, not a second ingestion path:
the trigger is durable and ordered the moment it is queued, and whether anyone
is blocked on it changes nothing about when Ego sees it.

A caller that gives up gets `queued` back and the cognition happens anyway.

**An interaction is complete only when it is answered.** The external worker
is asynchronous by construction — `io_submit` returns an id and the client
polls — so it waits for the answer across however many continuation turns the
thought needs, rather than giving up on the caller's behalf. If the answer
never arrives it **fails**, saying so and noting that the input is still
queued. Reporting an empty answer as a completed interaction would tell the
client, permanently, that nothing was the organism's reply (I79).
There is deliberately **no** path where an idle Ego is called directly while a
busy one goes through the mailbox — that would make conversational ordering a
race between whoever called while Ego happened to be free.

External clients still get only semantic I/O. Submitting input may cause Ego
to do a great deal on Amoeba's authority; it grants the caller none.

## 10. Stop reasons

| Reason | Terminal? |
|---|---|
| `model_stop` | yes — the thought finished |
| `max_output_tokens` | **no** |
| `token_budget_exhausted` | **no** |
| `tool_turn_limit_reached` | **no** |
| `context_pressure` | **no** |
| `deadline_reached` | yes |
| `cancelled` | yes |
| `backend_error` | yes |
| `role_failure` | yes |
| `no_environment` | yes |

Kept distinct because they drive different behaviour. They are detected from
the substrate — the backend's finish reason, the loop's own bounds — never
from the model cooperating. Requiring a magic phrase to survive truncation
would make the guarantee depend on the thing that failed.

## 11. Continuation

A non-terminal stop earns another turn, decided by the **Harness**. A thought
cut off by an output ceiling cannot be relied on to ask for its own
continuation, because being cut off is what stopped it.

A continuation is a **new bounded turn**, not an invisible extension:
`parent_turn` records the link, so the reasoning chain is inspectable rather
than pretending several generations were one event.

**Bounded** at `max_continuations` (default 3). This was found by running it:
with a backend that always truncated, every turn scheduled a successor that
also truncated, and the organism burned its context until inference refused
the prompt. An unbounded continuation policy is a token furnace. When the
chain stops, that is recorded rather than silent.

**A continuation resumes the message it continues** when the parent was cut
off by its ceiling, was the role's last turn, carries nothing else, and the
session is still exactly as long as the parent left it. Generation simply
carries on from that token -- no new message, no re-rendered environment -- so
the pieces join byte for byte. Otherwise it asks visibly, as a new message (I109).

**The answer is the interaction's, not the last turn's.** Every piece, in
order, once, assembled along the parent links (I107). Only a thought the model
ended is `answered`; one stopped by the continuation limit, a deadline or a
failure is `incomplete`, with everything it said and why it stopped (I108).

## 12. Context homeostasis

**Rejuvenation is Harness-initiated.** `context_rejuvenate` appears in no role
scope: a role that decided it needed more room cannot simply take it, and
cannot rewrite its own context at all. It reports `context_pressure` as a stop
reason and the Harness answers — reclaiming context between turns, handing the
role its replacement session, and letting the continuation run against it.

An earlier version of this work put `context_rejuvenate` in the role scope so
the role could rejuvenate itself. That granted Ego a power the verb's own
docstring forbids, and the existing prohibited-power test caught it.

Context pressure is a **turn outcome**, not a crash. On hitting it the role:

1. stops at the turn boundary and reports `context_pressure`;
2. the **Harness** rejuvenates (`context_rejuvenate`, `trim` mode);
3. the Harness hands the role its replacement session (`refresh_session`);
4. the continuation the stop reason earned runs against it.

Never mid-generation: rejuvenation replaces the inference session, and doing
that under an active turn would discard the reasoning in progress. Role
identity, incarnation, profile binding and mailbox all survive — a new session
is not a new mind, and triggers the turn never consumed stay queued.

## 13. Crash recovery

At-least-once, deliberately, with identity preserved.

A turn left `running` by a process that is gone is re-opened at supervisor
startup: its unconsumed triggers return to `queued` and `deliveries`
increments, so a replay is *visible* rather than looking like a new event.
Without this a crash would also wedge the role permanently, since one open
turn per role is a database constraint.

**Recovery runs whenever a role is restarted, not only at supervisor start.**
This matters more than it sounds: one open turn per role is a database
constraint, so a turn its owner never closed blocks *every* future turn for
that role. The role comes back, heartbeats, reports healthy, and never thinks
again while its mailbox fills. Recovery used to run only in
`Supervisor.start()`, while supervision restarts individual roles routinely.

**A hung role is swept, too.** A crash is recoverable because the process is
visibly gone; a hang is not, and it wedges the role identically. Any turn still
open `turn_wall_seconds + STALE_TURN_GRACE_SECONDS` after it began has already
ignored the deadline the role enforces on itself, and is abandoned.

A trigger that has been delivered `MAX_DELIVERIES` (3) times without a turn
surviving is `expired`. An undying poison message would be worse than a lost
one, and the expiry is recorded.

Exactly-once would need distributed-transaction machinery across a process
boundary. This does not claim it.

## 14. The side channel

Delivery may be transient. **Influence may not be unaudited.**

The backchannel used to push into an in-memory list on the role process,
bounded at 64 and dropping the oldest — and `IdProcess.status()` drained that
list into Id's cognition. A receipt-free message with no author, no body and
no record was shaping persistent cognition.

`side_channel` now queues a durable, attributable trigger by default, carrying
its author, its exact body and its consumption relationship. The transient
signal is still delivered for liveness-style nudges, and `durable=False` keeps
that behaviour — but a non-durable signal cannot enter a trigger bundle.

## 15. Provenance

`role_turns` records, per turn: role, incarnation, `profile_ref`, profile
digest, environment digest and blob, bundle id, **bundle digest and blob**,
trigger kinds and count, start and finish, status, stop reason, model
generation, tool call count, result digest, and `parent_turn`.

The bundle and environment are content-addressed **before** the turn runs, so
the exact inputs a past turn reasoned against are read back rather than
recomputed from state that has since moved. Historical turns are never
rewritten by later changes.

`role_turn(turn_id)` returns all of it, including the reconstructed bundle text.

## 16. Operator visibility

The `turns` panel shows, per role: state (`idle` / `queued` / `processing` /
`heartbeat_wait` / `recovering`), current turn, queued count and kinds, current
profile, last stop reason, and next heartbeat. Plus recent turns with their
trigger kinds, stop reasons and continuation links, and a drill-down into any
turn's exact bundle.

State is derived from durable rows — an open turn row is the ground truth for
"processing" — so a stale in-memory flag cannot claim a role is idle while a
turn is running.

The Operator can put a message in a role's mailbox. That is **input, not
authority**: it wakes the role and is attributable, and the role acts only
through its own effectors.

## 16a. An expired turn cannot act

Abandoning a turn releases its inputs to a replacement. That would be
dangerous on its own: the original turn may still be alive and about to wake
up, and two live turns from one lineage could both act.

`mailbox.complete` already refused a turn that is not `running`, so a late
turn could never commit its *result*. Its **side effects** were a different
matter — a role's effector call carried no turn identity whatsoever, so a
swept turn could still request work and record conclusions.

Every capability a role invokes now goes through `role_tool_invoke`, carrying
the turn it belongs to. The turn id is the fencing capability, exactly as a
neuocyte's fencing token is:

* it is minted by the Harness and handed only to the role that claimed it;
* it is readable nowhere a role can reach — the mailbox and turn views are
  operator-only — so there is no `role` argument to forge, and which role is
  asking is derived from the turn;
* a turn that is not `running` buys nothing;
* the verb must still be one that role is offered, so the capability boundary
  is unchanged.

## 16b. A role cannot forge attribution

`role_enqueue_trigger` takes `source` as an argument, so any holder can write
into a mailbox attributed to anyone. It is therefore **absent from every role
scope** — Ego holding it could queue "the operator says approve this" into Id's
cognition, and Ego is the component most exposed to a confident user.

Every legitimate caller is the Harness, holding the control token. Roles reach
each other through `ego_message_id` / `id_message_ego`, which attribute the
sender themselves.

## 16c. What lineage is, and is not

Lineage scopes what enters a **turn**. A continuation reads its own delegated
work, its own artifacts and messages about its own thought; another
interaction's evidence waits for a turn of its own, unless it is explicitly
ambient.

It is **not** a confidentiality boundary, and nothing here should be read as
one. Ego holds a persistent context and maintained state across interactions,
so information that entered its context in an earlier turn is still there in a
later one whoever asked. Lineage removes the sharp edge — one client's
evidence sitting in the same prompt as another client's question — and does
not, and cannot, make one persistent mind into several.

> One Amoeba is one cognitive trust domain. Two workloads that need real
> isolation get two Amoebas.

### Where a lineage comes from

Lineage is assigned by the producer of a trigger, at the point where the
organism still knows why the trigger exists. Nothing infers it later, because
by then the answer is always "nobody".

| trigger | lineage |
|---|---|
| a conversational request | the conversation, or the operation if there is none |
| an investigation, introspection or audit request | the operation that opened it |
| a work result | the operation that requested the work |
| a continuation | inherited from the turn it continues |
| a heartbeat or startup review | none, and explicitly ambient |
| an operator message, or one role messaging the other | none, and explicitly ambient — it addresses the role, not an interaction |

Declaring one or the other is mandatory: a trigger that names neither belongs
to nobody, and a turn already serving an interaction will not admit it. That
is the intended rule, and it makes an omission invisible at runtime — the
trigger simply waits. So the requirement is enforced in the test suite
instead, where forgetting fails the build rather than stalling a role.

The three questions stay independent. `expects_answer` is about who is owed a
reply; a work result is owed none and still belongs to exactly one
interaction. `ambient` is a statement that every turn may see something, not
a shrug about who owns it.


See `ARCHITECTURE.md`, "One Amoeba is one cognitive trust domain".

## 17. The substrate is not a cognitive component

Supervisor, scheduler, Arbiter, mailbox, bundling, wake policy and turn
lifecycle are **deterministic**. There is no scheduler neuocyte, no supervisor
neuocyte and no arbiter agent, and a test asserts that the scheduling modules
contain no inference calls at all.

`role_claim_turn`, `role_complete_turn` and `role_enqueue_trigger` are in the
role *process* scope and are **not** model-facing. Claiming your own next turn
is not a cognitive act, and a mind that could would be scheduling itself.

> Cognition may decide what it wants. Harness physics decides when and whether
> execution happens.

## 18. Configuration

```toml
[scheduler]
poll_seconds = 0.5              # role asks the Harness for a turn this often
id_heartbeat_seconds = 300.0    # 0 disables; Id becomes purely event-driven
id_heartbeat_max_seconds = 1800.0
id_heartbeat_backoff = 2.0
id_startup_turn = true
ego_startup_turn = false        # Ego does not talk to itself
max_continuations = 3
turn_wall_seconds = 180.0
submit_wait_seconds = 120.0     # how long a caller blocks, not how long work takes
```

## 19. Residual limits

* **Board and artifact waking is by ownership only.** A role is woken by what
  happens to work it originated. A post that merely *mentions* its work, or
  concerns a neighbouring topic, wakes nobody -- that would need a similarity
  judgement, which is the heuristic this deliberately does not make. See I92.
* **Polling, not pushing.** A role asks for a turn every `poll_seconds`, so
  there is up to half a second of latency between queueing and cognition. A
  push would need a second channel into the role process; the poll is one
  cheap query and was not worth the complexity yet.
* **At-least-once, not exactly-once.** Stated in §13 rather than claimed away.
* **Untagged evidence can wait a while.** A trigger that names no lineage is
  admitted by the first turn that is not already serving an interaction, which
  on a busy role may not be the next one. Every producer in the source
  declares a lineage or declares itself ambient, so this is reachable only by
  a new producer that forgets -- which fails a test rather than stalling
  quietly -- but the runtime behaviour is deferral, not an error.
* **Only the heartbeat yields to pressure**, and only for a bounded time. An
  event-driven turn is never held back at any pressure; the discretionary
  heartbeat is deferred while the pool is strained and runs regardless once
  the ceiling is reached, because Id's heartbeat is the homeostatic review and
  a review that never happens is worse than a turn that costs a prefill. See
  I99.
