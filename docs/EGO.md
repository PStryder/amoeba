# Ego: senses, effectors, and the outward interface

Ego interprets what the user wants and introduces that objective into the rest
of the organism. It is the outward conversational interface — not a mandatory
central planner.

**Bounded by capability physics, not by workflow.** Nothing constrains *how*
Ego thinks. It may answer directly, ask for one investigation, ask for five
independent ones, or hand over a worked plan when it genuinely has one. The
work system, the blackboard and the neuocytes are free to discover
decomposition, follow-up questions and intermediate structure during execution,
subject to Harness authority and resource policy.

What is fixed is what Ego can *reach*.

## Sensory surface

| Sense | Verb |
|---|---|
| Maintained state | `recall`, `get_memory` |
| History, provenance, receipts | `history`, `provenance`, `audit_dossier`, `get_conclusion` |
| Blackboard | `board_read`, `board_get_post`, `board_thread`, `board_stats` |
| Work state | `get_work` (incl. results), `queue_stats`, `ego_work_view` |
| Artifacts | `artifact_list`, `ego_artifact_evidence` |
| Resource identity | `ego_resource_identities` |
| Context | `context_report` |
| Backchannel | `ego_message_id`, and signals delivered to its own process |

`ego_work_view` is the light operational view: what is queued, running,
blocked, and which items Ego originated. It is **not** `system_pulse`. Ego does
not receive failure counters, context occupancy, storage pressure, security
policy or resource digests — handing the outward interface the organism's
health telemetry would blur it into the inward monitor and leave two components
reasoning about health with no agreed owner.

`ego_resource_identities` is the narrow exception: model generation, Ego's own
prompt digest (configured and embodied), and the neuocyte tool-surface digest.
A result produced under a different prompt is not strictly comparable to one
produced under this one, so Ego needs the identity — not the policy.

## Running compute scratch is not a communication channel

Ego may know a neuocyte exists, what work it holds, its board mode, its
capabilities and its status. It **cannot** read the sandbox.

Half-written scratch is not a claim anybody made. Reasoning over it would let
Ego consume something no neuocyte ever published, with no authorship and no
moment at which the worker stood behind it. Anything worth Ego's attention
crosses an explicit boundary with a name on it:

* a blackboard post,
* a work result,
* an artifact proposal (with its exact immutable evidence),
* durable evidence attached to either.

## Three ways Ego reaches work

Kept separate, because collapsing them would make the weakest one the effective
semantics of all three.

### 1. Admission — `ego_request_work`

Ego supplies intent: objective, class, replica count, board mode, independence,
constraints, evidence references. The Harness decides **everything** about
execution — whether and when to admit, which neuocyte gets it, which promoted
prompt version it receives, its budget and its capabilities.

```
ego_request_work(objective=…, replicas=3, independent=True)
  → three items admitted board-naive
  → Ego cannot choose the worker, the prompt, or the budget
```

There is no verb that instantiates a worker.

### 2. Blackboard — `board_post` / `board_read`

The ordinary collaborative surface, subject to each item's board policy. A
board-naive neuocyte does not receive Ego's posts merely because Ego is the
executive role.

### 3. Targeted work message — `ego_work_message`

A governed message to a **work item**, not to a process:

```
ego_work_message(work_id=…, message="prefer the 2024 data")
  → recorded by the Harness, provenance-bearing
  → collected by the neuocyte at a turn boundary
  → the original objective is unchanged
  → collection is recorded, so a later finding is marked as possibly influenced
```

**Refused when independence requires it.** A board-naive item was admitted
precisely so that whatever it concludes is its own. A clarification from the
executive role mid-flight would destroy exactly that property — quietly, and in
a way that still looks like independent replication afterwards. The refusal is
itself recorded.

Also refused once the item is no longer live.

## Effectors

| Effector | What Ego is asking for |
|---|---|
| `ego_request_work` | introduce an objective, optionally replicated or independent |
| `ego_work_message` | a governed mid-flight clarification |
| `ego_request_cancellation` | stop work **it originated** |
| `ego_propose_memory` | a maintained-state change, by supersession |
| `ego_message_id` | backchannel to Id |
| `ego_request_id_review` | ask Id for a second opinion |
| `board_post` | publish to the collaborative surface |
| `record_conclusion` | an auditable outward answer |
| `publish_ego_snapshot` | publish a prefix of its own context |

Cancellation is scoped to Ego's own requests: cancelling someone else's work is
an organism-level intervention, which belongs to Id or the operator. The
Harness still performs the transition and may refuse.

`ego_propose_memory` supersedes rather than edits. Ego is the component most
exposed to a confident user, so it is the most likely to acquire a belief
nobody checked — which is exactly why it goes through the same governed path as
everyone else's.

## What Ego cannot do

Id-only maintenance verbs, `system_pulse`, `id_health`, rejuvenation execution,
prompt editing or promotion, scheduler policy, security or filespace
configuration, artifact promotion, direct belief authoring (`remember`),
`tool_invoke`, any sandbox verb, another role's private context, and anything
that widens its own scope.

If Ego needs one of these, it asks Id, the Harness, or the operator through a
governed mechanism.

## Capability isolation

Identical in mechanism to the Id boundary (`ID.md`): one method table per scope
in `scopes.py`, and the scope is decided by the **secret presented**, not by any
field in the request. There is no role, actor, caller, `from_role` or
`origin_actor` parameter that changes what a connection can reach.

* Ego-only verbs appear in no other scope — not Id's, not a neuocyte's.
* Id-only verbs appear in no other scope — not Ego's.
* A neuocyte cannot name, list or dispatch either set.
* Unknown-method errors do not enumerate the table.
* `tool_invoke` builds only a neuocyte registry, so a model-supplied string
  cannot reach a role verb.
* Re-presenting a different token mid-connection does not widen a scope.

A message's author is the authenticated scope, never a parameter — so
attribution is a fact rather than a claim.

If a neuocyte needs something Ego can do, it says so through a work result, a
board post, or an artifact proposal. It does not borrow Ego's authority.

## Provenance

`what Ego knew → what Ego requested → what ran → what came back → what Ego
concluded` is reconstructible from the event log:

| Event | Records |
|---|---|
| `work.requested_by_ego` | the objective, replicas, board mode, admitted ids |
| `work.message_sent` | the message, its kind, the target item |
| `work.message_consumed` | which neuocyte collected it, and when |
| `work.message_refused` | refusals, with the reason |
| `ego.review_requested` | what Ego asked Id to look at |
| `board.posted`, `memory.created`, `conclusion.recorded` | the usual paths |

All attributed to `ego`, all in the hash chain.

## What Ego is given each turn

Three separate things, and Ego's doctrine is only the first:

* **Profile** — `ego@N` from the Prompt Library, bound at incarnation and
  frozen. Ego does not get a configured prompt appended to it; that path is
  closed (I63).
* **Environment** — a `role_environment` declaration built by the Harness at
  the start of every bounded turn: the cognitive profiles currently available
  to delegate to, the capabilities Ego may invoke right now, and the resource
  identities in force. Rebuilt per turn, frozen within one.
* **Turn input** — the message, question or trigger.

Ego's capabilities in that declaration are derived from the same scope table
that gates dispatch, so Ego is never told it has something its credential
cannot reach (I58). Ego executes them through a bounded Harness-mediated tool
loop: one call, the Harness validates and runs it, the result comes back, and
generation resumes (I59).

Ego does **not** receive Id's physiological telemetry, Id's effectors, or any
Operator verb, and its environment lists only `ego.*` profiles.

A newly approved `ego.neuocyte.research` appears in Ego's next environment with
Ego's own prompt untouched. That is the point: the world changing must not
require rewriting the constitution (I62).
