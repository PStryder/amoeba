# The tool execution loop

A model can *request* a tool call. It cannot perform one.

That sentence was previously true because nothing executed tools at all. It is
now true because of where the code lives: the loop runs in the neuocyte
process, and the *authority* runs in the Harness. The neuocyte parses a request
out of generated text and sends it to `tool_invoke`; it never holds a handler.

```
neuocyte process                    Harness (supervisor)
────────────────                    ────────────────────
generate ──► text
parse  ──► {name, arguments}
             │
             ├──── tool_invoke ────► check lease + fencing token
             │                       read capabilities from the work row
             │                       build the registry for THIS work item
             │                       validate arguments against the schema
             │                       execute
             │                       emit tool.requested + tool.result/rejected
             ◄──── outcome ─────────  receipt
append result to session
generate ──► ... (repeat, bounded)
```

## What a neuocyte may call

Always: `recall_memory`, `read_history`, `get_conclusion`, `list_open_work`,
`current_state_version` — all read-only.

Only when the work item was admitted with `sandbox_allowed=true`: `run_code`,
`write_file`, `read_file`, `list_files`, `propose_artifact`.

## Four things that are true by construction

**The capability comes from durable state.** `sandbox_allowed` is read from the
work row at the moment of the call, never from the request. When it is false
the sandbox tools are *not registered*, so there is no handler to reach — not a
guard that could be bypassed by a bug in the guard. Arguments that look like a
grant (`sandbox_allowed: true`, a different `work_id`) are simply unknown
arguments to a tool that does not exist for this caller.

**A model cannot name a sandbox.** No tool takes a sandbox id. The sandbox is
resolved from `work_id` server-side, created on first use and destroyed when
the work item reaches a terminal state. There is no argument in which to put
another neuocyte's sandbox.

**A fenced neuocyte cannot still run code.** The lease, the owner and the
fencing token are checked on the work row before any tool runs. A neuocyte that
was killed, expired or superseded is refused — killing one actually stops it,
including mid-loop.

**Scratch does not outlive its work item.** Promotion is the only way out, and
promotion is a Harness act with its own receipt that re-hashes the content at
the moment of copying.

## Three independent bounds

A model that keeps calling tools is an expected outcome, not a malfunction, so
the loop is bounded three ways and reports which bound bit:

| Bound | Source | `stop_reason` |
|---|---|---|
| Turns | `arbiter.max_tool_turns` (default 6) | `turn_limit_reached` |
| Tokens | the work item's granted budget | `token_budget_exhausted` |
| Wall clock | the work item's deadline | `deadline_reached` |

A tool requested on the *last* available turn is not executed: its result could
never be read, so running it would be a side effect with no purpose. This is
deliberately redundant with the loop bound — two reasons that happen to
overlap — and the mutation harness negates both together. Do not "simplify"
either away.

## Everything is recorded, including refusals

Every call emits `tool.requested`, then `tool.result` or `tool.rejected`, and
returns a receipt. A refusal is a fact about how the mind governed itself, so
it is recorded rather than merely returned. Large arguments are logged as a
digest plus a head, not in full — the payload belongs in the sandbox.

A refusal is also fed *back to the model*, saying what was refused and why.
Hiding it would leave the model guessing at why its request vanished.

## Testing note

The three layers fail differently, so they are tested separately rather than
through one end-to-end path that would hide which part is doing the work:
authorisation against a live Harness with no model involved, loop control
against fake RPC clients so the bounds are hit precisely, and one end-to-end
path through a real AppContainer.

Hashed deterministic text never contains a tool call, so the deterministic
backend grew a `script_responses` affordance. It exists only on that backend
and the inference service exposes the verb only if the backend has it — with a
real model loaded it does not exist. Scripted replies still carry the
`[SIMULATED]` label: a canned reply is no more model inference than a hashed
one, and the label is what stops either being mistaken for it.
