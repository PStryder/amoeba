# The Prompt Library: Amoeba's cognitive family tree

```
ego@5                                    id@2
 │  "You are Ego, the outward-facing…"     │  "You are Id, the inward half…"
 │  temperature 0.7, top_p 0.95            │  temperature 0.3, top_p 0.9
 │                                          │
 └── ego.neuocyte@7          (append)       └── id.neuocyte@1      (replace)
      │  "You are now a bounded neuocyte…"       "You are a bounded
      │  temperature 0.4, max_output 512          maintenance neuocyte…"
      │
      └── ego.neuocyte.research@3  (append)
           "Specialise in reading code and citing file:line."
           temperature 0.15

                    ego.neuocyte.research@3.7.5
                    └── leaf first ──┘ │ │ └ ego@5
                                       │ └── ego.neuocyte@7
                                       └──── research@3
```

A profile is not a string in a config file. It is a **node in a versioned
family tree**, with an ancestry, a governance state, and a record of every mind
that was ever born from it.

---

## 1. Namespaces are ancestry

`ego.neuocyte.research` means `ego` → `ego.neuocyte` → `ego.neuocyte.research`.
Parent first, specialisation afterwards. The name *is* the family tree, so a
node cannot be reparented by renaming it and ancestry cannot disagree with the
hierarchy.

Two roots exist — `ego` and `id` — and only bootstrap may establish them.

## 2. Roots are bootstrap-only, structurally

The runtime creation path, `create_runtime_version`, has no parameter that
permits a top-level root. A caller that names `godmode`, or names `ego` and
explicitly passes the private `_allow_root` flag, does not get refused by a
policy check — the flag is dropped before the call, and the creation path
refuses a root outright. It is not a request the runtime can express.

This does not depend on a model instruction or a caller-supplied boolean, and
it is mutation-verified: removing *both* defences is what it takes to make the
test go green (I49).

> **Known over-restriction.** The same guard currently also refuses a new
> *version* of an existing `ego` or `id`, which was not the intent. A changing
> environment should never require new root doctrine, but a genuine change to
> Ego's or Id's doctrine should be proposable through ordinary governance —
> candidate, evaluation, Operator approval, no self-promotion — exactly like
> any descendant. The two prohibitions are enforced by *independent* checks:
> `validate_namespace` rejects a name whose root is not `ego` or `id` before
> `_allow_root` is consulted at all, and a separate `is_root` test in
> `create_version` blocks root versions. Relaxing the second cannot weaken the
> first. See §14.

## 3. Versions are immutable and pin their parent

Every namespace has monotonically increasing local integer versions. A version
is immutable once written — there is no update path for a definition, and a
"change" always produces a new version.

Critically, **a child version pins the exact parent version it inherits from**.
Approving a new `ego` changes nothing about any existing descendant. Ancestry
is resolved by walking the stored bindings, never by consulting what is
selected now, which is the entire reason the binding is stored (I51).

## 4. Lineage vectors

A fully resolved profile is identified leaf-to-root, one component per level:

```
ego.neuocyte.research@3.7.5
  research@3  →  neuocyte@7  →  ego@5
```

Leaf-first because the part that changes most often is the part you read first,
and it matches how the namespace is written.

A reference is **self-checking**: component count must equal namespace depth.
`ego.neuocyte.research@3.7` is malformed, not merely incomplete — resolving it
would mean guessing which level was omitted, and guessing is exactly how an
explicit request for a historical profile quietly becomes a current one. A
complete reference whose ancestry does not match is refused, and the refusal
names the actual lineage (I52).

## 5. Inheritance

**Ordinary properties** take the nearest ancestor that defines them. A leaf
setting `temperature` shadows its grandparent's; a leaf that is silent
inherits. The resolved profile records *which* level supplied every value, and
`explain_profile` reports which descendant later shadowed it.

**Prompt text** composes through each level's declared mode:

| Mode | Result |
|---|---|
| `inherit` | the parent's effective text, unchanged (local text is refused) |
| `append` | parent, blank line, local |
| `prepend` | local, blank line, parent |
| `replace` | local only |

The separator is exactly one blank line, never leading or trailing, and never
inserted when one side is empty — so a root establishing the base prompt is
byte-identical to its own text.

A child inherits its parent's **effective** text, not just its local fragment,
so contribution accumulates down the whole lineage.

Mode and text must agree. `inherit` with local text would silently discard it;
`append` with no text is a no-op wearing a mode's name. Both are refused.

## 6. Model variables

Exactly the six the inference backend applies:

| Variable | Backend argument |
|---|---|
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `max_output_tokens` | `max_tokens` |
| `seed` | `seed` |
| `stop_sequences` | `stop_strings` |

An unknown name is **refused, not dropped**. A silently discarded
`repetition_penalty` would be a profile claiming to have shaped cognition that
it did not. The map to backend arguments is total over the variable set, and a
test says so — a variable the backend cannot apply would be recorded, reported
and then ignored.

Harness constraints narrow a profile and never widen it: a profile states a
ceiling, the Harness may impose a lower one. Constraints are **not** a
parameter any caller supplies — a caller that could raise its own ceiling by
asking would not be constrained.

## 7. Bootstrap: files propose, the database decides

Files live in `src/amoeba/promptlib/prompts/<namespace>.md`:

```
# comments are allowed in the header
mode: append
temperature: 0.4
max_output_tokens: 512
---
The prompt body, taken verbatim.
```

Header keys are `mode`, `rationale`, and the model variable names. Anything
else is **refused**, so a typo like `temprature` fails loudly instead of
silently configuring nothing.

At every startup each file is compared against the library:

| Outcome | Meaning |
|---|---|
| `baseline` | the namespace does not exist; version 1 is created, approved and selected |
| `matched` | byte-identical to the selected version |
| `present` | some version has this definition but it is not the one running |
| `delta` | no version has this definition; a **candidate** is created |

The `delta` case is the point. Editing a prompt file and restarting must not
quietly change how the organism thinks: the change becomes a governed candidate
that somebody has to approve, and the selected version keeps running until then
(I50). `operator_prompt_bootstrap_report` answers "I edited a prompt file and
restarted — why is nothing different".

Comparison is against *every* version, not only the selected one, so restarts
are idempotent and an unselected namespace does not accumulate one identical
candidate per restart.

## 8. Governance

```
candidate → validated → evaluated → proposed → production_approved
                                             → experimental_approved
        ↘ rejected (terminal)        retired (terminal) ↙
```

**Id evaluates and proposes. The Operator decides.** Id can read the whole
tree, compare lineages, record a verdict (`endorse` / `concern` / `oppose`) and
author a candidate. Approving, selecting and cascading appear in **no** scope
table at all, so there is no secret Id could present that resolves to them. An
endorsement that promoted would make Id the approver by a longer route (I55).

**Approval is not selection.** An approved version is merely *selectable*.
Selection chooses which approved version new incarnations are born with, and
any approved version may be chosen — including an older one, because rolling
back is selecting a historical lineage rather than reconstructing it.

**Selection is not installation.** A running Ego, Id or neuocyte keeps the
profile it was bound to, because its context was primed with those bytes
(I53).

## 9. Cascade: propagating a parent change

Approving a new `ego` does nothing to its descendants. Cascade is how you give
that up on purpose:

| Mode | Effect |
|---|---|
| `none` | nothing; descendants keep their pins |
| `queue` | each descendant gets a **candidate** rebased onto the new parent |
| `approve` | the same rebase, approved and selected in one act |

A rebase **never edits a local definition**. The new version's `local_sha256`
is byte-identical to the one it was rebased from — the local digest
deliberately excludes the parent binding, which is what makes "only the parent
changed" a checkable fact rather than a promise (I54).

Cascade descends level by level: rebasing `ego.neuocyte` onto `ego@2` produces
`ego.neuocyte@2`, and `ego.neuocyte.research` sees nothing until it is in turn
rebased onto *that*. A plan is computed first and shown to the Operator, and
it reports what it **skipped** and why — an empty cascade must never look like
a complete one.

The plan is recomputed at execution time rather than taken from the caller: a
plan the console rendered a minute ago describes a library that may have moved.

## 10. Incarnation binding: what a mind actually received

At birth, every cognition-producing mind resolves its profile and freezes it:

```
bind_profile(namespace="ego", actor_id="ego", actor_kind="ego")
  → prompt_text, profile_ref, prompt_sha256, config_sha256,
    profile_sha256, effective_settings, lineage
```

The binding stores the **resolved digests and the full lineage vector**, not a
pointer to be re-resolved. Cognition that happened must stay explicable from
what the organism held at the time; re-resolving against a library that has
since moved would quietly rewrite history (I57).

Roles bind *before* registering, because the digest reported at registration
has to be the digest of the text the incarnation is actually about to prime its
context with.

**Forked neuocytes.** A worker forked from an Ego snapshot already physically
holds its ancestors' text. It injects only the suffix below `ego`, and the
binding records the injected digest and the inherited prefix **separately** —
"this mind received profile P" and "these are the bytes that were injected" are
two different statements, and conflating them would make one of them false.

Neuocytes had no profile at all before this: their instructions were module
constants, so a worker could not be specialised and there was no record of what
any of them had been told. They now descend from the same tree —
`ego.neuocyte` for Ego-derived work, `id.neuocyte` for maintenance — which is
what makes `ego.neuocyte.research` expressible at all.

## 11. Who can reach what

| Caller | Read tree | Evaluate | Propose | Approve / select / cascade | Bind own profile |
|---|---|---|---|---|---|
| Operator | ✓ | — | ✓ | ✓ | — |
| Id | ✓ | ✓ | ✓ | **absent** | ✓ |
| Ego | — | — | — | **absent** | ✓ |
| Neuocyte | — | — | — | **absent** | ✓ |
| MCP / API client | **absent** | **absent** | **absent** | **absent** | **absent** |

Ego is deliberately excluded from governing its own prompt: it is the component
most exposed to a confident user.

The external surface is untouched. Reading the library would expose the
organism's cognitive configuration; proposing to it would be writing the
organism's mind through a public door. Defended in depth — the adapter
allowlist and the credential scope are independent lists, and both would have
to be widened (I56).

## 12. Verbs

**Read** (Operator, Id)
`prompt_tree` · `prompt_versions` · `prompt_resolve` · `prompt_diff` ·
`explain_profile` · `prompt_incarnations`

**Id**
`id_evaluate_prompt` · `id_propose_profile`

**Operator**
`operator_prompt_author` · `operator_prompt_state` · `operator_prompt_select` ·
`operator_prompt_cascade_plan` · `operator_prompt_cascade` ·
`operator_prompt_bootstrap_report`

**Birth** (every role and neuocyte)
`bind_profile`

`prompt_diff` answers the question that actually matters after a cascade:
identical prompt and config digests mean two lineages produce the same
cognition despite different version numbers.

## 13. What `id_propose_prompt` is now

> Shaped around the over-restriction in §2, and expected to change with it.


The older verb targets `ego` and `id`, which are bootstrap-only roots. It
therefore cannot and does not change the library. It records Id's suggested
wording, content-addressed, where the Operator will find it; adopting it means
editing the shipped file, which makes the change reviewable in the repository
before it ever runs, and it then arrives as a governed candidate at the next
start.

For anything below a root, `id_propose_profile` creates a real governed
candidate.

## 14. Residual limits

* **Root versions cannot be proposed at runtime.** See §2. `ego` and `id`
  accept new versions only from the bootstrap files, so a doctrine change
  currently requires a file edit rather than a governed proposal. Creating a
  *new* top-level namespace must remain impossible and is blocked separately.
* **Ego and Id receive no dynamic environment manifest.** A role's context is
  primed once, at birth, with its profile text, and each turn carries only its
  trigger — a user message, a dossier, measured state. Nothing tells a role
  which productive profiles exist, which effectors it currently has, or which
  resources are registered. Neuocytes already have this separation: they are
  given a Harness-built tool block per turn, recorded as `tools_offered` in the
  work result, and their prompt therefore cannot advertise a capability the
  work row does not carry. Roles have no equivalent, so a newly approved
  `ego.neuocyte.do_thing` is invisible to Ego without editing Ego's own
  doctrine — which is precisely the coupling this library exists to remove.
* **No A/B evaluation.** `experimental_approved` and the `experimental`
  selection purpose exist and are honoured, but nothing measures whether an
  experimental profile performs better. Id records verdicts; those are
  judgements, not measurements.
* **No harness ceiling yet.** `_harness_constraints` returns `{}`. The
  mechanism is real and applied — a constraint placed there narrows the profile
  and shows up in `effective_settings` — but the Harness currently imposes no
  generation ceiling of its own, and inventing one so the field looked used
  would be the kind of decoration this library exists to avoid.
* **Prompt files are ordinary files.** They live in the source tree and are
  protected by the repository, not by `security.py`. A process running as the
  Amoeba account can edit them — but doing so produces a *candidate*, not a
  change in behaviour, which is exactly the point.
* **Sampling reaches the backend, with caller and budget narrowing it.** A
  role's generation takes `temperature`, `seed` and `max_tokens` from the bound
  profile; an explicit argument at a call site still wins for temperature and
  seed, while `max_tokens` is *narrowed* rather than replaced, so a call site
  cannot ask for more than the profile allows. A neuocyte's ceiling is narrowed
  again by the Arbiter's remaining token budget, which always wins. `top_p`,
  `top_k` and `stop_sequences` are resolved and recorded but not yet passed by
  these two call sites — they are available through `backend_arguments` and
  reach the engine only where a caller forwards them.
