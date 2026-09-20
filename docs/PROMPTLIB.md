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

Two roots exist — `ego` and `id`. Only bootstrap may *establish* a
top-level namespace; new *versions* of an existing root are ordinary
governance (§2).

## 2. A root namespace is bootstrap-only; a root *version* is not

Two different operations, and only the first is reserved:

| Operation | Who |
|---|---|
| create a new top-level namespace (`godmode`, `foo`) | **bootstrap alone** |
| create a new version of an existing root (`ego@N+1`) | ordinary governance |

**Establishing a namespace** is `establish_root`, which no scope table names
and no RPC verb calls. The permission is *which function you can reach*, not a
flag you decline to pass — an earlier design gated it on a private boolean,
which meant the guarantee rested on every caller choosing not to set it.
`create_version` has no power to establish anything and no argument that grants
it one.

Defended twice, independently: `validate_namespace` refuses a name whose root
is not `ego` or `id`, and `create_version` refuses a top-level namespace with
no versions. The second is what stops a *known* root being conjured on an empty
library, where the first would let it through (I49).

**Versioning an existing root** is ordinary governance. Ego's and Id's doctrine
has to be able to change, and `ego@2` is a candidate like any other: validated,
evaluated, approved by the Operator, then selected. Id may propose one through
`id_propose_prompt` or `id_propose_profile`; Id may not approve it (I49b).

An earlier version of this library forbade both, having conflated "no new
top-level namespace" with "no new version of a root". The effect was that
changing Ego's doctrine required editing a shipped file. Negating either
guarantee now leaves the other standing, which is checked rather than assumed.

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

### Upgrading shipped doctrine

This has a consequence worth stating plainly, because it is the mechanism
working rather than a bug. When a release changes `ego.md` or `id.md`:

* **a fresh state directory** bootstraps the new text as `ego@1` — immediately
  usable, nothing to approve;
* **an existing state directory** keeps running the version it already
  selected and records the new text as a candidate. The organism does not
  change how it thinks because a package was upgraded.

To adopt it, approve the candidate the normal way:

```
operator_prompt_bootstrap_report()            # which files differ
operator_prompt_state(version_id=..., state="validated")
operator_prompt_state(version_id=..., state="proposed")
operator_prompt_state(version_id=..., state="production_approved")
operator_prompt_select(namespace="ego", version_id=...)
```

A role reborn after that selection gets the new doctrine. Until then it keeps
what it was bound to, which is I53 rather than a delay to work around.

Note that the *environment* is supplied either way: the Harness injects the
manifest regardless of which doctrine version is running. An un-upgraded Ego
still receives its capabilities; it just has not been told how to read the
declaration, which degrades gracefully rather than breaking.

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

## 10b. Profile, environment, turn input

A profile is one of three things that reach a mind, and the Prompt Library
governs exactly one of them:

```
PROFILE      who am I, how should I think     Prompt Library   bound at birth
ENVIRONMENT  what exists, what can I do now   Harness          rebuilt per turn
TURN INPUT   what should I think about now    the trigger      one turn
```

All three roles now work this way. A neuocyte always did — its tool block is
built by the Harness per work item and recorded as `tools_offered`. Ego and Id
had a profile and a turn input and nothing in between, which meant a newly
approved `ego.neuocyte.research` was invisible to Ego unless somebody rewrote
Ego's root prompt: constitutional doctrine was the only channel for
environmental fact.

`role_environment(role)` is the Harness verb that builds it. It is **not**
`system_pulse`: the pulse is Id's live physiological telemetry, the manifest is
the cognitive operating environment — what profiles exist, what this role may
invoke, and which resource identities produced the turn.

A root prompt teaches a role how to *read* the declaration. It never enumerates
what happens to exist today. See `ARCHITECTURE.md` I58–I63 for the guarantees.

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

**Per turn** (Ego and Id)
`role_environment` — the Harness builds, records and returns the turn's
authoritative environment. Not model-facing: the role asks for it on the
model's behalf and hands the answer over.

`prompt_diff` answers the question that actually matters after a cascade:
identical prompt and config digests mean two lineages produce the same
cognition despite different version numbers.

## 13. `id_propose_prompt`

A real Prompt Library candidate: `ego@N` becomes a proposed `ego@N+1` that goes
through validation, evaluation and Operator approval like any other version. Id
cannot approve, select or cascade it.

It is the role-shaped entry point; `id_propose_profile` is the general one and
reaches any namespace, roots included. Both go through the same store, so there
is one creation path rather than two governance schemes.

While the library forbade every root version, this verb could not do what its
name said: it recorded Id's wording as a note and told the Operator to go and
edit a file. That is fixed.

## 14. Residual limits

* **The manifest costs context every turn.** Ego's runs around 2.7 KB and Id's
  larger, injected into an accumulating context on every turn. That is a real
  cost, accepted because an environment that is sometimes stale is worse than
  one that is always paid for; context pressure is what homeostasis exists to
  handle. Compressing it to a digest-plus-delta would make reconstruction
  subtler and was not worth it before measuring.
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
