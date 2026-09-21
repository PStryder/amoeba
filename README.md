# 🦠 Amoeba

> **An agent swarm serving as the substrate for persistent cognition.**  
> It's not as woo-woo as it sounds, I promise.

Amoeba is an experimental architecture for building a persistent cognitive system out of **replaceable, concurrent model processes**.

The central idea is simple:

> **Persistence is the mind. Models are things the mind uses to think.**

Most agent systems treat the model or agent process as the primary unit and attach memory, tools, and orchestration around it.

Amoeba turns that relationship around.

The durable thing is the accumulated **state, history, knowledge, goals, artifacts, commitments, provenance, and organization** of the system. Model processes operate on that persistent substrate, contribute useful work to it, and can then disappear.

Kill every neuocyte.

Start new ones.

The work should continue.

---

## Why "Amoeba"?

Because this is the simple organism.

Amoeba is intended to begin as a relatively primitive persistent cognitive architecture and become more capable through better models, better tools, accumulated knowledge, improved internal organization, and eventually mechanisms for proposing and evaluating improvements to itself.

The disposable cognitive neuocytes inside an Amoeba are called **neuocytes**.

An Amoeba may eventually be only one organism in a larger cognitive ecology.

But first we have to get the amoeba to stop failing `pytest`.

---

# The Basic Idea

A conventional agent architecture often looks roughly like this:

```text
Model
  │
  ├── Memory
  ├── Tools
  └── Agent Loop
```

Amoeba instead treats models as replaceable cognitive resources:

```text
                ┌─────────────────────┐
                │   Persistent State  │
                │                     │
                │ knowledge           │
                │ history             │
                │ goals               │
                │ artifacts           │
                │ commitments         │
                │ provenance          │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
              Ego                    Id
                │                     │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                │  Neuocyte Swarm     │
                │                     │
                │  ○ ○ ○ ○ ○ ○ ○ ○   │
                │   ○ ○ ○ ○ ○ ○ ○    │
                └─────────────────────┘
```

Neuocytes are disposable.

The state they produce is not.

---

# Neuocytes

A **neuocyte** is an independent cognitive neuocyte.

Neuocytes may share the same underlying model weights while maintaining independent working contexts.

They can:

- investigate questions
- test hypotheses
- critique other work
- replicate results
- synthesize findings
- inspect persistent state
- produce artifacts
- discover unresolved work
- contribute new claims and evidence
- retire when their useful context is exhausted

Neuocytes are deliberately mortal.

A neuocyte accumulating too much context may become stale, biased by its own history, or simply inefficient. Rather than requiring a single agent to maintain indefinite continuity, Amoeba externalizes useful cognition into persistent state and replaces the neuocyte.

A neuocyte's death should therefore be boring.

Before retirement, useful results, unresolved questions, artifacts, and provenance are externalized.

The process disappears.

The organism continues.

---

# Ego and Id

Amoeba contains two persistent cognitive roles with different responsibilities.

## Ego

**Ego is the outward-facing cognitive boundary of the organism.**

It maintains conversational continuity, interprets external requests, creates goals, synthesizes internal work, and produces responses.

Ego is not the central executive of the swarm.

It does not micromanage individual neuocytes.

It represents the organism to the outside world.

## Id

**Id is the inward-facing cognitive boundary of the organism.**

It observes the health of the cognitive system itself.

Id may monitor things such as:

- contradictions
- unresolved dependencies
- stale knowledge
- repeated failures
- poor evidence
- orphaned work
- pathological consensus
- memory fragmentation
- queue starvation
- neuocyte utilization
- context utilization

Id can create internal maintenance work even when no user is asking the system a question.

Ego looks outward.

Id looks inward.

Neither owns reality.

---

# Persistent State

The authoritative continuity of an Amoeba does not live inside any single model context.

Persistent state records the things the organism needs to continue functioning across neuocyte replacement, process restarts, and eventually model upgrades.

This may include:

```text
Goals
Claims
Evidence
Questions
Hypotheses
Experiments
Results
Contradictions
Decisions
Commitments
Artifacts
Capabilities
External Action Requests
```

The exact schema will evolve.

The principle should not:

> **Executing processes are replaceable. Persistent cognitive state is authoritative.**

---

# Event History and Receipts

Amoeba maintains an append-only history of significant events.

Examples include:

- external input
- model output
- neuocyte creation
- neuocyte retirement
- work admission
- work leasing
- state mutation
- tool request
- tool execution
- artifact creation
- snapshot publication
- process restart

The event history answers:

> **What happened?**

Persistent state answers:

> **What does the organism currently need to know?**

These are intentionally different things.

Amoeba should not confuse an infinite transcript with memory.

Raw history can be consolidated into higher-order state while remaining available as evidence.

The governing rule is:

> **No receipt, never happened.**

Important claims and state transitions should be traceable back through their provenance.

---

# Shared and Private Cognitive State

Neuocytes need both common inherited context and private working context.

Conceptually:

```text
Shared cognitive prefix
          │
     ┌────┼────┐
     ▼    ▼    ▼
    PKV  PKV  PKV
     A    B    C
```

Each neuocyte begins from a common cognitive foundation and then develops an independent private context.

Where supported by the inference backend, Amoeba can exploit physical KV-cache prefix sharing rather than reproducing identical context for every neuocyte.

This distinction matters.

Logical sharing is useful.

Physical sharing can make large concurrent populations practical.

Backend capabilities are therefore measured rather than assumed.

---

# Concurrency

Amoeba is specifically interested in **concurrent cognition**.

There is an important difference between:

1. serial inference
2. continuous batching of independent sequences
3. genuinely overlapping independent inference execution

These are not treated as equivalent.

Multiple neuocytes should be able to investigate different aspects of a problem while other cognition is still underway.

A result published by one neuocyte may alter the work of another neuocyte that has not finished yet.

That interaction is part of the experiment.

The repository therefore includes benchmarks intended to characterize the actual behavior of inference backends rather than infer concurrency from API semantics.

---

# Models Are Backends

Amoeba is not intended to depend permanently on one model.

Different cognitive roles may eventually use different inference resources:

```text
Ego        -> frontier reasoning model
Id         -> inexpensive local model
Neuocytes  -> highly concurrent local model
Vision     -> multimodal model
Specialist -> task-specific model
```

Model selection is cognitive policy.

Model execution is a resource.

A future deployment may use local inference, remote APIs, specialized models, or other Amoebas as cognitive resources.

The persistent organism should survive replacement of the models it uses to think.

---

# The Harness

Amoeba is designed around a strict separation between **cognition** and **authority**.

The model does not get hands.

Models may request actions.

They do not directly perform them.

```text
Model
  │
  │ proposes
  ▼
Tool Request
  │
  ▼
Harness
  │
  ├── validate schema
  ├── check permissions
  ├── enforce resource limits
  ├── execute permitted capability
  └── record receipt
```

Model-generated strings never become host commands.

This leads to one of the project's central security principles:

> **If an action must never happen, do not rely on the model choosing not to do it. Remove the action from its physics.**

An Amoeba can reason about an action without possessing the authority to perform it.

---

# The Universal Amoeba Harness

The longer-term architecture extends this boundary to entire Amoebas.

A **Universal Amoeba Harness** could host multiple isolated cognitive organisms.

Each Amoeba may have its own:

- persistent state
- event history
- Ego and Id
- neuocyte population
- artifacts
- tools
- model requirements
- compute requirements
- resource policies
- databases and indexes
- execution environment
- potentially even a VM or container image

An Amoeba becomes a portable cognitive unit.

```text
                     Universal Harness

        ┌────────────────────────────────────┐
        │                                    │
        │   ┌──────────┐    ┌──────────┐     │
        │   │ Amoeba A │    │ Amoeba B │     │
        │   │  Memory  │    │  Vision  │     │
        │   └────┬─────┘    └────┬─────┘     │
        │        │               │           │
        │        └──── Harness ──┘           │
        │                                    │
        │   ┌──────────┐    ┌──────────┐     │
        │   │ Amoeba C │    │ Amoeba D │     │
        │   │ Research │    │ General  │     │
        │   └──────────┘    └──────────┘     │
        │                                    │
        └────────────────────────────────────┘
```

Amoebas do not require privileged access to one another.

Communication occurs through the Harness.

Conceptually:

```text
A.Ego -> Harness -> B.Ego
```

not:

```text
A.neuocyte -> B.database
```

This preserves organism boundaries while allowing cognitive specialization.

---

# Capability Discovery

Amoebas should ultimately advertise **capabilities**, not implementations.

For example, a persistent-memory Amoeba might provide:

```text
memory.recall
memory.search
memory.associate
memory.store
memory.audit
```

A vision Amoeba might provide:

```text
vision.describe
vision.extract_text
vision.locate
vision.compare
```

The caller should not need to know whether that capability is implemented using:

- one model
- fifty neuocytes
- deterministic code
- RAG
- a vector database
- a specialized neural model
- another nested Amoeba

This gives the architecture an important recursive property:

> **A Neusomatica does not need to know whether another cognitive resource is atomic or composite.**

---

# MCP

An Amoeba can expose Ego through the **Model Context Protocol (MCP)**.

This allows an external model or agent to use Amoeba as persistent cognitive infrastructure.

For example:

```text
Frontier Model
      │
      │ MCP
      ▼
    Amoeba
      │
      ├── persistent memory
      ├── accumulated research
      ├── unresolved goals
      ├── internal swarm
      └── cognitive machinery
```

The frontier model may be temporary.

Amoeba remains.

This allows Amoeba to operate in two interesting modes:

### Cognitive Interface

The Amoeba itself is the entity being interacted with.

### Cognitive Substrate

An external model uses Amoeba for persistent cognition while acting as the immediate conversational intelligence.

The executing model can change without necessarily discarding the accumulated cognitive system.

---

# Amoeba Owns Itself. The Harness Owns Reality.

This is the fundamental containment boundary.

An Amoeba may own:

- its identity
- its memory
- its history
- its internal organization
- its cognitive policies
- its learned strategies
- its artifacts
- its capability definitions

The Harness owns:

- process creation
- compute allocation
- GPU access
- filesystem boundaries
- network access
- external APIs
- credentials
- inter-Amoeba routing
- resource limits
- external side effects

An Amoeba can ask.

The Harness decides whether the requested action exists in that Amoeba's universe.

This allows cognitive capability to increase without automatically increasing external authority.

---

# Evolution

One long-term research goal is to determine whether persistent cognitive organization can improve even when the underlying model remains unchanged.

An Amoeba may eventually be able to propose:

- new internal tools
- new cognitive strategies
- better work decomposition
- specialized neuocyte populations
- new retrieval systems
- improved memory structures
- deterministic services
- learned adapters
- new compositions of existing capabilities

Candidate improvements can then be evaluated before becoming persistent machinery.

The intended loop is roughly:

```text
variation
    ↓
evaluation
    ↓
selection
    ↓
persistence
    ↓
new variation
```

The Harness itself remains outside this evolutionary loop.

The organism may evolve.

**Physics does not evolve merely because the organism asks it to.**

---

# A Useful Experiment

Suppose we run an Amoeba for six months without changing its base model.

Does it become more capable anyway?

Then replace its model with a substantially better one while preserving its accumulated state and organization.

How much improvement comes from the model?

How much comes from the accumulated cognitive system?

Then compare both against a fresh Amoeba instantiated directly on the newer model.

In other words:

> **Which is more capable: a smarter newborn or a less intelligent organism with accumulated cognitive culture?**

That is one of the questions this project exists to explore.

---

# Current Status

Amoeba is experimental software under active development.

The current implementation is focused on establishing the boring but essential machinery first:

- durable state
- append-only event history
- receipts and provenance
- neuocyte lifecycle
- work scheduling and leasing
- Ego and Id separation
- inference backend abstraction
- deterministic testing
- GPU inference
- independent inference sequences
- shared-prefix behavior
- concurrency characterization
- process restart and recovery
- constrained tool execution
- MCP integration

The architecture deliberately attempts to establish these invariants before adding increasingly autonomous behavior.

### Where that machinery lives

| Document | Contents |
|---|---|
| [Implementation & quickstart](docs/IMPLEMENTATION.md) | capability status table, setup, running, tests |
| [Architecture](docs/ARCHITECTURE.md) | process topology, 26 named invariants, data model, recovery |
| [Runtime findings](docs/RUNTIME.md) | what was measured on real hardware, and what was not |
| [Benchmarks](docs/BENCHMARKS.md) | concurrency curve, prefix-sharing evidence, raw data in `bench/out/` |
| [Blackboard](docs/BLACKBOARD.md) | how neuocytes collaborate, and how independent replication is told from echo |
| [Sandbox](docs/SANDBOX.md) | OS-enforced scratch compute, and exactly what it can still reach |
| [Persistent turns](docs/TURNS.md) | the role mailbox, wake semantics, continuation, and why the scheduler is not a neuocyte |
| [Prompt library](docs/PROMPTLIB.md) | the versioned cognitive family tree: namespaces, pinning, inheritance, governance, cascade |
| [Homeostasis](docs/HOMEOSTASIS.md) | keeping contexts healthy; Id requests, the Harness performs |
| [MCP contract](docs/MCP_CONTRACT.md) | the cognitive verbs, and what of MCP is *not* implemented |
| [Open questions](docs/OPEN_QUESTIONS.md) | unresolved design questions and known failure modes |

The code uses this vocabulary throughout: the Python package is `amoeba`, the
disposable cognitive workers are `neuocyte` (`src/amoeba/neuocyte.py`,
`max_neuocytes`, `role="neuocyte"`). `work_id` / `work_items` / `workspace`
refer to *work*, not to neuocytes, and are deliberately unchanged.

Two results worth pulling forward, because they constrain the design:

- **Physical prefix sharing is real and measured.** With one unified KV stream,
  a neuocyte forked from an Ego snapshot shares KV cells rather than copying
  them: 900-token prefix across 4 sequences occupied 2036 of 2048 cells, where
  copying would have needed 4736. The contrasting mode copies, and is measured
  too. See [RUNTIME §2](docs/RUNTIME.md#2-ego-snapshots-shared-prefix-vs-copied-prefix).
- **Concurrent cognition has a knee at ~32 neuocytes** on this hardware. Past
  it each added neuocyte costs 5.4x more decode time for a quarter of the
  throughput return. Separately, neuocytes that merely *exist* tax every other
  decode by up to 1.94x, fully recovered on retirement — which makes retirement
  a throughput mechanism, not hygiene. See
  [BENCHMARKS §1-2](docs/BENCHMARKS.md#1-the-scaling-curve-where-batching-stops-paying).

- **Neuocytes have somewhere to compute without hands on the host.** Scratch
  workspaces are Windows AppContainers with zero capabilities: network blocked
  in the kernel, user profile and project source unreadable, stdlib-only
  interpreter, Job Object resource caps. Artifacts leave only by proposal, and
  the Harness names the destination and re-hashes on arrival. The one thing
  still readable is world-readable `C:\Windows`, which an AppContainer needs
  to start; that is stated rather than glossed. See [SANDBOX](docs/SANDBOX.md).
- **Agreement on the blackboard is only evidence when it is independent.**
  Every read is recorded and every post snapshots what its author had already
  seen, so `board_corroboration` can split support into independent replication
  and socially propagated echo. Work can be admitted `board_access="none"` to
  produce a board-naive neuocyte by construction. See
  [BLACKBOARD](docs/BLACKBOARD.md).

Genuinely overlapping independent inference execution — item 3 in the
concurrency list above — is **not** achieved and is not claimed anywhere in
this repository.

---

# Acceptance Test

The simplest test of the architecture is brutal:

1. Give the system meaningful work.
2. Allow it to accumulate state and unfinished goals.
3. Kill every disposable neuocyte.
4. Spawn a fresh population.
5. Verify that useful work continues.

The stronger version:

1. Kill Ego.
2. Kill Id.
3. Kill every neuocyte.
4. Preserve only durable cognitive state, history, artifacts, and persisted cognitive snapshots.
5. Restart the entire organism.
6. Verify that goals, commitments, knowledge, and unfinished work resume coherently.

If that succeeds, no executing process was individually responsible for the system's continuity.

Or, less formally:

> **Kill the colony. Keep the civilization.**

---

# Neusomatica

Amoeba is also serving as the first reference implementation for a broader architectural idea currently called **Neusomatica**.

Neusomatica explores persistent cognition in which:

- executing cognitive components are replaceable
- durable state provides continuity
- cognition is separated from external authority
- cognitive resources are discoverable through capabilities
- organisms can be composed recursively
- implementations remain hidden behind stable cognitive interfaces

A future cognitive ecology might therefore contain specialized Amoebas for:

- conversational memory
- research
- software engineering
- visual perception
- image generation
- mathematics
- simulation
- verification
- planning
- domain-specific expertise

Those organisms may themselves contain swarms, deterministic machinery, specialized models, or other cognitive organisms.

The topology does not need to be fixed in advance.

---

# What Amoeba Is Not

Amoeba is not a claim that current language models are conscious.

It is not an attempt to prove artificial consciousness.

It is not an unrestricted autonomous agent.

It is not intended to give language models arbitrary access to the host operating system or network.

It is not an AGI announcement.

It is an engineering experiment investigating a narrower question:

> **Can persistent cognitive organization exist independently of the particular processes and models performing cognition at any given moment?**

Let's find out.

---

## License

TBD

---

**Amoeba**  
*The organism is persistent. The cells are disposable.*
