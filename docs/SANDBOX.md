# Sandboxed compute workspace

Somewhere for the swarm to turn fuzzy cognition into deterministic machinery,
without giving it hands on the host.

**The boundary is a Windows AppContainer** — an OS-enforced security boundary,
not a policy check inside Python that the sandboxed code could bypass. It works
without administrator rights, which is why it is usable here at all.

---

## What was measured

Every row below is asserted by `tests/test_sandbox.py` (29 tests), which tries
to break the boundary rather than trusting this table.

| Attempt from inside | Result |
|---|---|
| Connect to the internet (`1.1.1.1:53`) | **blocked** |
| Connect to the LAN (`192.168.1.1:80`) | **blocked** |
| Connect to host loopback | **blocked** |
| Resolve DNS | **blocked** |
| List `C:\Users` | **blocked** (`PermissionError`) |
| Read the project source (`config.toml`) | **blocked** |
| Read the state database | **blocked** |
| Write into the user profile | **blocked** |
| Import `numpy` / `requests` / `torch` / `mcp` | **not found** |
| Read another sandbox's files | **blocked** |
| Read world-readable `C:\Windows` system files | **ALLOWED** — see caveat |
| Write inside its own scratch | allowed |
| Spawn a child process | allowed |
| Compute | allowed |

### The caveat, stated plainly

An AppContainer must read system DLLs to start at all, so Windows grants
`ALL APPLICATION PACKAGES` read access to parts of `C:\Windows`. Nothing
user-specific, nothing project-specific and no credential is reachable through
it — but **"no host filesystem access whatsoever" would be a false claim**, and
it is not made. `test_windows_system_files_remain_readable_and_this_is_documented`
pins the actual behaviour so this document cannot drift away from it, and the
caveat is returned in `sandbox_capabilities()` rather than living only in prose.

---

## How the boundary is built

Four layers, all enforced outside the sandboxed process:

1. **AppContainer with zero capabilities.** No `internetClient`, which is what
   makes network calls fail in the kernel rather than in a wrapper. Each
   sandbox gets its own container profile and SID.
2. **Filesystem ACLs.** The container SID is granted full control of its own
   scratch directory and read+execute on the runtime — and nothing else.
3. **A stdlib-only Python runtime.** Built once into a tree the Harness owns,
   deliberately excluding `site-packages`. No third-party library is reachable
   by construction, so there is nothing to sandbox in the first place. (The
   system Python could not be used: its install directory cannot be ACL'd
   without admin, and shipping `site-packages` would hand the sandbox every
   installed library including a CUDA-capable torch.)
4. **A Job Object.** Caps active processes, committed memory and CPU time, with
   `KILL_ON_JOB_CLOSE` so nothing outlives the sandbox. A fork bomb is bounded;
   a wall-clock overrun is terminated.

## Paths

The caller is ultimately a language model, so every path is resolved strictly
inside the scratch root. Absolute paths, drive letters, UNC paths and `..`
traversal are **rejected rather than normalised** — a path trying to escape is
a signal, not a typo.

---

## Promotion: the only way out

A neuocyte **proposes**; the Harness **disposes**.

```
artifact_propose(sandbox_id, path, rationale, proposed_by)
    -> records an intention. Copies nothing. Grants nothing.

artifact_promote(artifact_id, decided_by)
    -> the Harness copies, re-hashes and receipts.

artifact_reject(artifact_id, reason)
    -> also receipted; a refusal is a fact about how the mind governed itself.
```

Three things the Harness does that the proposer cannot influence:

- **It names the destination.** The caller never supplies a host path; the file
  lands in the durable workspace as `<artifact_id>_<basename>`.
- **It re-hashes on arrival.** The receipt reports `sha256_at_proposal`,
  `sha256_on_arrival` and `content_changed_since_proposal`, so a file edited
  between proposal and decision is visible rather than silently promoted.
- **It enforces a type allowlist** and a size cap.

The bytes also land in the content-addressed blob store, so a promoted artifact
is referenced by digest from the event log.

---

## Limits

Configured under `[sandbox]`:

| Setting | Default |
|---|---|
| `wall_seconds` | 60 |
| `cpu_seconds` | 60 |
| `memory_bytes` | 1 GiB |
| `max_processes` | 8 |
| `max_output_bytes` | 256 KiB |
| `max_scratch_bytes` | 256 MiB |
| `max_artifact_bytes` | 16 MiB |
| `max_concurrent` | 4 |

---

## Not exposed over MCP

Sandbox creation and execution are **not** MCP verbs. The facade exposes
cognitive verbs; requesting scratch compute is a resource request made by a
neuocyte through the Harness. An external client can see the consequences
(`id_health` reports sandbox capabilities; artifacts appear in the event log)
but cannot ask the mind to run code on its behalf.

## What this is not

This is a containment boundary against a **mistaken or over-eager model**,
hardened by an OS mechanism that also resists deliberate attempts to leave.
It is not a claim that the boundary is unbreakable against a determined
attacker with a Windows kernel exploit. If the threat model ever becomes
adversarial code rather than an erring model, the honest answer is a VM, not a
stronger ACL.
