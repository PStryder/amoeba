# Compute sandbox

> The compute sandbox is ephemeral execution scratch, never
> authoritative storage. See `STORES.md` for how it differs from
> Filespace, the blob store, and accepted artifacts.

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
  lands in the durable artifact store as `<artifact_id>_<basename>`, or in a
  filespace root the decider named.
- **It promotes the proposal blob, not the scratch file.** A proposal is
  content-addressed when it is made, so the reviewed digest names immutable
  bytes and those are what land. A scratch file edited after the proposal
  cannot influence the result; the divergence is recorded as a fact about the
  neuocyte.
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

## The other direction: protecting the sandbox from the host

Everything above is about what code inside the container can reach. The
opposite question — what the rest of the machine can do to the sandbox — was
open until filesystem hardening was added, and it mattered more than it looks.

The sandbox root is created under the install directory, whose parent
granted
`Everyone:(OI)(CI)(F)`. Directories inherit their parent's DACL, so *measured,
not assumed*, every Amoeba directory was world-writable. Three consequences:

| | What it meant |
|---|---|
| **Injection** | The runtime is a copied CPython tree. Any process could overwrite a stdlib `.py` file or DLL, and the next neuocyte run would execute it *inside* the container, with the container's rights. The AppContainer bounds what that code reaches; it says nothing about which code runs. |
| **Exfiltration** | Scratch contents — including whatever a neuocyte was asked to work on — were readable by any process on the machine. The containment boundary held the code and leaked the data. |
| **Tampering** | The event log, the content-addressed blobs and the promoted workspace were writable by any process. The hash chain detects that, but detection after the fact is not prevention. |

### What hardening does

Each Amoeba directory has inheritance removed and an explicit DACL:

| Principal | Gets |
|---|---|
| The account Amoeba runs as | full control |
| `SYSTEM` | full control |
| `Administrators` | full control |
| The AppContainer SID | full control on **its own** scratch; read+execute on the shared runtime |
| Everyone / Users / Authenticated Users | **nothing** |

Two details are load-bearing and both were learned by getting them wrong:

* It is **one** `icacls` invocation. Splitting it into `/inheritance:r` then
  `/grant` leaves a window in which a failure strips the DACL and locks out the
  owner. That is not hypothetical — it happened during development and
  recovering needed `/reset /T`.
* It is applied to the **directory only**, never with `/T`. `(OI)(CI)` are
  *inheritance* flags and convey nothing on a file, so applying the grant to
  children produces files with an ACE that grants no access while `icacls`
  reports success. Children re-inherit from the hardened directory instead, and
  the permissive ACEs vanish with the inheritance that carried them.

Hardening is verified immediately after it is applied. If the Amoeba account
can no longer use the directory, inheritance is restored — fail safe, not fail
locked.

### The runtime is shared; scratch is not

Scratch dies with its sandbox, so code a neuocyte writes there cannot outlive
it. The runtime is shared by every sandbox, which makes it the only
*persistent* injection path available from inside: a neuocyte that could append
to a stdlib module would be running that code in every future sandbox,
including ones created for unrelated work. The container therefore gets
read+execute, never write, and that grant is revoked when the container is
destroyed — otherwise the runtime DACL would accumulate one ACE per sandbox
ever created, and a retired container's SID would keep access. AppContainer
SIDs derive from the container name, so a stale grant is a grant to whoever
next claims that name.

This is asserted by attempting the write *from inside the container* and
requiring the kernel to refuse it, rather than by reading back the ACL we just
wrote.

### Residual risk, stated plainly

**A process running as the same account can undo all of this.** The Amoeba user
owns these directories, and on Windows an object's owner implicitly holds
`WRITE_DAC`: it can rewrite the ACL and restore its own access. Hardening
therefore stops *other* accounts, and stops casual or accidental access from
this one — a backup agent, a search indexer, a careless script, another tool.
It is not a boundary against a determined process running as you.

The fix for that threat is a dedicated low-privilege service account for
Amoeba, which is a deployment decision rather than something the code can
impose on its own host. Until then `audit_paths` reports the exposure — it is
surfaced in `id_health` under `filesystem` — so the gap stays visible instead
of being quietly assumed away.

## What this is not

This is a containment boundary against a **mistaken or over-eager model**,
hardened by an OS mechanism that also resists deliberate attempts to leave.
It is not a claim that the boundary is unbreakable against a determined
attacker with a Windows kernel exploit. If the threat model ever becomes
adversarial code rather than an erring model, the honest answer is a VM, not a
stronger ACL.
