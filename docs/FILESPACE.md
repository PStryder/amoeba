# Host files

Amoeba can produce files you actually use, and can be handed files to work on.
Both go through the Harness, and both are bounded by an allowlist.

## Configuration

No roots are configured by default, and with none configured Amoeba has no host
filesystem access at all. Roots are added explicitly:

```toml
[[filespace.roots]]
name = "out"
path = "F:/hexylab/amoeba-out"
mode = "read_write"
description = "where Amoeba puts things it produces"

[[filespace.roots]]
name = "pcdc"
path = "F:/hexylab/pcdc"
mode = "read_only"
```

A path that does not resolve inside one of these is **refused** — not
sanitised, not clamped into the nearest root. There is no default destination,
so a bug that loses the root name produces an error rather than a write
somewhere arbitrary.

## Verbs

| Verb | Who | What |
|---|---|---|
| `file_roots` | anyone | what Amoeba may touch, and how |
| `file_list` / `file_read` | Harness | read inside a root |
| `file_write` / `file_delete` | Harness | write inside a read-write root |
| `file_versions` / `file_restore` | Harness | list and recover earlier versions |
| `file_attach` | you | hand a named file to a work item |
| `artifact_promote(root=, path=)` | Harness | turn a proposal into a real file |

All are exposed over MCP as `mind_file_*` and `mind_artifact_promote`.

## No write destroys

Before any overwrite, delete, or promotion-over, the existing bytes are
content-addressed into the blob store and the digest goes into the event log as
`file.superseded`. Every version stays recoverable:

```
mind_file_versions(root="out", path="report.md")
  → [{sha256: "a1b2…", kind: "file.written",    restorable: true},
     {sha256: "9f0e…", kind: "file.superseded", restorable: true}]

mind_file_restore(root="out", path="report.md", sha256="9f0e…")
```

Restoring is itself a write, so whatever it replaces is snapshotted too — undo
is not a way to lose the current version.

This is what makes it safe to let a mind write to disk: a neuocyte's work can
be wrong, and being wrong supersedes rather than destroys.

## What a neuocyte can and cannot do

A neuocyte has **no verb that reaches the host filesystem**. Its tools address
its own sandbox, and nothing in its vocabulary names a filespace root. Files
leave a sandbox one way:

```
neuocyte: propose_artifact(path="work/checker.py", rationale="…")
          ↓ (a proposal; nothing is copied)
you:      mind_artifact_promote(artifact_id=…, root="out", path="checker.py")
```

The promoter names the destination. The neuocyte that produced the bytes never
sees it, and the content is re-hashed at the moment of copying — if it changed
since it was proposed, the promotion is refused outright.

Input is symmetric: nothing is read that was not named. `file_attach` takes an
explicit path, requires it to be inside a root, content-addresses it on the way
in, and copies it into that work item's sandbox. Because the bytes are stored
by digest, a finding about a file can later be checked against exactly the
content that produced it.

## Path resolution

The caller is ultimately a language model, and Windows has more ways to name a
path than POSIX. Each of these is refused explicitly:

| Trap | Why it matters |
|---|---|
| `..` traversal | the obvious one |
| absolute paths, drive letters, `\\server\share` | ignore the root entirely |
| `\\?\` long-path prefix | bypasses normalisation Win32 would otherwise do |
| alternate data streams (`notes.txt:hidden`) | writes content no listing shows |
| device names (`CON`, `NUL`, `COM1`) | open a device, not a file |
| trailing dots and spaces | Windows strips them, so `x.txt.` and `x.txt` are one file that compares as two names |
| symlinks and junctions | a link inside the root can point anywhere |

Whitespace is deliberately *not* stripped from a path. Stripping would turn
`notes.txt ` into `notes.txt` silently — resolving the ambiguity instead of
refusing it, which is the opposite of the job.

Links are the interesting case, because the string looks fine: a junction
inside a root is an ordinary-looking name that resolves elsewhere. Paths are
resolved fully and re-checked for containment, listings skip links rather than
walking through them, and writes refuse to go through one at all.

## Residual risk, stated plainly

**Resolution and the subsequent open are not atomic.** A link swapped in
between the two would be followed. Closing that properly needs `O_NOFOLLOW`
semantics Windows does not offer through `pathlib`. It requires an attacker
already running as this account and racing a specific operation — and the same
account can rewrite the roots' ACLs anyway (see `SANDBOX.md`), so this is not
the weakest link, but it is a gap rather than a guarantee.

**Configured roots are not hardened.** Unlike Amoeba's own state tree, these
are your directories. Locking down a directory you work in would be a
surprising side effect of pointing Amoeba at it, so `harden_state_tree` leaves
them alone.

**A read-write root is genuinely writable.** Amoeba can create, overwrite and
delete files there. The protection is that every version is recoverable, not
that changes are prevented. Point a read-write root at a directory you are
willing to see change; use `read_only` for anything you are not.
