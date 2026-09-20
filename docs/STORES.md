# Where bytes live

Four stores, four lifetimes, four owners. Conflating any two is how a "safe to
destroy" claim quietly becomes false, so each has one name and that name is
used everywhere.

| Name | Where | Lifetime | Written by | Authoritative? |
|---|---|---|---|---|
| **Filespace** | configured host roots | yours; outlives Amoeba | the Harness only | yes — it's your data |
| **Blob store** | `state_dir/blobs/ab/cd/<sha256>.blob` | durable, content-addressed | the Harness only | as *evidence* |
| **Compute sandbox** | `state_dir/sandbox/<sandbox_id>/` | one work item, then destroyed | code running inside it | **no** |
| **Accepted artifact** | a filespace root, or `state_dir/artifacts/` | durable | the Harness, on promotion | yes |

A word that used to mean two of these: **"workspace"** described both the
ephemeral compute sandbox and the durable artifact store — opposite lifetimes
under one name. It is no longer used for either. `cfg.workspace_dir` is now
`cfg.artifact_dir` at `state_dir/artifacts`, and the sandbox is a *compute
sandbox*, never a workspace.

## The compute sandbox is a disposable laboratory

Not authoritative storage. Nothing that matters may exist only there.

```
                    file_attach                     propose → promote
  Filespace  ────────────────────────►  compute  ──────────────────────►  accepted
  blob store ◄──── content-addressed     sandbox   ◄──── re-hashed ────    artifact
                   on the way in                        on the way out
```

Both arrows are Harness acts. Code inside the sandbox has no verb that reaches
any other store, and no argument in which to name one.

## Destroying a sandbox

Four things are true at once afterwards, and all four are needed:

| | after destruction |
|---|---|
| scratch copy | **gone** |
| proposal record | **lapsed** / never accepted |
| proposal bytes | **preserved as evidence**, retrievable by digest |
| accepted artifact | **does not exist** |

A proposal's bytes are content-addressed the moment it is made, so the record
can say *both* things at once:

> "This artifact was never accepted."
> "Here are exactly the bytes the neuocyte proposed."

Keeping only the first leaves a rationale describing content nobody can ever
see. Keeping the row as `proposed` asserts it still awaits a decision when it
can never be promoted — a state that lies, which is worse than one that says it
lost. Recovering the evidence later is a fresh Harness act, not a promotion:
the record stays `lapsed`.

Storage note: proposing durably stores up to `MAX_PROMOTED_BYTES` per proposal.
That is the price of the evidence, and it is bounded by the tool-turn limit and
the artifact size cap.

## Lifecycle of a work product

```
1. code in the compute sandbox writes  work/checker.py     (scratch, disposable)
2. propose_artifact                     → status "proposed"
                                        → bytes content-addressed  (evidence)
3a. artifact_promote(root=…, path=…)    → status "promoted"
                                        → bytes re-hashed and copied out
                                        → now an accepted artifact
3b. artifact_reject                     → status "rejected"   (evidence kept)
3c. sandbox destroyed, no decision      → status "lapsed"     (evidence kept)
```

Only step 3a produces something authoritative, and only the Harness performs
it. The neuocyte that wrote the bytes never names the destination.

## What was verified, not merely asserted

Both invariants were measured **from inside the container**, because a check
run from outside tests the Harness's opinion of the boundary rather than the
boundary itself.

Sandboxed code attempting to read *and* write each store:

```
filespace root (dir)     read=DENIED  write=DENIED
filespace file           read=DENIED  write=DENIED
blob store               read=DENIED  write=DENIED
state database           read=DENIED  write=DENIED
artifact store           read=DENIED  write=DENIED
event log dir            read=DENIED  write=DENIED
other work item scratch  read=DENIED  write=DENIED
own scratch (control)    read=ok      write=ok
```

This holds even though the parent of the state tree grants `Everyone` full
control: an AppContainer token is not satisfied by `Everyone`, so the denial is
the container itself, not the ACLs. ACL hardening is also in place — see
`SANDBOX.md` — but it is the second line, not the first.

After destroying the sandbox: the filespace input is byte-identical, the hash
chain verifies with nothing missing, the attached input is still retrievable by
digest, the accepted artifact is on disk, and the abandoned proposal reads
`lapsed` with its bytes still recoverable.

See `tests/test_store_boundaries.py`; invariants I42, I42b, I42c and I43 in
`ARCHITECTURE.md`, each mutation-verified.
