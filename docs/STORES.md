# Where bytes live

Four stores, four lifetimes, four owners. Conflating any two is how a "safe to
destroy" claim quietly becomes false, so each has one name and that name is
used everywhere.

| Name | Where | Lifetime | Written by | Authoritative? |
|---|---|---|---|---|
| **Filespace** | configured host roots | yours; outlives Amoeba | the Harness only | yes — it's your data |
| **Blob store** | `state_dir/blobs/ab/cd/<sha256>.blob` | durable, content-addressed | the Harness only | as *evidence*, and as the promotable copy of a proposal |
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
               file_attach                    propose            promote
  Filespace ──────────────────► compute ─────────────────► blob ──────────► accepted
            (content-addressed   sandbox   (content-addressed  store          artifact
             on the way in)                 at proposal time)
```

Both ends are Harness acts. Note where promotion reads from: the **blob**, not
the sandbox. The sandbox is where the file was made; it is never on the path
from a decision to a durable artifact. Code inside it has no verb that reaches
any other store, and no argument in which to name one.

## Destroying a sandbox

Teardown is not a verdict. It removes the laboratory and decides nothing:

| | after destruction |
|---|---|
| scratch copy | **gone** |
| proposal record | **still pending, still promotable** |
| proposal bytes | **preserved**, and are what promotion uses |
| accepted artifact | **only if someone promotes it** |

A proposal's bytes are content-addressed when it is made, so the compute
sandbox is where the file was *made*, not where the promotable copy lives.
Destroying scratch removes a copy, not *the* copy.

An earlier version *lapsed* proposals on destruction, on the reasoning that
scratch held the only promotable copy. That stopped being true the moment
proposals became durable evidence, and keeping it would have coupled a decision
to an unrelated lifetime — a sandbox dying is not an opinion about whether an
artifact is worth keeping. Only `artifact_promote` and `artifact_reject`
decide.

`EventKind.ARTIFACT_LAPSED` is retired: nothing emits it, and the constant
stays only so historical events name a known kind.

Storage note: proposing durably stores up to `MAX_PROMOTED_BYTES` per proposal,
including proposals nobody ever decides on. There is deliberately no
sandbox-driven expiry. If pending proposals need pruning it should be its own
age- or quota-based policy, not a side effect of teardown.

## Lifecycle of a work product

```
1. code in the compute sandbox writes  work/checker.py     (scratch, disposable)
2. propose_artifact                     → status "proposed"
                                        → bytes content-addressed  (evidence,
                                           and the promotable copy)
3a. artifact_promote(root=…, path=…)    → status "promoted"
                                        → the *blob* is materialised at the
                                          destination the decider named
3b. artifact_reject                     → status "rejected"   (evidence kept)
3c. sandbox destroyed, no decision      → status unchanged: still "proposed",
                                          still promotable, indefinitely
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
digest, the accepted artifact is on disk, and an undecided proposal is still
`proposed` and still promotable from its blob.

See `tests/test_store_boundaries.py`; invariants I42, I42b, I42c and I43 in
`ARCHITECTURE.md`, each mutation-verified.
