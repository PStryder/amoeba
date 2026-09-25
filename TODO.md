# Known defects and deferred work

Things that are wrong, or owed, and not yet fixed. Each entry says where it
is, how to see it, and what a fix has to do — enough that somebody who did not
find it can act on it.

---

## Confirmed defects

### 1. A rejected control token raises `TypeError`, not `RpcError`

**Where:** `src/amoeba/rpc.py:245`, in `RpcClient.connect()`.

```python
if not resp.get("ok"):
    raise RpcError("control handshake rejected", **(resp.get("error") or {}))
```

`RpcError` inherits `MindError.__init__(self, message: str, **details)`. The
server's error object always carries its own `message` (`rpc.py:79` sends
`{"code": "unauthorized", "message": "bad control token"}`), so spreading it
supplies `message` a second time:

```
TypeError: MindError.__init__() got multiple values for argument 'message'
```

**Why it matters.** This is the authentication path. Every rejected token
raises `TypeError`, so a caller that catches `RpcError` does not catch it, and
whoever is holding the wrong token is shown a Python argument-binding
complaint instead of "bad control token".

**The fix** is to match what `call()` already does forty lines below
(`rpc.py:304`): extract `message` rather than spreading it, spread only
`err["details"]`, and rename `code` to `remote_code` — `MindError.code` is a
class attribute and letting a remote value share the name is how the next one
of these starts.

```python
err = resp.get("error") or {}
raise RpcError(err.get("message", "control handshake rejected"),
               remote_code=err.get("code"), **(err.get("details") or {}))
```

**A test that bites:** connect with a bad token and assert
`pytest.raises(RpcError)`. It fails with `TypeError` before the fix. Nothing
currently exercises a rejected handshake end to end, which is why neither the
suite nor the sweep sees this.

Found from outside this repository, 2026-09-25, and reproduced here.

---

### 2. `stop_sequences` longer than eight are silently truncated

**Where:** `src/amoeba/promptlib/model.py:292`, in `validate_model_vars()`.

```python
out[name] = [str(v) for v in value][:8]
```

Twelve in, eight out, nothing said:

```
asked for 12 stop sequences
got back  8
discarded: ['STOP8', 'STOP9', 'STOP10', 'STOP11']
```

**Why it matters.** The same function's docstring states the principle this
breaks, four lines above the offending line:

> Unknown names are refused rather than dropped: a profile that silently
> discarded `repetition_penalty` would look like it configured something.

`docs/ARCHITECTURE.md:794` records it as a guarantee. A profile declaring
twelve stop sequences is a profile claiming to have shaped generation in a way
it did not — the same failure the surrounding code exists to refuse, and the
same shape as I139 one layer down: a claim with nothing under it.

**The fix** is to refuse rather than truncate, and to name the limit in the
refusal — `8` is currently an undocumented magic number, and a caller who is
only told "too many" learns the bound by being refused twice.

**Safe to change here.** `stop_sequences` is the only `string_list` variable,
nothing tracked sets one, and `docs/PROMPTLIB.md:406` notes it is resolved and
recorded but not yet passed to the backend. Lists of eight or fewer are
unaffected, so no existing profile changes behaviour.

**A test that bites:** `validate_model_vars({"stop_sequences": [...9 items]})`
and assert it raises. Nothing currently passes more than eight.

Found from outside this repository, 2026-09-25, and reproduced here.

---

## Deferred

Judged worth doing, not yet done.

- **The migration list is maintained by hand.** `Database._ADDED_COLUMNS`
  carries every column added to a table that already existed. I84 proves the
  migration *runs*; nothing proves the list is *complete*. Adding
  `blocks_answer` without adding it there passed the whole suite — tests build
  fresh databases — and then refused to start against a real one. A per-table
  column digest, checked in, would force a decision whenever the schema
  changes.
- **Worker context is as fresh as the last rejuvenation.** Neuocytes fork the
  most recently published Ego snapshot, which is published on rejuvenation and
  nothing else. At low context occupancy that can be hours stale, so a worker
  reasons from what Ego knew rather than what it knows.
- **A disagreement between findings is not recorded as one.** Two board posts
  can carry contradictory numbers with no relation between them, so nothing
  reaches `disagreements` and Id is never asked to adjudicate. Seen live:
  41,675,000,250 against a measured 41,679,167,500, both standing.
- **No linter is configured**, despite `# noqa` markers throughout. A one-shot
  `pyflakes` pass on 2026-09-25 found two real defects — a `NameError` on a
  live dispatch path and a closure over an `except ... as` name.
- **Per-spawn neuocyte credential.** A worker currently uses the shared
  neuocyte scope token rather than one minted for its own lifetime.
