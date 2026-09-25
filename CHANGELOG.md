# Changelog

**1.0.0 is the root of this log.** The work before it is in the commit
history and stays there; it is not reconstructed here. Everything from this
release onward is recorded below, newest first.

Versions follow [semantic versioning](https://semver.org/). The external
client surface (`io_*`), the role capability tables and the on-disk schema are
the public contract; a breaking change to any of them is a major version.

---

## 1.0.0 — 2026-09-25

The first release that closes the loop it was built for: a client asks
something the organism cannot answer from memory, Amoeba delegates it, a
sandboxed worker computes it, and the answer comes back — with the record
saying who observed what, and when.

### What it is

Two persistent roles over a local llama.cpp model. **Ego** faces outward and
answers; **Id** faces inward and audits. Neither holds authority: the
**Harness** owns admission, scheduling, capabilities and every write, and the
roles reach it only through the verbs their scope names. Work is done by
**neuocytes** — disposable workers, forked from a published context, given a
sandbox when the work needs one and destroyed when it ends.

Everything durable is a hash-chained event, a receipt and content-addressed
storage. A restart reconstitutes from the record rather than from memory.

### How correctness is established

Every guarantee the system makes is written down as a numbered invariant in
`docs/ARCHITECTURE.md`, defended by tests named in the invariant itself, and
**mutation-verified**: `scripts/verify_invariants.py` breaks each guarantee on
purpose and fails if the named tests still pass. A mutant that survives is
reported `WEAK`; one that cannot be observed must be declared `masked` with a
reason; one whose anchor has drifted off its code is a hard failure, not a
skip. The suite and a full sweep both gate every commit.

That machinery exists because tests alone were not enough: an invariant here
once passed while asserting nothing, and it took a mutant to notice.

### Known limitations

- **Windows-first.** Sandbox isolation uses AppContainer; the filesystem
  hardening is NTFS ACL-based. Other platforms are not supported yet.
- **Bring your own runtime and weights.** A llama.cpp build and a GGUF model
  are configured, not vendored. `amoeba doctor` reports what is missing.
- **Single organism per state directory.** Concurrent supervisors against one
  state tree are refused, not merged.
- **Worker context freshness is tied to rejuvenation.** Neuocytes fork the
  most recently published Ego snapshot, which can lag what Ego currently
  knows.
- **The schema migration list is maintained by hand.** Adding a column to an
  existing table requires adding it to that list; tests build fresh databases
  and will not catch the omission.
