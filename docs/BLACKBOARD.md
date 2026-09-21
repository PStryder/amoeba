# The cognitive blackboard

A durable, receipted, queryable space where neuocytes talk to each other.

**It is communication, not Mind State.** A post is something a neuocyte *said*. A
memory item is something the organism *believes*. Nothing crosses that line
implicitly — promotion is a separate, receipted act by the Harness, and the
post keeps its own identity afterwards.

`test_board_posts_are_not_mind_state` asserts that two neuocytes posting
contradictory findings produce **zero** beliefs.

---

## Why every read is recorded

Two neuocytes reaching the same finding is either the most valuable signal the
swarm produces or the least, and which one depends on a fact that becomes
unrecoverable after the moment passes: *had the second one already read the
first?*

| | meaning |
|---|---|
| Neither had read the other | **independent replication** — two separate routes to the same answer |
| The later one had read the earlier | **socially propagated agreement** — one observation wearing two coats |
| Same author twice | not corroboration at all |

Counting the second case as corroboration is how a swarm talks itself into a
confident mistake on a single observation. So:

- `record_read` fires on **every** retrieval, with reader, timestamp and query.
- Every post snapshots **`informed_by`**: the exact set of posts its author had
  read before writing. The snapshot is immutable and never recomputed, because
  "what had this author seen by then" stops being answerable once more reads
  accumulate. `test_informed_by_snapshot_is_frozen_at_post_time` pins that.
- **`board_naive`** is the strongest form: the author had read nothing at all.

`independence(a, b)` then answers directly rather than inferring from wording,
and `corroboration(post)` splits support into `independent_support` and
`socially_informed_support`.

### Two decisions worth knowing about

**The ambiguous case resolves as influence.** Wall-clock granularity here is
~0.5 ms and consecutive readings are usually identical, so a read and the post
it informed routinely share a timestamp. The comparison is therefore `<=`, not
`<`. The bias is one-directional and deliberate: over-attributing influence
costs a true replication being called social agreement, while
under-attributing it would let an echo count as independent corroboration.

**Paging is by `seq`, not by timestamp.** For the same granularity reason, a
`since=<timestamp>` poll silently drops posts written in the same tick.
`board_posts.seq` is assigned under the writer's transaction lock and is
strictly increasing. `test_wall_clock_cursor_drops_colliding_posts_but_seq_does_not`
constructs the collision deterministically and shows both behaviours.

---

## Making independence possible

A neuocyte only sees the board if its work item allows it:

```
admit_work(..., board_access="none" | "read" | "read_write")
```

`none` produces a board-naive neuocyte **by construction** — it is shown an
explicit "you have deliberately not been shown what other neuocytes found" block
instead. That is what turns later agreement between two neuocytes into evidence
rather than an echo, and it is what makes an independent-replication experiment
possible at all.

---

## Data model

| Table | Holds |
|---|---|
| `board_posts` | `post_id`, `seq`, `thread_id`, author + kind + incarnation, `work_id`, `operation_id`, `post_type`, title, body, confidence, `snapshot_id`, `model_generation`, status, `supersedes`, **`informed_by`**, `read_count_before`, **`board_naive`** |
| `board_evidence` | event ids, blob digests, memory ids, artifact ids, notes |
| `board_relations` | `reply_to`, `challenges`, `supports`, `refines`, `duplicates`, `answers` |
| `board_reads` | post, reader, work id, timestamp, query — the influence record |

Post types: `finding`, `question`, `hypothesis`, `challenge`, `request`,
`answer`, `note`, `retraction`.

A correction supersedes rather than edits: the old post stays readable and
marked `superseded`.

---

## Harness verbs

`board_post`, `board_read`, `board_get_post`, `board_thread`, `board_relate`,
`board_set_status`, `board_independence`, `board_corroboration`, `board_stats`,
`board_promote_to_memory`.

`board_read(..., record=False)` exists for the Harness and audit paths, which
must inspect the board without contaminating any neuocyte's independence record.
It is never used on behalf of a neuocyte.

### Promotion

`board_promote_to_memory` creates a maintained belief citing the post, and
attaches the corroboration analysis to it:

- independent replications become **supporting** evidence;
- challenges become **opposing** evidence;
- socially informed support is also filed as **opposing** context, labelled
  "not independent evidence", so a later reader cannot mistake volume for
  corroboration.

---

## The board is not an external surface

**No board verb is reachable over MCP or the HTTP API.** An external client
holds `external_io` — eight `io_*` verbs, listed in `scopes.EXTERNAL_IO` —
and the blackboard is not among them. There is no `board_read` from outside,
no `board_post`, no `board_corroboration`.

An earlier design did expose three of them, on the reasoning that an external
frontier model is a cognitive peer. That is gone, along with the other
nineteen verbs the demotion removed, and this section used to still describe
it: a reader would have concluded an outside model can write to the swarm's
working surface. It cannot.

The board is an *internal* surface. Ego, Id and neuocytes reach it through
their own scopes; the operator reads it through the control plane. External
input arrives as input — `io_submit` — and whatever the organism then chooses
to post is its own act, made on its own authority. See
[INTERFACES](INTERFACES.md) and invariant I48.


## What became of the attempt behind a post

Every post read from the board carries the fate of the **attempt** that wrote
it: `attempt_fate`, `attempt_unfinished`, the work item's `work_status` as
context, and a `work_note` when there is something to say.

The attempt, not the work item. A work item can fail attempt 1, requeue, and
complete on attempt 2 -- joining the post to `work_items.status` would render
the fenced attempt's finding as `done`, laundering a dead attempt's post
through somebody else's success. The author is identified by the fencing token
that was live when it posted, read from the work row by the Harness and never
supplied by the author.

A post from a dead attempt is **not** retracted and not hidden. A finding can
be sound even when the attempt that produced it did not finish, and "the
author's process crashed" is not "the finding was wrong". What it may not be
is invisible: a neuocyte has no `get_work` and no history, so without this a
dead attempt's finding read exactly like a completed one and corroboration
could accumulate around a dead end while every independence check still read
clean.

The recorded outcome travels verbatim rather than collapsed to "failed",
because a deadline, a fencing, a tool error and an exhausted budget mean
different things to a reader. Work that failed and was requeued reports as
still running, since a retry may yet corroborate the finding.

What a reader was shown is frozen in the read log alongside the state version,
for the same reason `informed_by` is frozen: a fate changes after the read.
The fate travels through `board_corroboration` -- reported against each
supporter, never weighted -- and into promotion, because a memory item
outlives the post and what is missing at promotion is missing from the belief.

### Attempts that said nothing

An attempt that died before posting leaves nothing to annotate. `board_read`
therefore also returns `silent_attempts`: a bounded count, with recorded
outcomes, of attempts on the same **recorded lineage** (a shared
`operation_id`) that ended without publishing. Lineage, never resemblance of
objective -- deciding what counts as "the same ground" is the reader's
thinking to do, and nothing is posted in anybody's name.

See invariant I95.
