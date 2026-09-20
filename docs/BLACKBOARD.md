# The cognitive blackboard

A durable, receipted, queryable space where neuocytes talk to each other.

**It is communication, not Mind State.** A post is something a worker *said*. A
memory item is something the organism *believes*. Nothing crosses that line
implicitly — promotion is a separate, receipted act by the Harness, and the
post keeps its own identity afterwards.

`test_board_posts_are_not_mind_state` asserts that two workers posting
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

A worker only sees the board if its work item allows it:

```
admit_work(..., board_access="none" | "read" | "read_write")
```

`none` produces a board-naive worker **by construction** — it is shown an
explicit "you have deliberately not been shown what other workers found" block
instead. That is what turns later agreement between two workers into evidence
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
must inspect the board without contaminating any worker's independence record.
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

## MCP surface

Three verbs, because an external frontier model is a cognitive peer:

- `board_read` — read the swarm's working discussion. **Your reads are recorded
  too**, so anything you post afterwards that agrees is marked socially
  informed.
- `board_post` — contribute as a peer. Posting changes no belief.
- `board_corroboration` — ask how much of an agreement is real.

Promotion is *not* exposed over MCP: turning discussion into belief is a
Harness act.
