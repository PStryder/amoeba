"""The cognitive blackboard: how neuocytes talk to each other.

**The board is communication, not Mind State.** A post is something a neuocyte
*said*. A memory item is something the organism *believes*. Nothing crosses
that line implicitly: promoting a post into maintained memory is a separate,
receipted act performed by the Harness, and the post keeps its own identity
afterwards.

## Why every read is recorded

Two neuocytes reaching the same finding is either the most valuable signal the
swarm produces or the least, and which one depends entirely on a fact that is
invisible after the fact: *had the second one already read the first?*

- Neither had read the other -> **independent replication**. Two separate
  routes to the same answer.
- The second had read the first -> **socially propagated agreement**. It may
  still be correct, but it is one observation wearing two coats, and counting
  it twice is how a swarm talks itself into a confident mistake.

So `record_read` is called on every retrieval, and every post snapshots
`informed_by`: the exact set of posts its author had read *before* it was
written. That snapshot is immutable. :meth:`independence` then answers the
question directly rather than guessing from timestamps later.

`board_naive` is the strongest form of the signal: the author had read nothing
at all from the board before posting.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Sequence

from ..errors import InvalidInput, NotFound
from ..ids import new_id
from .blobs import BlobStore
from .db import Database
from .events import EventKind
from .writer import Mutation, Receipt, StateWriter

POST_TYPES = (
    "finding", "question", "hypothesis", "challenge", "request", "answer",
    "note", "retraction",
)
RELATIONS = ("reply_to", "challenges", "supports", "refines", "duplicates", "answers")
AUTHOR_KINDS = ("ego", "id", "neuocyte", "operator")
POST_STATUSES = ("open", "resolved", "retracted", "superseded")


class BoardRepo:
    def __init__(self, db: Database, blobs: BlobStore, writer: StateWriter) -> None:
        self.db = db
        self.conn = db.conn
        self.blobs = blobs
        self.writer = writer

    # ------------------------------------------------------------------
    # reading (which is itself recorded)
    # ------------------------------------------------------------------
    def read(
        self,
        *,
        reader: str,
        thread_id: str | None = None,
        post_types: Sequence[str] | None = None,
        query: str | None = None,
        since_seq: int | None = None,
        since: float | None = None,
        limit: int = 20,
        work_id: str | None = None,
        record: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch posts and, by default, record that ``reader`` saw them.

        ``record=False`` exists for the Harness and for audit paths, which must
        be able to inspect the board without contaminating the independence
        record of any neuocyte. It is never used on behalf of a neuocyte.
        """
        clauses = ["status != 'retracted'"]
        params: list[Any] = []
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if post_types:
            bad = set(post_types) - set(POST_TYPES)
            if bad:
                raise InvalidInput("unknown post type", unknown=sorted(bad),
                                   allowed=list(POST_TYPES))
            clauses.append("post_type IN (%s)" % ",".join("?" * len(post_types)))
            params.extend(post_types)
        if since_seq is not None:
            clauses.append("seq > ?")
            params.append(int(since_seq))
        if since is not None:
            # Approximate by construction; see the seq column comment.
            clauses.append("created_at > ?")
            params.append(float(since))
        if query:
            clauses.append("(title LIKE ? OR body LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        params.append(int(limit))
        rows = self.conn.execute(
            "SELECT * FROM board_posts WHERE %s ORDER BY seq DESC LIMIT ?"
            % " AND ".join(clauses), params,
        ).fetchall()
        posts = [self._hydrate(r) for r in rows]
        if record and posts:
            self.record_read(reader=reader, post_ids=[p["post_id"] for p in posts],
                             work_id=work_id, query=query,
                             rendered={p["post_id"]: p for p in posts})
        return posts

    def get_post(self, post_id: str, *, reader: str | None = None,
                 work_id: str | None = None) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM board_posts WHERE post_id = ?", (post_id,)).fetchone()
        if row is None:
            raise NotFound("unknown board post", post_id=post_id)
        if reader:
            self.record_read(reader=reader, post_ids=[post_id], work_id=work_id)
        return self._hydrate(row)

    # Recorded facts, not verdicts: a finding from an attempt that died may
    # be perfectly good, and the reader is the one to decide.
    # The statuses that mean the attempt may still reach an end. Everything
    # else terminal is reported as unfinished, and `fenced` is derived from
    # the token rather than from any status, because a superseded attempt
    # leaves no mark on the work row at all.
    WORK_IN_FLIGHT = ("queued", "leased")

    def _work_provenance(self, work_id: str | None,
                         author_token: int | None = None) -> dict[str, Any]:
        """What became of the attempt that wrote this post.

        The attempt, not the work item. A work item can fail one attempt and
        complete on the next, and reporting the item's status would render a
        fenced attempt's finding as `done` -- laundering a dead attempt's post
        through a later attempt's success.

        Always present, including as nulls, because a missing key reads as
        "nothing to see here" and an absent provenance is exactly what made a
        dead attempt look finished.
        """
        out: dict[str, Any] = {
            "attempt_fate": None, "attempt_unfinished": None,
            "work_status": None, "work_note": None,
        }
        if not work_id:
            # An operator note or a role's own post belongs to no work item;
            # there is no fate to report rather than an unknown one.
            return out
        row = self.conn.execute(
            "SELECT status, attempt, fencing_token, failure FROM work_items"
            " WHERE work_id = ?", (work_id,)).fetchone()
        if row is None:
            out["work_note"] = "the work item this was posted against is gone"
            return out

        status = row["status"]
        out["work_status"] = status
        current = row["fencing_token"]

        # Token 0 means the author held no lease -- a role or the operator
        # posting against a work item rather than an attempt at it. That is
        # not a fenced attempt; it is not an attempt.
        if not author_token:
            out["work_note"] = (
                "this was not written by an attempt at the work, so there is "
                f"no attempt fate; the work item is {status!r}")
            return out

        if current is not None and author_token < current:
            # A later attempt superseded this author. Whatever became of the
            # work afterwards was not this attempt's doing, and its finding
            # was never corroborated by the attempt that made it finishing.
            out["attempt_fate"] = "fenced"
            out["attempt_unfinished"] = True
            out["work_note"] = (
                f"the attempt that wrote this was superseded (token "
                f"{author_token} < {current}); the work itself is now "
                f"{status!r}, which a later attempt achieved, not this one")
            return out

        if status == "done":
            out["attempt_fate"] = "completed"
            out["attempt_unfinished"] = False
        elif status in self.WORK_IN_FLIGHT:
            # A failure that requeued lands here: the finding may yet be
            # corroborated by a retry, and discounting it would be the
            # original error pointing the other way.
            out["attempt_fate"] = "running"
            out["attempt_unfinished"] = False
            out["work_note"] = out["work_note"] or (
                "the work that produced this post is still running")
        else:
            out["attempt_fate"] = status
            out["attempt_unfinished"] = True
            why = (row["failure"] or "").strip()
            out["work_note"] = (
                f"the attempt that wrote this {status}"
                + (f": {why[:200]}" if why else "")
                + "; the post stands because a finding can be sound even when "
                  "the attempt that made it did not finish, but nothing has "
                  "corroborated it")
        return out

    def _hydrate(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        item["informed_by"] = json.loads(item.get("informed_by") or "[]")
        item["board_naive"] = bool(item.get("board_naive", 0))
        item.update(self._work_provenance(item.get("work_id"),
                                          item.get("author_fencing_token")))
        item["evidence"] = [
            dict(r) for r in self.conn.execute(
                "SELECT event_id, event_seq, blob_sha256, memory_id, artifact_id, note"
                " FROM board_evidence WHERE post_id = ? ORDER BY id", (item["post_id"],))
        ]
        item["relations"] = [
            dict(r) for r in self.conn.execute(
                "SELECT to_post, relation FROM board_relations WHERE from_post = ?",
                (item["post_id"],))
        ]
        item["replies"] = [
            dict(r) for r in self.conn.execute(
                "SELECT from_post, relation FROM board_relations WHERE to_post = ?",
                (item["post_id"],))
        ]
        return item

    #: Most silent attempts reported in one read. A bound, not a judgement --
    #: the count is exact even when the list is truncated.
    MAX_SILENT_ATTEMPTS = 10

    def silent_attempts(self, work_id: str | None, *, limit: int | None = None
                        ) -> dict[str, Any]:
        """Attempts on the same lineage that ended without posting anything.

        Annotating posts only helps where there is a post. An attempt that
        died before publishing leaves no trace at all, so a neuocyte can
        re-run ground its siblings already died on and never know.

        Scoped by `operation_id` -- recorded lineage, the same identifier the
        turn machinery uses -- and never by resemblance of objective. A
        similarity judgement would be the Harness deciding what counts as the
        same ground, which is the neuocyte's thinking to do.

        Facts only: what died and what it recorded on the way out. No advice,
        no verdict, and nothing posted to the board in anybody's name.
        """
        out: dict[str, Any] = {"count": 0, "attempts": [], "truncated": False,
                               "operation_id": None}
        if not work_id:
            return out
        row = self.conn.execute(
            "SELECT operation_id FROM work_items WHERE work_id = ?",
            (work_id,)).fetchone()
        operation_id = row["operation_id"] if row else None
        if not operation_id:
            return out
        out["operation_id"] = operation_id

        cap = int(limit or self.MAX_SILENT_ATTEMPTS)
        base = (" FROM work_items w WHERE w.operation_id = ?"
                "   AND w.work_id != ?"
                "   AND w.status IN ('failed', 'cancelled')"
                "   AND NOT EXISTS (SELECT 1 FROM board_posts p"
                "                   WHERE p.work_id = w.work_id)")
        total = self.conn.execute(
            "SELECT COUNT(*)" + base, (operation_id, work_id)).fetchone()[0]
        rows = self.conn.execute(
            "SELECT w.work_id, w.status, w.attempt, w.failure, w.objective"
            + base + " ORDER BY w.updated_at DESC LIMIT ?",
            (operation_id, work_id, cap)).fetchall()

        out["count"] = int(total)
        out["truncated"] = int(total) > len(rows)
        out["attempts"] = [
            {"work_id": r["work_id"], "status": r["status"],
             "attempt": r["attempt"],
             "objective": (r["objective"] or "")[:200],
             # The recorded reason, verbatim. Deadline, fencing, a tool error
             # and an exhausted budget mean different things to somebody
             # judging whether to try again, and collapsing them to "failed"
             # throws that away.
             "recorded_outcome": (r["failure"] or "")[:300] or None}
            for r in rows]
        if out["count"]:
            out["note"] = (
                f"{out['count']} attempt(s) on this same operation ended "
                "without posting anything. Recorded so they are not invisible; "
                "what that means for your own attempt is yours to judge")
        return out

    def record_read(self, *, reader: str, post_ids: Sequence[str],
                    work_id: str | None = None, query: str | None = None,
                    rendered: dict[str, dict[str, Any]] | None = None) -> int:
        """Append read receipts. Cheap, frequent, and not version-bumping.

        Reads are evidence about influence, not state changes, so they do not
        take the writer's transaction path. They are still durable.

        `rendered` is the provenance as this reader was shown it. A fate
        changes after the read -- an attempt that was running when it was read
        can be fenced an hour later -- so recording only the post id would
        leave "what was this reader actually influenced by" unanswerable, in
        exactly the way `informed_by` exists to prevent.
        """
        if not post_ids:
            return 0
        now = time.time()
        shown = rendered or {}
        version = self.conn.execute(
            "SELECT version FROM state_version WHERE id = 1").fetchone()
        version = int(version["version"]) if version else None
        with self.writer.tx_lock:
            self.conn.executemany(
                "INSERT INTO board_reads(post_id, reader, work_id, read_at, query,"
                " attempt_fate_at_read, work_status_at_read, state_version_at_read)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(pid, reader, work_id, now, query,
                  (shown.get(pid) or {}).get("attempt_fate"),
                  (shown.get(pid) or {}).get("work_status"),
                  version)
                 for pid in post_ids],
            )
            self.conn.commit()
        return len(post_ids)

    def posts_read_by(self, reader: str, *, at_or_before: float | None = None
                      ) -> list[str]:
        """Posts this reader had seen by a given moment.

        The comparison is ``<=``, not ``<``, and that choice matters. Wall-clock
        granularity here is ~0.5ms and consecutive readings are usually
        identical, so a read and the post it informed routinely share a
        timestamp. With ``<`` those reads vanish and the post is wrongly marked
        board-naive.

        The bias is deliberate and one-directional: an ambiguous timestamp is
        resolved as *influence*. Over-attributing influence costs a true
        replication being called social agreement. Under-attributing it would
        let an echo be counted as independent corroboration, which is exactly
        how a swarm convinces itself of something on one observation.
        """
        sql = "SELECT DISTINCT post_id FROM board_reads WHERE reader = ?"
        params: list[Any] = [reader]
        if at_or_before is not None:
            sql += " AND read_at <= ?"
            params.append(at_or_before)
        return [r["post_id"] for r in self.conn.execute(sql, params)]

    # ------------------------------------------------------------------
    # posting
    # ------------------------------------------------------------------
    def post(
        self,
        *,
        author: str,
        author_kind: str,
        post_type: str,
        body: str,
        title: str | None = None,
        thread_id: str | None = None,
        work_id: str | None = None,
        operation_id: str | None = None,
        confidence: float | None = None,
        evidence: Iterable[dict[str, Any]] = (),
        relations: Iterable[dict[str, str]] = (),
        snapshot_id: str | None = None,
        model_generation: str | None = None,
        author_incarnation: int | None = None,
        supersedes: str | None = None,
        mutation_id: str | None = None,
    ) -> tuple[str, Receipt]:
        """Publish a post, snapshotting what its author had already read.

        The ``informed_by`` snapshot is taken here, at post time, from the
        durable read log. It is never recomputed later, because "what had this
        author seen by then" stops being answerable once more reads accumulate.
        """
        if post_type not in POST_TYPES:
            raise InvalidInput("unknown post type", post_type=post_type,
                               allowed=list(POST_TYPES))
        if author_kind not in AUTHOR_KINDS:
            raise InvalidInput("unknown author kind", author_kind=author_kind,
                               allowed=list(AUTHOR_KINDS))
        if not body or not body.strip():
            raise InvalidInput("post body must be non-empty")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise InvalidInput("confidence must be in [0, 1]", confidence=confidence)
        relations = list(relations)
        for rel in relations:
            if rel.get("relation") not in RELATIONS:
                raise InvalidInput("unknown relation", relation=rel.get("relation"),
                                   allowed=list(RELATIONS))

        post_id = new_id("post")
        thread = thread_id or post_id
        evidence = list(evidence)
        now = time.time()
        informed_by = sorted(self.posts_read_by(author, at_or_before=now))

        def body_fn(m: Mutation) -> None:
            row = m.sql("SELECT COALESCE(MAX(seq), 0) AS s FROM board_posts").fetchone()
            seq = int(row["s"]) + 1
            if supersedes:
                prev = m.sql("SELECT post_id FROM board_posts WHERE post_id = ?",
                             (supersedes,)).fetchone()
                if prev is None:
                    raise NotFound("post to supersede does not exist", post_id=supersedes)
                m.sql("UPDATE board_posts SET status = 'superseded' WHERE post_id = ?",
                      (supersedes,))
            # Which attempt is writing this, taken from the work row rather
            # than from the author. Nothing to forge, for the same reason the
            # fencing token works as a fence.
            attempt_token = attempt_no = None
            if work_id:
                wrow = m.sql("SELECT fencing_token, attempt FROM work_items"
                             " WHERE work_id = ?", (work_id,)).fetchone()
                if wrow is not None:
                    attempt_token = wrow["fencing_token"]
                    attempt_no = wrow["attempt"]
            m.sql(
                "INSERT INTO board_posts(post_id, seq, thread_id, author, author_kind,"
                " author_incarnation, author_fencing_token, author_attempt,"
                " work_id, operation_id, post_type, title, body,"
                " confidence, snapshot_id, model_generation, status, supersedes,"
                " informed_by, read_count_before, board_naive, created_at, state_version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (post_id, seq, thread, author, author_kind, author_incarnation,
                 attempt_token, attempt_no, work_id,
                 operation_id, post_type, title, body, confidence, snapshot_id,
                 model_generation, "open", supersedes, json.dumps(informed_by),
                 len(informed_by), 1 if not informed_by else 0, now, m.prior_version + 1),
            )
            for ev in evidence:
                m.sql(
                    "INSERT INTO board_evidence(post_id, event_id, event_seq, blob_sha256,"
                    " memory_id, artifact_id, note) VALUES (?,?,?,?,?,?,?)",
                    (post_id, ev.get("event_id"), ev.get("event_seq"),
                     ev.get("blob_sha256"), ev.get("memory_id"), ev.get("artifact_id"),
                     ev.get("note")),
                )
            for rel in relations:
                m.sql(
                    "INSERT OR IGNORE INTO board_relations(from_post, to_post, relation,"
                    " created_at) VALUES (?,?,?,?)",
                    (post_id, rel["to_post"], rel["relation"], now),
                )
            m.emit(EventKind.BOARD_POSTED, {
                "post_id": post_id, "seq": seq, "thread_id": thread, "author": author,
                "post_type": post_type, "title": title, "work_id": work_id,
                "confidence": confidence, "read_count_before": len(informed_by),
                "board_naive": not informed_by, "evidence_count": len(evidence),
                "relations": [r["relation"] for r in relations],
            })

        receipt, _ = self.writer.apply(
            body_fn, actor=author, operation_id=operation_id,
            mutation_id=mutation_id or f"board-post:{post_id}",
        )
        return post_id, receipt

    def relate(self, *, from_post: str, to_post: str, relation: str, actor: str
               ) -> Receipt:
        if relation not in RELATIONS:
            raise InvalidInput("unknown relation", relation=relation,
                               allowed=list(RELATIONS))

        def body(m: Mutation) -> None:
            for pid in (from_post, to_post):
                if m.sql("SELECT 1 FROM board_posts WHERE post_id = ?", (pid,)).fetchone() is None:
                    raise NotFound("unknown board post", post_id=pid)
            m.sql(
                "INSERT OR IGNORE INTO board_relations(from_post, to_post, relation,"
                " created_at) VALUES (?,?,?,?)", (from_post, to_post, relation, time.time()))
            m.emit(EventKind.BOARD_RELATED,
                   {"from_post": from_post, "to_post": to_post, "relation": relation})

        receipt, _ = self.writer.apply(
            body, actor=actor, bump_version=False,
            mutation_id=f"board-rel:{from_post}:{to_post}:{relation}")
        return receipt

    def set_status(self, *, post_id: str, status: str, actor: str,
                   reason: str | None = None) -> Receipt:
        if status not in POST_STATUSES:
            raise InvalidInput("unknown status", status=status,
                               allowed=list(POST_STATUSES))

        def body(m: Mutation) -> None:
            row = m.sql("SELECT status FROM board_posts WHERE post_id = ?",
                        (post_id,)).fetchone()
            if row is None:
                raise NotFound("unknown board post", post_id=post_id)
            m.sql("UPDATE board_posts SET status = ? WHERE post_id = ?", (status, post_id))
            m.emit(EventKind.BOARD_STATUS_CHANGED,
                   {"post_id": post_id, "from": row["status"], "to": status,
                    "reason": reason})

        receipt, _ = self.writer.apply(
            body, actor=actor, mutation_id=f"board-status:{post_id}:{status}",
            bump_version=False)
        return receipt

    # ------------------------------------------------------------------
    # the point of all the bookkeeping
    # ------------------------------------------------------------------
    def independence(self, post_a: str, post_b: str) -> dict[str, Any]:
        """Were these two posts arrived at independently?

        Answered from the immutable ``informed_by`` snapshots plus the read
        log, never inferred from wording similarity.
        """
        a = self.get_post(post_a)
        b = self.get_post(post_b)
        earlier, later = (a, b) if a["created_at"] <= b["created_at"] else (b, a)

        later_saw_earlier = earlier["post_id"] in later["informed_by"]
        # Belt and braces: the read log is the ground truth if a snapshot is
        # ever absent (e.g. a post written before this bookkeeping existed).
        if not later_saw_earlier:
            row = self.conn.execute(
                "SELECT 1 FROM board_reads WHERE post_id = ? AND reader = ?"
                " AND read_at <= ? LIMIT 1",
                (earlier["post_id"], later["author"], later["created_at"]),
            ).fetchone()
            later_saw_earlier = row is not None

        same_author = a["author"] == b["author"]
        if same_author:
            verdict = "same_author"
        elif later_saw_earlier:
            verdict = "socially_informed"
        else:
            verdict = "independent"

        return {
            "post_a": post_a, "post_b": post_b,
            "earlier_post": earlier["post_id"], "later_post": later["post_id"],
            "later_author_had_read_earlier": later_saw_earlier,
            "later_author_board_naive": later["board_naive"],
            "same_author": same_author,
            "verdict": verdict,
            "explanation": {
                "independent": "neither author had read the other's post; agreement "
                               "here is replication",
                "socially_informed": "the later author had already read the earlier "
                                     "post; agreement is not independent evidence",
                "same_author": "both posts have the same author; not corroboration "
                               "at all",
            }[verdict],
        }

    def corroboration(self, post_id: str) -> dict[str, Any]:
        """Who agrees with this post, and how much of that agreement is real?"""
        post = self.get_post(post_id)
        supporters = [
            dict(r) for r in self.conn.execute(
                "SELECT from_post FROM board_relations WHERE to_post = ?"
                " AND relation = 'supports'", (post_id,))
        ]
        challengers = [
            dict(r) for r in self.conn.execute(
                "SELECT from_post FROM board_relations WHERE to_post = ?"
                " AND relation = 'challenges'", (post_id,))
        ]
        independent, informed = [], []
        for s in supporters:
            verdict = self.independence(post_id, s["from_post"])["verdict"]
            (independent if verdict == "independent" else informed).append(s["from_post"])

        # Reported, never weighted. A supporter whose attempt was fenced is
        # not a second mind agreeing; it is a dead attempt's post still
        # sitting there. Whether that agreement counts is a judgement, and
        # judgements belong to the reader -- so no count is adjusted here and
        # no supporter is dropped.
        def _fate(pid: str) -> dict[str, Any]:
            row = self.conn.execute(
                "SELECT work_id, author_fencing_token FROM board_posts"
                " WHERE post_id = ?", (pid,)).fetchone()
            prov = self._work_provenance(
                row["work_id"] if row else None,
                row["author_fencing_token"] if row else None)
            return {"post_id": pid, "attempt_fate": prov["attempt_fate"],
                    "attempt_unfinished": prov["attempt_unfinished"],
                    "work_status": prov["work_status"]}

        support_provenance = [_fate(pid) for pid in independent + informed]
        unfinished_support = [f["post_id"] for f in support_provenance
                              if f["attempt_unfinished"]]
        return {
            "support_provenance": support_provenance,
            "unfinished_support": unfinished_support,
            "post_id": post_id,
            "author": post["author"],
            "supporting_posts": [s["from_post"] for s in supporters],
            "independent_support": independent,
            "socially_informed_support": informed,
            "challenges": [c["from_post"] for c in challengers],
            "independent_support_count": len(independent),
            "note": ("only independent_support counts as corroboration; "
                     "socially informed support is one observation restated"),
        }

    # ------------------------------------------------------------------
    def thread(self, thread_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM board_posts WHERE thread_id = ? ORDER BY seq ASC LIMIT ?",
            (thread_id, limit)).fetchall()
        return [self._hydrate(r) for r in rows]

    def latest_seq(self) -> int:
        """The cursor a neuocyte should remember to poll for new posts."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS s FROM board_posts").fetchone()
        return int(row["s"])

    def stats(self) -> dict[str, Any]:
        by_type = {r["post_type"]: r["n"] for r in self.conn.execute(
            "SELECT post_type, COUNT(*) AS n FROM board_posts GROUP BY post_type")}
        by_status = {r["status"]: r["n"] for r in self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM board_posts GROUP BY status")}
        naive = self.conn.execute(
            "SELECT COUNT(*) AS n FROM board_posts WHERE board_naive = 1").fetchone()["n"]
        total = sum(by_type.values())
        return {
            "posts": total, "by_type": by_type, "by_status": by_status,
            "threads": self.conn.execute(
                "SELECT COUNT(DISTINCT thread_id) AS n FROM board_posts").fetchone()["n"],
            "reads": self.conn.execute(
                "SELECT COUNT(*) AS n FROM board_reads").fetchone()["n"],
            "relations": self.conn.execute(
                "SELECT COUNT(*) AS n FROM board_relations").fetchone()["n"],
            "board_naive_posts": naive,
            "board_naive_fraction": (naive / total) if total else 0.0,
            "latest_seq": self.latest_seq(),
            "note": "the board is communication, not maintained memory",
        }
