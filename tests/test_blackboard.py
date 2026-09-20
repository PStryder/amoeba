"""The blackboard is communication; Mind State is belief. And agreement is only
evidence when it is independent.
"""

from __future__ import annotations

import pytest

from synthetic_mind.errors import InvalidInput, NotFound
from synthetic_mind.store.events import EventKind


def post(mind, author, body, **kw):
    kw.setdefault("author_kind", "worker")
    kw.setdefault("post_type", "finding")
    return mind.board.post(author=author, body=body, **kw)[0]


# ---------------------------------------------------------------------------
# Posting, provenance, receipts
# ---------------------------------------------------------------------------
def test_post_carries_author_time_work_type_and_receipt(mind):
    pid, receipt = mind.board.post(
        author="wk_1", author_kind="worker", post_type="finding",
        title="KV pool fills", body="Occupancy above 70% halves decode rate.",
        work_id="work_abc", confidence=0.8,
        evidence=[{"event_id": "ev_1", "note": "bench run"}],
    )
    assert receipt.outcome == "committed"
    p = mind.board.get_post(pid)
    assert p["author"] == "wk_1" and p["author_kind"] == "worker"
    assert p["post_type"] == "finding" and p["work_id"] == "work_abc"
    assert p["confidence"] == 0.8 and p["created_at"] > 0
    assert p["evidence"][0]["event_id"] == "ev_1"
    assert p["thread_id"] == pid          # a new post opens its own thread
    assert p["status"] == "open"


def test_posting_is_recorded_in_history(mind):
    pid = post(mind, "wk_1", "something")
    rows = mind.db.conn.execute(
        "SELECT payload_inline FROM events WHERE kind = ?", (EventKind.BOARD_POSTED,)
    ).fetchall()
    assert len(rows) == 1 and pid in rows[0]["payload_inline"]


def test_invalid_post_type_and_relation_rejected(mind):
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="worker", post_type="gossip", body="x")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="alien", post_type="note", body="x")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="worker", post_type="note", body="  ")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="worker", post_type="note", body="x",
                        confidence=5.0)


def test_replies_and_relations_are_navigable(mind):
    a = post(mind, "wk_1", "the cache is the bottleneck")
    b = mind.board.post(author="wk_2", author_kind="worker", post_type="challenge",
                        body="measured the opposite", thread_id=a,
                        relations=[{"to_post": a, "relation": "challenges"}])[0]
    pa, pb = mind.board.get_post(a), mind.board.get_post(b)
    assert pb["relations"] == [{"to_post": a, "relation": "challenges"}]
    assert pa["replies"] == [{"from_post": b, "relation": "challenges"}]
    assert [p["post_id"] for p in mind.board.thread(a)] == [a, b]


def test_supersede_marks_the_old_post_without_deleting_it(mind):
    a = post(mind, "wk_1", "first answer")
    b = mind.board.post(author="wk_1", author_kind="worker", post_type="finding",
                        body="corrected answer", supersedes=a)[0]
    assert mind.board.get_post(a)["status"] == "superseded"
    assert mind.board.get_post(a)["body"] == "first answer"   # still readable
    assert mind.board.get_post(b)["supersedes"] == a


# ---------------------------------------------------------------------------
# The board is NOT Mind State
# ---------------------------------------------------------------------------
def test_posting_creates_no_belief(mind):
    post(mind, "wk_1", "the port is 9090")
    post(mind, "wk_2", "the port is 8080")
    assert mind.memory.recall(scope="all") == []
    assert mind.board.stats()["posts"] == 2


def test_board_stats_say_what_the_board_is(mind):
    post(mind, "wk_1", "x")
    assert "communication, not maintained memory" in mind.board.stats()["note"]


# ---------------------------------------------------------------------------
# Read tracking -- the reason the board exists in this form
# ---------------------------------------------------------------------------
def test_reading_is_recorded(mind):
    a = post(mind, "wk_1", "finding one")
    assert mind.board.posts_read_by("wk_2") == []
    mind.board.read(reader="wk_2", limit=10)
    assert mind.board.posts_read_by("wk_2") == [a]


def test_harness_can_read_without_contaminating_the_record(mind):
    a = post(mind, "wk_1", "finding one")
    mind.board.read(reader="supervisor", limit=10, record=False)
    assert mind.board.posts_read_by("supervisor") == []
    assert a  # still there


def test_post_snapshots_what_its_author_had_read(mind):
    a = post(mind, "wk_1", "alpha")
    b = post(mind, "wk_2", "beta")
    # wk_3 reads both, then posts.
    mind.board.read(reader="wk_3", limit=10)
    c = post(mind, "wk_3", "gamma")
    pc = mind.board.get_post(c)
    assert sorted(pc["informed_by"]) == sorted([a, b])
    assert pc["read_count_before"] == 2
    assert pc["board_naive"] is False


def test_a_naive_author_is_flagged(mind):
    post(mind, "wk_1", "alpha")
    b = post(mind, "wk_2", "beta")          # wk_2 never read anything
    pb = mind.board.get_post(b)
    assert pb["board_naive"] is True
    assert pb["informed_by"] == []


def test_informed_by_snapshot_is_frozen_at_post_time(mind):
    a = post(mind, "wk_1", "alpha")
    b = post(mind, "wk_2", "beta")          # naive
    # wk_2 reads afterwards; the earlier post must not retroactively change.
    mind.board.read(reader="wk_2", limit=10)
    assert mind.board.get_post(b)["informed_by"] == []
    assert mind.board.get_post(b)["board_naive"] is True
    assert a in mind.board.posts_read_by("wk_2")


# ---------------------------------------------------------------------------
# Independent replication vs socially propagated agreement
# ---------------------------------------------------------------------------
def test_two_naive_workers_agreeing_is_independent_replication(mind):
    a = post(mind, "wk_1", "the knee is at 32 sessions")
    b = post(mind, "wk_2", "the knee is at 32 sessions")
    r = mind.board.independence(a, b)
    assert r["verdict"] == "independent"
    assert r["later_author_had_read_earlier"] is False
    assert "replication" in r["explanation"]


def test_agreement_after_reading_is_not_independent(mind):
    a = post(mind, "wk_1", "the knee is at 32 sessions")
    mind.board.read(reader="wk_2", limit=10)            # wk_2 sees it first
    b = post(mind, "wk_2", "agreed, the knee is at 32")
    r = mind.board.independence(a, b)
    assert r["verdict"] == "socially_informed"
    assert r["later_author_had_read_earlier"] is True
    assert "not independent evidence" in r["explanation"]


def test_same_author_agreeing_with_itself_is_not_corroboration(mind):
    a = post(mind, "wk_1", "claim")
    b = post(mind, "wk_1", "claim again")
    r = mind.board.independence(a, b)
    assert r["verdict"] == "same_author"
    assert "not corroboration" in r["explanation"]


def test_independence_uses_the_read_log_when_no_snapshot_exists(mind):
    """Robustness: a post written before this bookkeeping existed."""
    a = post(mind, "wk_1", "alpha")
    mind.board.read(reader="wk_2", limit=10)
    b = post(mind, "wk_2", "beta")
    # Erase the snapshot to simulate a legacy row; the read log must still answer.
    mind.db.conn.execute(
        "UPDATE board_posts SET informed_by = '[]', board_naive = 1 WHERE post_id = ?",
        (b,))
    mind.db.conn.commit()
    assert mind.board.independence(a, b)["verdict"] == "socially_informed"


def test_corroboration_separates_real_support_from_echo(mind):
    claim = post(mind, "wk_1", "the fork shares KV cells")

    # wk_2 verifies without looking at the board: independent.
    indep = mind.board.post(
        author="wk_2", author_kind="worker", post_type="finding",
        body="measured cell occupancy; shared", thread_id=claim,
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    # wk_3 reads the board first, then agrees: an echo.
    mind.board.read(reader="wk_3", limit=10)
    echo = mind.board.post(
        author="wk_3", author_kind="worker", post_type="note",
        body="I concur with the above", thread_id=claim,
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    c = mind.board.corroboration(claim)
    assert set(c["supporting_posts"]) == {indep, echo}
    assert c["independent_support"] == [indep]
    assert c["socially_informed_support"] == [echo]
    assert c["independent_support_count"] == 1
    assert "one observation restated" in c["note"]


def test_corroboration_counts_challenges_too(mind):
    claim = post(mind, "wk_1", "claim")
    ch = mind.board.post(author="wk_2", author_kind="worker", post_type="challenge",
                         body="no", relations=[{"to_post": claim,
                                                "relation": "challenges"}])[0]
    assert mind.board.corroboration(claim)["challenges"] == [ch]


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
def test_filter_by_type_and_query_and_since(mind):
    q = mind.board.post(author="wk_1", author_kind="worker", post_type="question",
                        body="what limits concurrency?")[0]
    post(mind, "wk_1", "unrelated finding about caches")
    only_q = mind.board.read(reader="wk_9", post_types=["question"], record=False)
    assert [p["post_id"] for p in only_q] == [q]
    hits = mind.board.read(reader="wk_9", query="concurrency", record=False)
    assert [p["post_id"] for p in hits] == [q]
    cursor = mind.board.latest_seq()
    later = post(mind, "wk_1", "after the cutoff")
    fresh = mind.board.read(reader="wk_9", since_seq=cursor, record=False)
    assert [p["post_id"] for p in fresh] == [later]


def test_retracted_posts_are_hidden_from_reads_but_still_resolvable(mind):
    a = post(mind, "wk_1", "wrong thing")
    mind.board.set_status(post_id=a, status="retracted", actor="wk_1",
                          reason="measurement error")
    assert mind.board.read(reader="wk_2", record=False) == []
    assert mind.board.get_post(a)["status"] == "retracted"


def test_unknown_post_lookups_raise(mind):
    with pytest.raises(NotFound):
        mind.board.get_post("post_nope")
    with pytest.raises(NotFound):
        mind.board.relate(from_post="post_a", to_post="post_b", relation="supports",
                          actor="wk")


def test_seq_is_strictly_increasing_and_pages_without_loss(mind):
    cursor = mind.board.latest_seq()
    ids = [post(mind, f"wk_{i}", f"burst {i}") for i in range(12)]
    seqs = [r["seq"] for r in mind.db.conn.execute(
        "SELECT seq FROM board_posts ORDER BY seq")]
    assert seqs == sorted(set(seqs)), "seq must be strictly increasing and unique"
    fresh = mind.board.read(reader="wk_poll", since_seq=cursor, limit=100, record=False)
    assert {p["post_id"] for p in fresh} == set(ids)


def test_wall_clock_cursor_drops_colliding_posts_but_seq_does_not(mind):
    """Why the cursor is seq and not a timestamp.

    time.time() on this platform has ~0.5ms granularity and returns identical
    values on consecutive calls, so posts written in a burst can share a
    created_at. The collision is constructed here rather than raced for, so the
    test demonstrates the hazard deterministically instead of depending on
    timing luck.
    """
    a = post(mind, "wk_1", "first")
    b = post(mind, "wk_2", "second")
    stamp = mind.db.conn.execute(
        "SELECT created_at FROM board_posts WHERE post_id = ?", (a,)).fetchone()[0]
    mind.db.conn.execute("UPDATE board_posts SET created_at = ? WHERE post_id = ?",
                         (stamp, b))
    mind.db.conn.commit()

    # A wall-clock cursor taken at `a` silently loses `b`.
    by_time = mind.board.read(reader="wk_poll", since=stamp, record=False)
    assert b not in {p["post_id"] for p in by_time}

    # The seq cursor does not.
    seq_a = mind.db.conn.execute(
        "SELECT seq FROM board_posts WHERE post_id = ?", (a,)).fetchone()[0]
    by_seq = mind.board.read(reader="wk_poll", since_seq=seq_a, record=False)
    assert b in {p["post_id"] for p in by_seq}


def test_read_and_post_in_the_same_tick_still_counts_as_informed(mind):
    """The ambiguous case is resolved as influence, never as independence."""
    a = post(mind, "wk_1", "alpha")
    mind.board.read(reader="wk_2", limit=10)
    b = post(mind, "wk_2", "beta")          # likely the same millisecond
    pb = mind.board.get_post(b)
    assert a in pb["informed_by"], "a same-tick read must not be lost"
    assert pb["board_naive"] is False
    assert mind.board.independence(a, b)["verdict"] == "socially_informed"
