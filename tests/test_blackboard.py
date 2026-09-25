"""The blackboard is communication; Mind State is belief. And agreement is only
evidence when it is independent.
"""

from __future__ import annotations

import pytest

from amoeba.errors import InvalidInput, NotFound
from amoeba.store.events import EventKind


def post(mind, author, body, **kw):
    kw.setdefault("author_kind", "neuocyte")
    kw.setdefault("post_type", "finding")
    return mind.board.post(author=author, body=body, **kw)[0]


def _observe(mind, actor, *, tool="board_get_post", arguments=None, result=None):
    """Record that `actor` acquired something, the way the Harness does.

    Corroboration rests on acquisitions rather than on who spoke (I138), so a
    test about corroboration has to make observations happen.
    """
    import json

    from amoeba.results import issue_result

    return issue_result(mind, json.dumps(result or {"seen": True}),
                        role=actor, tool=tool,
                        arguments=arguments if arguments is not None
                        else {"post_id": "p_source"}, actor="harness")


# ---------------------------------------------------------------------------
# Posting, provenance, receipts
# ---------------------------------------------------------------------------
def test_post_carries_author_time_work_type_and_receipt(mind):
    pid, receipt = mind.board.post(
        author="wk_1", author_kind="neuocyte", post_type="finding",
        title="KV pool fills", body="Occupancy above 70% halves decode rate.",
        work_id="work_abc", confidence=0.8,
        evidence=[{"event_id": "ev_1", "note": "bench run"}],
    )
    assert receipt.outcome == "committed"
    p = mind.board.get_post(pid)
    assert p["author"] == "wk_1" and p["author_kind"] == "neuocyte"
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
        mind.board.post(author="a", author_kind="neuocyte", post_type="gossip", body="x")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="alien", post_type="note", body="x")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="neuocyte", post_type="note", body="  ")
    with pytest.raises(InvalidInput):
        mind.board.post(author="a", author_kind="neuocyte", post_type="note", body="x",
                        confidence=5.0)


def test_a_relation_with_the_wrong_keys_is_not_called_an_unknown_relation(mind):
    """The refusal must name the mistake that was made, not a different one.

    Live on 2026-09-25 Ego passed `{"target": ..., "relation_type":
    "supports"}` and was told "unknown relation (allowed: ... supports ...)",
    because `.get("relation")` was None rather than because `supports` was
    wrong. Its value was correct and the refusal pointed straight at it, so it
    spent four attempts cycling through relation names it had never got wrong
    and gave up believing the board could not link a finding to its support.
    """
    claim = post(mind, "wk_1", "the cache is cold")
    with pytest.raises(InvalidInput) as exc:
        mind.board.post(author="wk_2", author_kind="neuocyte", post_type="finding",
                        body="agreed", thread_id=claim,
                        relations=[{"target": claim, "relation_type": "supports"}])

    said = str(exc.value)
    assert "unknown relation" not in said, (
        "the refusal blamed the value, which was never wrong")
    assert "'to_post'" in said and "'relation'" in said, (
        "the refusal did not name the keys that were missing")

    # What the role reads is the message plus the details, so the shape it
    # needs has to be in one of them.
    details = exc.value.details
    assert "supports" in str(details.get("allowed")), (
        "the refusal did not say what a relation may be")
    assert "to_post" in str(details.get("hint")), (
        "the refusal did not show the shape it wanted")

    # And the shape it names is one the board actually accepts.
    ok = mind.board.post(author="wk_2", author_kind="neuocyte", post_type="finding",
                         body="agreed", thread_id=claim,
                         relations=[{"to_post": claim, "relation": "supports"}])[0]
    assert ok


def test_relating_to_a_post_that_does_not_exist_says_so(mind):
    """Not "IntegrityError: FOREIGN KEY constraint failed".

    Live on 2026-09-25, one refusal after the keys were fixed, Ego linked its
    finding to a worker post that had not been written yet and was handed the
    raw constraint failure as the reason. An internal error standing in for
    the mistake the model made is the thing I127 exists to prevent, and
    `board_relate` already checked this -- posting did not.
    """
    with pytest.raises(NotFound) as exc:
        mind.board.post(author="wk_1", author_kind="neuocyte", post_type="finding",
                        body="linked to something that isn't there",
                        relations=[{"to_post": "post_does_not_exist",
                                    "relation": "supports"}])

    said = str(exc.value)
    assert "FOREIGN KEY" not in said and "IntegrityError" not in said, (
        "the model was shown a database failure instead of its mistake")
    assert "unknown board post" in said
    assert exc.value.details.get("post_id") == "post_does_not_exist", (
        "the refusal did not name the post that was missing")


def test_a_genuinely_unknown_relation_still_says_so(mind):
    """Control: the new branch must not swallow the mistake it replaced."""
    claim = post(mind, "wk_1", "the cache is cold")
    with pytest.raises(InvalidInput) as exc:
        mind.board.post(author="wk_2", author_kind="neuocyte", post_type="finding",
                        body="agreed", thread_id=claim,
                        relations=[{"to_post": claim, "relation": "vibes"}])
    assert "unknown relation" in str(exc.value)


def test_replies_and_relations_are_navigable(mind):
    a = post(mind, "wk_1", "the cache is the bottleneck")
    b = mind.board.post(author="wk_2", author_kind="neuocyte", post_type="challenge",
                        body="measured the opposite", thread_id=a,
                        relations=[{"to_post": a, "relation": "challenges"}])[0]
    pa, pb = mind.board.get_post(a), mind.board.get_post(b)
    assert pb["relations"] == [{"to_post": a, "relation": "challenges"}]
    assert pa["replies"] == [{"from_post": b, "relation": "challenges"}]
    assert [p["post_id"] for p in mind.board.thread(a)] == [a, b]


def test_supersede_marks_the_old_post_without_deleting_it(mind):
    a = post(mind, "wk_1", "first answer")
    b = mind.board.post(author="wk_1", author_kind="neuocyte", post_type="finding",
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
def test_two_reasoners_agreeing_from_the_same_evidence_is_not_corroboration(mind):
    """This used to assert the opposite, and it was the category error.

    Two authors, neither having read the other, neither having observed
    anything: the old rule called that "independent replication" because the
    speakers were different. Agreement here says something about the
    reasoning -- and nothing further about the evidence, which is the only
    thing corroboration is about (I138). Reported as `concurring_reasoning`
    so the signal is not lost, and kept out of `independent_support` so it
    cannot be mistaken for a second observation.
    """
    a = post(mind, "wk_1", "the knee is at 32 sessions")
    b = post(mind, "wk_2", "the knee is at 32 sessions")
    r = mind.board.independence(a, b)

    assert r["verdict"] == "shared_evidence"
    assert r["later_author_had_read_earlier"] is False
    assert r["distinct_roots"] == []
    assert "not a second observation" in r["explanation"]


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

    # wk_2 goes and measures it for itself: a distinct acquisition, and the
    # only thing here that is corroboration.
    _observe(mind, "wk_2", tool="run_code", arguments={"code": "measure()"},
             result={"occupancy": 2036})
    indep = mind.board.post(
        author="wk_2", author_kind="neuocyte", post_type="finding",
        body="measured cell occupancy; shared", thread_id=claim,
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    # wk_3 reads the board first, then agrees: an echo.
    mind.board.read(reader="wk_3", limit=10)
    echo = mind.board.post(
        author="wk_3", author_kind="neuocyte", post_type="note",
        body="I concur with the above", thread_id=claim,
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    c = mind.board.corroboration(claim)
    assert set(c["supporting_posts"]) == {indep, echo}
    assert c["independent_support"] == [indep]
    assert c["socially_informed_support"] == [echo]
    assert c["independent_support_count"] == 1
    assert "distinct acquisitions, not speakers" in c["note"]


def test_support_from_an_attempt_that_never_finished_is_flagged(mind):
    """Corroboration reports the fate of its supporters, not just their count.

    A supporter whose own attempt died is weaker evidence than one whose
    attempt completed, and a reader counting `independent_support_count`
    cannot tell them apart. `unfinished_support` is what says so, and nothing
    asserted it: the mutant that empties the list survived every board test.
    """
    claim = post(mind, "wk_1", "the fork shares KV cells")

    supporter_work = _admit(mind, objective="verify independently")
    lease = mind.work.lease(neuocyte_id="wk_2", work_id=supporter_work)
    supporting = mind.board.post(
        author="wk_2", author_kind="neuocyte", post_type="finding",
        body="measured it myself; shared", thread_id=claim,
        work_id=supporter_work,
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    # The attempt that wrote it then died without finishing.
    mind.work.fail(work_id=supporter_work, neuocyte_id="wk_2",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    c = mind.board.corroboration(claim)

    assert supporting in c["supporting_posts"]
    assert c["unfinished_support"] == [supporting], (
        "support from an attempt that never finished was reported as though "
        "its author had completed")
    fates = {f["post_id"]: f for f in c["support_provenance"]}
    assert fates[supporting]["attempt_unfinished"] is True


def test_corroboration_counts_challenges_too(mind):
    claim = post(mind, "wk_1", "claim")
    ch = mind.board.post(author="wk_2", author_kind="neuocyte", post_type="challenge",
                         body="no", relations=[{"to_post": claim,
                                                "relation": "challenges"}])[0]
    assert mind.board.corroboration(claim)["challenges"] == [ch]


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
def test_filter_by_type_and_query_and_since(mind):
    q = mind.board.post(author="wk_1", author_kind="neuocyte", post_type="question",
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


# ---------------------------------------------------------------------------
# A post carries what became of the work that produced it
# ---------------------------------------------------------------------------
def _posted_against(mind, *, work_id, body="the cache is cold on boot"):
    post_id, _ = mind.board.post(author="nc_1", author_kind="neuocyte",
                                 post_type="finding", body=body,
                                 work_id=work_id)
    return post_id


def _admit(mind, *, objective="look into the cache"):
    work_id, _ = mind.work.admit(objective=objective, work_class="user",
                                 origin_actor="ego")
    return work_id


def test_a_finding_from_failed_work_is_still_readable(mind):
    """It is not retracted and not hidden.

    A neuocyte can discover something true and then die for reasons that have
    nothing to do with the finding -- a deadline, a fencing token, a tool
    error. Dropping those posts would throw away real evidence, and "the
    author's process crashed" is not "the finding was wrong".
    """
    work_id = _admit(mind)
    post_id = _posted_against(mind, work_id=work_id)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    posts = mind.board.read(reader="nc_2", record=False)
    assert [p["post_id"] for p in posts] == [post_id], "the finding vanished"


def test_a_finding_from_failed_work_does_not_look_like_one_that_succeeded(mind):
    """The defect: indistinguishable from a completed attempt's finding.

    A neuocyte's window onto other work is almost entirely this board -- no
    `get_work`, no history, no provenance. So if the board does not say the
    work failed, nothing does, and corroboration accumulates around a dead end
    while every independence check still reads clean.
    """
    failed_work = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=failed_work)
    failed_post = _posted_against(mind, work_id=failed_work, body="from a failure")
    mind.work.fail(work_id=failed_work, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    good_work = _admit(mind, objective="something else")
    lease2 = mind.work.lease(neuocyte_id="nc_2", work_id=good_work)
    good_post = _posted_against(mind, work_id=good_work, body="from a success")
    mind.work.complete(work_id=good_work, neuocyte_id="nc_2",
                       fencing_token=lease2["fencing_token"], result={"ok": True})

    by_id = {p["post_id"]: p for p in mind.board.read(reader="nc_3", record=False)}
    bad, good = by_id[failed_post], by_id[good_post]

    assert bad["attempt_fate"] == "failed" and bad["attempt_unfinished"] is True
    assert "failed" in (bad["work_note"] or "")
    assert good["attempt_fate"] == "completed" and good["attempt_unfinished"] is False
    assert good["work_note"] is None, "a completed attempt needs no annotation"


def test_a_post_belonging_to_no_work_reports_no_fate(mind):
    """An operator note has no work item, which is not the same as unknown.

    The fields are present as nulls rather than absent, because a missing key
    reads as "nothing to see here" -- and an absent provenance is exactly what
    made a failed attempt look finished.
    """
    post_id, _ = mind.board.post(author="operator", author_kind="operator",
                                 post_type="note", body="a standing note")
    post = {p["post_id"]: p for p in mind.board.read(reader="nc_1", record=False)}[post_id]
    assert "attempt_fate" in post and post["attempt_fate"] is None
    assert post["work_status"] is None
    assert post["attempt_unfinished"] is None
    assert post["work_note"] is None


def test_work_still_running_is_reported_as_such(mind):
    """In flight is not the same as finished, and not the same as failed."""
    work_id = _admit(mind)
    mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    post_id = _posted_against(mind, work_id=work_id)

    post = {p["post_id"]: p for p in mind.board.read(reader="nc_2", record=False)}[post_id]
    assert post["work_status"] == "leased"
    assert post["attempt_fate"] == "running"
    assert post["attempt_unfinished"] is False, (
        "work that is still running was reported as unfinished; a reader "
        "would discount a finding that may yet be corroborated")
    assert "still running" in (post["work_note"] or "")


def test_cancelled_work_is_reported_like_failed_work(mind):
    """Cancelled is unfinished for the same reason failed is.

    Nothing corroborated the finding by the work reaching its end, and that is
    the fact a reader needs -- why it stopped is a separate question.
    """
    work_id = _admit(mind)
    mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    post_id = _posted_against(mind, work_id=work_id)
    mind.work.cancel(work_id=work_id, reason="superseded", actor="ego")

    post = {p["post_id"]: p for p in mind.board.read(reader="nc_2", record=False)}[post_id]
    assert post["work_status"] == "cancelled"
    assert post["attempt_fate"] == "cancelled"
    assert post["attempt_unfinished"] is True


def test_reading_a_thread_says_the_same_as_reading_a_query(mind):
    """One post, one answer, whichever route reached it.

    Two read paths deciding separately what a post says is two answers, and
    the disagreement would surface as a neuocyte trusting a finding its
    sibling discounted.
    """
    work_id = _admit(mind)
    post_id = _posted_against(mind, work_id=work_id)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    via_read = {p["post_id"]: p for p in mind.board.read(reader="x", record=False)}[post_id]
    via_get = mind.board.get_post(post_id)
    via_thread = {p["post_id"]: p
                  for p in mind.board.thread(via_read["thread_id"])}[post_id]

    for other in (via_get, via_thread):
        assert other["attempt_fate"] == via_read["attempt_fate"]
        assert other["attempt_unfinished"] == via_read["attempt_unfinished"]


def test_a_failure_that_will_be_retried_is_not_reported_as_unfinished(mind):
    """A requeued attempt is still in flight, and the distinction matters.

    `fail` requeues for another attempt rather than ending the work, so the
    finding may yet be corroborated by a retry that succeeds. Reporting it as
    unfinished would have a reader discount evidence that is still live --
    which is the same error as the original defect, pointing the other way.
    """
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    post_id = _posted_against(mind, work_id=work_id)
    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="first try")

    post = {p["post_id"]: p for p in mind.board.read(reader="nc_2", record=False)}[post_id]
    assert post["work_status"] == "queued", "the retry was not queued"
    assert post["attempt_fate"] == "running"
    assert post["attempt_unfinished"] is False
    assert "still running" in (post["work_note"] or "")


def test_a_later_attempt_s_success_does_not_launder_a_fenced_attempt_s_post(mind):
    """The case a join on `work_items.status` passes while the guarantee is absent.

    A work item can fail attempt 1, requeue, and complete on attempt 2. The
    post written by the fenced attempt 1 must not render as `done`: that
    attempt's finding was never corroborated by the attempt that made it
    finishing, and the success belongs to somebody else.

    This is the whole reason the fate reported is the *attempt's* and not the
    work item's, and it is why the token -- which the system already maintains
    to stop a superseded neuocyte committing -- is what identifies the author.
    """
    work_id = _admit(mind)
    lease1 = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    doomed = _posted_against(mind, work_id=work_id, body="from attempt 1")
    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease1["fencing_token"], failure="died")

    lease2 = mind.work.lease(neuocyte_id="nc_2", work_id=work_id)
    assert lease2["fencing_token"] > lease1["fencing_token"]
    survivor = _posted_against(mind, work_id=work_id, body="from attempt 2")
    mind.work.complete(work_id=work_id, neuocyte_id="nc_2",
                       fencing_token=lease2["fencing_token"], result={"ok": True})

    by_id = {p["post_id"]: p for p in mind.board.read(reader="nc_3", record=False)}
    dead, alive = by_id[doomed], by_id[survivor]

    assert dead["attempt_fate"] == "fenced", (
        "a fenced attempt's post was laundered through a later success: "
        f"{dead['attempt_fate']!r}")
    assert dead["attempt_unfinished"] is True
    assert "superseded" in (dead["work_note"] or "")
    # The work item's own status is still reported, because "this attempt was
    # fenced but the work later succeeded" is more use than either half alone.
    assert dead["work_status"] == "done"

    assert alive["attempt_fate"] == "completed"
    assert alive["attempt_unfinished"] is False


def test_the_fate_a_reader_was_shown_is_recorded(mind):
    """A fate changes after the read, so the read log must freeze what it showed.

    `informed_by` is snapshotted for the same reason: "what had this author
    seen by then" stops being answerable once the world moves on. A reader
    influenced by `attempt: running` must stay distinguishable from one
    influenced by `attempt: fenced`, and only the read log can say which.
    """
    work_id = _admit(mind)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    post_id = _posted_against(mind, work_id=work_id)

    # Read it while the attempt is alive.
    mind.board.read(reader="early_reader")

    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)
    mind.board.read(reader="late_reader")

    rows = {r["reader"]: r for r in mind.db.conn.execute(
        "SELECT reader, attempt_fate_at_read, work_status_at_read,"
        " state_version_at_read FROM board_reads WHERE post_id = ?", (post_id,))}

    assert rows["early_reader"]["attempt_fate_at_read"] == "running", (
        "the early reader's view was not recorded as it was shown")
    assert rows["late_reader"]["attempt_fate_at_read"] == "failed"
    assert rows["early_reader"]["state_version_at_read"] is not None


def test_corroboration_reports_a_dead_supporter_without_discounting_it(mind):
    """Reported, never weighted.

    A supporter whose attempt was fenced is not a second mind agreeing. But
    whether that agreement counts is a judgement, so the count is untouched
    and the supporter is not dropped -- only the fact travels.
    """
    claim_work = _admit(mind)
    mind.work.lease(neuocyte_id="nc_1", work_id=claim_work)
    claim = _posted_against(mind, work_id=claim_work, body="the claim")

    sup_work = _admit(mind, objective="replicate it")
    lease = mind.work.lease(neuocyte_id="nc_2", work_id=sup_work)
    support = _posted_against(mind, work_id=sup_work, body="I saw it too")
    mind.board.relate(from_post=support, to_post=claim, relation="supports",
                      actor="nc_2")
    mind.work.fail(work_id=sup_work, neuocyte_id="nc_2",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    corr = mind.board.corroboration(claim)
    assert support in (corr["independent_support"] + corr["socially_informed_support"]), (
        "the supporter was dropped; that is a verdict, not a fact")
    assert support in corr["unfinished_support"]
    fates = {f["post_id"]: f for f in corr["support_provenance"]}
    assert fates[support]["attempt_fate"] == "failed"
    assert fates[support]["attempt_unfinished"] is True


def test_attempts_that_died_before_posting_are_reported(mind):
    """The other half of the asymmetry: silence cannot be annotated.

    An attempt that failed before publishing leaves nothing to mark, so
    without this a neuocyte re-runs ground its siblings already died on and
    has no way to know. Scoped by recorded lineage -- the shared
    `operation_id` -- never by resemblance of objective, because deciding what
    counts as "the same ground" is the neuocyte's thinking to do.
    """
    op = "op-shared"
    dead, _ = mind.work.admit(objective="try the cold path", work_class="user",
                              origin_actor="ego", operation_id=op)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=dead)
    mind.work.fail(work_id=dead, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"],
                   failure="deadline_reached after 60s", requeue=False)

    mine, _ = mind.work.admit(objective="try the cold path again",
                              work_class="user", origin_actor="ego",
                              operation_id=op)

    out = mind.board.silent_attempts(mine)
    assert out["count"] == 1, out
    only = out["attempts"][0]
    assert only["work_id"] == dead and only["status"] == "failed"
    assert "deadline_reached" in (only["recorded_outcome"] or ""), (
        "the recorded reason was collapsed to 'failed'")


def test_an_attempt_that_posted_is_not_reported_as_silent(mind):
    """It left a trace, and that trace is annotated instead.

    Counting it here as well would report the same death twice, in two
    different vocabularies.
    """
    op = "op-two"
    spoke, _ = mind.work.admit(objective="noisy", work_class="user",
                               origin_actor="ego", operation_id=op)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=spoke)
    mind.board.post(author="nc_1", author_kind="neuocyte", post_type="finding",
                    body="I got this far", work_id=spoke)
    mind.work.fail(work_id=spoke, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    mine, _ = mind.work.admit(objective="again", work_class="user",
                              origin_actor="ego", operation_id=op)
    assert mind.board.silent_attempts(mine)["count"] == 0


def test_silence_on_another_lineage_is_not_reported(mind):
    """Lineage, not likeness.

    Two work items with near-identical objectives on different operations are
    different ground. Reporting across them would be the Harness judging
    topical similarity, which is exactly the heuristic it declines to make.
    """
    stranger, _ = mind.work.admit(objective="try the cold path", work_class="user",
                                  origin_actor="ego", operation_id="op-a")
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=stranger)
    mind.work.fail(work_id=stranger, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"], failure="died",
                   requeue=False)

    mine, _ = mind.work.admit(objective="try the cold path", work_class="user",
                              origin_actor="ego", operation_id="op-b")
    assert mind.board.silent_attempts(mine)["count"] == 0


# ===========================================================================
# Withdrawal, and disputes that end because the record moved
# ===========================================================================
def _concl(mind, claim="the cache is cold on boot", by="ego"):
    cid, _ = mind.memory.record_conclusion(claim=claim, produced_by=by)
    return cid


def _dispute(mind, conclusion_id, *, basis="basis-1"):
    did, _ = mind.memory.open_disagreement(
        subject_kind="conclusion", subject_id=conclusion_id,
        claim_a="the cache is cold on boot", actor_a="ego",
        claim_b="the evidence does not establish that", actor_b="id",
        evidence_basis_digest=basis)
    return did


def _status(mind, did):
    return mind.db.conn.execute(
        "SELECT status, resolution, recurrences FROM disagreements"
        " WHERE disagreement_id = ?", (did,)).fetchone()


def test_withdrawing_the_claim_ends_the_dispute_about_it(mind):
    """The subject may close a dispute by changing the record, not by assertion.

    Ego retracting its own conclusion is Ego changing its mind, and the
    dispute has nothing left to be about. That is categorically different
    from Ego declaring the audit finding dismissed -- which it has no verb
    for and cannot do.
    """
    cid = _concl(mind)
    did = _dispute(mind, cid)
    assert _status(mind, did)["status"] == "open"

    mind.memory.withdraw_conclusion(conclusion_id=cid, actor="ego",
                                    reason="I could not support it")
    row = _status(mind, did)
    assert row["status"] == "retracted", "the dispute outlived the claim"

    concl = mind.memory.get_conclusion(cid)
    assert concl["standing"] == "retracted"
    assert concl["claim"], "the row was deleted rather than withdrawn"


def test_superseding_the_claim_ends_the_dispute_about_it(mind):
    """A claim is replaced by the claim that replaces it."""
    old = _concl(mind)
    did = _dispute(mind, old)
    new, _ = mind.memory.record_conclusion(
        claim="the cache is cold only after a cold boot", produced_by="ego",
        supersedes=old)

    assert _status(mind, did)["status"] == "superseded"
    prior = mind.memory.get_conclusion(old)
    assert prior["standing"] == "superseded" and prior["superseded_by"] == new


class _EgoSup:
    """Enough supervisor for the Ego verbs to be built against."""

    def __init__(self, mind):
        self.mind = mind
        self.cfg = mind.cfg
        self.log = __import__("logging").getLogger("test")

    def methods(self):
        return {}


def test_only_the_author_may_withdraw_a_claim(mind):
    """An auditor that can edit the record it audits is not an auditor.

    The check is on the stored `produced_by`, not on anything the caller
    says, so there is no phrasing that turns editing somebody else's record
    into changing your own mind.
    """
    from amoeba.errors import InvalidInput
    from amoeba import ego_api

    verbs = ego_api.build(_EgoSup(mind))
    mine = _concl(mind, by="ego")
    theirs = _concl(mind, claim="something Id concluded", by="id")

    with pytest.raises(InvalidInput) as exc:
        verbs["ego_withdraw_conclusion"](conclusion_id=theirs, reason="no")
    assert "only by whoever made it" in str(exc.value)
    assert mind.memory.get_conclusion(theirs)["standing"] == "active"

    out = verbs["ego_withdraw_conclusion"](conclusion_id=mine,
                                           reason="I could not support it")
    assert out["standing"] == "retracted"
    assert mind.memory.get_conclusion(mine)["standing"] == "retracted"


def test_withdrawing_twice_is_a_no_op_that_keeps_the_first_reason(mind):
    """A repeat is idempotent rather than an error, and does not rewrite why.

    The withdrawal carries a mutation id derived from the conclusion, so the
    writer returns the stored receipt instead of running it again. That is the
    right shape -- withdrawing something already withdrawn is a no-op, not a
    fault -- and it must not quietly replace the recorded reason with a later
    one.
    """
    cid = _concl(mind)
    mind.memory.withdraw_conclusion(conclusion_id=cid, actor="ego",
                                    reason="first reason")
    mind.memory.withdraw_conclusion(conclusion_id=cid, actor="ego",
                                    reason="a different second reason")
    row = mind.db.conn.execute(
        "SELECT standing, withdrawn_reason FROM conclusions"
        " WHERE conclusion_id = ?", (cid,)).fetchone()
    assert row["standing"] == "retracted"
    assert row["withdrawn_reason"] == "first reason"


def test_a_superseded_claim_cannot_then_be_withdrawn(mind):
    """The standing guard, on the path idempotency does not short-circuit.

    A claim that was replaced has already stopped being made, by somebody
    else's act. Withdrawing it afterwards would record a second, contradictory
    account of how it ended.
    """
    from amoeba.errors import InvalidInput

    old_id = _concl(mind)
    mind.memory.record_conclusion(claim="a better claim", produced_by="ego",
                                  supersedes=old_id)
    assert mind.memory.get_conclusion(old_id)["standing"] == "superseded"
    with pytest.raises(InvalidInput) as exc:
        mind.memory.withdraw_conclusion(conclusion_id=old_id, actor="ego",
                                        reason="also withdrawing it")
    assert "no longer being made" in str(exc.value)


def test_a_repeated_contradiction_does_not_open_a_second_dispute(mind):
    """One unresolved issue must not look like twenty.

    Every adverse audit called `open_disagreement` unconditionally, so
    revisiting one conclusion twenty times produced twenty open rows and the
    count stopped describing anything.
    """
    cid = _concl(mind)
    first = _dispute(mind, cid)
    again = _dispute(mind, cid)
    assert again == first, "a rival dispute was opened about the same claim"

    row = _status(mind, first)
    assert row["recurrences"] == 1, "the repeat was not recorded"
    n = mind.db.conn.execute(
        "SELECT COUNT(*) FROM disagreements WHERE subject_id = ?", (cid,)
    ).fetchone()[0]
    assert n == 1


def test_a_closed_dispute_does_not_block_a_later_one(mind):
    """The uniqueness is on *open* disputes, not on the subject forever."""
    cid = _concl(mind)
    first = _dispute(mind, cid)
    mind.memory.resolve_disagreement(disagreement_id=first,
                                     resolution="closed_by_operator",
                                     actor="operator", detail="decided")
    second = _dispute(mind, cid, basis="basis-2")
    assert second != first
    assert _status(mind, second)["status"] == "open"


def test_a_dispute_cannot_be_closed_twice(mind):
    """A second closure would overwrite how the first one ended."""
    from amoeba.errors import InvalidInput

    cid = _concl(mind)
    did = _dispute(mind, cid)
    mind.memory.resolve_disagreement(disagreement_id=did, resolution="retracted",
                                     actor="ego", detail="withdrawn")
    with pytest.raises(InvalidInput):
        mind.memory.resolve_disagreement(disagreement_id=did,
                                         resolution="closed_by_operator",
                                         actor="operator", detail="again")


# ---------------------------------------------------------------------------
# Resolution by audit requires the ground to have moved
# ---------------------------------------------------------------------------
def test_a_supporting_audit_on_a_changed_basis_settles_the_dispute(mind):
    """The contradiction really has changed, and the Harness can see it.

    New evidence arrived, the fresh audit was performed against it, and the
    verdict reversed. That is the ground moving, which is what a resolution
    is supposed to mean.
    """
    cid = _concl(mind)
    did = _dispute(mind, cid, basis="basis-at-open")

    mind.memory.resolve_disagreement(
        disagreement_id=did, resolution="resolved_supported", actor="harness",
        detail="a later audit supported the claim against a changed basis")
    row = _status(mind, did)
    assert row["status"] == "resolved_supported"


def test_the_opening_evidence_basis_is_recorded(mind):
    """Without it there is nothing to compare a later audit against.

    The whole rule depends on knowing what the dispute was opened against, so
    the digest is stored at open time rather than reconstructed later -- the
    dossier will have moved on by then, which is precisely the thing being
    measured.
    """
    cid = _concl(mind)
    did = _dispute(mind, cid, basis="basis-at-open")
    row = mind.db.conn.execute(
        "SELECT evidence_basis_digest FROM disagreements"
        " WHERE disagreement_id = ?", (did,)).fetchone()
    assert row["evidence_basis_digest"] == "basis-at-open"


def test_the_evidence_basis_digest_reflects_the_measured_dossier(mind):
    """Taken from what was measured, not from what the adjudicator says.

    The point is to tell a changed world from a changed mind, so the digest
    has to come from the Harness's own dossier rather than from anything Id
    reports about what it looked at.
    """
    from amoeba.supervisor_api import _evidence_basis_digest

    a = _evidence_basis_digest({"evidence": ["e17"], "resolved_from": "record"})
    b = _evidence_basis_digest({"resolved_from": "record", "evidence": ["e17"]})
    c = _evidence_basis_digest({"evidence": ["e17", "e18"],
                                "resolved_from": "record"})
    assert a == b, "key order changed the basis; it must not"
    assert a != c, "adding evidence did not change the basis"


def test_a_reversal_on_an_unchanged_basis_does_not_settle_the_dispute(mind):
    """The adjudicator changing its mind is not the ground moving.

    Id opens a dispute, nothing about the evidence changes, and a later audit
    says supported. Closing on that would let the auditor open a dispute and
    then quietly certify it away -- which is precisely what separating Ego
    from Id exists to prevent.
    """
    from amoeba.supervisor_api import _settle_if_the_ground_moved

    cid = _concl(mind)
    did = _dispute(mind, cid, basis="same-basis")

    out = _settle_if_the_ground_moved(mind, cid, "same-basis", "audit-2")
    assert out["disagreement_resolved"] is False
    assert out["self_contradicted"] is True
    assert _status(mind, did)["status"] == "open", "the dispute was settled"


def test_a_reversal_on_an_unchanged_basis_is_recorded_not_discarded(mind):
    """Two opposite verdicts against identical evidence is worth keeping.

    It is a fact about the organism's own reasoning, and smoothing it over
    would be the one kind of forgetting this system refuses.
    """
    from amoeba.store.events import read_events
    from amoeba.supervisor_api import _settle_if_the_ground_moved

    cid = _concl(mind)
    _dispute(mind, cid, basis="same-basis")
    _settle_if_the_ground_moved(mind, cid, "same-basis", "audit-2")

    noticed = [e for e in read_events(mind.db.conn)
               if e.kind == "audit.self_contradicted"]
    assert noticed, "the contradiction was discarded rather than recorded"


def test_a_reversal_after_the_evidence_moved_does_settle_it(mind):
    """New evidence, fresh audit, reversed verdict -- the contradiction changed."""
    from amoeba.supervisor_api import _settle_if_the_ground_moved

    cid = _concl(mind)
    did = _dispute(mind, cid, basis="basis-at-open")

    out = _settle_if_the_ground_moved(mind, cid, "a-different-basis", "audit-2")
    assert out["disagreement_resolved"] is True
    row = _status(mind, did)
    assert row["status"] == "resolved_supported"


def test_a_supporting_audit_with_no_open_dispute_changes_nothing(mind):
    """The ordinary case: most audits have no dispute to settle."""
    from amoeba.supervisor_api import _settle_if_the_ground_moved

    cid = _concl(mind)
    assert _settle_if_the_ground_moved(mind, cid, "any-basis", "audit-1") == {}
