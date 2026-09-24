"""A review that reports no change is only as true as its watermark.

Id is woken by conclusions, messages, work it owns and the operator. Nothing
wakes it for a failure, for resource pressure, or for a contradiction nobody
announced: the heartbeat is the only sense for unannounced state. So it
cannot simply be made rarer -- but it was paying about two thousand tokens a
time, 1793 of them tool results, to re-read state that had not moved. The
trigger said "nothing has woken you, check the organism's internal state" and
carried nothing, so Id spent five calls discovering a row of zeros.

The digest carries what changed, measured from an event watermark. Its
load-bearing property is completeness: every event since the watermark is
counted, by kind, so the digest cannot quietly omit a kind nobody thought to
include -- which is the way a cheap heartbeat would make Id blind. It
measures and never interprets, and Id keeps every sense it had.
"""

from __future__ import annotations

import pytest

from amoeba import heartbeat, mailbox
from amoeba.store.events import EventKind


def _emit(mind, kind, payload=None):
    mind.writer.apply(lambda m: m.emit(kind, payload or {}), actor="test",
                      bump_version=False)


def _seq(mind):
    return int(mind.db.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events")
               .fetchone()[0])


# ---------------------------------------------------------------------------
# Completeness: the property the whole cut depends on
# ---------------------------------------------------------------------------
def test_every_kind_since_the_watermark_is_counted(mind):
    """Nothing is filtered by what the Harness thinks matters."""
    mark = _seq(mind)
    for kind in (EventKind.ROLE_TURN_ENDED, EventKind.CONTEXT_PRESSURE,
                 EventKind.ROLE_TOOL_INVOKED, EventKind.ROLE_TOOL_INVOKED):
        _emit(mind, kind)
    digest = heartbeat.measure(mind.db.conn, "id", since=mark)

    assert digest["events"] == 4
    assert digest["by_kind"][EventKind.ROLE_TOOL_INVOKED] == 2
    assert digest["by_kind"][EventKind.CONTEXT_PRESSURE] == 1
    assert sum(digest["by_kind"].values()) + digest["other_events"] == digest["events"]
    assert heartbeat.quiet(digest) is False


def test_what_is_not_itemised_is_still_counted(mind):
    """A long tail is summarised by count, never dropped."""
    mark = _seq(mind)
    kinds = [k for k in vars(EventKind).values()
             if isinstance(k, str) and "." in k][:heartbeat.MAX_KINDS_SHOWN + 6]
    assert len(kinds) > heartbeat.MAX_KINDS_SHOWN
    for kind in kinds:
        _emit(mind, kind)
    digest = heartbeat.measure(mind.db.conn, "id", since=mark)

    assert len(digest["by_kind"]) == heartbeat.MAX_KINDS_SHOWN
    assert digest["other_kinds"] == len(kinds) - heartbeat.MAX_KINDS_SHOWN
    assert sum(digest["by_kind"].values()) + digest["other_events"] == digest["events"]
    assert digest["events"] == len(kinds)
    assert "not itemised" in heartbeat.render(digest, interval_seconds=300)


def test_a_first_review_does_not_claim_a_quiet_organism(mind):
    """No watermark is unknown, and unknown is not zero."""
    _emit(mind, EventKind.ROLE_TURN_ENDED)
    digest = heartbeat.measure(mind.db.conn, "id", since=None)
    assert digest["first_review"] is True
    assert heartbeat.quiet(digest) is False
    assert "first review" in heartbeat.render(digest, interval_seconds=300)


def test_the_watermark_is_read_from_the_previous_review(mind):
    """From the record, so a restarted supervisor does not restart the window."""
    assert heartbeat.watermark_of(mind.db.conn, mind.blobs, "id") is None
    mark = _seq(mind)
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="heartbeat",
                                  source="scheduler", summary="review",
                                  payload={"measured_to_seq": mark}),
        actor="harness", bump_version=False)
    assert heartbeat.watermark_of(mind.db.conn, mind.blobs, "id") == mark


def test_what_is_owed_is_reported_even_when_nothing_happened(mind):
    """An organism where nothing moved but something is outstanding is not quiet."""
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="operator_message",
                                  source="operator", summary="a question",
                                  expects_answer=True, lineage="op-1"),
        actor="test", bump_version=False)
    # Watermarked *after* the arrival, so nothing has happened since: the
    # only thing keeping this review from being quiet is what is still owed.
    digest = heartbeat.measure(mind.db.conn, "id", since=_seq(mind))
    assert digest["events"] == 0
    assert digest["attention"]["unanswered_requests"] == 1
    assert heartbeat.quiet(digest) is False
    assert "unanswered requests 1" in heartbeat.render(digest, interval_seconds=300)


def test_a_genuinely_quiet_organism_says_so_briefly(mind):
    mark = _seq(mind)
    digest = heartbeat.measure(mind.db.conn, "id", since=mark)
    assert heartbeat.quiet(digest) is True
    text = heartbeat.render(digest, interval_seconds=1800)
    assert "no events since seq" in text and "nothing outstanding" in text
    assert len(text) < 300, "a quiet review must not cost what it saves"


def test_the_digest_measures_and_does_not_interpret(mind):
    """No verdicts: Id decides what a count means."""
    mark = _seq(mind)
    _emit(mind, EventKind.CONTEXT_PRESSURE)
    text = heartbeat.render(heartbeat.measure(mind.db.conn, "id", since=mark),
                            interval_seconds=300).lower()
    for verdict in ("healthy", "fine", "nothing to worry", "no action",
                    "you should", "recommend", "ignore"):
        assert verdict not in text
    assert "check anything you doubt" in text


# ---------------------------------------------------------------------------
# What a quiet review is allowed to cost
# ---------------------------------------------------------------------------
def test_a_quiet_review_carries_a_ceiling_and_a_busy_one_does_not(mind):
    """The ceiling rides with the inputs, and only when they all agree."""
    quiet = {"payload_sha256": mind.blobs.put_json({"output_ceiling": 128})}
    busy = {"payload_sha256": mind.blobs.put_json({"reason": "review"})}
    assert mailbox.output_ceiling_for([quiet], mind.blobs) == 128
    assert mailbox.output_ceiling_for([quiet, busy], mind.blobs) is None, \
        "a real question bundled with a quiet review must not be shortened"
    assert mailbox.output_ceiling_for([], mind.blobs) is None


def test_a_claimed_turn_carries_the_ceiling_its_inputs_agreed(mind):
    from test_persistent_turns import _claim

    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="heartbeat",
                                  source="scheduler", summary="review",
                                  payload={"message": "no events",
                                           "output_ceiling": 128}),
        actor="harness", bump_version=False)
    turn = _claim(mind, "id")
    assert turn["output_ceiling"] == 128
    assert "no events" in turn["text"], "the digest must reach the role"


# ---------------------------------------------------------------------------
# live: what a review actually costs the session it runs in
# ---------------------------------------------------------------------------
@pytest.mark.skipif(__import__("sys").platform != "win32",
                    reason="live stack fixtures are Windows-only here")
def test_a_live_quiet_review_is_cheap(tmp_path):
    """The regression guard on the whole point of this.

    Measured before: 2315 tokens for a review that found nothing, of which
    1793 were tool results re-reading unchanged state.
    """
    import time as _time

    from conftest import start_stack

    stack = start_stack(tmp_path, scheduler={"id_heartbeat_seconds": 4.0})
    try:
        deadline = _time.time() + 90
        while _time.time() < deadline:
            beats = [t for t in stack.call("role_turns", role="id")["turns"]
                     if "heartbeat" in t["trigger_kinds"]
                     and t["status"] != "running"]
            if beats:
                break
            _time.sleep(0.3)
        assert beats, "Id never finished a heartbeat turn"
        beat = beats[0]
        detail = stack.call("role_turn", turn_id=beat["turn_id"])
        # The digest itself has to reach the role: without it the review is
        # cheap because it says nothing, which is the wrong kind of cheap.
        read = detail["bundle"]["text"]
        assert "attention:" in read and "every event since the watermark" in read
        assert detail["token_start"] is not None, "the turn measured no span"
        span = detail["token_end"] - detail["token_start"]
        assert span < 400, f"a quiet review cost {span} tokens"
        body = "\n".join(t.get("summary") or "" for t in detail["triggers"])
        assert "homeostatic" in body
        # It arrived knowing what changed, so it had no reason to go looking.
        assert beat["tool_call_count"] == 0
    finally:
        stack.stop()
