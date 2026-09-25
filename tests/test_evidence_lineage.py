"""Corroboration is a property of evidence lineage, not speaker count.

Found live on 2026-09-24, one probe after worker lineage was fixed. Ego read
the board, forked an `ego.neuocyte`, and the worker -- which inherited Ego's
context and called nothing -- posted a finding at confidence 0.98 citing
"board_read returned no posts matching..." and "board_stats show 2 findings".
It had made no observation at all. Under the old rule it would have counted as
independent corroboration of Ego's own claim, because it had different
authorship and had never *read a post*.

Independence used to mean "the later author had not read the earlier post".
That is a fact about reading, and a forked worker never reads: it inherits the
observation itself. It now means "the later author observed a source lineage
the earlier one did not", which is a fact about evidence.

Two identities are kept apart, and conflating them would break this in both
directions:

    evidence_root   which acquisition it was -- the source lineage
    sha256          what the acquisition returned -- the payload

Two independent computations that both print "0" share a payload and are still
two observations. One source consulted twice yields two payloads and is still
one observation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from amoeba.results import acquisition_roots, evidence_root, inherit_roots, issue_result

sys.path.insert(0, str(Path(__file__).parent))
from test_identifier_discovery import _turn, _verbs  # noqa: E402
from test_work_lifecycle import _sup  # noqa: E402


def _post(mind, author, body, **kw):
    kw.setdefault("author_kind", "neuocyte")
    kw.setdefault("post_type", "finding")
    return mind.board.post(author=author, body=body, **kw)[0]


def _observe(mind, actor, *, tool="board_get_post", arguments=None, result=None):
    """An acquisition, recorded the way the Harness records one."""
    return issue_result(mind, json.dumps(result or {"seen": True}),
                        role=actor, tool=tool,
                        arguments=arguments if arguments is not None
                        else {"post_id": "p_source"}, actor="harness")


def _supports(mind, author, body, claim):
    return mind.board.post(
        author=author, author_kind="neuocyte", post_type="finding", body=body,
        thread_id=claim, relations=[{"to_post": claim, "relation": "supports"}])[0]


# ---------------------------------------------------------------------------
# What a root is, and what it is not
# ---------------------------------------------------------------------------
def test_two_computations_with_the_same_output_are_two_observations():
    """The payload is not the identity. Rooting this in the digest would have
    made every worker that ever printed "0" a single witness."""
    first = evidence_root("run_code", {"code": "print(0)"})
    second = evidence_root("run_code", {"code": "print(0)"})
    assert first != second


def test_one_source_consulted_twice_is_one_observation():
    """And the converse: the same source re-read is one root, whoever reads
    it and whatever it says the second time."""
    assert (evidence_root("board_get_post", {"post_id": "p1"})
            == evidence_root("board_get_post", {"post_id": "p1"}))
    assert (evidence_root("board_get_post", {"post_id": "p1"})
            != evidence_root("board_get_post", {"post_id": "p2"}))


def test_an_unclassified_tool_does_not_manufacture_independence():
    """The conservative direction: a tool nobody classified is treated as a
    retrieval, so it cannot invent corroboration by being unknown."""
    assert evidence_root("something_new", {"x": 1}).startswith("src:")
    assert (evidence_root("something_new", {"x": 1})
            == evidence_root("something_new", {"x": 1}))


# ---------------------------------------------------------------------------
# The five cases
# ---------------------------------------------------------------------------
def test_a_worker_repeating_what_it_inherited_adds_nothing(mind):
    """Case 1, and the live failure. Ego observes; the worker inherits and
    restates; support stays at one."""
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    claim = _post(mind, "ego", "the board holds no unfinished findings",
                  author_kind="ego")
    inherit_roots(mind, heir="nc_1", source="ego")

    echo = _supports(mind, "nc_1", "I concur: none unfinished", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support"] == []
    assert echo in c["concurring_reasoning"]
    assert c["independent_support_count"] == 0


def test_two_workers_inheriting_the_same_observation_add_nothing(mind):
    """Case 2. Two heirs of one observation are not two witnesses."""
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    claim = _post(mind, "ego", "no unfinished findings", author_kind="ego")
    for worker in ("nc_1", "nc_2"):
        inherit_roots(mind, heir=worker, source="ego")
        _supports(mind, worker, "agreed", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support"] == []
    assert len(c["concurring_reasoning"]) == 2


def test_a_worker_rereading_the_same_source_adds_nothing(mind):
    """Case 3. Going and looking yourself is honest, and it is the same
    source: one lineage, however many actors consult it."""
    _observe(mind, "ego", tool="board_get_post", arguments={"post_id": "p_a"})
    claim = _post(mind, "ego", "post p_a says the cache is cold", author_kind="ego")

    _observe(mind, "nc_1", tool="board_get_post", arguments={"post_id": "p_a"})
    _supports(mind, "nc_1", "I read p_a too; it says the cache is cold", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support"] == [], (
        "re-reading one source counted as a second observation of it")


def test_a_worker_observing_something_else_corroborates(mind):
    """Case 4. A distinct acquisition is the thing that counts."""
    _observe(mind, "ego", tool="board_get_post", arguments={"post_id": "p_a"})
    claim = _post(mind, "ego", "the cache is cold on boot", author_kind="ego")

    _observe(mind, "nc_1", tool="run_code", arguments={"code": "measure_cache()"},
             result={"cold": True})
    real = _supports(mind, "nc_1", "measured it: cold", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support"] == [real]
    assert c["independent_support_count"] == 1


def test_inherited_plus_newly_acquired_counts_only_the_new(mind):
    """Case 5. Both paths survive in the record; only the new one supports."""
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    claim = _post(mind, "ego", "the cache is cold", author_kind="ego")

    inherit_roots(mind, heir="nc_1", source="ego")
    _observe(mind, "nc_1", tool="run_code", arguments={"code": "measure()"})
    both = _supports(mind, "nc_1", "inherited the reading, then measured", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support"] == [both]

    held = acquisition_roots(mind.db.conn, "nc_1", first_hand_only=False)
    first_hand = acquisition_roots(mind.db.conn, "nc_1")
    assert len(held) > len(first_hand), "the inherited root was not kept"
    assert len(first_hand) == 1, "only the new acquisition is its own"


def test_distinct_authors_cannot_manufacture_independence(mind):
    """Case 6. Speaker count is not evidence."""
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    claim = _post(mind, "ego", "the knee is at 32", author_kind="ego")
    for worker in ("nc_1", "nc_2", "nc_3", "nc_4"):
        inherit_roots(mind, heir=worker, source="ego")
        _supports(mind, worker, "the knee is at 32", claim)

    c = mind.board.corroboration(claim)
    assert c["independent_support_count"] == 0, (
        "four voices repeating one observation became corroboration")


# ---------------------------------------------------------------------------
# It survives the things that rewrite context
# ---------------------------------------------------------------------------
def test_inheritance_is_durable_and_stays_inherited(mind):
    """Case 7. A restart re-reads the record; it does not re-grade it."""
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    inherit_roots(mind, heir="nc_1", source="ego")

    rows = [dict(r) for r in mind.db.conn.execute(
        "SELECT actor, acquired, inherited_from FROM acquisitions"
        " WHERE actor = 'nc_1'")]
    assert rows and all(r["acquired"] == "inherited" for r in rows)
    assert all(r["inherited_from"] == "ego" for r in rows)
    assert acquisition_roots(mind.db.conn, "nc_1") == set(), (
        "inherited evidence became first-hand")


def test_invoking_a_tool_records_what_the_role_observed(mind):
    """The other end of the same wire.

    Every other test here makes observations happen by calling `issue_result`.
    That proves the rule and not that anything feeds it: if the tool loop
    stopped recording acquisitions, no actor would ever hold a first-hand root,
    nothing would ever be corroboration, and every test above would still pass
    because they all supply their own.
    """
    verbs = _verbs(mind, {"system_pulse": lambda: {"awake": True}})
    before = acquisition_roots(mind.db.conn, "id")

    out = verbs["role_tool_invoke"](turn_id=_turn(mind), name="system_pulse",
                                    arguments={})
    assert out["accepted"] is True, out["reason"]

    gained = acquisition_roots(mind.db.conn, "id") - before
    assert len(gained) == 1, "the tool loop recorded no observation"


def test_a_fork_passes_its_forkers_observations_on_as_inherited(mind):
    """The path the live defect actually came down.

    The other tests call `inherit_roots` directly, which proves the rule but
    not that anything invokes it. A neuocyte receives Ego's context by
    acquiring a snapshot of it, and that is the moment the observations behind
    it change hands. If the fork passes nothing on, the worker starts with an
    empty record, and every observation it then restates from the inherited
    context looks like its own.
    """
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})
    mind.work.publish_snapshot(
        actor="ego", model_generation="gen_a", token_count=3, tokens=[1, 2, 3],
        text="ctx", kv_mode="shared_prefix", backend_handle="sess_ego")

    _sup(mind).methods()["acquire_snapshot"](holder="nc_1")

    assert acquisition_roots(mind.db.conn, "nc_1", first_hand_only=False), (
        "the fork passed none of Ego's observations on")
    assert acquisition_roots(mind.db.conn, "nc_1") == set(), (
        "what the worker inherited was recorded as its own observation")


def test_an_actor_cannot_claim_a_root_it_did_not_acquire(mind):
    """A model may say anything in a post body; roots come from the Harness."""
    claim = _post(mind, "ego", "the cache is cold", author_kind="ego")
    _observe(mind, "ego", tool="board_read", arguments={"limit": 20})

    liar = mind.board.post(
        author="nc_1", author_kind="neuocyte", post_type="finding",
        body="independently confirmed", thread_id=claim,
        evidence=[{"note": "I ran my own measurement",
                   "blob_sha256": "f" * 64, "event_id": "evt_invented"}],
        relations=[{"to_post": claim, "relation": "supports"}])[0]

    c = mind.board.corroboration(claim)
    assert liar not in c["independent_support"], (
        "a post claimed corroboration into existence")


def test_supersession_still_behaves(mind):
    """Case 8: the relations logic this rides on is untouched."""
    first = _post(mind, "nc_1", "the knee is at 30")
    second = _post(mind, "nc_1", "the knee is at 32", supersedes=first)

    posts = {p["post_id"]: p for p in mind.board.read(reader="nc_2", record=False)}
    assert second in posts
    assert posts.get(first, {}).get("status", "superseded") == "superseded"
