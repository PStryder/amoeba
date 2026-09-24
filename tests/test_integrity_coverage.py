"""A clean bill of health has to say what it examined.

An independent audit on 2026-09-24 deleted a claimed turn's `bundle_blob` from
the store and asked for integrity. Deep integrity reported zero missing
content, because the reference inventory covered six columns out of twenty and
`role_turns` was not among them. Shallow integrity also reported zero -- it
substituted an empty list, so "nothing is missing" and "nothing was looked at"
were the same answer, and `id_health` asks the shallow way.

Two things follow, and both are tested here: the inventory is complete, and it
stays complete as the schema grows.
"""

from __future__ import annotations

import pytest

from amoeba.store.events import (CONTENT_REFERENCES, NOT_CONTENT_REFERENCES,
                                 candidate_reference_columns)


def _turn_with_a_bundle(mind):
    from amoeba import mailbox

    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  payload={"message": "what is the state?"},
                                  expects_answer=True, lineage="op-1"),
        actor="test", bump_version=False)
    mind.work.register_agent(agent_id="ego", role="ego", session_handle="s")
    _r, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role="ego", incarnation=1,
                                profile_ref="ego@1", profile_sha256="p",
                                environment_sha256="e", environment_blob=None),
        actor="ego", bump_version=False)
    return turn


# ---------------------------------------------------------------------------
# The reported defect
# ---------------------------------------------------------------------------
def test_a_missing_turn_bundle_is_reported(mind):
    """`role_turns` held four references and none of them were checked."""
    turn = _turn_with_a_bundle(mind)
    digest = mind.db.conn.execute(
        "SELECT bundle_blob FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone()["bundle_blob"]
    assert digest, "the turn stored no bundle; nothing to lose"

    mind.blobs.path_for(digest).unlink()

    report = mind.verify_integrity(deep=True)
    assert report["missing_content_count"] == 1, report["missing_content"]
    assert report["missing_content"][0]["sha256"] == digest
    assert report["missing_content"][0]["referrer_kind"] == "role_turns.bundle_blob"


def test_a_shallow_check_does_not_report_a_clean_bill(mind):
    """It is the path `id_health` takes, and it examined nothing."""
    turn = _turn_with_a_bundle(mind)
    digest = mind.db.conn.execute(
        "SELECT bundle_blob FROM role_turns WHERE turn_id = ?",
        (turn["turn_id"],)).fetchone()["bundle_blob"]
    mind.blobs.path_for(digest).unlink()

    shallow = mind.verify_integrity()
    assert shallow["content_checked"] is False
    assert shallow["missing_content_count"] is None, (
        "a shallow check reported a count it never computed")

    deep = mind.verify_integrity(deep=True)
    assert deep["content_checked"] is True
    assert deep["content_references_checked"] == len(CONTENT_REFERENCES)


def test_a_healthy_store_says_how_much_it_checked(mind):
    _turn_with_a_bundle(mind)
    report = mind.verify_integrity(deep=True)
    assert report["missing_content_count"] == 0
    assert report["content_references_checked"] >= 20, (
        "a zero that covers six references is not the same claim")


# ---------------------------------------------------------------------------
# And it stays complete
# ---------------------------------------------------------------------------
def test_every_digest_column_is_either_checked_or_excused(mind):
    """The inventory is derived from the schema, not remembered.

    A table added later is a candidate by default, so leaving it unchecked
    takes a deliberate entry in `NOT_CONTENT_REFERENCES` with a reason. The
    previous inventory was hand-written, which is how it came to cover six
    references out of twenty without anyone noticing.
    """
    checked = {(t, c) for t, _id, c in CONTENT_REFERENCES}
    excused = set(NOT_CONTENT_REFERENCES)
    unclassified = candidate_reference_columns(mind.db.conn) - checked - excused

    assert not unclassified, (
        "these columns hold a digest and are neither checked for content nor "
        f"excused with a reason: {sorted(unclassified)}")


def test_nothing_is_both_checked_and_excused(mind):
    checked = {(t, c) for t, _id, c in CONTENT_REFERENCES}
    assert not checked & set(NOT_CONTENT_REFERENCES)


def test_every_excused_column_says_why(mind):
    for key, reason in NOT_CONTENT_REFERENCES.items():
        assert reason.strip(), f"{key} is excused without a reason"


def test_the_inventory_names_columns_that_exist(mind):
    """An inventory can also drift by naming something that is gone."""
    present = candidate_reference_columns(mind.db.conn)
    for table, _idcol, col in CONTENT_REFERENCES:
        assert (table, col) in present, f"{table}.{col} is not in the schema"
