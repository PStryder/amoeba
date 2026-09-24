"""A count comes with a way to reach the thing counted.

Live, the new heartbeat digest told Id "unaudited conclusions: 17" and Id
immediately called `get_conclusion`, `get_memory` and `audit_dossier` with
`c8f050bd638e10dc` -- the reference from a bounded tool result it had just
been shown. Before that it had used its own session id with a `con_` prefix
glued on. Every call came back "unknown conclusion", which is true and
useless.

The cause was the surface, not the model. Id can fetch a conclusion by id
and cannot list conclusions at all; `audit_dossier()` with no argument --
the question it actually had -- refused with "no operation to resolve". A
mind handed a count with no route to the things counted manufactures the
route.

Nothing here guesses which identifier was meant. Guessing would turn a
mind's mistake into the Harness's claim.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from amoeba import heartbeat, identifiers, mailbox, turn_api
from amoeba.errors import NotFound

CONCL = "concl_01M34JP5HHF1PS7QKTTFFQMG7X"
SESSION = "sess_01M34JP5HHF1PS7QKTTFFQMG7X"
RESULT_REF = "c8f050bd638e10dc"


# ---------------------------------------------------------------------------
# What a value is
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value, said", [
    (RESULT_REF, "a result reference"),
    (SESSION, "an inference session identifier"),
    ("post_01M34JP5HHF1PS7QKTTFFQMG7X", "a blackboard post identifier"),
    (1, "a number"),
    ("the one about clock drift", "not an identifier this organism issues"),
])
def test_a_value_is_described_as_what_it_is(value, said):
    assert identifiers.looks_like(value) == said


def test_a_well_formed_identifier_of_the_right_kind_is_not_a_misuse():
    """Then it is simply absent, and "no such conclusion" is the whole truth."""
    assert identifiers.misuse("conclusion_id", CONCL) is None
    assert identifiers.misuse("conclusion_id", RESULT_REF) is not None
    assert identifiers.misuse("conclusion_id", SESSION) is not None


def test_nothing_guesses_which_identifier_was_meant(mind):
    """The refusal names a path, never a candidate."""
    said = identifiers.explain("conclusion_id", RESULT_REF)
    assert "audit_dossier()" in said and "history(" in said
    assert "did you mean" not in said.lower()
    assert CONCL not in said


def test_every_kind_a_role_can_name_says_where_one_comes_from():
    for argument in ("conclusion_id", "memory_id", "post_id", "work_id"):
        assert identifiers.where_from(argument), argument


# ---------------------------------------------------------------------------
# The refusal a role actually reads
# ---------------------------------------------------------------------------
def _verbs(mind, methods):
    sup = SimpleNamespace(mind=mind, cfg=mind.cfg, log=logging.getLogger("t"),
                          methods=lambda: methods,
                          note_trigger=lambda role: None,
                          note_turn_finished=lambda *a, **k: None,
                          next_heartbeat=lambda role: None)
    return turn_api.build(sup)


def _turn(mind, role="id"):
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind="heartbeat",
                                  source="scheduler", summary="review"),
        actor="harness", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role=role, incarnation=1,
                                profile_ref=f"{role}@1", profile_sha256="p",
                                environment_sha256="e", environment_blob="eb"),
        actor=role, bump_version=False)
    return turn["turn_id"]


def test_a_refusal_says_what_the_value_was_and_where_a_real_one_lives(mind):
    """The live mistake, replayed through the path the model reads."""
    def get_conclusion(*, conclusion_id):
        raise NotFound("unknown conclusion", conclusion_id=conclusion_id)

    verbs = _verbs(mind, {"get_conclusion": get_conclusion})
    out = verbs["role_tool_invoke"](turn_id=_turn(mind), name="get_conclusion",
                                    arguments={"conclusion_id": RESULT_REF})
    assert out["accepted"] is False
    assert "unknown conclusion" in out["reason"]
    assert "is a result reference" in out["reason"]
    assert "audit_dossier()" in out["reason"]


def test_a_missing_but_well_formed_identifier_is_not_called_a_misuse(mind):
    def get_conclusion(*, conclusion_id):
        raise NotFound("unknown conclusion", conclusion_id=conclusion_id)

    verbs = _verbs(mind, {"get_conclusion": get_conclusion})
    out = verbs["role_tool_invoke"](turn_id=_turn(mind), name="get_conclusion",
                                    arguments={"conclusion_id": CONCL})
    assert "is a result reference" not in out["reason"]
    assert "identifier" not in out["reason"].split("(")[0]
    assert "audit_dossier()" in out["reason"], "still says where one comes from"


# ---------------------------------------------------------------------------
# The question Id actually has
# ---------------------------------------------------------------------------
def _conclusion(mind, claim="gw-3 drifted"):
    conclusion_id, _receipt = mind.memory.record_conclusion(
        claim=claim, produced_by="ego")
    return conclusion_id


def test_audit_dossier_with_no_argument_resolves_the_oldest_unaudited(mind, monkeypatch):
    from amoeba import supervisor_api

    first = _conclusion(mind, "the first claim")
    _conclusion(mind, "a later claim")
    sup = SimpleNamespace(mind=mind, cfg=mind.cfg, log=logging.getLogger("t"))
    dossier = supervisor_api.build(sup)["audit_dossier"]()
    assert dossier["conclusion"]["conclusion_id"] == first
    assert dossier["resolved_by"] == "the oldest conclusion nobody has audited"


def test_when_nothing_is_waiting_it_says_so_plainly(mind):
    from amoeba import supervisor_api

    sup = SimpleNamespace(mind=mind, cfg=mind.cfg, log=logging.getLogger("t"))
    with pytest.raises(NotFound) as refused:
        supervisor_api.build(sup)["audit_dossier"]()
    assert "nothing is waiting to be audited" in refused.value.message


def test_the_digest_names_the_conclusion_it_is_counting(mind):
    """Seventeen unaudited conclusions and no way to reach one is what did this."""
    first = _conclusion(mind, "the first claim")
    _conclusion(mind, "a later claim")
    digest = heartbeat.measure(mind.db.conn, "id", since=0)
    assert digest["attention"]["unaudited_conclusions"] == 2
    assert digest["oldest_unaudited"] == first
    assert f"unaudited conclusions 2 (oldest: {first})" in heartbeat.render(
        digest, interval_seconds=1800)
