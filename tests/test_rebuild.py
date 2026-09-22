"""A rebuilt context is one the conversation could have reached by itself.

Measured live on Id, positional trim kept a 512-token head that cut the
capability declaration off after 227 tokens, kept two "the declaration given
earlier still applies" references to text it had just removed, and resumed
the tail in the middle of a tool result. The rule now:

    Environment is reconstructed. Cognition is preserved selectively.
    Neither is token-spliced.

Settled work may be removed whole; work still owed an answer must remain
sufficient to continue, so an oversized result in it is re-rendered as a
bounded projection of its exact stored copy rather than deleted.
"""

from __future__ import annotations

import json

import pytest

from amoeba import mailbox, reconstitution as rc
from amoeba.errors import InvalidInput
from amoeba.homeostasis import ContextHomeostasis, HomeostasisConfig
from chat_fixture import (END, START, ChatInference, Conversation, detokenize,
                          messages_of, record_turn, render, tokenize)

SYSTEM = "You are a test mind. " * 8


def _history(n=10):
    return [{"seq": j, "kind": "x" * 20} for j in range(n)]


def _homeo(mind, conv, *, role="ego", fraction=0.40, owed_last=False,
           record=True, governed=None):
    inf = ChatInference(sessions=[{"session_id": "sess_a", "role": role,
                                   "n_past": len(conv.tokens), "prefix_len": 0,
                                   "snapshot_id": None}],
                        tokens={"sess_a": list(conv.tokens)})
    mind.work.register_agent(agent_id=role, role=role, session_handle="sess_a")
    ids = []
    if record:
        for i, t in enumerate(conv.turns):
            owed = owed_last and i == len(conv.turns) - 1
            ids.append(record_turn(mind, role, "sess_a", t, lineage=f"op-{i}",
                                   settled=not owed))
    h = ContextHomeostasis(HomeostasisConfig(min_seconds_between_rejuvenations=0.0,
                                             rebuild_keep_fraction=fraction),
                           mind=mind, inference=lambda: inf)
    if governed is not None:
        h.governed_prompt = lambda _role: governed
    h.fake, h.turn_ids = inf, ids
    return h


def _rebuilt(h, out):
    return h.fake.tokens[out["new_session_id"]]


def _eight(pad=200, **kw):
    c = Conversation(SYSTEM)
    for i in range(8):
        c.turn(f"question {i}", pad=pad, calls=[("history", _history())], **kw)
    return c


# ---------------------------------------------------------------------------
# Nothing is spliced
# ---------------------------------------------------------------------------
def test_every_message_of_a_rebuilt_session_is_whole(mind):
    """No message starts or ends anywhere but at a boundary the session had."""
    c = _eight()
    h = _homeo(mind, c)
    out = h.rejuvenate(role="ego", reason="t")
    msgs = messages_of(_rebuilt(h, out))
    original = messages_of(c.tokens)
    assert out["tokens_after"] < out["tokens_before"]
    for m in msgs:
        assert m.startswith(START) and m.rstrip("\n").endswith(END), m[:80]
        # Whole and verbatim, or an original opening with its environment
        # removed -- nothing else is allowed to differ.
        assert m in original or any(rc.strip_environment(o)[0] == m for o in original), m[:120]


def test_no_environment_block_or_reference_survives_a_rebuild(mind):
    """The dangling reference is what made the old trim incoherent.

    Every environment block goes -- the full declaration and every "given
    earlier" reference -- because the next turn of a new session is given the
    current declaration in full (I118), immediately before it is used.
    """
    c = _eight()
    h = _homeo(mind, c, fraction=0.9)
    out = h.rejuvenate(role="ego", reason="t")
    text = detokenize(_rebuilt(h, out))
    assert "<role_environment" not in text
    assert "given earlier" not in text
    assert out["environments_removed"] >= 1
    assert "<turn_input>" in text, "the questions themselves must survive"


def test_the_governed_prompt_is_rendered_fresh(mind):
    c = _eight()
    h = _homeo(mind, c, governed="The governed doctrine.")
    out = h.rejuvenate(role="ego", reason="t")
    first = messages_of(_rebuilt(h, out))[0]
    assert first == render([{"role": "system", "content": "The governed doctrine."}])
    assert out["system_prompt"] == "reconstructed"
    assert out["system_prompt_changed"] is True


def test_without_a_binding_the_primed_prompt_is_kept_whole(mind):
    c = _eight()
    h = _homeo(mind, c)
    out = h.rejuvenate(role="ego", reason="t")
    assert messages_of(_rebuilt(h, out))[0] == messages_of(c.tokens)[0]
    assert out["system_prompt"] == "verbatim"


def test_positional_trim_and_eviction_are_gone(mind):
    """Not kept as options: the rule is that nothing is token-spliced."""
    h = _homeo(mind, _eight())
    for mode in ("trim", "evict"):
        with pytest.raises(InvalidInput):
            h.rejuvenate(role="ego", reason="t", mode=mode)


# ---------------------------------------------------------------------------
# Settled work goes whole, oldest first, only as far as needed
# ---------------------------------------------------------------------------
def test_settled_turns_go_oldest_first_and_only_as_far_as_needed(mind):
    c = _eight()
    h = _homeo(mind, c)
    out = h.rejuvenate(role="ego", reason="t")
    dropped = [d["turn_ids"][0] for d in out["dropped_units"]]
    assert dropped == h.turn_ids[:len(dropped)], "not oldest first"
    assert 0 < len(dropped) < len(h.turn_ids), "dropped nothing, or everything"
    assert out["reached_target"] is True
    # One fewer dropped unit would not have fitted: only what was needed.
    assert out["tokens_after"] <= out["target_tokens"]


def test_a_small_context_loses_no_turn(mind):
    c = Conversation(SYSTEM)
    c.turn("just one", pad=10)
    h = _homeo(mind, c)
    out = h.rejuvenate(role="ego", reason="t")
    assert out["dropped_units"] == []
    assert "just one" in detokenize(_rebuilt(h, out))


# ---------------------------------------------------------------------------
# Owed work stays sufficient to continue
# ---------------------------------------------------------------------------
def test_a_turn_still_owed_an_answer_is_never_removed(mind):
    """Dropping a live thought is the failure I80 exists to prevent."""
    c = _eight()
    c.turn("the live question", pad=10, calls=[("history", _history(3))])
    h = _homeo(mind, c, fraction=0.05, owed_last=True)
    out = h.rejuvenate(role="ego", reason="t")
    text = detokenize(_rebuilt(h, out))
    assert "the live question" in text
    assert h.turn_ids[-1] not in [t for d in out["dropped_units"] for t in d["turn_ids"]]
    # Its call and its result are both still there, in order.
    assert text.index('"name": "history"', text.index("the live question")) < \
        text.index('<tool_result name="history">', text.index("the live question"))


def test_a_turn_being_continued_is_active_even_when_nobody_awaits_an_answer(mind):
    """Measured live: a heartbeat cut off by pressure, with its continuation queued.

    A heartbeat expects no answer, so its lineage is never owed -- and the
    rebuild reported `units_owed: 0` for the very turn its continuation was
    about to carry on from. It survived only because it was newest.
    """
    c = _eight()
    c.turn("the heartbeat review", pad=10, calls=[("history", _history(3))])
    h = _homeo(mind, c, fraction=0.05)
    last = c.turns[-1]
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="heartbeat", source="scheduler",
                                  summary="review", expects_answer=False),
        actor="test", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role="ego", incarnation=1, profile_ref="ego@1",
                                profile_sha256="p", environment_sha256="e",
                                environment_blob="eb"),
        actor="ego", bump_version=False)
    mind.writer.apply(
        lambda m: mailbox.complete(m, mind, turn_id=turn["turn_id"],
                                   stop_reason="max_output_tokens",
                                   session_handle="sess_a", token_start=last["start"],
                                   token_end=last["end"]),
        actor="harness", bump_version=False)
    assert turn["turn_id"] in mailbox.active_turns(mind.db.conn, "ego")
    out = h.rejuvenate(role="ego", reason="t")
    assert out["units_owed"] == 1
    assert "the heartbeat review" in detokenize(_rebuilt(h, out))


def test_an_owed_result_is_projected_never_deleted_and_its_exact_copy_is_there(mind):
    """I called X, here is a bounded view of what it returned, the rest is retrievable."""
    big = {"items": [{"n": j, "note": "y" * 60} for j in range(60)]}
    c = Conversation(SYSTEM)
    c.turn("the live question", pad=10, calls=[("history", big)])
    h = _homeo(mind, c, fraction=0.3, owed_last=True)
    out = h.rejuvenate(role="ego", reason="t")
    text = detokenize(_rebuilt(h, out))
    assert '"name": "history"' in text, "the call was removed"
    assert '<tool_result name="history">' in text, "the result was removed"
    assert out["results_projected"], "an oversized owed result was left oversized"
    body = rc.result_body([m for m in messages_of(_rebuilt(h, out))
                           if "<tool_result" in m][0])[2]
    view = json.loads(body)
    assert view["complete"] is False and view["list"]["of"] == 60
    assert view["list"]["returned"] < 60
    from amoeba.results import issued_digest

    digest = issued_digest(mind, view["result_ref"], role="ego")
    assert digest, "the projection names a copy that was never issued to this role"
    assert json.loads(mind.blobs.get(digest).decode()) == big


def test_a_rebuild_that_cannot_reach_its_target_says_so_and_cuts_nothing(mind):
    c = Conversation(SYSTEM)
    c.turn("the live question " + "z" * 3000, pad=10)
    h = _homeo(mind, c, fraction=0.01, owed_last=True)
    out = h.rejuvenate(role="ego", reason="t")
    assert out["reached_target"] is False
    assert "z" * 3000 in detokenize(_rebuilt(h, out))


def test_a_turn_the_record_cannot_place_is_kept(mind):
    """"I do not know what this is" must not resolve to "so remove it"."""
    c = _eight()
    h = _homeo(mind, c, fraction=0.05, record=False)
    out = h.rejuvenate(role="ego", reason="t")
    assert out["dropped_units"] == []
    assert out["units_unknown"] == 8
    for i in range(8):
        assert f"question {i}" in detokenize(_rebuilt(h, out))


def test_a_span_that_fits_no_unit_places_nothing(mind):
    c = _eight()
    units = rc.group_units([rc.Message(start=a, end=b, tokens=c.tokens[a:b],
                                       text=detokenize(c.tokens[a:b]), role="user",
                                       kind="opening", terminated=True)
                            for a, b in ((t["start"], t["end"]) for t in c.turns)])
    straddling = [{"turn_id": "t", "lineage": "op", "start": c.turns[0]["start"],
                   "end": c.turns[1]["end"]}]
    rc.attribute(units, straddling, owed=set())
    assert all(u.status == "unknown" for u in units)


def test_an_empty_generation_prompt_is_not_carried(mind):
    c = _eight()
    c.turn("the live question", pad=10, open_prompt=True)
    h = _homeo(mind, c, fraction=0.9, owed_last=True)
    out = h.rejuvenate(role="ego", reason="t")
    msgs = messages_of(_rebuilt(h, out))
    assert out["empty_generation_prompts_removed"] == 1
    assert msgs[-1] != f"{START}assistant\n"


# ---------------------------------------------------------------------------
# Coordinates survive the rebuild that made them
# ---------------------------------------------------------------------------
def test_a_second_rebuild_can_still_remove_carried_settled_turns(mind):
    """Carried turns are placed in the new session, not left unknown.

    Without their new coordinates every carried turn would be `unknown`, kept
    forever, and a role's context could only grow across rebuilds -- the
    way whole-turn eviction stopped working after its first use.
    """
    c = _eight()
    h = _homeo(mind, c, fraction=0.6)
    first = h.rejuvenate(role="ego", reason="one")
    carried = mailbox.session_spans(mind.db.conn, "ego", first["new_session_id"])
    assert carried, "no carried turn was placed in the new session"
    h.cfg.rebuild_keep_fraction = 0.25
    second = h.rejuvenate(role="ego", reason="two")
    assert second["units_unknown"] == 0
    assert second["dropped_units"], "carried settled turns could not be removed"
    assert second["tokens_after"] < first["tokens_after"]


def test_spans_of_a_closed_session_never_place_a_turn_in_its_successor(mind):
    c = _eight()
    h = _homeo(mind, c, fraction=0.6)
    first = h.rejuvenate(role="ego", reason="one")
    new = mailbox.session_spans(mind.db.conn, "ego", first["new_session_id"])
    old = mailbox.session_spans(mind.db.conn, "ego", "sess_a")
    assert len(old) == 8, "the historical record was rewritten"
    new_len = len(h.fake.tokens[first["new_session_id"]])
    assert all(s["end"] <= new_len for s in new)
    assert {s["turn_id"] for s in new} < {s["turn_id"] for s in old}


def test_the_turn_boundary_asks_for_a_rebuild(mind):
    """The path every pressure stop takes, which used to ask for positional trim."""
    import logging

    from amoeba import turn_api

    asked: dict = {}

    class _Sup:
        def __init__(self):
            self.mind, self.cfg = mind, mind.cfg
            self.log = logging.getLogger("t")

        def methods(self):
            def context_rejuvenate(**kw):
                asked.update(kw)
                return {"new_session_id": "sess_b"}
            return {"context_rejuvenate": context_rejuvenate,
                    "harness_commit_audit": lambda **kw: None}

        def hand_over_session(self, *a, **k): return True
        def note_turn_finished(self, *a, **k): pass
        def note_trigger(self, role): return None
        def next_heartbeat(self, role): return None

    verbs = turn_api.build(_Sup())
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="id", kind="user_input", source="operator",
                                  summary="q", lineage="op-b", expects_answer=True),
        actor="test", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role="id", incarnation=1, profile_ref="id@1",
                                profile_sha256="p", environment_sha256="e",
                                environment_blob="eb"),
        actor="id", bump_version=False)
    out = verbs["role_complete_turn"](turn_id=turn["turn_id"], stop_reason="context_pressure")
    assert out["rejuvenation"]["performed"] is True
    assert asked["mode"] == "rebuild"
