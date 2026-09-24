"""The backchannel room: Ego, Id and the Operator, live.

The audit that preceded this work found the cognitive loop entirely real and
the page over it entirely blind. `ego_message_id` and `id_message_ego` queue a
durable `role_message` trigger in the other role's mailbox and it does enter
the other role's bundle -- verified by putting a marker through and finding it
in the turn. But neither verb emits an event, and the console's transcript was
assembled from three event kinds, none of which either verb produces. The page
called "ego <-> id backchannel" had never in its life shown an Ego-to-Id
message.

It also had no room. `side_channel` requires `to_role` in `{ego, id}` and
refuses anything else, so an Operator message reached exactly one mind and the
other never learned it happened. Three participants, three private wires.

What these tests hold:

  * the loop really is a loop -- each direction lands in the other role's
    bundle, which is what "consumed into cognition" has to mean;
  * the Operator addresses the room, and the Harness delivers to both minds as
    two attributed triggers rather than one invisible broadcast;
  * authorship is structural, not a field -- there is no parameter with which
    the Operator could post as Ego or Id;
  * the room buffer is bounded, ephemeral, and honest about being so: it dies
    with the process, while what a message *influenced* survives in its
    trigger and in the turn that consumed it.

The last point is the one worth being strict about. An ephemeral view is fine.
An ephemeral view that quietly becomes the only record of a causal input would
not be.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from amoeba.room import MAX_TEXT, Room
from conftest import start_stack
from test_converse_panel import _script

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")
needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node to drive the panel")

HARNESS = Path(__file__).with_name("dashboard_panel_harness.js")


# ---------------------------------------------------------------------------
# The buffer itself
# ---------------------------------------------------------------------------
def test_the_room_is_bounded():
    """A viewport that grows without bound is a leak wearing a feature."""
    room = Room(maxlen=5)
    for i in range(50):
        room.post(author="ego", text=f"message {i}")
    entries = room.since(0)
    assert len(entries) == 5
    assert [e["text"] for e in entries] == [f"message {i}" for i in range(45, 50)]
    assert room.state()["capacity"] == 5


def test_a_reader_gets_only_what_it_has_not_seen():
    room = Room()
    a = room.post(author="ego", text="first")
    b = room.post(author="id", text="second")
    assert [e["seq"] for e in room.since(0)] == [a["seq"], b["seq"]]
    assert [e["text"] for e in room.since(a["seq"])] == ["second"]
    assert room.since(b["seq"]) == []


def test_an_author_nobody_can_be_held_to_is_refused():
    """An unattributable room entry is worse than no entry."""
    room = Room()
    with pytest.raises(ValueError):
        room.post(author="supervisor", text="who said this?")
    with pytest.raises(ValueError):
        room.post(author="", text="or this?")


def test_a_signal_with_no_prose_is_not_part_of_a_conversation():
    room = Room()
    assert room.post(author="ego", text="") is None
    assert room.post(author="ego", text="   \n ") is None
    assert room.since(0) == []


def test_one_speaker_cannot_flood_the_room():
    room = Room()
    entry = room.post(author="ego", text="x" * (MAX_TEXT * 4))
    assert len(entry["text"]) == MAX_TEXT


# ---------------------------------------------------------------------------
# The loop, against a running organism
# ---------------------------------------------------------------------------
def _bundles(stack, markers: dict[str, str]) -> dict[str, set[str]]:
    """Which role's turns saw which marker."""
    seen: dict[str, set[str]] = {"ego": set(), "id": set()}
    for t in (stack.call("role_turns", limit=60).get("turns") or []):
        detail = stack.call("role_turn", turn_id=t["turn_id"])
        text = json.dumps(detail.get("bundle") or {})
        for name, marker in markers.items():
            if marker in text:
                seen.setdefault(detail.get("role"), set()).add(name)
    return seen


@live
def test_each_direction_of_the_backchannel_lands_in_the_other_mind(tmp_path):
    """Ego to Id and Id to Ego: authored in the room, consumed in a bundle."""
    stack = start_stack(tmp_path)
    markers = {"ego_to_id": "EGOMARK-zebra-7 a word from Ego",
               "id_to_ego": "IDMARK-quartz-9 a word from Id"}
    try:
        stack.wait_for_children(timeout=90)
        time.sleep(4)   # let the startup turns drain

        stack.call("ego_message_id", kind="notice", message=markers["ego_to_id"])
        stack.call("id_message_ego", kind="notice", message=markers["id_to_ego"])
        time.sleep(10)

        room = stack.call("operator_backchannel", since=0)
        by_author = {e["author"]: e["text"] for e in room["entries"]}
        assert by_author.get("ego") == markers["ego_to_id"], room["entries"]
        assert by_author.get("id") == markers["id_to_ego"], room["entries"]

        seen = _bundles(stack, markers)
        assert "ego_to_id" in seen["id"], "Id never saw what Ego said"
        assert "id_to_ego" in seen["ego"], "Ego never saw what Id said"
        # And not to themselves.
        assert "ego_to_id" not in seen["ego"]
        assert "id_to_ego" not in seen["id"]
    finally:
        stack.stop()


@live
def test_the_operator_speaks_to_the_room_and_both_minds_hear_it(tmp_path):
    """One utterance, two attributed deliveries, nothing hidden."""
    stack = start_stack(tmp_path)
    marker = "OPMARK-falcon-3 a word from the Operator"
    try:
        stack.wait_for_children(timeout=90)
        time.sleep(4)

        out = stack.call("operator_backchannel", message=marker)
        assert {d["role"] for d in out["delivered"]} == {"ego", "id"}
        assert all(d["trigger_id"] for d in out["delivered"]), out["delivered"]

        # One entry for one utterance, however many deliveries it took.
        operator_entries = [e for e in out["entries"] if e["author"] == "operator"]
        assert len(operator_entries) == 1, operator_entries
        assert operator_entries[0]["text"] == marker

        time.sleep(10)
        seen = _bundles(stack, {"op": marker})
        assert "op" in seen["ego"], "Ego never heard the Operator"
        assert "op" in seen["id"], "Id never heard the Operator"
    finally:
        stack.stop()


@live
def test_the_operator_cannot_post_as_ego_or_id(tmp_path):
    """Structural, not validated: there is no parameter to misuse."""
    stack = start_stack(tmp_path)
    try:
        stack.wait_for_children(timeout=90)
        for forged in ("from_role", "actor", "author", "as_role"):
            with pytest.raises(Exception) as caught:
                stack.call("operator_backchannel", message="I am Ego.",
                           **{forged: "ego"})
            # Refused, and told which argument was refused. Calls are
            # checked before dispatch now, so the wording is the Harness's
            # own rather than Python's; what matters is that there is no
            # parameter to misuse and the forgery never lands.
            assert forged in str(caught.value), forged

        # And there is no recipient to choose either: the room is the address.
        with pytest.raises(Exception) as caught:
            stack.call("operator_backchannel", message="x", to_role="ego")
        assert "to_role" in str(caught.value)

        out = stack.call("operator_backchannel", message="plainly the operator")
        assert [e["author"] for e in out["entries"]
                if e["text"] == "plainly the operator"] == ["operator"]
    finally:
        stack.stop()


@live
def test_a_room_message_is_durably_recorded_even_though_the_view_is_not(tmp_path):
    """The view may forget. What influenced cognition may not.

    This is the line the ephemeral buffer must not cross: it is a viewport,
    and a viewport that became the only record of a causal input would make
    the organism's history quietly incomplete.
    """
    stack = start_stack(tmp_path)
    marker = "DURABLE-MARK-11 said into the room"
    try:
        stack.wait_for_children(timeout=90)
        out = stack.call("operator_backchannel", message=marker)
        trigger_ids = [d["trigger_id"] for d in out["delivered"]]

        # One durable trigger per delivery, each addressed to a real mailbox.
        for role, trigger_id in zip(("ego", "id"), trigger_ids):
            answer = stack.call("role_answer", trigger_id=trigger_id)
            assert answer["trigger_id"] == trigger_id

        # And the Operator's own act is in the event ledger, where governance
        # lives -- separately from the room, which is only a view.
        blob = json.dumps(stack.call("history", limit=200))
        assert marker in blob, "the operator's act left no durable trace"
    finally:
        stack.stop()


@live
def test_the_room_does_not_survive_a_restart_but_the_record_does(tmp_path):
    """Explicitly ephemeral, and not masquerading as history."""
    marker = "EPHEMERAL-MARK-5 only this runtime"
    stack = start_stack(tmp_path)
    try:
        stack.wait_for_children(timeout=90)
        out = stack.call("operator_backchannel", message=marker)
        assert any(e["text"] == marker for e in out["entries"])
    finally:
        stack.stop()

    again = start_stack(tmp_path)
    try:
        again.wait_for_children(timeout=90)
        room = again.call("operator_backchannel", since=0)
        assert all(e["text"] != marker for e in room["entries"]), \
            "the room came back from the dead"
        assert room["held"] == len(room["entries"])

        # The durable side is untouched: the message still happened.
        assert marker in json.dumps(again.call("history", limit=200)),             "the record forgot too"
    finally:
        again.stop()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def _drive(room: list, actions: list, **extra) -> dict:
    scenario = {"panel": "backchannel", "session": "operator-token",
                "responses": [], "room": room, "actions": actions, **extra}
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "dashboard.js"
        js.write_text(_script(), encoding="utf-8")
        scn = Path(tmp) / "scenario.json"
        scn.write_text(json.dumps(scenario), encoding="utf-8")
        r = subprocess.run([shutil.which("node"), str(HARNESS), str(js),
                            str(scn)],
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=120)
    assert r.returncode == 0, r.stderr.strip()
    return json.loads(r.stdout)


def _room(*pairs) -> list:
    return [{"seq": i + 1, "ts": 0, "author": a, "text": t}
            for i, (a, t) in enumerate(pairs)]


@needs_node
def test_the_page_shows_the_room_as_a_conversation():
    out = _drive(_room(("ego", "I found a contradiction."),
                       ("id", "That conflicts with the audit."),
                       ("operator", "Check the evidence basis.")), [])

    assert [(t["who"], t["text"]) for t in out["transcript"]] == [
        ("ego", "I found a contradiction."),
        ("id", "That conflicts with the audit."),
        ("operator", "Check the evidence basis.")]
    # Each speaker is distinguishable by class rather than by decoration.
    assert [t["side"] for t in out["transcript"]] == ["ego", "id", "operator"]
    assert out["requests"][0]["method"] == "operator_backchannel"
    assert out["requests"][0]["params"] == {"since": 0}


@needs_node
def test_new_entries_appear_without_a_refresh():
    """The whole point of the page: watch it happen."""
    out = _drive(_room(("ego", "first thing")),
                 [{"type": "wait", "untilRequests": 3}],
                 injections=[{"beforeRequest": 2, "author": "id",
                              "text": "a reply that arrived on its own"}])

    texts = [t["text"] for t in out["transcript"]]
    assert texts == ["first thing", "a reply that arrived on its own"], texts
    # Polled incrementally: it asked for what it had not seen, not for
    # everything, and did not append the same entry twice.
    assert out["requests"][1]["params"]["since"] >= 1


@needs_node
def test_an_operator_message_goes_to_the_room_with_no_recipient():
    out = _drive(_room(("ego", "hello")),
                 [{"type": "type", "text": "look at the disagreement"},
                  {"type": "key", "key": "Enter"}])

    posts = [r for r in out["requests"] if (r["params"] or {}).get("message")]
    assert len(posts) == 1, posts
    assert posts[0]["method"] == "operator_backchannel"
    assert posts[0]["params"]["message"] == "look at the disagreement"
    # No recipient, and nothing that could name a speaker.
    for forbidden in ("to_role", "from_role", "actor", "author"):
        assert forbidden not in posts[0]["params"]

    assert ("operator", "look at the disagreement") in [
        (t["who"], t["text"]) for t in out["transcript"]]
    assert out["composer"] == ""


@needs_node
def test_the_operators_words_come_back_through_the_room():
    """Not echoed locally: what is shown is what the Harness carried."""
    out = _drive([], [{"type": "type", "text": "said once"},
                      {"type": "key", "key": "Enter"},
                      {"type": "wait", "untilRequests": 4}])
    said = [t for t in out["transcript"] if t["text"] == "said once"]
    assert len(said) == 1, "the message was shown twice"
    assert said[0]["who"] == "operator"


@needs_node
def test_enter_sends_and_shift_enter_does_not():
    out = _drive([], [{"type": "type", "text": "half a thought"},
                      {"type": "key", "key": "Enter", "shift": True}])
    assert [r for r in out["requests"] if (r["params"] or {}).get("message")] == []
    assert out["composer"] == "half a thought"


@needs_node
def test_a_multiline_room_message_keeps_its_shape():
    text = "One point.\n\nAnother point,\n  with a continuation."
    out = _drive([], [{"type": "type", "text": text},
                      {"type": "key", "key": "Enter"}])
    assert [t["text"] for t in out["transcript"]] == [text]


@needs_node
def test_the_room_carries_no_mechanics():
    """Backchannel answers one question, and it is not an observability page."""
    out = _drive(_room(("ego", "I will ask Id."), ("id", "Looking now.")), [])
    blob = json.dumps(out["transcript"])
    for leak in ("trg_", "turn_", "op_", "seq", "lineage", "receipt",
                 "sha256", "to_role", "from_role", "kind"):
        assert leak not in blob, f"{leak!r} reached the room"
    assert out["pills"] == [
        "the live room for this runtime — it is not history, and a "
        "restart starts it empty"]


@needs_node
def test_a_reader_who_scrolled_up_is_left_alone():
    out = _drive(_room(("ego", "old")),
                 [{"type": "wait", "untilRequests": 3}],
                 injections=[{"beforeRequest": 2, "author": "id",
                              "text": "new"}],
                 scroll={"scrollTop": 0, "scrollHeight": 4000,
                         "clientHeight": 400})
    assert out["scrollTop"] == 0
    assert [t["text"] for t in out["transcript"]] == ["old", "new"]
