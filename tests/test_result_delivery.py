"""A result too large to show is never chopped, and the rest is reachable.

Two findings from the context accounting of Id's session:

- Every role tool result was delivered unbounded. The Harness computed the
  bounded text and `RoleProcess._invoke` dropped it, so `_feed_tool_result`
  re-serialized the whole result: one `prompt_incarnations(limit=5)` put 1827
  tokens into Id, `prompt_resolve` another 694.
- Where the bound did apply it cut the serialized JSON at 2000 characters --
  half a value in the context, and priced in characters where the context
  spends tokens.

Now a result that fits is shown whole; one that does not is shown as a whole
JSON projection that says how much of what it omits, with a reference to the
exact stored copy that `result_read` opens for the role it was issued to.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from types import SimpleNamespace

import pytest

from amoeba import mailbox, turn_api
from amoeba.errors import InvalidInput, NotFound
from amoeba.results import issue_result
from amoeba.rpc import RpcClient, read_or_create_token
from conftest import start_stack

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")


def _turn_for(mind, role):
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role=role, kind="user_input", source="operator",
                                  summary="q", lineage="op-r", expects_answer=True),
        actor="test", bump_version=False)
    _, turn = mind.writer.apply(
        lambda m: mailbox.claim(m, mind, role=role, incarnation=1,
                                profile_ref=f"{role}@1", profile_sha256="p",
                                environment_sha256="e", environment_blob="eb"),
        actor=role, bump_version=False)
    return turn["turn_id"]


def _read(mind):
    sup = SimpleNamespace(mind=mind, cfg=None, log=logging.getLogger("t"))
    return turn_api.build(sup)["result_read"]


FULL = {"bindings": [{"actor_id": f"a{i}", "note": "n" * 50} for i in range(17)],
        "detail": "full"}


def test_result_read_returns_the_exact_stored_result_and_any_part_of_it(mind):
    ref = issue_result(mind, json.dumps(FULL), role="id", tool="prompt_incarnations")[:16]
    read = _read(mind)
    turn = _turn_for(mind, "id")
    assert read(result_ref=ref, turn_id=turn) == {"path": "", "value": FULL}
    page = read(result_ref=ref, path="bindings", offset=2, limit=3, turn_id=turn)
    assert page["of"] == 17 and page["offset"] == 2
    assert page["items"] == FULL["bindings"][2:5]
    one = read(result_ref=ref, path="bindings[4].actor_id", turn_id=turn)
    assert one["value"] == "a4"


def test_a_reference_opens_only_for_the_role_it_was_issued_to(mind):
    """A reference is a capability that was handed over, not a digest to guess."""
    ref = issue_result(mind, json.dumps(FULL), role="id", tool="prompt_incarnations")[:16]
    read = _read(mind)
    with pytest.raises(NotFound):
        read(result_ref=ref, turn_id=_turn_for(mind, "ego"))


def test_result_read_belongs_to_a_turn(mind):
    ref = issue_result(mind, json.dumps(FULL), role="id", tool="t")[:16]
    with pytest.raises(InvalidInput):
        _read(mind)(result_ref=ref)


def test_a_path_that_is_not_there_is_refused_in_words(mind):
    ref = issue_result(mind, json.dumps(FULL), role="id", tool="t")[:16]
    turn = _turn_for(mind, "id")
    with pytest.raises(InvalidInput) as exc:
        _read(mind)(result_ref=ref, path="bindingz", turn_id=turn)
    assert "bindings" in exc.value.details["allowed"]


# ---------------------------------------------------------------------------
# live: the role path, where the bound was being thrown away
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = start_stack(tmp_path_factory.mktemp("deliver"),
                    scheduler={"id_startup_turn": False})
    s.wait_for_children(timeout=90)
    yield s
    s.stop()


def _inference(stack):
    cfg = stack.cfg
    inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                    read_or_create_token(cfg.token_path), timeout=60)
    inf.connect(retries=20, delay=0.5)
    return inf


def _converse(stack, message):
    env = stack.call("ego_converse", message=message, wait=False)
    deadline = time.time() + 120
    while stack.call("role_answer", trigger_id=env["result"]["trigger_id"])[
            "status"] not in ("completed", "incomplete", "unanswerable"):
        assert time.time() < deadline
        time.sleep(0.3)
    return env


class _Harness:
    """The supervisor side of `role_tool_invoke`, as the role process sees it."""

    def __init__(self, reply):
        self.reply = reply

    def call(self, method, **kw):
        assert method == "role_tool_invoke"
        return self.reply


def test_a_role_is_shown_the_bounded_result_not_the_whole_one():
    """The defect, at the layer it lived in: what the role feeds its session.

    Real `_turn`, `_invoke` and `_feed_tool_result`; only the connections are
    fakes. The simulator's tokenizer is a hash, so a live session's text
    cannot be read back -- this is where the result text is decided anyway.
    """
    from test_role_affordances import _looping_role

    call = '<tool_call>{"name": "history", "arguments": {"limit": 200}}</tool_call>'
    role = _looping_role([call, "Seen."])
    role.current_turn_id = "turn_x"
    role._offered = lambda: {"history"}
    role._sanitise = lambda name, arguments: arguments
    whole = {"events": [{"seq": i, "kind": "UNBOUNDED" * 5} for i in range(200)]}
    view = '{"complete": false, "result_ref": "0123456789abcdef", "result": "VIEW"}'
    role.sup = _Harness({"accepted": True, "result": whole, "result_text": view,
                         "reason": None})
    env = {"manifest": {"environment_sha256": "e",
                        "capabilities": [{"verb": "history"}]},
           "text": "VERBS", "environment_blob": None}
    role._turn("look", environment=env)
    fed = "\n".join(role.inf.fed)
    assert "UNBOUNDED" not in fed, "the whole result was fed to the role"
    assert '"result_ref": "0123456789abcdef"' in fed


@live
def test_the_compact_views_answer_the_usual_question(stack):
    summary = stack.call("prompt_incarnations", limit=5)
    assert summary["detail"] == "summary"
    for b in summary["bindings"]:
        assert set(b) <= {"actor_id", "actor_kind", "incarnation", "profile_ref",
                          "prompt_sha256", "created_at", "work_id"}
    full = stack.call("prompt_incarnations", limit=5, detail="full")
    assert "effective_settings" in full["bindings"][0]
    lean = stack.call("prompt_resolve", namespace="id")
    assert "prompt_text" not in lean and lean["prompt_chars"] > 0
    whole = stack.call("prompt_resolve", namespace="id", include_text=True)
    assert len(whole["prompt_text"]) == lean["prompt_chars"]


def test_the_harness_bounds_a_role_result_by_that_roles_budget_and_issues_it(mind):
    """`role_tool_invoke` itself: the budget, the projection, the issued copy."""
    whole = {"events": [{"seq": i, "kind": "k" * 30} for i in range(200)]}

    class _Sup:
        def __init__(self):
            self.mind, self.cfg = mind, mind.cfg
            self.log = logging.getLogger("t")

        def methods(self):
            return {"history": lambda **kw: whole}

        def note_trigger(self, role): return None
        def note_turn_finished(self, *a, **k): pass
        def next_heartbeat(self, role): return None

    mind.cfg.ego.tool_result_budget_tokens = 300
    verbs = turn_api.build(_Sup())
    turn = _turn_for(mind, "ego")
    out = verbs["role_tool_invoke"](turn_id=turn, name="history", arguments={})
    view = json.loads(out["result_text"])
    assert out["result_complete"] is False
    assert len(out["result_text"]) <= 300        # no tokenizer here: characters
    assert view["list"]["of"] == 200 and view["result_ref"] == out["result_ref"]
    back = verbs["result_read"](result_ref=out["result_ref"], turn_id=turn)
    assert back["value"] == whole, "the issued copy is not the exact result"
