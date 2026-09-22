"""What a capability accepts is declared, and a refusal says so.

In the first live pressure test Ego reached for its affordances and missed:
`post_type="incident_synthesis"`, `kind="incident_synthesis"`, `evidence` as
a string, `confidence` as "high". Every refusal was correct, and every one was
uninformative -- the checks carried the allowed values and the dispatcher
dropped them, so Ego was told "unknown post type" and never the list. The
tool surface was present and behaviourally undiscoverable.

And the one affordance that would have produced an organism -- another mind
working on something -- was described as "introduce an objective into the
productive-work system". Ego never used it once.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import time
from pathlib import Path

import pytest

from amoeba import role_env
from amoeba.rpc import RpcClient, read_or_create_token
from amoeba.vocabularies import ARGUMENT_VOCABULARIES
from conftest import start_stack

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")
SRC = Path(__file__).resolve().parents[1] / "src" / "amoeba"


# ---------------------------------------------------------------------------
# one vocabulary, read by both the check and the declaration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(ARGUMENT_VOCABULARIES), ids=lambda k: ".".join(k))
def test_every_declared_vocabulary_is_the_one_enforced(key):
    """The declared tuple is the enforced tuple -- the same object, checked at source.

    Two copies drift; one cannot. So the table names the constant, the
    constant is what the check reads, and this holds both halves.
    """
    values, path, name = ARGUMENT_VOCABULARIES[key]
    module = importlib.import_module(
        "amoeba." + path.removesuffix(".py").replace("/", "."))
    assert getattr(module, name) is values, f"{key} declares a copy, not the constant"
    source = (SRC / path).read_text(encoding="utf-8")
    assert re.search(rf"not in {name}\b|set\({name}\)", source), \
        f"{path} does not check against {name}"


def test_no_vocabulary_is_still_spelled_inline_where_it_is_checked():
    """The three hoisted tuples stay hoisted."""
    ego = (SRC / "ego_api.py").read_text(encoding="utf-8")
    ident = (SRC / "id_api.py").read_text(encoding="utf-8")
    assert 'not in ("conclusion", "work", "memory"' not in ego
    assert 'not in ("user", "maintenance")' not in ego
    assert 'not in ("none", "read", "read_write")' not in ego
    assert 'not in ("notice", "concern", "urgent")' not in ident
    assert 'not in ("ego", "id")' not in ident


def test_an_argument_renders_its_values_or_its_kind():
    assert role_env._render_argument(
        {"name": "post_type", "required": True, "allowed": ["a", "b"]}) == "post_type:{a|b}"
    assert role_env._render_argument(
        {"name": "evidence", "required": False, "kind": "list"}) == "evidence:list?"
    assert role_env._render_argument({"name": "body", "required": True}) == "body"
    assert role_env._kind_of("Sequence[dict[str, Any]]") == "list"
    assert role_env._kind_of("float | None") == "number"
    assert role_env._kind_of("str | None") is None


# ---------------------------------------------------------------------------
# live: what the roles are actually told
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = start_stack(tmp_path_factory.mktemp("vocab"),
                    scheduler={"id_startup_turn": False})
    s.wait_for_children(timeout=90)
    yield s
    s.stop()


def _caps(stack, role):
    env = stack.call("role_environment", role=role)
    return {c["verb"]: {a["name"]: a for a in c["arguments"]}
            for c in env["manifest"]["capabilities"]}, env


@live
def test_the_roles_are_told_the_values_they_are_checked_against(stack):
    ego, env = _caps(stack, "ego")
    ident, _ = _caps(stack, "id")
    from amoeba.store.board_repo import POST_TYPES
    from amoeba.store.memory_repo import MEMORY_KINDS

    assert ego["board_post"]["post_type"]["allowed"] == list(POST_TYPES)
    assert ego["ego_propose_memory"]["kind"]["allowed"] == list(MEMORY_KINDS)
    assert ident["id_escalate_to_operator"]["severity"]["allowed"] == \
        ["notice", "concern", "urgent"]
    # And the kinds that were guessed wrong live.
    assert ego["record_conclusion"]["evidence"]["kind"] == "list"
    assert ego["ego_propose_memory"]["confidence"]["kind"] == "number"
    text = env["text"]
    assert "post_type:{finding|" in text and "confidence:number?" in text


@live
def test_the_delegation_affordance_says_what_it_gets_you(stack):
    """Described as another mind, without being told when to want one."""
    env = stack.call("role_environment", role="ego")["manifest"]
    summary = next(c["summary"] for c in env["capabilities"]
                   if c["verb"] == "ego_request_work").lower()
    assert "neuocyte" in summary and "delegate" in summary
    for prescriptive in ("should", "must", "always", "whenever"):
        assert prescriptive not in summary, prescriptive


@live
def test_a_refusal_says_what_would_have_been_accepted(stack):
    """The live miss, replayed: the model is now told the list."""
    cfg = stack.cfg
    inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                    read_or_create_token(cfg.token_path), timeout=60)
    inf.connect(retries=20, delay=0.5)
    call = json.dumps({"name": "board_post", "arguments": {
        "author": "ego", "author_kind": "ego", "post_type": "incident_synthesis",
        "body": "the synthesis"}})
    inf.call("script_responses", responses=[
        {"role": "ego", "text": f"<tool_call>{call}</tool_call>",
         "finish_reason": "stop", "continues": True},
        {"role": "ego", "text": "Posted nothing.", "finish_reason": "stop",
         "continues": True}])
    env = stack.call("ego_converse", message="post it", wait=False)
    deadline = time.time() + 120
    while stack.call("role_answer", trigger_id=env["result"]["trigger_id"])[
            "status"] not in ("completed", "incomplete", "unanswerable"):
        assert time.time() < deadline
        time.sleep(0.3)
    refused = [json.loads(e["payload_inline"]) for e in
               stack.call("history", kinds=["role.tool_invoked"], limit=20)
               if '"board_post"' in (e.get("payload_inline") or "")]
    assert refused and refused[-1]["accepted"] is False
    assert "allowed: finding, question, hypothesis" in refused[-1]["reason"]
