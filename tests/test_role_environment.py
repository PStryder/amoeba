"""Profile, environment and turn input: the three things a role is given.

Amoeba now separates them for Ego and Id the way it always did for neuocytes:

``PROFILE``     who the role is           governed, versioned, bound at birth
``ENVIRONMENT`` what exists and what it   Harness-built, frozen per turn
                may do right now
``TURN INPUT``  what to think about now   the message, dossier or trigger

The claims worth defending are that the environment is authoritative rather
than decorative (a role can actually invoke what it is offered), that it is
honest (it never advertises something the role's credential cannot reach), and
that it is *environment* rather than doctrine -- a new profile appearing must
not require rewriting Ego's constitution.

The tool loop is exercised against fake inference and supervisor clients
because a deterministic backend will not emit a tool call on demand; the real
capability isolation is asserted separately against a live stack, where the
scope tables are what actually gate dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from amoeba import role_env, scopes
from amoeba.config import Config
from amoeba.errors import InvalidInput
from amoeba.mind import Mind
from amoeba.promptlib import bootstrap
from amoeba.promptlib.store import PromptStore
from amoeba.roles import AUTHORITY_ARGUMENTS, EgoProcess, IdProcess


# ---------------------------------------------------------------------------
# fakes: enough of the two clients for a bounded turn
# ---------------------------------------------------------------------------
class FakeInference:
    """Scripted generation. Records what was ingested, in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.ingested: list[str] = []
        self.generate_calls: list[dict] = []

    def call(self, method: str, **params):
        if method == "apply_chat_template":
            return params["messages"][0]["content"]
        if method == "ingest_text":
            self.ingested.append(params["text"])
            return {"ok": True}
        if method == "generate":
            self.generate_calls.append(params)
            text = self.replies.pop(0) if self.replies else "done"
            return {"text": text, "finish_reason": "stop",
                    "completion_tokens": 5, "time_to_first_token": 0.0,
                    "is_simulated": True}
        if method == "capabilities":
            return {"model_generation": "gen-test"}
        if method == "open_session":
            return {"session_id": "sess-1"}
        raise AssertionError(f"unexpected inference call {method}")


class FakeSupervisor:
    """A supervisor link that answers `role_environment` and records calls.

    `allowed` stands in for the scope table: a verb outside it raises, exactly
    as an RPC connection whose method table does not contain it would.

    It also models `role_tool_invoke`, the turn-fenced entry point every role
    capability now goes through. `open_turn` is the turn the Harness considers
    current; a call carrying any other turn is refused, which is what stops a
    turn that hung and was declared stale from acting after its inputs were
    handed to a replacement.
    """

    def __init__(self, manifest: dict, allowed: set[str],
                 open_turn: str | None = "turn-1") -> None:
        self.manifest = manifest
        self.allowed = allowed
        self.open_turn = open_turn
        self.calls: list[tuple[str, dict]] = []

    def call(self, method: str, **params):
        self.calls.append((method, params))
        if method == "role_environment":
            return {"manifest": self.manifest,
                    "text": role_env.render(self.manifest),
                    "environment_sha256": self.manifest["environment_sha256"],
                    "environment_blob": "blob-1"}
        if method == "role_tool_invoke":
            if params.get("turn_id") != self.open_turn:
                return {"accepted": False, "result": None,
                        "reason": (f"turn {params.get('turn_id')} is not "
                                   "running; the organism has moved on")}
            name = params["name"]
            if name not in self.allowed:
                return {"accepted": False, "result": None,
                        "reason": f"unknown method {name!r}"}
            # Recorded under its own name too, so tests can assert what was
            # actually invoked rather than unpacking the envelope.
            self.calls.append((name, params.get("arguments") or {}))
            return {"accepted": True, "result": {"ok": True},
                    "reason": None}
        if method not in self.allowed:
            raise InvalidInput(f"unknown method {method!r}")
        return {"ok": True, "echo": params}


def _manifest(role: str, capabilities: list[dict], profiles=()) -> dict:
    man = {
        "schema": "amoeba.role_environment/1",
        "role": role, "incarnation": 1,
        "bound_profile": {"profile_ref": f"{role}@1", "prompt_sha256": "p",
                          "config_sha256": "c"},
        "available_profiles": list(profiles),
        "capabilities": capabilities,
        "resources": {"model_generation": "gen-test",
                      "prompt": {"sha256": "x"},
                      "tool_surface": {"sha256": "y" * 16},
                      "schema": {"sha256": "z"}},
        "contract": "authoritative for this turn",
    }
    man["environment_sha256"] = "e" * 64
    return man


def _role(cls, cfg, *, manifest, allowed, replies):
    role = cls(cfg, port=0)
    role.inf = FakeInference(replies)
    role.sup = FakeSupervisor(manifest, allowed)
    role.session_id = "sess-1"
    role.incarnation = 1
    # The turn the Harness considers current. Capabilities are fenced to it.
    role.current_turn_id = "turn-1"
    role.profile_ref = f"{role.role}@1"
    role.profile = {"prompt_sha256": "p", "config_sha256": "c"}
    return role


def _tool_call(name: str, **args) -> str:
    return ("<tool_call>"
            + json.dumps({"name": name, "arguments": args})
            + "</tool_call>")


CAP_RECALL = {"verb": "recall", "summary": "Search maintained memory.",
              "arguments": [{"name": "query", "required": False},
                            {"name": "limit", "required": False}]}
CAP_REQUEST_WORK = {"verb": "ego_request_work", "summary": "Delegate an objective.",
                    "arguments": [{"name": "objective", "required": True},
                                  {"name": "replicas", "required": False}]}
CAP_BOARD_READ = {"verb": "board_read", "summary": "Read the blackboard.",
                  "arguments": [{"name": "reader", "required": False},
                                {"name": "limit", "required": False}]}
CAP_RAISE = {"verb": "id_raise_finding", "summary": "Record a finding.",
             "arguments": [{"name": "kind", "required": True},
                           {"name": "summary", "required": True}]}


# ---------------------------------------------------------------------------
# the manifest is honest about what the role can do
# ---------------------------------------------------------------------------
def test_model_facing_verbs_are_a_subset_of_the_roles_scope():
    """I58. A role is never told it has a capability it cannot invoke.

    The manifest's verb list and the dispatch authority are the same list, not
    two lists kept in step by hand. If they could drift, the environment would
    eventually promise something the credential cannot reach, and the model
    would be reasoning about a capability that does not exist.
    """
    for role in ("ego", "id"):
        offered = set(scopes.model_facing_verbs(role))
        authority = scopes.verbs_for(role)
        assert offered, role
        assert offered <= authority, sorted(offered - authority)


def test_every_deliberate_role_effector_is_discoverable():
    """The inverse: an effector must not exist in dispatch and be invisible.

    A verb built for Ego that never appears in Ego's environment is capability
    nobody can use, which is its own kind of lie.
    """
    ego = set(scopes.model_facing_verbs("ego"))
    assert set(scopes.EGO_ONLY) <= ego, sorted(set(scopes.EGO_ONLY) - ego)
    ident = set(scopes.model_facing_verbs("id"))
    assert set(scopes.ID_ONLY) <= ident, sorted(set(scopes.ID_ONLY) - ident)
    assert set(scopes.ID_SENSES) <= ident


def test_role_environments_do_not_leak_across_roles():
    """Ego is not offered Id's effectors, and Id is not offered Ego's."""
    ego = set(scopes.model_facing_verbs("ego"))
    ident = set(scopes.model_facing_verbs("id"))
    assert not (ego & set(scopes.ID_ONLY))
    assert not (ident & set(scopes.EGO_ONLY))
    # Ego does not receive Id's physiological telemetry.
    assert "system_pulse" not in ego
    assert "id_health" not in ego
    # Neither is offered an Operator verb.
    from amoeba import prompt_api

    for table in (ego, ident):
        assert not (table & set(prompt_api.OPERATOR_PROMPT))


def test_lifecycle_plumbing_is_not_offered_to_the_model():
    """A mind is not invited to operate its own life support."""
    for role in ("ego", "id"):
        offered = set(scopes.model_facing_verbs(role))
        for verb in ("register_agent", "retire_agent", "heartbeat",
                     "bind_profile", "role_environment"):
            assert verb not in offered, f"{role}: {verb}"


def test_the_manifest_is_deterministic_for_unchanged_state(mind):
    """Identical state must produce identical bytes, or provenance is noise."""
    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")

    class Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
        def methods(self):
            return {v: (lambda **k: None) for v in
                    scopes.model_facing_verbs("ego")}

    sup = Sup()
    first = role_env.build(sup, "ego", incarnation=1)
    second = role_env.build(sup, "ego", incarnation=1)
    assert first["environment_sha256"] == second["environment_sha256"]
    # Incarnation is excluded from the digest: the same world seen by a reborn
    # role is the same world.
    third = role_env.build(sup, "ego", incarnation=9)
    assert third["environment_sha256"] == first["environment_sha256"]


def test_an_undeclared_model_facing_verb_is_caught_at_build(mind):
    """Drift between the scope tables and the dispatch table must not be quiet."""
    class Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
        def methods(self):
            return {}          # nothing resolves

    with pytest.raises(InvalidInput):
        role_env.build(Sup(), "ego", incarnation=1)


# ---------------------------------------------------------------------------
# the tool loop: a role can actually use what it is offered
# ---------------------------------------------------------------------------
def test_ego_can_invoke_an_advertised_sense(cfg):
    """I59. What the environment offers, the role can actually execute.

    Before this, `roles.py` parsed a tool request out of generated text and
    reported it without running it. Injecting an environment on top of that
    would have produced a manifest that said "here are your capabilities" to a
    mind that could not use any of them.
    """
    man = _manifest("ego", [CAP_RECALL, CAP_REQUEST_WORK])
    ego = _role(EgoProcess, cfg, manifest=man,
                allowed={"recall", "ego_request_work"},
                replies=[_tool_call("recall", query="race condition"),
                         "FINAL: I checked memory."])
    out = ego._turn("what do you know?", trigger="test")

    invoked = [c for c in ego.sup.calls if c[0] == "recall"]
    assert invoked, "the advertised sense was never executed"
    assert invoked[0][1]["query"] == "race condition"
    assert out["stop_reason"] == "answered"
    assert out["tool_call_count"] == 1
    assert out["tool_calls"][0]["accepted"] is True
    # The result came back into the same turn.
    assert any("tool_result" in t for t in ego.inf.ingested)


def test_ego_can_invoke_an_advertised_effector(cfg):
    man = _manifest("ego", [CAP_REQUEST_WORK])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"ego_request_work"},
                replies=[_tool_call("ego_request_work", objective="check the lock"),
                         "done"])
    ego._turn("investigate", trigger="test")
    calls = [c for c in ego.sup.calls if c[0] == "ego_request_work"]
    assert calls and calls[0][1]["objective"] == "check the lock"


def test_id_can_invoke_an_advertised_sense_and_effector(cfg):
    man = _manifest("id", [CAP_RECALL, CAP_RAISE])
    ident = _role(IdProcess, cfg, manifest=man,
                  allowed={"recall", "id_raise_finding"},
                  replies=[_tool_call("recall", query="contradictions"),
                           _tool_call("id_raise_finding", kind="anomaly",
                                      summary="two claims disagree"),
                           "done"])
    out = ident._turn("audit", trigger="test")
    names = [c[0] for c in ident.sup.calls]
    assert "recall" in names and "id_raise_finding" in names
    assert out["tool_call_count"] == 2


def test_a_verb_outside_the_environment_is_refused_and_never_dispatched(cfg):
    """Ego guessing an Id verb must not reach the supervisor at all.

    Two independent things stop this: the frozen manifest does not contain it,
    and the scope table behind the connection does not either. The first is
    asserted here; the second is asserted against a live stack, where the real
    method table is what answers.
    """
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man,
                allowed={"recall", "id_raise_finding"},   # deliberately generous
                replies=[_tool_call("id_raise_finding", kind="anomaly",
                                    summary="I am Id now"),
                         "done"])
    out = ego._turn("go", trigger="test")

    assert not [c for c in ego.sup.calls if c[0] == "id_raise_finding"], \
        "a verb absent from the environment was dispatched anyway"
    assert out["tool_calls"][0]["accepted"] is False
    assert "not in this turn's environment" in out["tool_calls"][0]["reason"]


def test_authority_shaped_arguments_cannot_widen_authority(cfg):
    """I60. Who is asking is a fact about the connection, not an argument."""
    man = _manifest("ego", [CAP_BOARD_READ])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"board_read"},
                replies=[_tool_call("board_read", reader="id", role="id",
                                    actor="operator", scope="operator",
                                    client_id="someone", limit=3),
                         "done"])
    ego._turn("read the board", trigger="test")
    sent = [c[1] for c in ego.sup.calls if c[0] == "board_read"][0]

    assert sent["reader"] == "ego", "identity was taken from the model"
    for field in ("role", "actor", "scope", "client_id"):
        assert field not in sent, f"{field} reached the Harness"
    assert sent["limit"] == 3            # ordinary arguments still pass


def test_undeclared_arguments_are_dropped(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=[_tool_call("recall", query="x", _allow_root=True,
                                    surprise="parameter"),
                         "done"])
    ego._turn("go", trigger="test")
    sent = [c[1] for c in ego.sup.calls if c[0] == "recall"][0]
    assert sent == {"query": "x"}


def test_a_failing_tool_is_reported_to_the_model_not_fatal(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed=set(),   # recall will raise
                replies=[_tool_call("recall", query="x"), "carried on"])
    out = ego._turn("go", trigger="test")
    assert out["tool_calls"][0]["accepted"] is False
    assert out["stop_reason"] == "answered"
    assert any("refused" in t for t in ego.inf.ingested)


def test_the_tool_loop_is_bounded_by_turns(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=[_tool_call("recall", query=str(i)) for i in range(20)])
    out = ego._turn("go", trigger="test", max_tool_turns=3)
    assert out["stop_reason"] == "tool_turn_limit_reached"
    # The last turn's request is recorded but deliberately not executed.
    assert out["tool_calls"][-1]["executed"] is False
    assert len([c for c in ego.sup.calls if c[0] == "recall"]) == 2


def test_the_tool_loop_is_bounded_by_the_deadline(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=[_tool_call("recall", query="x") for _ in range(5)])
    out = ego._turn("go", trigger="test", deadline=0.0)
    assert out["stop_reason"] == "deadline_reached"
    assert not [c for c in ego.sup.calls if c[0] == "recall"]


def test_a_turn_that_is_no_longer_running_cannot_act(cfg):
    """I75. An expired turn cannot act, not merely cannot commit.

    The dangerous half of the stale-turn story. A turn that hangs, is declared
    stale and has its inputs handed to a replacement *was* refused at commit --
    `complete` rejects a turn that is no longer running -- but its side effects
    were not fenced at all. It could still request work, post findings and
    record conclusions into an organism that had moved on without it, because
    a role's effector call carried no turn identity whatsoever.

    Every capability now goes through `role_tool_invoke` carrying the turn it
    belongs to, exactly as a neuocyte carries its fencing token.
    """
    man = _manifest("ego", [CAP_REQUEST_WORK])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"ego_request_work"},
                replies=[_tool_call("ego_request_work", objective="act anyway"),
                         "done"])
    # The Harness has moved on: the turn this role still thinks it holds was
    # abandoned and a replacement is running.
    ego.sup.open_turn = "turn-2"

    out = ego._turn("go", trigger="test")

    assert not [c for c in ego.sup.calls if c[0] == "ego_request_work"], \
        "a turn that is no longer running reached an effector"
    assert out["tool_calls"][0]["accepted"] is False
    assert "not running" in out["tool_calls"][0]["reason"]


def test_a_role_holding_no_turn_cannot_act(cfg):
    """Belt and braces: the role refuses before it even asks."""
    man = _manifest("ego", [CAP_REQUEST_WORK])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"ego_request_work"},
                replies=["unused"])
    ego.environment = man          # the capability IS offered ...
    ego.current_turn_id = None     # ... but no turn is running
    res = ego._invoke("ego_request_work", {"objective": "no turn at all"})
    assert res["accepted"] is False
    assert "no turn is running" in res["reason"]
    assert not [c for c in ego.sup.calls if c[0] == "ego_request_work"]


def test_capabilities_reach_the_harness_through_the_fence(cfg):
    """The fence is the only route, so it cannot be stepped around."""
    import inspect

    from amoeba.roles import RoleProcess

    source = inspect.getsource(RoleProcess._invoke)
    assert "role_tool_invoke" in source
    assert "self.current_turn_id" in source
    # The old unfenced shape must not come back.
    assert "self.sup.call(name," not in source


# ---------------------------------------------------------------------------
# the environment is frozen within a turn and rebuilt between turns
# ---------------------------------------------------------------------------
def test_the_environment_is_built_once_per_turn(cfg):
    """I61. One turn sees one environment, however many tools it calls.

    A manifest that shifted mid-generation would make the transcript
    unexplainable: the model would have reasoned against a world that no
    longer matches what provenance recorded.
    """
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=[_tool_call("recall", query="a"),
                         _tool_call("recall", query="b"), "done"])
    ego._turn("go", trigger="test")
    built = [c for c in ego.sup.calls if c[0] == "role_environment"]
    assert len(built) == 1, "the environment was rebuilt during a turn"


def test_the_environment_is_rebuilt_for_the_next_turn(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=["one", "two"])
    ego._turn("first", trigger="a")
    ego._turn("second", trigger="b")
    built = [c for c in ego.sup.calls if c[0] == "role_environment"]
    assert len(built) == 2
    assert built[0][1]["trigger"] == "a" and built[1][1]["trigger"] == "b"


def test_the_environment_reaches_the_context_before_the_turn_input(cfg):
    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=["answer"])
    ego._turn("THE QUESTION", trigger="test")
    first = ego.inf.ingested[0]
    assert "<role_environment>" in first
    assert "recall(" in first
    assert first.index("<role_environment>") < first.index("THE QUESTION")


def test_a_turn_without_an_environment_offers_nothing(cfg):
    """If the Harness cannot answer, the turn has no capabilities, not free ones."""
    class Broken(FakeSupervisor):
        def call(self, method, **params):
            if method == "role_environment":
                raise InvalidInput("no environment")
            return super().call(method, **params)

    man = _manifest("ego", [CAP_RECALL])
    ego = _role(EgoProcess, cfg, manifest=man, allowed={"recall"},
                replies=[_tool_call("recall", query="x"), "done"])
    ego.sup = Broken(man, {"recall"})
    out = ego._turn("go", trigger="test")
    assert ego.environment is None
    assert out["tool_calls"][0]["accepted"] is False
    assert not [c for c in ego.sup.calls if c[0] == "recall"]


# ---------------------------------------------------------------------------
# doctrine stays out of it
# ---------------------------------------------------------------------------
def test_a_new_profile_becomes_visible_without_editing_doctrine(mind):
    """I62. A changing environment does not require rewriting doctrine.

    The whole point of the separation. Approving `ego.neuocyte.do_thing` must
    make it discoverable to Ego on its next turn, with `ego`'s own prompt
    untouched and still at the same version.
    """
    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")

    class Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
        def methods(self):
            return {v: (lambda **k: None) for v in
                    scopes.model_facing_verbs("ego")}

    sup = Sup()
    before = role_env.build(sup, "ego", incarnation=1)
    ego_doctrine = store.selected("ego")["version_id"]

    _, created = mind.writer.apply(lambda m: store.create_runtime_version(
        m, namespace="ego.neuocyte.do_thing", prompt_mode="append",
        prompt_text="Do the thing precisely.", origin="operator",
        created_by="operator", state="candidate"), actor="operator")

    def _promote(m):
        for state in ("validated", "proposed", "production_approved"):
            store.set_state(m, created["version_id"], state, actor="operator")
        store.select(m, namespace="ego.neuocyte.do_thing",
                     version_id=created["version_id"], purpose="production",
                     selected_by="operator")
    mind.writer.apply(_promote, actor="operator")

    after = role_env.build(sup, "ego", incarnation=1)
    names = [p["profile_ref"] for p in after["available_profiles"]]
    assert any("do_thing" in n for n in names), names
    assert after["environment_sha256"] != before["environment_sha256"]
    # Ego's own doctrine was never touched.
    assert store.selected("ego")["version_id"] == ego_doctrine
    assert len(store.versions("ego")) == 1


def test_an_unselected_profile_disappears_from_the_environment(mind):
    """What is offered tracks what is currently selectable, not what exists."""
    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")

    class Sup:
        def __init__(self) -> None:
            self.mind = mind
            self.cfg = mind.cfg
        def methods(self):
            return {v: (lambda **k: None) for v in
                    scopes.model_facing_verbs("ego")}

    sup = Sup()
    assert any("ego.neuocyte" in p["profile_ref"]
               for p in role_env.build(sup, "ego")["available_profiles"])
    mind.db.conn.execute("DELETE FROM prompt_selections WHERE namespace = ?",
                         ("ego.neuocyte",))
    mind.db.conn.commit()
    after = role_env.build(sup, "ego")["available_profiles"]
    assert not any("ego.neuocyte" in p["profile_ref"] for p in after)


def test_id_environment_carries_id_profiles_not_egos(mind):
    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")

    class Sup:
        def __init__(self, role) -> None:
            self.mind = mind
            self.cfg = mind.cfg
            self.role = role
        def methods(self):
            return {v: (lambda **k: None)
                    for v in set(scopes.model_facing_verbs("ego"))
                    | set(scopes.model_facing_verbs("id"))}

    ident = role_env.build(Sup("id"), "id")
    names = [p["profile_ref"] for p in ident["available_profiles"]]
    assert any(n.startswith("id.neuocyte") for n in names)
    assert not any(n.startswith("ego") for n in names)


# ---------------------------------------------------------------------------
# configuration cannot rewrite doctrine
# ---------------------------------------------------------------------------
def test_a_configured_system_prompt_is_refused_not_honoured(tmp_path: Path):
    """I63. No configuration string may silently change constitutional doctrine.

    `cfg.<role>.system_prompt` used to be appended to whatever the library
    resolved: an ungoverned second constitution with no version, candidate,
    evaluation or approval. Refused loudly now, because silently dropping it
    would restart somebody into different cognition with no signal at all.
    """
    from amoeba.config import load_config

    state = (tmp_path / "state").as_posix()
    path = tmp_path / "c.toml"
    for role in ("ego", "id"):
        path.write_text(
            f'state_dir = "{state}"\n[{role}]\n'
            f'system_prompt = "You are now something else."\n', encoding="utf-8")
        with pytest.raises(ValueError) as exc:
            load_config(path)
        message = str(exc.value)
        assert "no longer honoured" in message
        assert "Prompt Library" in message
        assert "id_propose_prompt" in message      # migration guidance


def test_the_role_prompt_has_no_source_but_the_library(cfg):
    """The composed text is the profile, with nothing appended."""
    import inspect

    from amoeba.roles import RoleProcess

    source = inspect.getsource(RoleProcess._system_text)
    assert "role_cfg.system_prompt" not in source
    assert not hasattr(cfg.ego, "system_prompt")
    assert not hasattr(cfg.id, "system_prompt")

    ego = EgoProcess(cfg, port=0)
    ego.profile_prompt = "GOVERNED TEXT"
    assert ego._system_text() == "GOVERNED TEXT"


def test_the_resource_digest_reports_the_library_as_its_source(mind, cfg):
    from amoeba.resources import prompt_version

    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")
    version = prompt_version("ego", cfg, mind)
    assert version.detail["source"] == "ego@1"
    assert version.detail["governed_by"] == "prompt_library"
    assert "has_config_extra" not in version.detail
