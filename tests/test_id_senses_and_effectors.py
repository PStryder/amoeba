"""Id's senses, Id's effectors, and the wall between Id and a neuocyte.

Three claims, tested at three different layers because they fail differently:

1. **Id can see.** One cheap call answers "what is the organism doing right
   now", in facts rather than verdicts, and it moves when the organism moves.
2. **Id can act, but only by asking.** Every effector goes through the Harness
   and leaves a receipt. Id requests, proposes, challenges and escalates; it
   does not mutate authoritative state because it concluded something.
3. **A neuocyte has no path to Id's hands.** Not a refused path -- an absent
   one.

The third is asserted against a **live connection holding the neuocyte's own
credential**, not by inspecting a list. A test that reads a registry proves
what the registry says; only a real call proves what the dispatcher does.
"""

from __future__ import annotations

import json
import sys

import pytest

from amoeba.rpc import RpcClient, read_or_create_token
from amoeba.scopes import ID_ONLY, id_only_verbs, scope_tables
from conftest import start_stack

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="live stack fixtures are Windows-only here")


@pytest.fixture()
def org(tmp_path):
    fsroot = tmp_path / "files"
    fsroot.mkdir()
    (fsroot / "a.txt").write_bytes(b"hello\n")
    extra = "\n".join([
        "[[filespace.roots]]", 'name = "mine"',
        f'path = "{fsroot.as_posix()}"', 'mode = "read_write"',
    ])
    s = start_stack(tmp_path, extra_toml=extra)
    s.fsroot = fsroot                      # type: ignore[attr-defined]

    def as_scope(scope: str) -> RpcClient:
        tok = read_or_create_token(s.cfg.scope_token_path(scope))
        c = RpcClient(s.cfg.supervisor_host, s.cfg.supervisor_port, tok, timeout=120)
        c.connect(retries=20, delay=0.25)
        return c

    s.as_scope = as_scope                  # type: ignore[attr-defined]
    s.id = as_scope("id")                  # type: ignore[attr-defined]
    s.neuocyte = as_scope("neuocyte")      # type: ignore[attr-defined]
    s.ego = as_scope("ego")                # type: ignore[attr-defined]
    yield s
    s.stop()


# ===========================================================================
# 1. Id can see
# ===========================================================================
REQUIRED_SECTIONS = (
    "pulse_id", "captured_at", "state_version", "run_id", "harness", "roles",
    "inference", "model_generation", "neuocytes", "work", "scheduler",
    "context_pressure", "resources", "pending_decisions", "failures",
    "storage", "attention",
)


def test_id_can_obtain_the_complete_bounded_pulse(org):
    p = org.id.call("system_pulse")
    missing = [k for k in REQUIRED_SECTIONS if k not in p]
    assert not missing, f"pulse is missing sections Id needs: {missing}"

    # Identity and the state it observed, so a conclusion can be anchored.
    assert p["pulse_id"].startswith("pls_")
    assert isinstance(p["state_version"], int)

    # Resource versions, configured and embodied, kept distinct.
    configured = p["resources"]["configured"]
    for kind in ("prompt.ego", "prompt.id", "tools.neuocyte", "security.policy",
                 "filespace.config", "store.schema"):
        assert kind in configured, f"no version reported for {kind}"
        assert len(configured[kind]["sha256"]) == 64
    embodied = p["resources"]["embodied"]
    assert "prompt.ego" in embodied and "prompt.id" in embodied
    assert embodied["prompt.ego"]["matches_configured"] in (True, False, None)

    # Enough scheduler and work detail to reason about capacity.
    assert set(p["scheduler"]) >= {"max_neuocytes", "in_use", "available"}
    assert set(p["work"]) >= {"by_status", "in_flight", "oldest_queued_age_seconds",
                              "retried_items", "blocked"}
    assert set(p["failures"]) >= {"last_5m", "last_1h", "watermark_seq"}
    assert "free_bytes" in p["storage"] or "error" in p["storage"]


def test_the_pulse_excludes_bulky_content(org):
    """Cheap to poll is the point; bulk would quietly end that.

    The pulse says where to look. Board posts, artifact bodies, memory claims
    and exception text all have their own interfaces, and putting any of them
    here would make Id's cheapest sense its most expensive one.
    """
    p = org.id.call("system_pulse")
    blob = json.dumps(p)
    assert len(blob) < 64_000, f"pulse is {len(blob)} bytes; it must stay small"
    assert p["contract"]["excluded"]
    for banned in ("board_posts", "artifacts", "memories", "traceback",
                   "log_lines", "stdout"):
        assert banned not in blob, f"pulse carries bulky field {banned!r}"


def test_the_pulse_is_cheap_and_cached(org):
    org.id.call("system_pulse")
    warm = org.id.call("system_pulse")
    assert warm["cached"] is True
    fresh = org.id.call("system_pulse", max_age_seconds=0)
    assert fresh["cached"] is False
    assert fresh["pulse_id"] != warm["pulse_id"]


def test_the_pulse_reports_observations_not_verdicts(org):
    """No field may read as a health judgement.

    `ego_unhealthy=true` would move the cognition into the Harness and leave
    Id agreeing with a number it cannot inspect. The pulse reports the
    heartbeat, the occupancy and the failure counts; deciding what they add up
    to is Id's job.
    """
    p = org.id.call("system_pulse")

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                low = k.lower()
                assert not any(w in low for w in
                               ("unhealthy", "healthy", "degraded", "failing",
                                "is_ok", "is_fine", "verdict", "severity",
                                "should_", "needs_", "recommend")), (
                    f"pulse field {here!r} states a judgement rather than an "
                    "observation")
                walk(v, here)
        elif isinstance(node, list):
            for item in node[:8]:
                walk(item, path)

    walk(p)
    # One labelled classification is allowed and is named; everything else is
    # a measurement. The pulse used to claim it made no judgements at all
    # while deriving `thinking` from a threshold -- two documented policies
    # contradicting each other, resolved in I126's favour because a mind that
    # cannot think cannot be the one to notice it cannot think (item 12).
    reports = p["contract"]["reports"]
    assert "thinking" in reports and "measurement" in reports
    thinking = p["roles"]["id"]["thinking"] if "roles" in p else None
    assert thinking is None or isinstance(thinking, bool)


def test_the_pulse_moves_when_work_moves(org):
    before = org.id.call("system_pulse", max_age_seconds=0)
    admitted = org.call("admit_work", objective="watch me appear",
                        work_class="user", origin_actor="pete")
    after = org.id.call("system_pulse", max_age_seconds=0)

    assert after["state_version"] > before["state_version"]
    q_before = before["work"]["by_status"].get("queued", 0)
    q_after = after["work"]["by_status"].get("queued", 0)
    assert q_after > q_before, "a newly admitted item did not show in the pulse"

    org.call("lease_work", neuocyte_id="nc_pulse", work_id=admitted["work_id"])
    leased = org.id.call("system_pulse", max_age_seconds=0)
    in_flight = {w["work_id"] for w in leased["work"]["in_flight"]}
    assert admitted["work_id"] in in_flight, "leased work is not reported in flight"


def test_the_pulse_reports_execution_mode_of_running_work(org):
    """Board-naive or informed is what makes later agreement interpretable."""
    naive = org.call("admit_work", objective="independent", work_class="maintenance",
                     origin_actor="id", board_access="none")
    org.call("lease_work", neuocyte_id="nc_naive", work_id=naive["work_id"])
    p = org.id.call("system_pulse", max_age_seconds=0)
    row = next(w for w in p["work"]["in_flight"] if w["work_id"] == naive["work_id"])
    assert row["board_access"] == "none"


def test_the_pulse_moves_when_failures_happen(org):
    before = org.id.call("system_pulse", max_age_seconds=0)
    with pytest.raises(Exception):
        org.call("file_write", root="mine", path="../escape.txt",
                 content="nope", actor="pete")
    after = org.id.call("system_pulse", max_age_seconds=0)

    b = before["failures"]["last_5m"].get("file_denied", 0)
    a = after["failures"]["last_5m"].get("file_denied", 0)
    assert a > b, (
        f"a refused path did not move the failure counter: {b} -> {a}. "
        "Id cannot notice a change in behaviour it cannot count.")
    assert after["failures"]["watermark_seq"] >= before["failures"]["watermark_seq"]


def test_the_pulse_reports_inference_and_model_generation(org):
    p = org.id.call("system_pulse", max_age_seconds=0)
    assert p["inference"]["reachable"] is True
    assert p["model_generation"], "no model generation reported"
    assert p["inference"]["max_sessions"] is not None


def test_a_resource_version_changes_when_the_resource_does(cfg, mind):
    """At the layer the guarantee lives: the digest, not the transport.

    A behavioural shift after a prompt change is a different problem from one
    with no resource change at all, and Id can only tell them apart if the
    digest actually tracks the text.

    This used to change the prompt through `cfg.ego.system_prompt`. That field
    is gone -- it was an ungoverned way to rewrite constitutional doctrine --
    so the change is made the only way it can now be made: a new governed
    version, approved and selected.
    """
    from amoeba.promptlib import bootstrap
    from amoeba.promptlib.store import PromptStore
    from amoeba.resources import all_versions, prompt_version

    store = PromptStore(mind)
    mind.writer.apply(lambda m: bootstrap.ingest(m, store), actor="bootstrap")
    before = prompt_version("ego", cfg, mind).sha256

    _, created = mind.writer.apply(lambda m: store.create_runtime_version(
        m, namespace="ego", prompt_mode="replace",
        prompt_text="An additional instruction.", origin="operator",
        created_by="operator", state="candidate"), actor="operator")

    # A candidate changes nothing: the digest still describes what is running.
    assert prompt_version("ego", cfg, mind).sha256 == before

    def _promote(m):
        for state in ("validated", "proposed", "production_approved"):
            store.set_state(m, created["version_id"], state, actor="operator")
        store.select(m, namespace="ego", version_id=created["version_id"],
                     purpose="production", selected_by="operator")
    mind.writer.apply(_promote, actor="operator")

    after = prompt_version("ego", cfg, mind).sha256
    assert after != before, "approving a role prompt did not change its digest"

    v = all_versions(cfg, mind)
    assert v["prompt.ego"]["sha256"] == after
    assert v["prompt.ego"]["detail"]["source"] == "ego@2"
    cfg.filespace.allow_multiply_linked = True
    assert all_versions(cfg, mind)["security.policy"]["sha256"] \
        != v["security.policy"]["sha256"]


# ===========================================================================
# 2. Id can act, but only by asking
# ===========================================================================
def test_id_can_invoke_every_authorised_effector(org):
    """Each one, through the Harness, with a receipt or an explicit outcome."""
    p = org.id.call("system_pulse", max_age_seconds=0)
    pid = p["pulse_id"]

    cited = org.id.call("id_cite_pulse", pulse_id=pid,
                        state_version=p["state_version"],
                        captured_at=p["captured_at"], reason="effector sweep")
    assert cited["pulse_id"] == pid and cited["cited_by"] == "id"

    finding = org.id.call("id_raise_finding", claim="queue depth is unusual",
                          kind="anomaly", confidence=0.4, pulse_id=pid)
    assert finding["receipt_id"] and finding["memory_id"]
    assert finding["memory_kind"] == "interpretation", (
        "a finding must not become an organism belief")

    corrected = org.id.call("id_propose_memory_correction",
                            memory_id=finding["memory_id"],
                            claim="queue depth is ordinary after all",
                            confidence=0.6, rationale="recounted",
                            pulse_id=pid)
    assert corrected["supersedes"] == finding["memory_id"]
    assert corrected["receipt_id"]

    investigation = org.id.call("id_request_investigation",
                                objective="check the queue", replicas=2,
                                independent=True, pulse_id=pid)
    assert investigation["independent"] is True
    assert len(investigation["admitted"]) + len(investigation["refused"]) == 2

    rejuv = org.id.call("id_request_rejuvenation", target_role="ego",
                        reason="context climbing", pulse_id=pid)
    assert "performed" in rejuv or "refused" in rejuv

    prompt = org.id.call("id_propose_prompt", target_role="ego",
                         prompt="You are Ego. Be brief.",
                         rationale="verbosity", pulse_id=pid)
    # A real Prompt Library candidate now, not a note: `ego@2` exists and is
    # awaiting approval. It used to be recorded as a suggestion because the
    # library refused every root version, which was an over-restriction.
    assert prompt["status"] == "candidate" and prompt["receipt_id"]
    assert prompt["profile_ref"] == "ego@2"
    assert prompt["local_version"] == 2
    assert prompt["candidate_sha256"] != prompt["current_sha256"]
    # Proposing is not adopting, and Id cannot adopt.
    tree = {n["namespace"]: n for n in org.call("prompt_tree")["nodes"]}
    assert tree["ego"]["selected"]["profile_ref"] == "ego@1"
    assert any(p["local_version"] == 2 for p in tree["ego"]["pending"])

    msg = org.id.call("id_message_ego", kind="notice",
                      message="your context is climbing", pulse_id=pid)
    assert msg["delivered"] is True

    esc = org.id.call("id_escalate_to_operator", summary="integrity needs a look",
                      severity="concern", recommended_action="run a deep verify",
                      pulse_id=pid)
    assert esc["receipt_id"] and esc["severity"] == "concern"


def test_id_can_request_cancellation_of_problematic_work(org):
    p = org.id.call("system_pulse", max_age_seconds=0)
    work = org.call("admit_work", objective="runaway", work_class="user",
                    origin_actor="pete")
    out = org.id.call("id_request_work_intervention", work_id=work["work_id"],
                      action="cancel", reason="looping", pulse_id=p["pulse_id"])
    assert out["action"] == "cancel" and out["receipt_id"]
    assert org.call("get_work", work_id=work["work_id"])["status"] == "cancelled"


def test_id_cannot_change_scheduler_policy_or_requeue(org):
    """Id asks for interventions; it does not run the scheduler.

    Requeue is absent on purpose: returning a leased item to the queue is lease
    expiry, which the supervision loop owns, and a second actor forcing it
    would race that loop.
    """
    work = org.call("admit_work", objective="x", work_class="user",
                    origin_actor="pete")
    with pytest.raises(Exception) as exc:
        org.id.call("id_request_work_intervention", work_id=work["work_id"],
                    action="requeue", reason="redo it")
    assert "cancel" in str(exc.value)

    # And no scheduler-policy verb is reachable at all.
    for verb in ("admit_work", "kill_all_neuocytes", "shutdown"):
        with pytest.raises(Exception) as e:
            org.id.call(verb)
        assert "unknown method" in str(e.value), (
            f"Id can reach {verb}, which is scheduler or lifecycle authority")


def test_consequential_id_actions_are_receipted_and_attributed(org):
    p = org.id.call("system_pulse", max_age_seconds=0)
    org.id.call("id_raise_finding", claim="attribution check", kind="risk",
                confidence=0.3, pulse_id=p["pulse_id"])
    org.id.call("id_escalate_to_operator", summary="attribution check",
                severity="notice", pulse_id=p["pulse_id"])
    org.id.call("id_propose_prompt", target_role="id", prompt="Be careful.",
                rationale="attribution check", pulse_id=p["pulse_id"])

    events = org.call("history", limit=600)
    kinds = [e["kind"] for e in events]
    for kind in ("id.finding_raised", "operator.escalation", "prompt.proposed"):
        assert kind in kinds, f"no durable record of {kind}"

    for e in events:
        if e["kind"] in ("id.finding_raised", "operator.escalation",
                         "prompt.proposed"):
            assert e["actor_id"] == "id", (
                f"{e['kind']} was not attributed to Id: {e['actor_id']}")
            payload = json.loads(e["payload_inline"])
            assert payload.get("pulse_id") == p["pulse_id"], (
                "the action does not record the telemetry it was formed from")

    assert org.call("verify_integrity", deep=True)["hash_chain_ok"] is True


def test_id_proposals_do_not_install_themselves(org):
    """A proposal is a proposal. Prompts especially."""
    p = org.id.call("system_pulse", max_age_seconds=0)
    before = p["resources"]["configured"]["prompt.ego"]["sha256"]
    org.id.call("id_propose_prompt", target_role="ego", prompt="Completely different.",
                rationale="test", pulse_id=p["pulse_id"])
    after = org.id.call("system_pulse", max_age_seconds=0)
    assert after["resources"]["configured"]["prompt.ego"]["sha256"] == before, (
        "proposing a prompt changed the configured prompt")
    assert after["resources"]["embodied"]["prompt.ego"]["sha256"] == \
        p["resources"]["embodied"]["prompt.ego"]["sha256"]


# ===========================================================================
# 3. A neuocyte has no path to Id's hands
# ===========================================================================
def test_the_neuocyte_scope_contains_no_id_only_verb(org):
    """The declaration, checked as data."""
    neuocyte = set(scope_tables()["neuocyte"])
    assert neuocyte.isdisjoint(id_only_verbs())
    assert "system_pulse" not in neuocyte
    assert set(scope_tables()["ego"]).isdisjoint(id_only_verbs()), (
        "Ego shares plumbing with Id but must not share its authority")


@pytest.mark.parametrize("verb", sorted(ID_ONLY) + ["system_pulse", "id_health"])
def test_a_neuocyte_cannot_invoke_an_id_only_verb_by_name(org, verb):
    """Guessing the name is the attack, so the test guesses the name.

    Over a live connection holding the neuocyte's own credential. The verb is
    not refused -- it does not exist on that connection, which is a stronger
    statement and a different code path from a permission check.
    """
    with pytest.raises(Exception) as exc:
        org.neuocyte.call(verb)
    assert "unknown method" in str(exc.value), (
        f"a neuocyte reached {verb} with something other than absence")


def test_a_neuocyte_cannot_enumerate_the_methods_it_lacks(org):
    """An unknown-method error must not be a discovery oracle.

    It used to list every method on the server, so a caller could learn the
    name of everything it was not allowed to call by asking for something that
    does not exist.
    """
    with pytest.raises(Exception) as exc:
        org.neuocyte.call("no_such_method_at_all")

    # The payload, not an attribute: RpcError puts its keyword details in
    # `.details`, so `getattr(exc, "known")` is always None and a test written
    # that way passes whether or not the table leaks.
    details = dict(getattr(exc.value, "details", {}) or {})
    assert "known" not in details, (
        f"the error enumerated the method table: {details.get('known')}")
    listed = [k for k, v in details.items() if isinstance(v, (list, tuple))]
    assert not listed, f"the error returned a list of methods under {listed}"

    blob = f"{exc.value} {details}"
    for verb in ID_ONLY:
        assert verb not in blob, f"the refusal disclosed {verb}"


def test_a_neuocyte_cannot_spoof_its_way_into_id_scope(org):
    """The scope is what the secret is, not what the caller says.

    There is no role, actor, caller or work-class field in the handshake, so
    there is nothing to forge. Passing identity-shaped arguments changes
    nothing because the verb still does not exist on this connection.
    """
    for kwargs in ({}, {"actor": "id"}, {"role": "id"}, {"caller": "id"},
                   {"from_role": "id"}, {"work_class": "maintenance"}):
        with pytest.raises(Exception) as exc:
            org.neuocyte.call("id_raise_finding", claim="spoofed", **kwargs)
        assert "unknown method" in str(exc.value)

    # A bad credential buys nothing either.
    bad = RpcClient(org.cfg.supervisor_host, org.cfg.supervisor_port,
                    "not-a-real-token", timeout=10)
    with pytest.raises(Exception):
        bad.connect(retries=1, delay=0.1)


def test_a_neuocytes_model_facing_tool_list_contains_no_id_verb(org):
    """The other surface a model can see: its tool schemas."""
    work = org.call("admit_work", objective="tools", work_class="user",
                    origin_actor="pete", sandbox_allowed=True)
    schemas = org.call("tool_schemas", work_id=work["work_id"], role="neuocyte")
    names = {t["name"] for t in schemas["tools"]}
    assert names.isdisjoint(id_only_verbs())
    assert "system_pulse" not in names
    blob = json.dumps(schemas)
    for verb in ID_ONLY:
        assert verb not in blob, f"the tool surface mentions {verb}"


def test_the_generic_tool_dispatcher_cannot_reach_an_id_verb(org):
    """tool_invoke is the only generic dispatcher a model can drive.

    It builds a neuocyte registry and nothing else, so an Id verb name reaching
    it finds no tool -- there is no path from a model-supplied string to an Id
    effector.
    """
    work = org.call("admit_work", objective="dispatch", work_class="user",
                    origin_actor="pete", sandbox_allowed=True)
    item = org.call("lease_work", neuocyte_id="nc_disp", work_id=work["work_id"])
    for verb in list(ID_ONLY) + ["system_pulse"]:
        out = org.call("tool_invoke", neuocyte_id="nc_disp",
                       work_id=work["work_id"],
                       fencing_token=item["fencing_token"], name=verb,
                       arguments={})
        assert out["accepted"] is False, f"tool_invoke executed {verb}"
        assert "no such tool" in out["reason"]
        assert verb not in out["available_tools"]


def test_ego_cannot_reach_ids_effectors(org):
    """Checked separately: shared plumbing is not shared authority."""
    for verb in list(ID_ONLY) + ["system_pulse"]:
        with pytest.raises(Exception) as exc:
            org.ego.call(verb)
        assert "unknown method" in str(exc.value)


def test_a_neuocyte_keeps_the_verbs_it_legitimately_needs(org):
    """Control: the boundary must discriminate, not just deny.

    If the neuocyte scope were empty every isolation test above would pass and
    the system would not work at all.
    """
    work = org.call("admit_work", objective="still works", work_class="user",
                    origin_actor="pete")
    item = org.neuocyte.call("lease_work", neuocyte_id="nc_ok",
                             work_id=work["work_id"])
    assert item and item["work_id"] == work["work_id"]
    assert org.neuocyte.call("tool_schemas", work_id=work["work_id"],
                             role="neuocyte")["tools"]
