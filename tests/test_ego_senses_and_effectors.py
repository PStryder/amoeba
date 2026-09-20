"""Ego's surface: what it can see, what it can ask for, and what it cannot touch.

Ego is the outward conversational interface. It interprets what the user wants
and introduces that objective into the rest of the organism. The architecture
deliberately does **not** require it to plan first: it may hand over a one-line
objective and let the work system, the blackboard and the neuocytes discover
structure during execution.

So these tests are about capability physics, not workflow. Nothing here asserts
how Ego must think. What they pin down is:

* which surfaces Ego can read (published cognition, never live scratch),
* that Ego asks for work rather than instantiating workers,
* that a mid-flight message is governed and refused where independence
  requires it,
* and that no role gains power merely because it can name a verb.

Every isolation claim is checked over a **live connection holding that role's
own credential**. Reading a table proves what the table says; only a call
proves what the dispatcher does.
"""

from __future__ import annotations

import json
import sys

import pytest

from amoeba.rpc import RpcClient, read_or_create_token
from amoeba.scopes import EGO_ONLY, ID_ONLY, ego_only_verbs, id_only_verbs, scope_tables
from conftest import start_stack

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="live stack fixtures are Windows-only here")

# Powers Ego must not have, from the architectural prohibition list.
FORBIDDEN_FOR_EGO = sorted(set(ID_ONLY) | {
    "system_pulse", "id_health",                    # Id's telemetry
    "context_rejuvenate", "retire_session",         # homeostasis execution
    "sandbox_create", "sandbox_run", "sandbox_files",
    "sandbox_read", "sandbox_write", "sandbox_destroy",   # live scratch
    "file_read", "file_write", "file_delete", "file_roots",
    "file_attach", "file_restore",                  # filespace / security
    "artifact_promote", "artifact_reject",          # promotion authority
    "admit_work", "cancel_work", "kill_all_neuocytes",    # scheduler authority
    "shutdown", "ensure_snapshot",
    "remember",                                     # belief authoring
    "tool_invoke",                                  # neuocyte execution path
})


@pytest.fixture()
def org(tmp_path):
    s = start_stack(tmp_path)

    def as_scope(scope: str) -> RpcClient:
        tok = read_or_create_token(s.cfg.scope_token_path(scope))
        c = RpcClient(s.cfg.supervisor_host, s.cfg.supervisor_port, tok, timeout=120)
        c.connect(retries=20, delay=0.25)
        return c

    s.ego = as_scope("ego")            # type: ignore[attr-defined]
    s.idc = as_scope("id")             # type: ignore[attr-defined]
    s.neuocyte = as_scope("neuocyte")  # type: ignore[attr-defined]
    yield s
    s.stop()


# ===========================================================================
# Senses
# ===========================================================================
def test_ego_can_read_the_cognitive_and_productive_surfaces(org):
    """Everything Ego needs to interpret a result and answer with it."""
    assert org.ego.call("recall") is not None
    assert "by_status" in org.ego.call("queue_stats")
    assert org.ego.call("artifact_list") == [] or True
    assert "posts" in org.ego.call("board_read", reader="ego", limit=5)
    assert org.ego.call("context_report")["pool_capacity"] > 0
    assert org.ego.call("history", limit=5) is not None

    ident = org.ego.call("ego_resource_identities")
    assert ident["model_generation"]
    assert ident["prompt.ego"]["configured"]["sha256"]
    assert "tools.neuocyte" in ident


def test_ego_sees_work_state_without_id_telemetry(org):
    """A productive view, not the organism's health.

    Ego needs to know what it asked for and how it is going. Handing it failure
    counters, context occupancy, storage pressure and resource digests would
    blur the outward interface into the inward monitor -- and would mean two
    components reasoning about organism health with no agreed owner.
    """
    requested = org.ego.call("ego_request_work", objective="something to watch")
    work_id = requested["admitted"][0]["work_id"]

    view = org.ego.call("ego_work_view")
    assert work_id in view["queued"] or work_id in view["running"]
    assert work_id in view["originated_by_ego"]
    assert set(view) >= {"by_status", "running", "queued", "blocked",
                         "originated_by_ego", "items"}

    # And none of Id's telemetry rides along.
    blob = json.dumps(view)
    for leaked in ("failures", "storage", "context_pressure", "pulse_id",
                   "vram_free_bytes", "supervision_passes", "security.policy"):
        assert leaked not in blob, f"the Ego work view carries Id telemetry: {leaked}"

    org.call("lease_work", neuocyte_id="nc_w", work_id=work_id)
    after = org.ego.call("ego_work_view")
    assert work_id in after["running"]


def test_ego_can_follow_a_result_from_work_to_answer(org):
    """The explicit path: work completes, Ego reads the finding."""
    requested = org.ego.call("ego_request_work", objective="produce a finding")
    work_id = requested["admitted"][0]["work_id"]
    item = org.call("lease_work", neuocyte_id="nc_r", work_id=work_id)
    org.call("complete_work", work_id=work_id, neuocyte_id="nc_r",
             fencing_token=item["fencing_token"],
             result={"finding": "the answer is 42", "confidence": 0.8})

    got = org.ego.call("get_work", work_id=work_id)
    assert got["status"] == "done"
    assert got["result"]["finding"] == "the answer is 42"


def test_ego_sees_proposal_evidence_not_scratch(org):
    """Artifacts reach Ego as immutable evidence, not as files in a sandbox."""
    if not org.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work = org.call("admit_work", objective="make something", work_class="user",
                    origin_actor="pete", sandbox_allowed=True)
    item = org.call("lease_work", neuocyte_id="nc_a", work_id=work["work_id"])
    org.call("tool_invoke", neuocyte_id="nc_a", work_id=work["work_id"],
             fencing_token=item["fencing_token"], name="write_file",
             arguments={"path": "work/out.py", "content": "x = 1\n"})
    prop = org.call("tool_invoke", neuocyte_id="nc_a", work_id=work["work_id"],
                    fencing_token=item["fencing_token"], name="propose_artifact",
                    arguments={"path": "work/out.py", "rationale": "useful"})
    art_id = prop["result"]["artifact_id"]

    evidence = org.ego.call("ego_artifact_evidence", artifact_id=art_id)
    assert evidence["sha256"] == prop["result"]["sha256"]
    assert evidence["content"] == "x = 1\n"
    assert evidence["evidence_available"] is True
    org.call("cancel_work", work_id=work["work_id"], reason="test")


@pytest.mark.parametrize("verb", ["sandbox_files", "sandbox_read", "sandbox_run",
                                  "sandbox_list", "sandbox_create"])
def test_ego_cannot_inspect_compute_sandbox_scratch(org, verb):
    """Running scratch is not a communication channel.

    Half-written scratch is not a claim anybody made. Reasoning over it would
    let Ego consume something no neuocyte ever published, with no authorship
    and no moment at which the worker stood behind it.
    """
    with pytest.raises(Exception) as exc:
        org.ego.call(verb)
    assert "unknown method" in str(exc.value)


# ===========================================================================
# Effectors
# ===========================================================================
def test_ego_requests_work_and_cannot_instantiate_a_worker(org):
    """Intent in; the Harness owns every execution condition."""
    out = org.ego.call("ego_request_work", objective="answer the question",
                       replicas=2, constraints="be quick")
    assert len(out["admitted"]) + len(out["refused"]) == 2
    assert out["receipt_id"]
    for admitted in out["admitted"]:
        row = org.call("get_work", work_id=admitted["work_id"])
        assert row["origin_actor"] == "ego"
        assert "be quick" in row["objective"]

    # There is no verb that starts a worker.
    for verb in ("dispatch_neuocyte", "spawn_neuocyte", "admit_work",
                 "lease_work", "kill_all_neuocytes"):
        with pytest.raises(Exception) as exc:
            org.ego.call(verb)
        assert "unknown method" in str(exc.value), (
            f"Ego can reach {verb}, which is execution authority")


def test_ego_can_request_independent_replication(org):
    """Board-naive by construction, so later agreement means something."""
    out = org.ego.call("ego_request_work", objective="check independently",
                       replicas=3, independent=True)
    assert out["board_access"] == "none"
    for admitted in out["admitted"]:
        assert org.call("get_work",
                        work_id=admitted["work_id"])["board_access"] == "none"

    with pytest.raises(Exception) as exc:
        org.ego.call("ego_request_work", objective="incoherent",
                     independent=True, board_access="read_write")
    assert "board-naive by definition" in str(exc.value)


def test_ego_can_message_eligible_work_and_the_worker_collects_it(org):
    requested = org.ego.call("ego_request_work", objective="open work")
    work_id = requested["admitted"][0]["work_id"]
    item = org.call("lease_work", neuocyte_id="nc_m", work_id=work_id)

    sent = org.ego.call("ego_work_message", work_id=work_id,
                        message="prefer the 2024 data", kind="clarification")
    assert sent["receipt_id"] and sent["delivered"] == "pending"

    # The objective itself is untouched: a message is an addition.
    assert "2024" not in org.call("get_work", work_id=work_id)["objective"]

    got = org.neuocyte.call("work_messages", work_id=work_id,
                            neuocyte_id="nc_m",
                            fencing_token=item["fencing_token"])
    assert got["count"] == 1
    assert got["messages"][0]["body"] == "prefer the 2024 data"
    assert got["messages"][0]["from_role"] == "ego"

    # Collection is recorded, and the message is not handed out twice.
    again = org.neuocyte.call("work_messages", work_id=work_id,
                              neuocyte_id="nc_m",
                              fencing_token=item["fencing_token"])
    assert again["count"] == 0
    kinds = [e["kind"] for e in org.call("history", limit=400)]
    assert "work.message_sent" in kinds and "work.message_consumed" in kinds


def test_board_naive_work_refuses_mid_flight_messages(org):
    """The whole point of admitting it board-naive.

    A clarification from the executive role would destroy exactly the
    independence the item was admitted for -- quietly, and in a way that still
    looks like independent replication afterwards.
    """
    out = org.ego.call("ego_request_work", objective="sealed", independent=True)
    work_id = out["admitted"][0]["work_id"]
    org.call("lease_work", neuocyte_id="nc_sealed", work_id=work_id)

    with pytest.raises(Exception) as exc:
        org.ego.call("ego_work_message", work_id=work_id, message="psst")
    assert "board-naive" in str(exc.value)

    kinds = [e["kind"] for e in org.call("history", limit=400)]
    assert "work.message_refused" in kinds, (
        "a refused message must be recorded; a refusal is a fact about how the "
        "organism governed itself")


def test_a_message_to_finished_work_is_refused(org):
    requested = org.ego.call("ego_request_work", objective="already done")
    work_id = requested["admitted"][0]["work_id"]
    item = org.call("lease_work", neuocyte_id="nc_f", work_id=work_id)
    org.call("complete_work", work_id=work_id, neuocyte_id="nc_f",
             fencing_token=item["fencing_token"], result={"finding": "done"})
    with pytest.raises(Exception) as exc:
        org.ego.call("ego_work_message", work_id=work_id, message="too late")
    assert "not live" in str(exc.value)


def test_ego_can_cancel_its_own_work_but_not_anyone_elses(org):
    """Cancellation stays a Harness act, and only over Ego's own requests."""
    mine = org.ego.call("ego_request_work", objective="mine")["admitted"][0]
    out = org.ego.call("ego_request_cancellation", work_id=mine["work_id"],
                       reason="changed my mind")
    assert out["receipt_id"]
    assert org.call("get_work", work_id=mine["work_id"])["status"] == "cancelled"

    theirs = org.call("admit_work", objective="not ego's", work_class="user",
                      origin_actor="pete")
    with pytest.raises(Exception) as exc:
        org.ego.call("ego_request_cancellation", work_id=theirs["work_id"],
                     reason="meddling")
    assert "only request cancellation of work it originated" in str(exc.value)
    assert org.call("get_work", work_id=theirs["work_id"])["status"] != "cancelled"


def test_ego_proposes_memory_rather_than_authoring_it(org):
    """Supersession, through the same governed path as everyone else.

    Ego is the component most exposed to a confident user, so it is the most
    likely to acquire a belief nobody checked.
    """
    first = org.ego.call("ego_propose_memory", claim="the port is 8080",
                         kind="belief", confidence=0.6)
    assert first["memory_id"] and first["receipt_id"]
    assert first["proposed_by"] == "ego"

    second = org.ego.call("ego_propose_memory", claim="the port is 9090",
                          kind="belief", confidence=0.8,
                          supersedes=first["memory_id"],
                          rationale="checked the config")
    assert second["supersedes"] == first["memory_id"]
    old = org.ego.call("get_memory", memory_id=first["memory_id"])
    assert old["status"] == "superseded"
    assert old["claim"] == "the port is 8080", "supersession rewrote the original"

    # And the raw authoring verb is gone from Ego's reach.
    with pytest.raises(Exception) as exc:
        org.ego.call("remember", kind="belief", claim="x", confidence=1.0,
                     created_by="ego")
    assert "unknown method" in str(exc.value)


def test_ego_can_reach_id_through_the_backchannel_and_ask_for_review(org):
    msg = org.ego.call("ego_message_id", kind="notice",
                       message="the user contradicted himself")
    assert msg["delivered"] is True

    review = org.ego.call("ego_request_id_review", subject="general",
                          question="is my recall drifting?")
    assert review["receipt_id"] and review["request_id"]
    kinds = [e["kind"] for e in org.call("history", limit=400)]
    assert "ego.review_requested" in kinds


def test_ego_actions_are_attributable(org):
    """what Ego requested -> what ran -> what came back, reconstructible."""
    out = org.ego.call("ego_request_work", objective="traceable work")
    work_id = out["admitted"][0]["work_id"]
    item = org.call("lease_work", neuocyte_id="nc_t", work_id=work_id)
    org.ego.call("ego_work_message", work_id=work_id, message="a clarification")
    org.neuocyte.call("work_messages", work_id=work_id, neuocyte_id="nc_t",
                      fencing_token=item["fencing_token"])
    org.call("complete_work", work_id=work_id, neuocyte_id="nc_t",
             fencing_token=item["fencing_token"], result={"finding": "ok"})

    events = org.call("history", limit=600)
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)

    requested = by_kind["work.requested_by_ego"][-1]
    assert requested["actor_id"] == "ego"
    assert work_id in json.loads(requested["payload_inline"])["admitted"]

    sent = by_kind["work.message_sent"][-1]
    assert sent["actor_id"] == "ego"
    consumed = by_kind["work.message_consumed"][-1]
    assert json.loads(consumed["payload_inline"])["work_id"] == work_id

    assert org.call("verify_integrity", deep=True)["hash_chain_ok"] is True


# ===========================================================================
# Prohibitions
# ===========================================================================
@pytest.mark.parametrize("verb", FORBIDDEN_FOR_EGO)
def test_ego_cannot_reach_a_prohibited_power(org, verb):
    """Id telemetry, scheduler policy, security, promotion, live scratch.

    Absence, not refusal: for an Ego connection these verbs do not exist.
    """
    with pytest.raises(Exception) as exc:
        org.ego.call(verb)
    assert "unknown method" in str(exc.value), (
        f"Ego reached {verb} with something other than absence")


def test_ego_cannot_widen_its_own_capabilities(org):
    """No verb grants capability, and the scope is not negotiable.

    The scope comes from the presented secret, so there is nothing in the
    protocol to renegotiate mid-connection.
    """
    for verb in ("register_scope", "grant", "set_scope", "elevate"):
        with pytest.raises(Exception) as exc:
            org.ego.call(verb)
        assert "unknown method" in str(exc.value)

    # A second handshake frame does not change anything either.
    id_token = read_or_create_token(org.cfg.scope_token_path("id"))
    org.ego._write({"token": id_token})          # type: ignore[attr-defined]
    with pytest.raises(Exception) as exc:
        org.ego.call("system_pulse")
    assert "unknown method" in str(exc.value), (
        "re-presenting another scope's token mid-connection widened Ego")


# ===========================================================================
# Cross-role isolation
# ===========================================================================
def test_the_three_scopes_are_disjoint_where_it_matters(org):
    t = scope_tables()
    assert set(t["ego"]).isdisjoint(id_only_verbs())
    assert set(t["id"]).isdisjoint(ego_only_verbs())
    assert set(t["neuocyte"]).isdisjoint(ego_only_verbs())
    assert set(t["neuocyte"]).isdisjoint(id_only_verbs())
    assert "system_pulse" not in t["ego"] and "system_pulse" not in t["neuocyte"]


@pytest.mark.parametrize("verb", sorted(EGO_ONLY))
def test_a_neuocyte_cannot_invoke_an_ego_only_verb_by_name(org, verb):
    with pytest.raises(Exception) as exc:
        org.neuocyte.call(verb)
    assert "unknown method" in str(exc.value)


@pytest.mark.parametrize("verb", sorted(EGO_ONLY))
def test_id_cannot_invoke_an_ego_only_verb(org, verb):
    """Id monitors Ego; it does not act as Ego."""
    with pytest.raises(Exception) as exc:
        org.idc.call(verb)
    assert "unknown method" in str(exc.value)


def test_a_neuocyte_cannot_spoof_ego_identity(org):
    """Role authority comes from the credential, not from a request field."""
    for kwargs in ({}, {"actor": "ego"}, {"role": "ego"}, {"from_role": "ego"},
                   {"origin_actor": "ego"}, {"requested_by": "ego"}):
        with pytest.raises(Exception) as exc:
            org.neuocyte.call("ego_request_work", objective="spoofed", **kwargs)
        assert "unknown method" in str(exc.value)


def test_a_neuocyte_cannot_message_work_as_ego(org):
    """The message's author is the authenticated scope, never a parameter."""
    requested = org.ego.call("ego_request_work", objective="target")
    work_id = requested["admitted"][0]["work_id"]
    org.call("lease_work", neuocyte_id="nc_s", work_id=work_id)
    with pytest.raises(Exception) as exc:
        org.neuocyte.call("ego_work_message", work_id=work_id,
                          message="from a neuocyte", from_role="ego")
    assert "unknown method" in str(exc.value)


def test_a_neuocyte_cannot_collect_another_work_items_messages(org):
    """Message collection is fenced like everything else on a work item."""
    a = org.ego.call("ego_request_work", objective="a")["admitted"][0]["work_id"]
    b = org.ego.call("ego_request_work", objective="b")["admitted"][0]["work_id"]
    item_a = org.call("lease_work", neuocyte_id="nc_a2", work_id=a)
    org.call("lease_work", neuocyte_id="nc_b2", work_id=b)
    org.ego.call("ego_work_message", work_id=a, message="for a only")

    with pytest.raises(Exception):
        org.neuocyte.call("work_messages", work_id=a, neuocyte_id="nc_b2",
                          fencing_token=item_a["fencing_token"])
    with pytest.raises(Exception):
        org.neuocyte.call("work_messages", work_id=a, neuocyte_id="nc_a2",
                          fencing_token=item_a["fencing_token"] + 1)

    got = org.neuocyte.call("work_messages", work_id=a, neuocyte_id="nc_a2",
                            fencing_token=item_a["fencing_token"])
    assert got["count"] == 1


def test_ego_keeps_the_verbs_it_legitimately_needs(org):
    """Control: the boundary must discriminate, not just deny."""
    assert org.ego.call("recall") is not None
    assert org.ego.call("ego_work_view") is not None
    assert org.ego.call("board_post", author="ego", author_kind="ego",
                        post_type="note", body="a note")["post_id"]
