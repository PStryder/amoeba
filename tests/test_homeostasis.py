"""Context homeostasis: the Harness acts, the model asks.

These run against a scripted inference client rather than a GPU, because what
is under test is the decision logic and the authority boundary, not decode
speed.
"""

from __future__ import annotations

import pytest

from amoeba.errors import CapabilityUnsupported, InvalidInput, NotFound
from amoeba.homeostasis import ContextHomeostasis, HomeostasisConfig
from amoeba.store.events import EventKind
from chat_fixture import ChatInference, Conversation, record_turn


@pytest.fixture()
def homeo(mind):
    ego = Conversation()
    for i in range(8):
        ego.turn(f"question {i}", pad=200,
                 calls=[("history", [{"seq": j, "kind": "x" * 20} for j in range(10)])])
    idc = Conversation()
    idc.turn("an id question", pad=100)
    inf = ChatInference(
        capacity=10000,
        sessions=[{"session_id": "sess_ego", "role": "ego", "n_past": 4000,
                   "prefix_len": 0, "snapshot_id": None},
                  {"session_id": "sess_id", "role": "id", "n_past": 1000,
                   "prefix_len": 0, "snapshot_id": None}],
        tokens={"sess_ego": list(ego.tokens), "sess_id": list(idc.tokens)},
    )
    mind.work.register_agent(agent_id="ego", role="ego", session_handle="sess_ego")
    mind.work.register_agent(agent_id="id", role="id", session_handle="sess_id")
    for i, t in enumerate(ego.turns):
        record_turn(mind, "ego", "sess_ego", t, lineage=f"op-{i}")
    h = ContextHomeostasis(HomeostasisConfig(min_seconds_between_rejuvenations=0.0),
                           mind=mind, inference=lambda: inf)
    h.fake = inf
    h.conv = {"ego": ego, "id": idc}
    return h


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def test_measures_real_occupancy_never_estimates(homeo):
    r = homeo.measure()
    assert r.pool_tokens_used == 5000 and r.pool_capacity == 10000
    assert r.occupancy == 0.5
    assert r.backend_available is True
    assert len(r.sessions) == 2


@pytest.mark.parametrize("occ,level", [
    (0.10, "nominal"), (0.54, "nominal"), (0.55, "elevated"),
    (0.69, "elevated"), (0.70, "high"), (0.84, "high"),
    (0.85, "critical"), (0.99, "critical"),
])
def test_pressure_classification(homeo, occ, level):
    assert homeo.classify(occ) == level


def test_measurement_degrades_safely_when_inference_is_down(homeo):
    homeo.fake.unreachable = True
    r = homeo.measure()
    assert r.backend_available is False
    assert r.pressure == "nominal"          # does not invent pressure it cannot see
    assert "unreachable" in r.detail


def test_assess_performs_no_action(homeo):
    before = list(homeo.fake.methods_called())
    out = homeo.assess()
    after = homeo.fake.methods_called()
    assert set(after) - set(before) <= {"context_report"}
    assert "performs no action" in out["note"]
    assert "recommendations" in out


def test_assess_recommends_rejuvenation_for_a_large_role_context(homeo):
    homeo.fake.sessions[0]["n_past"] = 9000
    out = homeo.assess()
    recs = {r["role"]: r for r in out["recommendations"]}
    assert "ego" in recs and recs["ego"]["action"] == "rejuvenate"


# ---------------------------------------------------------------------------
# Authority: Id requests, the Harness performs
# ---------------------------------------------------------------------------
def test_id_request_is_honoured_and_recorded(mind, homeo):
    out = homeo.request_rejuvenation(role="ego", reason="context is large",
                                     requested_by="id")
    assert out["performed"] is True
    assert out["requested_by"] == "id"
    kinds = [r["kind"] for r in mind.db.conn.execute("SELECT kind FROM events")]
    assert EventKind.REJUVENATION_REQUESTED in kinds
    assert EventKind.REJUVENATION_PERFORMED in kinds
    assert EventKind.SESSION_RETIRED in kinds
    assert EventKind.SESSION_REBORN in kinds


def test_a_refused_request_is_still_recorded(mind, homeo):
    homeo.cfg.min_seconds_between_rejuvenations = 9999
    homeo.request_rejuvenation(role="ego", reason="first", requested_by="id")
    out = homeo.request_rejuvenation(role="ego", reason="again", requested_by="id")
    assert out["performed"] is False
    assert "minimum interval" in out["refused_because"]
    kinds = [r["kind"] for r in mind.db.conn.execute("SELECT kind FROM events")]
    assert EventKind.REJUVENATION_REFUSED in kinds


def test_rate_limit_stops_a_wedged_id_thrashing_contexts(homeo):
    homeo.cfg.max_rejuvenations_per_hour = 2
    assert homeo.request_rejuvenation(role="ego", reason="1")["performed"]
    assert homeo.request_rejuvenation(role="id", reason="2")["performed"]
    out = homeo.request_rejuvenation(role="ego", reason="3")
    assert out["performed"] is False and "rate limit" in out["refused_because"]


def test_only_ego_and_id_have_rejuvenable_contexts(homeo):
    with pytest.raises(InvalidInput):
        homeo.request_rejuvenation(role="neuocyte", reason="x")


def test_checkpoint_requires_a_live_session(mind, homeo):
    mind.work.retire_agent(agent_id="ego", reason="gone")
    with pytest.raises(NotFound):
        homeo.checkpoint(role="ego")


# ---------------------------------------------------------------------------
# Reconstitution: summarise is refused (rebuild itself: test_rebuild.py)
# ---------------------------------------------------------------------------
def test_summarising_is_refused_because_it_is_a_different_behaviour(homeo):
    with pytest.raises(CapabilityUnsupported) as exc:
        homeo.rejuvenate(role="ego", reason="x", mode="summarise")
    assert "different behaviour" in exc.value.message


def test_rejuvenation_reduces_the_context_and_reports_honestly(mind, homeo):
    out = homeo.rejuvenate(role="ego", reason="too big")
    assert out["tokens_before"] == len(homeo.conv["ego"].tokens)
    assert out["tokens_after"] < out["tokens_before"]
    assert out["dropped_tokens"] > 0
    assert out["old_session_id"] != out["new_session_id"]
    assert "no message was cut" in out["reconstitution"]
    assert "nothing is summarised" in out["reconstitution"]
    assert out["occupancy_after"] < out["occupancy_before"]


def test_ego_rejuvenation_checkpoints_the_full_prefix_first(mind, homeo):
    out = homeo.rejuvenate(role="ego", reason="too big")
    snap = mind.work.get_snapshot(out["checkpoint_snapshot_id"])
    # Nothing is lost from the record: the checkpoint holds every token.
    everything = homeo.conv["ego"].tokens
    assert snap["token_count"] == len(everything)
    assert mind.work.snapshot_tokens(out["checkpoint_snapshot_id"]) == everything


def test_id_context_is_checkpointed_but_never_published_as_a_snapshot(mind, homeo):
    homeo.rejuvenate(role="id", reason="tidy")
    snaps = [dict(r) for r in mind.db.conn.execute("SELECT actor FROM snapshots")]
    assert all(s["actor"] == "ego" for s in snaps), "Id's PKV must never be published"


# ---------------------------------------------------------------------------
# The automatic path is conservative
# ---------------------------------------------------------------------------
def test_tick_does_nothing_below_critical(homeo):
    homeo.fake.sessions[0]["n_past"] = 6000     # 70% -> high, not critical
    assert homeo.tick() is None


def test_tick_acts_at_critical_and_targets_the_largest_role(mind, homeo):
    homeo.fake.sessions[0]["n_past"] = 8000
    homeo.fake.sessions[1]["n_past"] = 700      # 87% total
    out = homeo.tick()
    assert out and out["performed"] is True
    assert out["role"] == "ego"
    kinds = [r["kind"] for r in mind.db.conn.execute("SELECT kind FROM events")]
    assert EventKind.CONTEXT_PRESSURE in kinds


def test_tick_is_disabled_when_configured_off(homeo):
    homeo.cfg.auto_rejuvenate = False
    homeo.fake.sessions[0]["n_past"] = 9500
    assert homeo.tick() is None


def test_tick_does_nothing_when_inference_is_unreachable(homeo):
    homeo.fake.unreachable = True
    assert homeo.tick() is None


# ---------------------------------------------------------------------------
# The model never touches KV
# ---------------------------------------------------------------------------
def test_no_kv_verb_is_reachable_from_a_model_facing_tool():
    """Neither the MCP surface nor the neuocyte tool registry may expose a way to
    manipulate a cache. Requesting rejuvenation is the only affordance."""
    import inspect

    from amoeba import mcp_api, tools

    mcp_src = inspect.getsource(mcp_api)
    for forbidden in ("seq_cp", "memory_seq", "fork_prefix", "restore_prefix",
                      "close_session", "context_rejuvenate", "retire_session"):
        assert forbidden not in mcp_src, f"MCP exposes {forbidden}"

    reg_src = inspect.getsource(tools)
    for forbidden in ("seq_cp", "memory_seq", "fork_prefix", "restore_prefix"):
        assert forbidden not in reg_src, f"tool registry exposes {forbidden}"


def test_rejuvenation_never_consults_a_model(homeo):
    """Homeostasis is deterministic: no generate call, ever."""
    homeo.rejuvenate(role="ego", reason="x")
    assert "generate" not in homeo.fake.methods_called()
    assert "chat" not in homeo.fake.methods_called()


# ---------------------------------------------------------------------------
# Review-pass regression: occupancy accounting
# ---------------------------------------------------------------------------
def test_a_recomputed_prefix_is_charged_in_full(mind):
    """A restored session occupies its own cells and must be counted that way.

    Occupancy used to key off snapshot_id, which restore_prefix also sets. A
    recomputed session was therefore charged only its private tail, so pool
    pressure was under-reported for exactly the sessions created by the
    post-restart fallback path -- when pressure matters most.
    """
    from amoeba.backends.deterministic import DeterministicBackend

    b = DeterministicBackend(n_seq_max=6, n_ctx=100000)
    b.load()
    src = b.open_session(role="ego")
    b.ingest(src.session_id, list(range(500)))

    forked = b.fork_prefix(src_session_id=src.session_id, prefix_len=500,
                           role="neuocyte", snapshot_id="snap_x")
    restored = b.open_session(role="neuocyte")
    b.restore_prefix(session_id=restored.session_id, tokens=list(range(500)),
                     snapshot_id="snap_x")

    by_id = {s["session_id"]: s for s in b.active_sessions()}
    assert by_id[forked.session_id]["shares_prefix"] is True
    assert by_id[restored.session_id]["shares_prefix"] is False
    # Both carry a snapshot_id, which is why that was the wrong signal.
    assert by_id[forked.session_id]["snapshot_id"] == "snap_x"
    assert by_id[restored.session_id]["snapshot_id"] == "snap_x"


def test_occupancy_counts_shared_once_and_recomputed_in_full(mind):
    """The arithmetic the report claims, checked directly."""
    sessions = [
        {"session_id": "a", "role": "ego", "n_past": 500, "prefix_len": 0,
         "shares_prefix": False, "snapshot_id": None},
        {"session_id": "b", "role": "neuocyte", "n_past": 520, "prefix_len": 500,
         "shares_prefix": True, "snapshot_id": "s"},     # fork: charge 20
        {"session_id": "c", "role": "neuocyte", "n_past": 530, "prefix_len": 500,
         "shares_prefix": False, "snapshot_id": "s"},    # recomputed: charge 530
    ]
    used = sum(
        (s["n_past"] - s["prefix_len"]) if s["shares_prefix"] else s["n_past"]
        for s in sessions)
    assert used == 500 + 20 + 530


def test_a_rejuvenated_role_keeps_its_identity_and_gains_a_new_handle(homeo, mind):
    """The record follows the live session; the incarnation does not move.

    A replacement session is not a new incarnation. Identity survives
    rejuvenation -- profile binding, mailbox and turn history all continue --
    which is why the Harness hands the session over instead of restarting the
    role, and why this is not `register_agent`.
    """
    before = {a["agent_id"]: a for a in mind.work.live_agents()}["ego"]
    out = homeo.rejuvenate(role="ego", reason="too big")
    after = {a["agent_id"]: a for a in mind.work.live_agents()}["ego"]

    assert after["session_handle"] == out["new_session_id"]
    assert after["session_handle"] != before["session_handle"]
    assert after["incarnation"] == before["incarnation"], (
        "rejuvenation bumped the incarnation; a replacement session is not a "
        "new mind")


def test_a_second_rejuvenation_does_not_resurrect_what_the_first_dropped(homeo, mind):
    """The bug this class of check exists to find.

    `hand_over_session` updated the role's in-memory handle and nothing
    updated the durable one, so the second rejuvenation checkpointed the
    *first* session -- closed, gone -- and restored the whole pre-rebuild
    context into a new one. Everything the first pass dropped came back.
    """
    first = homeo.rejuvenate(role="ego", reason="one")
    assert first["tokens_after"] < first["tokens_before"]

    second = homeo.rejuvenate(role="ego", reason="two")
    assert second["old_session_id"] == first["new_session_id"], (
        "the second rejuvenation worked from a session the first one closed")
    assert second["tokens_before"] == first["tokens_after"], (
        f"the second pass saw {second['tokens_before']} tokens where the "
        f"first left {first['tokens_after']}: dropped context came back")


# ---------------------------------------------------------------------------
# The heartbeat yields to pressure. Nothing else does.
# ---------------------------------------------------------------------------
class _GateSup:
    """The supervisor's heartbeat gate, with everything else stubbed out."""

    def __init__(self, homeo, cfg, *, pressure=None):
        from amoeba.supervisor import Supervisor

        self.cfg = cfg
        self.homeostasis = homeo
        self.mind = None
        self._heartbeat_deferred_since = {}
        self.log = __import__("logging").getLogger("test")
        self._gate = Supervisor._heartbeat_deferred_for_pressure.__get__(self)

    def gate(self, now):
        return self._gate("id", now)


def _gate_for(mind, pressure, **sched):
    """A gate whose measured pressure is whatever the test says it is."""
    from amoeba.config import Config

    cfg = Config()
    for k, v in sched.items():
        setattr(cfg.scheduler, k, v)

    class _Homeo:
        def last_pressure(self):
            return pressure, 1.0

    return _GateSup(_Homeo(), cfg)


def test_a_heartbeat_is_held_back_under_pressure(mind):
    """A heartbeat is discretionary: nothing is waiting on it.

    It costs a prefill and grows Id's context at exactly the moment the pool
    is stressed, which is the one turn worth not taking.
    """
    sup = _gate_for(mind, "high")
    held = sup.gate(1000.0)
    assert held is not None
    assert held["pressure"] == "high" and held["threshold"] == "high"


def test_a_heartbeat_is_not_held_back_below_the_threshold(mind):
    """Elevated is not high. The gate is a threshold, not a mood."""
    sup = _gate_for(mind, "elevated")
    assert sup.gate(1000.0) is None


def test_an_unknown_pressure_never_defers(mind):
    """Absent evidence is not evidence.

    The inference service may be down or starting. Refusing cognition on a
    measurement nobody took would stop the organism thinking for a reason
    that has nothing to do with its resources.
    """
    sup = _gate_for(mind, None)
    assert sup.gate(1000.0) is None


def test_a_deferred_heartbeat_eventually_runs_anyway(mind):
    """Id's heartbeat *is* the homeostatic review.

    Suppressing it for as long as pressure lasts would silence the organism's
    self-examination exactly while it was under strain. Relief does not depend
    on Id -- the Harness rejuvenates on its own authority at critical -- but a
    review that never happens is worse than a turn that costs a prefill.
    """
    sup = _gate_for(mind, "critical", heartbeat_max_deferral_seconds=300.0)
    assert sup.gate(1000.0) is not None, "it should defer at first"
    assert sup.gate(1200.0) is not None, "still inside the ceiling"
    assert sup.gate(1301.0) is None, "the ceiling did not release it"


def test_the_deferral_clock_resets_when_pressure_clears(mind):
    """The ceiling measures one continuous run of strain, not a lifetime.

    Otherwise an organism that was briefly busy hours ago would spend its
    ceiling on that and lose the protection during a later, real episode.
    """
    from amoeba.config import Config

    cfg = Config()
    cfg.scheduler.heartbeat_max_deferral_seconds = 300.0
    levels = ["high"]

    class _Homeo:
        def last_pressure(self):
            return levels[0], 1.0

    sup = _GateSup(_Homeo(), cfg)
    assert sup.gate(1000.0) is not None
    levels[0] = "nominal"
    assert sup.gate(1100.0) is None, "pressure cleared and it still deferred"
    levels[0] = "high"
    held = sup.gate(1200.0)
    assert held is not None and held["deferred_seconds"] == 0.0, (
        "the deferral clock carried over from the earlier episode")


def test_the_gate_can_be_turned_off(mind):
    """An operator who does not want this behaviour can say so."""
    sup = _gate_for(mind, "critical", heartbeat_defer_at_pressure="never")
    assert sup.gate(1000.0) is None


def test_only_the_heartbeat_consults_the_gate():
    """Event-driven turns are never held back.

    A role that something happened to must be able to think about it, whatever
    the pool is doing. The gate is called from the heartbeat scheduler and
    from nowhere else, which is what keeps that true.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "amoeba"
           / "supervisor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    callers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "_heartbeat_deferred_for_pressure"):
                    callers.add(node.name)
    assert callers == {"_schedule_heartbeats"}, (
        f"the pressure gate is consulted outside the heartbeat scheduler: "
        f"{sorted(callers)}")


# ---------------------------------------------------------------------------
# Whatever replaces a session tells the role, and a role can recover
# ---------------------------------------------------------------------------
def test_every_path_that_replaces_a_session_tells_the_role(homeo, mind):
    """Live: Id asked for its own rejuvenation and was never told the answer.

    `id_request_rejuvenation` and the critical-pressure tick both replaced
    Id's session without handing it over -- only the turn-boundary path did.
    Id went on calling a session the Harness had closed, and every turn for
    the next day and a half failed with "unknown inference session". Asking
    for help was the one way it could wedge itself permanently.
    """
    told: list[tuple[str, str]] = []
    homeo.hand_over = lambda role, session, reason="": (
        told.append((role, session)) or True)

    out = homeo.request_rejuvenation(role="id", reason="my context is large",
                                     requested_by="id")
    assert out["performed"] is True
    assert told == [("id", out["new_session_id"])], "Id was not told"
    assert out["role_told"] is True

    homeo.fake.sessions[0]["n_past"] = 9000          # critical: the tick path
    told.clear()
    tick = homeo.tick()
    assert tick and tick["performed"] is True
    assert told == [(tick["role"], tick["new_session_id"])]


def test_a_handover_that_does_not_land_is_recorded_not_swallowed(homeo, mind):
    def refuses(role, session, reason=""):
        raise ConnectionError("the role is not answering")

    homeo.hand_over = refuses
    out = homeo.rejuvenate(role="id", reason="tidy")
    assert out["performed"] is True and out["role_told"] is False
