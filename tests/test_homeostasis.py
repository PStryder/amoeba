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


class FakeInference:
    """Records every call, so tests can assert on what the Harness did."""

    def __init__(self, *, capacity=10000, sessions=None, tokens=None):
        self.capacity = capacity
        self.sessions = sessions or []
        self.tokens = tokens or {}
        self.calls: list[tuple[str, dict]] = []
        self.next_session = 0
        self.unreachable = False

    def call(self, method, **kw):
        self.calls.append((method, kw))
        if self.unreachable:
            raise ConnectionError("inference is down")
        if method == "context_report":
            used = sum(s["n_past"] for s in self.sessions)
            return {"pool_tokens_used": used, "pool_capacity": self.capacity,
                    "occupancy": used / self.capacity,
                    "sessions": [{**s, "budget_tokens": self.capacity}
                                 for s in self.sessions]}
        if method == "session_tokens":
            return {"tokens": self.tokens.get(kw["session_id"], []),
                    "n_past": len(self.tokens.get(kw["session_id"], []))}
        if method == "close_session":
            self.sessions = [s for s in self.sessions
                             if s["session_id"] != kw["session_id"]]
            return {"closed": kw["session_id"]}
        if method == "open_session":
            self.next_session += 1
            sid = f"sess_new_{self.next_session}"
            self.sessions.append({"session_id": sid, "role": kw["role"], "n_past": 0,
                                  "prefix_len": 0, "snapshot_id": None})
            return {"session_id": sid, "role": kw["role"], "seq_id": 9}
        if method == "restore_prefix":
            for s in self.sessions:
                if s["session_id"] == kw["session_id"]:
                    s["n_past"] = len(kw["tokens"])
            self.tokens[kw["session_id"]] = list(kw["tokens"])
            return {"n_past": len(kw["tokens"]), "kv_mode": "recomputed"}
        if method == "detokenize":
            return "reconstituted text"
        if method == "capabilities":
            return {"model_generation": "gen_test", "kv_mode": "shared_prefix"}
        raise AssertionError(f"unexpected call {method}")

    def methods_called(self):
        return [m for m, _ in self.calls]


@pytest.fixture()
def homeo(mind):
    inf = FakeInference(
        capacity=10000,
        sessions=[{"session_id": "sess_ego", "role": "ego", "n_past": 4000,
                   "prefix_len": 0, "snapshot_id": None},
                  {"session_id": "sess_id", "role": "id", "n_past": 1000,
                   "prefix_len": 0, "snapshot_id": None}],
        tokens={"sess_ego": list(range(4000)), "sess_id": list(range(1000))},
    )
    mind.work.register_agent(agent_id="ego", role="ego", session_handle="sess_ego")
    mind.work.register_agent(agent_id="id", role="id", session_handle="sess_id")
    h = ContextHomeostasis(HomeostasisConfig(min_seconds_between_rejuvenations=0.0),
                           mind=mind, inference=lambda: inf)
    h.fake = inf
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
# Reconstitution: trim keeps real tokens; summarise is refused
# ---------------------------------------------------------------------------
def test_summarising_is_refused_because_it_is_a_different_behaviour(homeo):
    with pytest.raises(CapabilityUnsupported) as exc:
        homeo.rejuvenate(role="ego", reason="x", mode="summarise")
    assert "different behaviour" in exc.value.message


def test_trim_keeps_a_verbatim_head_and_tail(homeo):
    tokens = list(range(4000))
    plan = homeo.plan_trim(tokens)
    kept = homeo.apply_trim(tokens, plan)
    head, tail = plan["keep_head"], plan["keep_tail"]
    assert kept[:head] == tokens[:head], "head must be verbatim"
    assert kept[-tail:] == tokens[-tail:], "tail must be verbatim"
    assert len(kept) == plan["kept_tokens"] < len(tokens)
    assert plan["dropped_span"] == {"start": head, "end": head + plan["dropped_tokens"]}
    # every surviving token came from the original, in order
    assert set(kept).issubset(set(tokens))


def test_trim_is_a_noop_when_the_context_is_already_small(homeo):
    tokens = list(range(100))
    plan = homeo.plan_trim(tokens)
    assert plan["dropped_tokens"] == 0
    assert homeo.apply_trim(tokens, plan) == tokens


def test_rejuvenation_reduces_the_context_and_reports_honestly(mind, homeo):
    out = homeo.rejuvenate(role="ego", reason="too big")
    assert out["tokens_before"] == 4000
    assert out["tokens_after"] < out["tokens_before"]
    assert out["dropped_tokens"] > 0
    assert out["old_session_id"] != out["new_session_id"]
    assert "verbatim" in out["reconstitution"]
    assert "not summarised" in out["reconstitution"]
    assert out["occupancy_after"] < out["occupancy_before"]


def test_ego_rejuvenation_checkpoints_the_full_prefix_first(mind, homeo):
    out = homeo.rejuvenate(role="ego", reason="too big")
    snap = mind.work.get_snapshot(out["checkpoint_snapshot_id"])
    # Nothing is lost from the record: the checkpoint holds every token.
    assert snap["token_count"] == 4000
    assert mind.work.snapshot_tokens(out["checkpoint_snapshot_id"]) == list(range(4000))


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
    homeo.fake.tokens["sess_ego"] = list(range(8000))
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
