"""Per-actor context budgets, and the physical pool they share.

Three numbers used to do this job badly. `arbiter.max_prompt_tokens` was one
global scalar with no idea who was asking; `[ego]/[id].max_context_tokens`
looked like ceilings and reached only the dashboard; and `role_context_high`
measured every role against the whole KV pool, so the proactive threshold sat
six times above the hard refusal and rejuvenation happened by collision.

What is asserted here is the separation that fixes it:

* **cognitive policy** -- `context_budget_tokens` and `budget_basis`, set when
  a session is created and never moved afterwards;
* **physical cost** -- derived from backend state, `shares_prefix ? private :
  total`, which is a different question with a different answer;
* **admission** -- measured for what exists, estimated for what does not, with
  a reserve held back for the decode that has not happened yet.

The case that forces the separation is the fork fallback: an Ego-derived
worker keeps its private-growth allowance when forking fails, and the
recomputed prefix is still charged in full. A single flag could not say both.
"""

from __future__ import annotations

import pytest

from amoeba.arbiter import Arbiter, ResourceSnapshot
from amoeba.backends.deterministic import DeterministicBackend
from amoeba.config import ArbiterConfig, Config
from amoeba.errors import ResourceExhausted


# ===========================================================================
# The session owns its policy
# ===========================================================================
@pytest.fixture()
def backend():
    # Large enough to hold the prefixes these tests build; the point here is
    # the budget arithmetic, not the simulated pool.
    b = DeterministicBackend(n_ctx=131072, n_seq_max=8)
    b.load()
    return b


def test_a_role_session_is_bounded_by_its_own_ceiling_not_a_global_one(backend):
    """Ego's ceiling is Ego's, and it is far above the old global 6144."""
    arb = Arbiter(ArbiterConfig())
    sess = backend.open_session(role="ego", context_budget_tokens=16384,
                                budget_basis="total")
    backend.ingest(sess.session_id, list(range(10000)))

    # 10000 would have been refused by the old global ceiling of 6144.
    assert sess.n_past == 10000
    out = arb.clamp_inference(prompt_tokens=sess.budgeted_tokens, max_tokens=64,
                              deadline=None,
                              budget_tokens=sess.context_budget_tokens,
                              budget_basis=sess.budget_basis)
    assert out["max_tokens"] == 64

    backend.ingest(sess.session_id, list(range(7000)))
    with pytest.raises(ResourceExhausted) as exc:
        arb.clamp_inference(prompt_tokens=sess.budgeted_tokens, max_tokens=64,
                            deadline=None,
                            budget_tokens=sess.context_budget_tokens,
                            budget_basis=sess.budget_basis)
    assert exc.value.details["max_prompt_tokens"] == 16384


def test_ids_ceiling_is_independent_of_egos(backend):
    """Changing one role's budget must not move the other's.

    They were a single global scalar; that they are now separate is the whole
    point, and a shared default would hide it.
    """
    arb = Arbiter(ArbiterConfig())
    ego = backend.open_session(role="ego", context_budget_tokens=16384)
    idd = backend.open_session(role="id", context_budget_tokens=8192)
    backend.ingest(ego.session_id, list(range(9000)))
    backend.ingest(idd.session_id, list(range(9000)))

    # The same occupancy: allowed for Ego, over budget for Id.
    arb.clamp_inference(prompt_tokens=ego.budgeted_tokens, max_tokens=8,
                        deadline=None, budget_tokens=ego.context_budget_tokens)
    with pytest.raises(ResourceExhausted):
        arb.clamp_inference(prompt_tokens=idd.budgeted_tokens, max_tokens=8,
                            deadline=None, budget_tokens=idd.context_budget_tokens)


def test_an_unbudgeted_session_falls_back_rather_than_becoming_unlimited(backend):
    """A session nobody gave a policy is a bug, not a privilege."""
    arb = Arbiter(ArbiterConfig())
    sess = backend.open_session(role="neuocyte")
    assert sess.context_budget_tokens is None
    backend.ingest(sess.session_id, list(range(7000)))
    with pytest.raises(ResourceExhausted) as exc:
        arb.clamp_inference(prompt_tokens=sess.budgeted_tokens, max_tokens=8,
                            deadline=None,
                            budget_tokens=sess.context_budget_tokens)
    assert exc.value.details["max_prompt_tokens"] == ArbiterConfig().max_prompt_tokens
    assert exc.value.details["budget_source"] == "global default"


# ===========================================================================
# An inherited prefix is not the worker's own growth
# ===========================================================================
def test_a_forked_worker_is_not_charged_for_the_prefix_it_inherited(backend):
    """The failure this design exists to prevent.

    Ego holds 10000 tokens; an Ego-derived worker is allowed 6144 of private
    growth. Judging its *total* against that allowance refuses it on its first
    generate, before it has thought anything at all.
    """
    arb = Arbiter(ArbiterConfig())
    ego = backend.open_session(role="ego", context_budget_tokens=16384)
    backend.ingest(ego.session_id, list(range(10000)))

    worker = backend.fork_prefix(src_session_id=ego.session_id, prefix_len=10000,
                                 role="neuocyte", context_budget_tokens=6144,
                                 budget_basis="private_growth")
    assert worker.n_past == 10000 and worker.prefix_len == 10000
    assert worker.private_tokens == 0
    assert worker.budgeted_tokens == 0, "the inherited prefix counted against it"

    # It can work, and it is bounded by its own growth rather than by what it
    # inherited.
    arb.clamp_inference(prompt_tokens=worker.budgeted_tokens, max_tokens=64,
                        deadline=None, budget_tokens=worker.context_budget_tokens,
                        budget_basis=worker.budget_basis)
    backend.ingest(worker.session_id, list(range(7000)))
    assert worker.private_tokens == 7000
    with pytest.raises(ResourceExhausted):
        arb.clamp_inference(prompt_tokens=worker.budgeted_tokens, max_tokens=8,
                            deadline=None,
                            budget_tokens=worker.context_budget_tokens,
                            budget_basis=worker.budget_basis)


def test_a_shared_prefix_is_charged_once_but_a_recomputed_one_in_full(backend):
    """Cost is measured from what the backend did, not from the policy.

    The same worker, the same allowance, two different physical outcomes. A
    design that answered both questions with one flag could not express this.
    """
    ego = backend.open_session(role="ego", context_budget_tokens=16384)
    backend.ingest(ego.session_id, list(range(10000)))

    forked = backend.fork_prefix(src_session_id=ego.session_id, prefix_len=10000,
                                 role="neuocyte", context_budget_tokens=6144,
                                 budget_basis="private_growth")
    backend.ingest(forked.session_id, list(range(500)))

    cold = backend.open_session(role="neuocyte", context_budget_tokens=6144,
                                budget_basis="private_growth")
    backend.restore_prefix(session_id=cold.session_id, tokens=list(range(10000)))
    backend.ingest(cold.session_id, list(range(500)))

    assert forked.shares_prefix is True and cold.shares_prefix is False
    assert forked.charged_tokens == 500, "a shared prefix was charged twice"
    assert cold.charged_tokens == 10500, "a recomputed prefix was treated as free"

    # ...and the cognitive allowance is identical for both, because the
    # fallback is a performance event and not a policy change.
    assert forked.budgeted_tokens == cold.budgeted_tokens == 500
    assert forked.context_budget_tokens == cold.context_budget_tokens == 6144


def test_a_cold_worker_is_judged_on_everything_it_holds(backend):
    """A maintenance worker inherits nothing, so total is what it means."""
    arb = Arbiter(ArbiterConfig())
    sess = backend.open_session(role="neuocyte", context_budget_tokens=4096,
                                budget_basis="total")
    backend.ingest(sess.session_id, list(range(5000)))
    assert sess.budgeted_tokens == 5000
    with pytest.raises(ResourceExhausted) as exc:
        arb.clamp_inference(prompt_tokens=sess.budgeted_tokens, max_tokens=8,
                            deadline=None,
                            budget_tokens=sess.context_budget_tokens,
                            budget_basis=sess.budget_basis)
    assert exc.value.details["max_prompt_tokens"] == 4096


# ===========================================================================
# The pool everybody shares
# ===========================================================================
def _snap(**kw):
    base = dict(kv_pool_capacity=49152, kv_tokens_used=0, outstanding_work=0)
    base.update(kw)
    return ResourceSnapshot(**base)


def test_admission_refuses_when_the_pool_has_no_room():
    """The check that did not exist before this pass.

    Admission weighed queue depth and slot reservations and read no token
    figure at all, so aggregate overcommit was prevented only by the numbers
    happening to be small.
    """
    arb = Arbiter(ArbiterConfig())
    ok = arb.admit(work_class="user", snapshot=_snap(kv_tokens_used=1000))
    assert ok.admitted, ok.reason

    full = arb.admit(work_class="user", snapshot=_snap(kv_tokens_used=41000))
    assert not full.admitted
    assert "KV pool" in full.reason


def test_the_reserve_is_held_back_from_new_work_only():
    """It is a margin for decode, not unavailable memory.

    A prompt that fits the pool exactly has left nowhere for its own answer,
    so admission stops short of the edge. Nothing stops a session already
    running from growing into that margin.
    """
    cfg = ArbiterConfig()
    arb = Arbiter(cfg)
    capacity = 49152
    reserve = int(capacity * cfg.kv_admission_reserve_fraction)
    usable = capacity - reserve
    want = cfg.ego_neuocyte_budget_tokens

    just_fits = arb.kv_admission(work_class="user",
                                 snapshot=_snap(kv_tokens_used=usable - want))
    assert just_fits["admit"]
    over = arb.kv_admission(work_class="user",
                            snapshot=_snap(kv_tokens_used=usable - want + 1))
    assert not over["admit"]
    assert over["kv_reserve"] == reserve


def test_admission_does_not_guess_when_the_pool_cannot_be_measured():
    """No measurement is not the same as no room.

    The inference service may be down or still starting. The queue is durable,
    so refusing on an absent number would stall work for a reason that is not
    about resources at all.
    """
    arb = Arbiter(ArbiterConfig())
    out = arb.kv_admission(work_class="user", snapshot=_snap(kv_pool_capacity=0))
    assert out["admit"] and out["measured"] is False


def test_maintenance_demand_is_estimated_from_its_own_budget():
    """The two classes do not cost the same, and admission knows it."""
    arb = Arbiter(ArbiterConfig())
    snap = _snap(kv_tokens_used=0)
    user = arb.kv_admission(work_class="user", snapshot=snap)
    maint = arb.kv_admission(work_class="maintenance", snapshot=snap)
    assert user["estimated_demand"] == ArbiterConfig().ego_neuocyte_budget_tokens
    assert maint["estimated_demand"] == ArbiterConfig().id_neuocyte_budget_tokens


# ===========================================================================
# One number, shown and enforced
# ===========================================================================
def test_the_shipped_policy_fits_the_pool_it_shares():
    """The configured ceilings are logical, but they still have to add up.

    Ego and Id at their ceilings plus three cold workers must fit the pool
    with the reserve intact, or the policy is writing cheques the KV cannot
    cash and the failure would appear as refused work under load.
    """
    from amoeba.config import load_config
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    arb = cfg.arbiter
    worst = (cfg.ego.max_context_tokens + cfg.id.max_context_tokens
             + arb.max_neuocytes * arb.id_neuocyte_budget_tokens)
    usable = cfg.backend.n_ctx * (1 - arb.kv_admission_reserve_fraction)
    assert worst <= usable, (
        f"worst-case logical demand {worst} exceeds usable pool {usable:.0f}")


def test_the_budget_a_role_reports_is_the_budget_it_is_held_to():
    """A shown number that is not the enforced number is worse than none.

    `max_context_tokens` used to reach the dashboard and the pulse and never
    the control loop, while a global scalar did the enforcing, and a reader
    had no way to tell which governed. Both paths now read the same field of
    the same object, which is what this asserts -- at the source, because the
    alternative is standing up a role process to compare two numbers.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src" / "amoeba"
           / "roles.py").read_text(encoding="utf-8")

    # What the operator is shown.
    assert '"max_context_tokens": self.role_cfg.max_context_tokens' in src, (
        "context_stats no longer reports the role's own configured budget")
    # What the Arbiter is told to enforce.
    assert 'getattr(self.role_cfg, "max_context_tokens", 0)' in src, (
        "_context_budget no longer reads the role's own configured budget")
    # And the session is opened with it rather than with something else.
    assert "context_budget_tokens=self._context_budget()" in src, (
        "the session is not opened with the budget that is reported")


# ===========================================================================
# The wiring, not just the arithmetic
# ===========================================================================
class _PoolBackend:
    """A backend that hands out one session, so the service path is exercised."""

    backend_kind = "fake"
    is_simulated = True
    model_generation = "fake-1"

    def __init__(self, sess):
        self.sess = sess

    def get_session(self, session_id):
        return self.sess

    def generate(self, session_id, **kw):
        from amoeba.backends.llama_engine import GenerationResult
        return GenerationResult(session_id=session_id, text="ok", tokens=[1],
                                finish_reason="stop", prompt_tokens=1,
                                completion_tokens=1, time_to_first_token=0.0,
                                total_seconds=0.0)

    def generate_batched(self, requests, **kw):
        from amoeba.errors import CapabilityUnsupported
        raise CapabilityUnsupported("no batching here")


def _service(tmp_path, sess):
    from amoeba.inference_service import InferenceService

    cfg = Config()
    cfg.state_dir = tmp_path / "state"
    cfg.runtime_dir = tmp_path / "runtime"
    cfg.models_dir = tmp_path / "models"
    cfg.ensure_dirs()
    svc = InferenceService(cfg)
    svc.backend = _PoolBackend(sess)
    return svc


def test_the_service_judges_a_forked_worker_on_its_growth_not_its_total(tmp_path, backend):
    """The wiring, which the arithmetic tests do not reach.

    Calling `clamp_inference` directly proves the Arbiter can do the right
    thing. It does not prove the inference service *asks* it the right thing,
    and reverting that one call site to `sess.n_past` passed every other test
    in this file. This is the test that fails when it does.
    """
    ego = backend.open_session(role="ego", context_budget_tokens=16384)
    backend.ingest(ego.session_id, list(range(10000)))
    worker = backend.fork_prefix(src_session_id=ego.session_id, prefix_len=10000,
                                 role="neuocyte", context_budget_tokens=6144,
                                 budget_basis="private_growth")

    svc = _service(tmp_path, worker)
    out = svc.generate(session_id=worker.session_id, max_tokens=8)
    assert out["text"] == "ok", "a forked worker was refused on its inherited prefix"

    # ...and the same session is refused once its own growth passes the budget.
    backend.ingest(worker.session_id, list(range(7000)))
    with pytest.raises(ResourceExhausted):
        svc.generate(session_id=worker.session_id, max_tokens=8)
