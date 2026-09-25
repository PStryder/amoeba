"""Whose cognition a worker is, and who gets to decide.

Found live on 2026-09-24. Ego asked for maintenance-shaped work and was handed
an `id.neuocyte@1.1` worker: Id's cognition, instantiated by Ego, with no Id
involvement anywhere in the record. The cause was one line binding two
unrelated questions to one model-authored field:

    maintenance = item.get("work_class") == "maintenance"
    base = "id.neuocyte" if maintenance else "ego.neuocyte"

`work_class` is an argument to `ego_request_work`. So the model chose not only
what kind of work to request but whose mind would perform it.

The two questions, kept apart now:

    work class      what KIND of work this is -- decides the execution shape
    worker lineage  WHOSE delegated cognition does it -- decides the profile

Lineage is the Harness's, derived from `origin_actor`, which is written at
admission and which no caller supplies. If Ego needs Id's cognition the path
is the explicit Ego -> Id request; Id then delegates into its own lineage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from amoeba.neuocyte import WORK_ORIGINS, Neuocyte, worker_lineage  # noqa: E402
from test_tool_loop import _BindingSup, _binding_neuocyte  # noqa: E402


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("work_class", ["user", "maintenance"])
def test_ego_delegates_only_into_its_own_lineage(work_class):
    """Including maintenance, which is what the live defect turned on.

    Ego can no longer *request* maintenance work at all, so this pairing is
    unreachable through the door. It is still asserted here: the derivation
    must not consult the class even when handed one, or closing the door
    would be the only thing standing between Ego and Id's cognition.
    """
    assert worker_lineage({"origin_actor": "ego",
                           "work_class": work_class}) == "ego"


@pytest.mark.parametrize("work_class", ["user", "maintenance"])
def test_id_delegates_only_into_its_own_lineage(work_class):
    assert worker_lineage({"origin_actor": "id",
                           "work_class": work_class}) == "id"


def test_the_work_class_does_not_choose_a_mind():
    """The same class, two origins, two lineages: the class decided nothing."""
    ego = worker_lineage({"origin_actor": "ego", "work_class": "maintenance"})
    id_ = worker_lineage({"origin_actor": "id", "work_class": "maintenance"})
    assert (ego, id_) == ("ego", "id")


def test_work_nobody_persistent_originated_takes_the_outward_lineage():
    """There is no role to inherit from, so it does not inherit Id's."""
    for origin in ("operator", "supervisor", "nc_7", "", None):
        assert worker_lineage({"origin_actor": origin,
                               "work_class": "maintenance"}) == "ego"


def test_only_the_persistent_roles_are_lineages():
    assert tuple(WORK_ORIGINS) == ("ego", "id")


# ---------------------------------------------------------------------------
# What a caller can and cannot reach
# ---------------------------------------------------------------------------
def test_a_specialisation_cannot_name_another_lineage(cfg):
    """A specialisation narrows within a lineage; it cannot cross to another.

    The lineage is always the prefix, so the worst a caller can do by naming
    `id.neuocyte` is ask for `ego.neuocyte.id.neuocyte` -- a leaf under its
    own root that does not exist, which falls back to its own base.
    """
    sup = _BindingSup({"ego.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"origin_actor": "ego", "work_class": "user",
                              "specialisation": "id.neuocyte"}, work_id="w1")

    assert bound["profile_ref"] == "ego.neuocyte@1"
    assert all(ns.startswith("ego.neuocyte") for ns in sup.attempts), sup.attempts
    assert "id.neuocyte" not in sup.attempts


def test_ego_asking_for_maintenance_still_gets_an_ego_worker(cfg):
    """The live defect, at the binding layer."""
    sup = _BindingSup({"ego.neuocyte", "id.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"origin_actor": "ego",
                              "work_class": "maintenance"}, work_id="w2")

    assert bound["profile_ref"] == "ego.neuocyte@1"
    assert sup.attempts == ["ego.neuocyte"], (
        "Ego's maintenance work reached into Id's family tree")


def test_id_maintenance_work_binds_under_id(cfg):
    """The path that was right all along, and stays right."""
    sup = _BindingSup({"id.neuocyte.integrity"})
    nc = _binding_neuocyte(cfg, sup)

    bound = nc._bind_profile({"origin_actor": "id", "work_class": "maintenance",
                              "specialisation": "integrity"}, work_id="w3")

    assert bound["profile_ref"] == "id.neuocyte.integrity@1"
    assert sup.attempts == ["id.neuocyte.integrity"]


def test_a_worker_never_inherits_another_roles_prefix(cfg):
    """What is physically inherited follows the lineage too."""
    inherited = {}

    class _Sup(_BindingSup):
        def call(self, method, **params):
            if method == "bind_profile":
                inherited[params["namespace"]] = params.get("inherited_namespace")
            return super().call(method, **params)

    sup = _Sup({"ego.neuocyte", "id.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)
    nc._bind_profile({"origin_actor": "id", "work_class": "user"}, work_id="w4")

    assert inherited["id.neuocyte"] == "id", (
        "an Id worker was handed Ego's prefix to inherit")


# ---------------------------------------------------------------------------
# It survives everything that re-derives it
# ---------------------------------------------------------------------------
def test_a_retry_binds_the_same_lineage(mind, cfg):
    """Lineage comes from the durable row, so an attempt cannot move it."""
    work_id, _ = mind.work.admit(objective="look into it", work_class="user",
                                 origin_actor="ego")
    for _ in range(2):
        lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
        row = mind.work.get_work(work_id)
        assert worker_lineage(row) == "ego", "a retry changed whose worker it is"
        mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                       fencing_token=lease["fencing_token"], failure="again")


def test_expiry_and_recovery_keep_the_lineage(mind):
    """An expired lease returns the item; it does not re-home it."""
    import time

    work_id, _ = mind.work.admit(objective="look into it", work_class="maintenance",
                                 origin_actor="id")
    mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.expire_leases(now=time.time() + 10_000)

    assert worker_lineage(mind.work.get_work(work_id)) == "id"


def test_nothing_rewrites_who_originated_work():
    """Structural: `origin_actor` is written once, at admission."""
    import re

    source = (Path(__file__).resolve().parents[1] / "src" / "amoeba").rglob("*.py")
    offenders = []
    for path in source:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"UPDATE work_items SET[^\"']*origin_actor", line):
                offenders.append(f"{path.name}:{n}")
    assert not offenders, f"origin_actor is rewritten at {offenders}"


# ---------------------------------------------------------------------------
# The record says whose it was
# ---------------------------------------------------------------------------
def test_the_record_shows_the_origin_and_the_lineage_it_produced(mind, cfg):
    """Provenance has to carry both, or the pairing cannot be audited."""
    work_id, _ = mind.work.admit(objective="check something",
                                 work_class="user", origin_actor="ego")
    row = mind.work.get_work(work_id)

    assert row["origin_actor"] == "ego"
    assert worker_lineage(row) == "ego"

    sup = _BindingSup({"ego.neuocyte"})
    nc = _binding_neuocyte(cfg, sup)
    bound = nc._bind_profile(row, work_id=work_id)
    assert bound["profile_ref"].startswith("ego.neuocyte")


# ---------------------------------------------------------------------------
# A role delegates work of its own kind
# ---------------------------------------------------------------------------
def _delegating(mind):
    """Ego's door and the Harness's, over a real arbiter."""
    import logging
    from types import SimpleNamespace

    from amoeba import ego_api, supervisor_api
    from amoeba.arbiter import Arbiter, ResourceSnapshot

    holder: dict[str, object] = {}
    cache: dict[str, object] = {}

    def all_methods():
        if not cache:
            cache.update(supervisor_api.build(holder["sup"]))
            cache.update(ego_api.build(holder["sup"]))
        return cache

    sup = SimpleNamespace(
        mind=mind, cfg=mind.cfg, log=logging.getLogger("t"),
        methods=all_methods, arbiter=Arbiter(mind.cfg.arbiter),
        resource_snapshot=lambda: ResourceSnapshot(),
        note_trigger=lambda role: None,
        note_turn_finished=lambda *a, **k: None,
        next_heartbeat=lambda role: None,
        release_work_sandbox=lambda *a, **k: None, neuocytes={})
    holder["sup"] = sup
    return sup.methods()



def test_ego_cannot_ask_for_maintenance_work(mind):
    """Noticing that something needs tending is a reason to ask Id.

    Lineage derivation already stops Ego receiving an `id.neuocyte`. This
    closes the other half: an `ego.neuocyte` running maintenance-shaped work
    would be a worker doing a kind of work its own profile does not describe.
    """
    from amoeba.errors import InvalidInput

    verbs = _delegating(mind)
    with pytest.raises(InvalidInput) as caught:
        verbs["ego_request_work"](objective="tidy the snapshots",
                                  work_class="maintenance")

    assert "Id's work" in caught.value.message
    assert "ego_request_id_review" in str(caught.value.details)


def test_the_harness_refuses_the_pairing_too(mind):
    """The verb is the model's door; admission is the Harness's."""
    from amoeba.errors import InvalidInput

    verbs = _delegating(mind)
    with pytest.raises(InvalidInput):
        verbs["admit_work"](objective="tidy up", work_class="maintenance",
                            origin_actor="ego")
    with pytest.raises(InvalidInput):
        verbs["admit_work"](objective="answer this", work_class="user",
                            origin_actor="id")


def test_id_still_delegates_its_own_maintenance(mind):
    """The Ego -> Id -> id.neuocyte path is the intended one and still works."""
    verbs = _delegating(mind)
    out = verbs["admit_work"](objective="check for stale snapshots",
                              work_class="maintenance", origin_actor="id")

    assert out["admitted"], out
    row = mind.work.get_work(out["work_id"])
    assert row["origin_actor"] == "id"
    assert worker_lineage(row) == "id"


def test_work_nobody_persistent_originated_is_left_alone(mind):
    """I92 documents operator-originated work; admission must still take it."""
    verbs = _delegating(mind)
    out = verbs["admit_work"](objective="an errand", work_class="user",
                              origin_actor="operator")
    assert out["admitted"], out

