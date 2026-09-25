"""An interaction does not settle while work its answer needs is still running.

Live on 2026-09-25, timed from the event chain:

    09:14:33.135  ego  work.admitted     the worker is delegated
    09:14:44.770  ego  board_read        last look -- nothing there yet
    09:14:45.871  nc   board.posted      41679167500, the right answer
    09:14:49.759  ego  output.emitted    completed, carrying the wrong one

Ego delegated a computation, exhausted its tool-turn budget, and the
interaction completed 1.1 seconds after its own worker posted the correct
number. The client was told 41,675,000,250. The worker had measured
41,679,167,500. Ego then went and read the board twice more -- at 09:14:51 and
09:15:00, exactly as it had promised -- and had nowhere to put what it found,
because `interaction.completed` had already fired.

Two rules, one for each half:

  I140  an interaction cannot settle while any work admitted as an answer
        dependency remains non-terminal
  I141  completion of answer-dependent work makes the originating interaction
        eligible to resume, including after a restart

The first without the second is a nicer hang. Parking is bounded because work
is: every item reaches done, failed or cancelled, and a role is woken for all
three, because "the computation failed" is an answer and silence is not.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

from amoeba import io_api, mailbox
from amoeba.store.events import EventKind

sys.path.insert(0, str(Path(__file__).parent))
from test_external_delivery import _interaction, _row, _sup  # noqa: E402
from test_persistent_turns import _claim, _complete  # noqa: E402

OP = "op_dep"


def _ask(mind, *, operation_id=OP):
    """A client request bound to an operation, as the real path binds one."""
    interaction_id = _interaction(mind)
    mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=True, lineage=operation_id,
                                  operation_id=operation_id),
        actor="test", bump_version=False)
    return interaction_id


def _work(mind, verbs, *, blocks, operation_id=OP, objective="compute it"):
    """Admitted through the repository, as the other lifecycle tests do."""
    work_id, _ = mind.work.admit(objective=objective, work_class="user",
                                 origin_actor="ego", operation_id=operation_id,
                                 blocks_answer=blocks)
    return work_id


def _with_arbiter(mind):
    """A sup whose `admit_work` runs, for the tests about who decides blocking.

    The lifecycle fixture carries a stub arbiter because nothing else here
    needs one; `ego_request_work` reaches it, so these tests supply the
    smallest thing that admits.
    """
    from types import SimpleNamespace
    decision = SimpleNamespace(admitted=True, reason=None, detail=None,
                               granted_budget_tokens=2048,
                               granted_deadline=None)
    from amoeba import ego_api
    sup = _sup(mind)
    sup.arbiter = SimpleNamespace(admit=lambda **kw: decision,
                                  note_served=lambda *a, **k: None)
    sup.resource_snapshot = lambda: None
    sup.client = lambda name: SimpleNamespace(
        call=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no inference")))
    verbs = dict(sup.methods())
    verbs.update(ego_api.build(sup))
    return verbs


def _answer(mind, interaction_id, verbs, text="A worker has been delegated."):
    """Ego finishes its turn and the answer is delivered from the record."""
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": text})
    trigger_id = [r["trigger_id"] for r in mind.db.conn.execute(
        "SELECT trigger_id FROM role_triggers WHERE target_role = 'ego'")][-1]
    mind.writer.apply(
        lambda m: m.sql("UPDATE interactions SET trigger_id = ?"
                        " WHERE interaction_id = ?", (trigger_id, interaction_id)),
        actor="test", bump_version=False)
    return verbs["io_reconcile"]()


def _finish(mind, verbs, work_id, *, status="done"):
    lease = mind.work.lease(neuocyte_id="nc_x", work_id=work_id)
    if status == "done":
        mind.work.complete(work_id=work_id, neuocyte_id="nc_x",
                           fencing_token=lease["fencing_token"],
                           result={"value": 41679167500})
    else:
        mind.work.cancel(work_id=work_id, reason="test", actor="harness")


# ---------------------------------------------------------------------------
# I140: it does not settle while a dependency is outstanding
# ---------------------------------------------------------------------------
def test_an_interaction_does_not_settle_while_its_work_runs(mind):
    """The live failure, in one test."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    _work(mind, verbs, blocks=True)

    _answer(mind, interaction_id, verbs)

    row = _row(mind, interaction_id)
    assert row["status"] == "awaiting_work", (
        "the interaction settled over the top of its own dependency")
    assert row["completed_at"] is None, "a parked interaction was given a finish time"


def test_the_interim_answer_is_kept_not_discarded(mind):
    """What the thought got to is real, and a client may read it meanwhile."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    _work(mind, verbs, blocks=True)

    _answer(mind, interaction_id, verbs, text="I delegated it; no result yet.")

    assert "delegated" in (_row(mind, interaction_id)["output_preview"] or "")
    status = verbs["io_status"](interaction_id=interaction_id,
                                client_id="client_a")
    assert status["status"] == "awaiting_work"
    assert len(status["awaiting_work"]) == 1, "the client is not told what it waits on"


def test_one_of_three_finishing_does_not_release_it(mind):
    """The gate is every dependency terminal, not any one of them landing."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    ids = [_work(mind, verbs, blocks=True, objective=f"part {i}") for i in range(3)]

    _answer(mind, interaction_id, verbs)
    assert _row(mind, interaction_id)["status"] == "awaiting_work"

    _finish(mind, verbs, ids[0])
    verbs["io_reconcile"]()
    assert _row(mind, interaction_id)["status"] == "awaiting_work", (
        "one worker finishing released an interaction waiting on three")

    _finish(mind, verbs, ids[1])
    verbs["io_reconcile"]()
    assert _row(mind, interaction_id)["status"] == "awaiting_work", (
        "two of three was treated as all of them")

    _finish(mind, verbs, ids[2])
    out = verbs["io_reconcile"]()
    assert out["resumed_count"] == 1
    assert _row(mind, interaction_id)["status"] == "running"


def test_background_work_never_blocks_an_answer(mind):
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    _work(mind, verbs, blocks=False)

    _answer(mind, interaction_id, verbs)

    assert _row(mind, interaction_id)["status"] == "complete", (
        "work the answer does not depend on held the answer up")


def test_work_for_another_operation_does_not_block_this_answer(mind):
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    _work(mind, verbs, blocks=True, operation_id="op_somebody_else")

    _answer(mind, interaction_id, verbs)

    assert _row(mind, interaction_id)["status"] == "complete"


# ---------------------------------------------------------------------------
# How the classification is decided, and that it cannot be undone
# ---------------------------------------------------------------------------
def test_omitting_background_means_the_answer_waits(mind):
    """Delegating in order to answer is the ordinary case, so it is the default."""
    verbs = _with_arbiter(mind)
    sig = inspect.signature(verbs["ego_request_work"])
    assert sig.parameters["background"].default is False

    out = verbs["ego_request_work"](objective="compute it", operation_id=OP)
    work_id = out["admitted"][0]["work_id"]
    assert mind.db.conn.execute(
        "SELECT blocks_answer FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()["blocks_answer"] == 1


def test_background_false_is_the_same_as_omitting_it(mind):
    verbs = _with_arbiter(mind)
    out = verbs["ego_request_work"](objective="compute it", operation_id=OP,
                                    background=False)
    work_id = out["admitted"][0]["work_id"]
    assert mind.db.conn.execute(
        "SELECT blocks_answer FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()["blocks_answer"] == 1


def test_background_true_is_the_only_way_out(mind):
    verbs = _with_arbiter(mind)
    out = verbs["ego_request_work"](objective="index the archive",
                                    operation_id=OP, background=True)
    work_id = out["admitted"][0]["work_id"]
    assert mind.db.conn.execute(
        "SELECT blocks_answer FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()["blocks_answer"] == 0


def test_nothing_can_reclassify_work_after_it_is_admitted(mind):
    """A model must not escape the wait by relabelling what it already asked for.

    Enforced by there being no such path at all, which is a claim about the
    source and is checked as one: `blocks_answer` is written by the INSERT
    that admits the work and by nothing else, ever.
    """
    from amoeba.store import work_repo

    source = inspect.getsource(work_repo)
    assert "blocks_answer" in source, "the column vanished; this guard is vacuous"

    # Every statement that rewrites a work row, and whether any touches it.
    updates = re.findall(r"UPDATE work_items SET(.*?)(?:WHERE|\")", source,
                         re.DOTALL)
    assert updates, "no UPDATE statements found; this guard is vacuous"
    culprits = [u for u in updates if "blocks_answer" in u]
    assert not culprits, (
        f"an answer dependency can be rewritten after admission: {culprits}")


def test_a_retry_keeps_the_classification(mind):
    """Attempts, expiry and recovery all re-read the durable row."""
    verbs = _sup(mind).methods()
    work_id = _work(mind, verbs, blocks=True)
    lease = mind.work.lease(neuocyte_id="nc_1", work_id=work_id)
    mind.work.fail(work_id=work_id, neuocyte_id="nc_1",
                   fencing_token=lease["fencing_token"],
                   failure="first attempt died", requeue=True)

    assert mind.db.conn.execute(
        "SELECT blocks_answer, status FROM work_items WHERE work_id = ?",
        (work_id,)).fetchone()["blocks_answer"] == 1, (
        "a retry lost the answer dependency")


# ---------------------------------------------------------------------------
# I141: finishing makes it eligible to resume, however it finished
# ---------------------------------------------------------------------------
def test_finished_work_resumes_the_interaction(mind):
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)

    _finish(mind, verbs, work_id)
    out = verbs["io_reconcile"]()

    assert out["resumed_count"] == 1
    row = _row(mind, interaction_id)
    assert row["status"] == "running", "the parked interaction was never given back"
    assert row["trigger_id"], "nothing was queued for Ego to answer"


def test_the_resuming_request_expects_an_answer(mind):
    """Otherwise Ego is told the work finished and answers nobody."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)
    _finish(mind, verbs, work_id)
    verbs["io_reconcile"]()

    trigger_id = _row(mind, interaction_id)["trigger_id"]
    row = dict(mind.db.conn.execute(
        "SELECT expects_answer, operation_id FROM role_triggers"
        " WHERE trigger_id = ?", (trigger_id,)).fetchone())
    assert row["expects_answer"] == 1
    assert row["operation_id"] == OP, "the resumed request left its operation"


def test_work_that_failed_still_wakes_the_role(mind):
    """"The computation failed" is an answer. Silence is not."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)

    _finish(mind, verbs, work_id, status="cancelled")
    out = verbs["io_reconcile"]()

    assert out["resumed_count"] == 1, "a failed dependency parked it forever"
    assert _row(mind, interaction_id)["status"] == "running"

    # And it is told *what happened*, not merely that something did. Ego can
    # only report a failure it was given; a resumption carrying none of the
    # unsuccessful work would wake it to say nothing useful.
    trigger_id = _row(mind, interaction_id)["trigger_id"]
    digest = mind.db.conn.execute(
        "SELECT payload_sha256 FROM role_triggers WHERE trigger_id = ?",
        (trigger_id,)).fetchone()["payload_sha256"]
    payload = mind.blobs.get_json(digest)
    reported = {w["work_id"]: w["status"] for w in payload["work"]}
    assert reported.get(work_id) == "cancelled", (
        f"the resumed request did not carry the failed work: {reported}")


def test_a_parked_interaction_resumes_with_nobody_waiting(mind):
    """The crash-safe half: resumption is the reconciler's, not a thread's.

    `io_reconcile` is what runs after a restart, and it is the only thing
    that resumed the interaction in every test here -- no thread was ever
    holding it.
    """
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)
    _finish(mind, verbs, work_id)

    # A fresh set of verbs: nothing in memory knows about this interaction.
    fresh = _sup(mind).methods()
    assert fresh["io_reconcile"]()["resumed_count"] == 1
    assert _row(mind, interaction_id)["status"] == "running"


def test_resuming_is_recorded(mind):
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)
    _finish(mind, verbs, work_id)
    verbs["io_reconcile"]()

    kinds = [r["kind"] for r in mind.db.conn.execute(
        "SELECT kind FROM events WHERE kind IN (?, ?)",
        (EventKind.INTERACTION_AWAITING_WORK, EventKind.INTERACTION_RESUMED))]
    assert EventKind.INTERACTION_AWAITING_WORK in kinds
    assert EventKind.INTERACTION_RESUMED in kinds


def test_the_final_answer_settles_the_interaction(mind):
    """The whole loop: park, work lands, Ego answers again, client is told."""
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs, text="delegated; nothing back yet")
    _finish(mind, verbs, work_id)
    verbs["io_reconcile"]()

    _answer(mind, interaction_id, verbs, text="the worker measured 41679167500")
    row = _row(mind, interaction_id)
    assert row["status"] == "complete", "it never settled after the work landed"
    assert "41679167500" in (row["output_preview"] or "")
    out = verbs["io_output"](interaction_id=interaction_id, client_id="client_a")
    assert "41679167500" in (out["answer"] or ""), (
        "the client was never given the number it waited for")


# ---------------------------------------------------------------------------
# The check is where it has to be
# ---------------------------------------------------------------------------
def test_the_dependency_check_happens_inside_the_settling_mutation():
    """Otherwise work admitted between the check and the commit is missed.

    A claim about *where* the question is asked cannot be observed from
    outside, because the window is a scheduling accident. It is checked as a
    claim about the source instead: the query that decides whether to park
    must sit inside the function the writer applies, not before it.
    """
    source = inspect.getsource(io_api.build)
    start = source.index("def _publish(")
    end = source.index("def io_reconcile(", start)
    publish = source[start:end]

    body = publish.index("def body(")
    check = publish.index("AND blocks_answer = 1")
    assert check > body, (
        "the dependency check runs before the mutation that settles, so work "
        "admitted in between would be settled straight over")


def test_a_programming_error_in_the_resume_path_is_not_swallowed(mind,
                                                                 monkeypatch):
    """Operational failures may be survivable. A NameError is not one.

    While this was being built, `_resume_parked` caught `Exception` around the
    enqueue and continued. A reference to a constant that did not exist raised
    `NameError`, was swallowed, and every pass reported nothing to resume --
    so parked interactions stayed parked and the record said the reconciler
    had looked and found no work to do. An internal mistake must not be
    indistinguishable from an empty queue.
    """
    verbs = _sup(mind).methods()
    interaction_id = _ask(mind)
    work_id = _work(mind, verbs, blocks=True)
    _answer(mind, interaction_id, verbs)
    _finish(mind, verbs, work_id)

    def boom(*a, **k):
        raise NameError("name 'MAX_TRIGGER_SUMMARY' is not defined")

    monkeypatch.setattr(io_api.mailbox, "enqueue", boom)
    with pytest.raises(NameError):
        verbs["io_reconcile"]()

    assert _row(mind, interaction_id)["status"] == "awaiting_work", (
        "the interaction was unparked by a pass that failed")


def test_nothing_in_the_resume_path_swallows_exceptions(mind):
    """The guard above, kept honest at the source.

    A behavioural test can only catch the swallow it happens to trigger. This
    says there is nowhere for one to hide.
    """
    source = inspect.getsource(io_api.build)
    start = source.index("def _resume_parked(")
    end = source.index("def _wait_expired(", start)
    resume = source[start:end]

    assert "except" not in resume, (
        "the resume path catches something; an internal error there would be "
        "reported as nothing to do")
