"""A conclusion is a claim Ego chooses to make, and Id can audit it for real.

The first live pressure test found the most developed epistemic machinery in
the organism -- audits, disagreements, settlement on changed ground (I100 to
I102) -- entirely unreachable in operation:

  * `id_audit` existed only behind the control token. Neither Id nor the
    operator could start an audit, so none ever happened.
  * Even reachable, it committed a verdict only inside a caller that waited.
    An audit nobody waited on -- the only kind a wake can produce -- had its
    verdict generated and then dropped.
  * Every finished Ego answer was recorded as an auditable "conclusion",
    including a complaint that a tool refused and a self-report about memory.
    Seven conclusions in one run, none of them reviewed, the wrong ones never
    superseded: an audit queue made of noise.

Now a conclusion is something Ego records on purpose. Recording one wakes Id
to audit it, the verdict is committed when Id answers whether or not anyone
is waiting, and the operator can ask for an audit too.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
from urllib.parse import urlparse

import pytest

from amoeba import mailbox
from amoeba.rpc import RpcClient, read_or_create_token
from amoeba.supervisor_api import commit_audit
from conftest import start_stack
from test_persistent_turns import _claim, _complete

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")

CLAIM = "gw-3's clock running 241 s fast caused the token_expired rejections."
CONTESTED = ("VERDICT: contested\n"
             "FINDING: the record shows the skew only on gw-3's own report\n"
             "UNRESOLVED: whether gw-3 was measured independently")
SUPPORTED = ("VERDICT: supported\n"
             "FINDING: chronyc and the load-balancer timestamps agree\n"
             "UNRESOLVED: why the migration stepped the clock")


def _request(mind, operation_id="op_answering"):
    _, out = mind.writer.apply(
        lambda m: mailbox.enqueue(m, role="ego", kind="user_input",
                                  source="operator", summary="a question",
                                  expects_answer=True, operation_id=operation_id,
                                  lineage=operation_id),
        actor="test", bump_version=False)
    return out


def _record(mind, row):
    return mind.db.conn.execute(
        "SELECT answer_sha256 FROM role_triggers WHERE trigger_id = ?",
        (row["trigger_id"],)).fetchone()["answer_sha256"]


def _conclusions(mind):
    return [dict(r) for r in mind.db.conn.execute(
        "SELECT conclusion_id, claim, produced_by, operation_id FROM conclusions")]


# ---------------------------------------------------------------------------
# D6: answering is not concluding
# ---------------------------------------------------------------------------
def test_answering_is_not_concluding(mind):
    """A finished answer is a reply, not a claim put into the record."""
    req = _request(mind)
    turn = _claim(mind, "ego")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": "Startup takes about ten seconds.",
                      "segment": "Startup takes about ten seconds."})
    assert _conclusions(mind) == [], "an answer was recorded as a conclusion"
    record = mind.blobs.get_json(_record(mind, req))
    assert record["conclusion_ids"] == [] and record["conclusion_id"] is None


def test_the_answer_reports_the_conclusion_ego_chose_to_record(mind):
    """What Ego recorded while answering is named by the answer, and only that."""
    req = _request(mind, operation_id="op_q1")
    turn = _claim(mind, "ego")
    cid, _ = mind.memory.record_conclusion(claim=CLAIM, produced_by="ego",
                                           operation_id="op_q1")
    # Another interaction's conclusion is not this answer's.
    mind.memory.record_conclusion(claim="unrelated", produced_by="ego",
                                  operation_id="op_elsewhere")
    _complete(mind, turn["turn_id"], stop_reason="model_stop",
              result={"answer": "It was the clock.", "segment": "It was the clock."})
    record = mind.blobs.get_json(_record(mind, req))
    assert record["conclusion_ids"] == [cid]
    assert record["conclusion_id"] == cid


# ---------------------------------------------------------------------------
# D3: the verdict is committed, whoever asked
# ---------------------------------------------------------------------------
def _dossier():
    return {"events": [], "hash_chain_ok": True, "unresolved_content": []}


def test_a_contested_verdict_opens_a_disagreement(mind):
    cid, _ = mind.memory.record_conclusion(claim=CLAIM, produced_by="ego")
    out = commit_audit(mind, _dossier(), conclusion_id=cid, operation_id=None,
                       focus="", text=CONTESTED, op_id="op_audit")
    assert out["verdict"] == "contested" and out["verdict_stated"] is True
    assert out["audit_id"] and out["disagreement_id"]
    opened = mind.memory.get_disagreements(status="open", limit=10)
    assert [d["subject_id"] for d in opened] == [cid]


def test_an_unstated_verdict_is_recorded_as_unparsed_not_judged(mind):
    cid, _ = mind.memory.record_conclusion(claim=CLAIM, produced_by="ego")
    out = commit_audit(mind, _dossier(), conclusion_id=cid, operation_id=None,
                       focus="", text="I looked and it seems fine.", op_id="op_a")
    assert out["verdict"] == "inconclusive" and out["verdict_stated"] is False
    assert "disagreement_id" not in out


# ---------------------------------------------------------------------------
# live: the whole lifecycle, from supported surfaces only
# ---------------------------------------------------------------------------
def _inference(stack):
    cfg = stack.cfg
    c = RpcClient(cfg.supervisor_host, cfg.inference_port,
                  read_or_create_token(cfg.token_path), timeout=60)
    c.connect(retries=20, delay=0.5)
    return c


def _until(predicate, timeout=120, step=0.3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = predicate()
        if got:
            return got
        time.sleep(step)
    return None


@pytest.fixture()
def quiet(tmp_path):
    # Id's startup and heartbeat stay off, so the only thing that can wake Id
    # is the event under test.
    stack = start_stack(tmp_path, scheduler={"id_startup_turn": False})
    stack.wait_for_children(timeout=90)
    yield stack
    stack.stop()


@live
def test_a_conclusion_ego_records_wakes_id_and_the_verdict_lands(quiet):
    """Record -> wake -> audit -> disagreement, with nobody waiting on the audit."""
    call = json.dumps({"name": "record_conclusion",
                       "arguments": {"claim": CLAIM,
                                     "evidence": [{"note": "chronyc on gw-3"}]}})
    _inference(quiet).call("script_responses", responses=[
        {"role": "ego", "text": f"<tool_call>{call}</tool_call>",
         "finish_reason": "stop", "continues": True},
        {"role": "ego", "text": "It was gw-3's clock.", "finish_reason": "stop",
         "continues": True},
        {"role": "id", "text": CONTESTED, "finish_reason": "stop", "continues": True},
    ])
    env = quiet.call("ego_converse", message="what caused it?", wait=False)
    trigger_id = env["result"]["trigger_id"]
    state = _until(lambda: (lambda s: s if s["status"] in ("completed", "incomplete")
                            else None)(quiet.call("role_answer", trigger_id=trigger_id)))
    assert state and state["answer"] == "It was gw-3's clock."

    recorded = quiet.call("history", operation_id=env["operation_id"],
                          kinds=["conclusion.recorded"], limit=10)
    assert len(recorded) == 1, "Ego's intentional conclusion was not recorded once"
    cid = json.loads(recorded[0]["payload_inline"])["conclusion_id"] \
        if recorded[0].get("payload_inline") else None
    concl = quiet.call("get_conclusion", conclusion_id=cid)
    assert concl["claim"] == CLAIM and concl["produced_by"] == "ego"
    assert concl["operation_id"] == env["operation_id"]

    # Id was woken by the claim, by the Harness, and said so honestly.
    woke = _until(lambda: [t for t in quiet.call("role_turns", role="id")["turns"]
                           if t["trigger_kinds"] == ["conclusion_recorded"]
                           and t["status"] == "completed"])
    assert woke, "recording a conclusion did not wake Id"
    detail = quiet.call("role_turn", turn_id=woke[0]["turn_id"])
    assert detail["triggers"][0]["source"] == "harness"

    # And its verdict became durable though nobody waited for it.
    opened = _until(lambda: [d for d in quiet.call("disagreements", status="open")
                             if d.get("status") == "open"] or None)
    assert opened, "the contested verdict was produced and then dropped"
    assert [d["subject_id"] for d in opened] == [cid]


@live
def test_the_operator_can_ask_for_an_audit_and_its_verdict_is_recorded(quiet):
    """Through the operator HTTP surface, as the dashboard does."""
    # A conclusion as Ego records one: under the operation that asked. The
    # dossier resolves a claim through its operation, and refuses one it
    # cannot resolve -- before waking Id, as the next test holds.
    asked = quiet.call("ego_converse", message="what caused it?", wait=False)
    concl = quiet.call("record_conclusion", claim=CLAIM, produced_by="ego",
                       operation_id=asked["operation_id"])
    _until(lambda: quiet.call("role_answer", trigger_id=asked["result"]["trigger_id"])
           ["status"] in ("completed", "incomplete"))
    _inference(quiet).call("script_responses", responses=[
        {"role": "id", "text": SUPPORTED, "finish_reason": "stop", "continues": True}])
    cfg = quiet.cfg
    base = urlparse(f"http://{cfg.api_host}:{cfg.api_port}")
    token = cfg.operator_session_path.read_text(encoding="utf-8").strip()
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "id_audit",
                       "params": {"conclusion_id": concl["conclusion_id"],
                                  "wait": False}})
    conn = http.client.HTTPConnection(base.hostname, base.port, timeout=60)
    try:
        conn.request("POST", "/operator/rpc", body=body, headers={
            "Content-Type": "application/json", "X-Amoeba-Operator": token,
            "Content-Length": str(len(body))})
        reply = json.loads(conn.getresponse().read())
    finally:
        conn.close()
    assert "error" not in reply, reply

    audits = _until(lambda: [e for e in quiet.call("history", kinds=["audit.recorded"],
                                                   limit=20)
                             if concl["conclusion_id"] in (e.get("payload_inline") or "")])
    assert audits, "the operator's audit was produced and then dropped"
    assert '"verdict":"supported"' in audits[0]["payload_inline"]


@live
def test_an_unauditable_target_is_refused_without_waking_id(quiet):
    """Measured before queueing: a claim the record cannot resolve costs no turn.

    A conclusion recorded outside any operation is *not* an example of this:
    since I127 it resolves a dossier of its own evidence, because a claim Id
    was told to audit has to be reachable. Something the record cannot find
    at all still costs nobody a turn.
    """
    before = len(quiet.call("role_turns", role="id")["turns"])
    out = quiet.call("id_audit", conclusion_id="concl_01M34JP5HHF1PS7QKTTFFQMG7X",
                     wait=False)
    assert out["status"] == "failed"
    assert "unknown conclusion" in json.dumps(out)
    time.sleep(3)
    assert len(quiet.call("role_turns", role="id")["turns"]) == before, \
        "Id was woken for an audit that could never be committed"
    assert quiet.call("role_mailbox", role="id")["id"]["queued"] == 0


@live
def test_a_claim_recorded_outside_an_operation_is_still_auditable(quiet):
    """The other half of I127: told to audit it, Id must be able to reach it.

    This used to refuse with "no operation to resolve", so a conclusion the
    digest counts could not be opened at all.
    """
    concl = quiet.call("record_conclusion", claim="with no operation",
                       produced_by="ego")
    dossier = quiet.call("audit_dossier", conclusion_id=concl["conclusion_id"])
    assert dossier["conclusion"]["conclusion_id"] == concl["conclusion_id"]
    assert dossier["events"] == []
    assert "no operation trail" in dossier["note"]

    out = quiet.call("id_audit", conclusion_id=concl["conclusion_id"], wait=False)
    assert out["status"] != "failed", "Id was not woken for an auditable claim"
