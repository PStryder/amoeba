"""Arguments are checked before dispatch; a failed call leaves nothing behind.

Two correctness holes from the first live pressure test.

**D4.** A role's tool call reached the verb with whatever the model wrote. Ego
passed `evidence` as a string where a list of objects was wanted, and the
Harness replied `AttributeError: 'str' object has no attribute 'get'` -- an
internal failure, reported to the model as the reason.

**D8.** `RpcClient` kept its connection after a call failed mid-flight. A
timed-out call's reply arrives later, and the *next* call on that connection
read it as its own. Observed live as "cannot read from timed out object", and
able to make a wedged child look reachable, so health could lie.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

from collections.abc import Sequence
from typing import Any

import pytest

from amoeba.argcheck import _matches_any, argument_problem
from amoeba.rpc import RpcClient, RpcError
from conftest import start_stack

live = pytest.mark.skipif(sys.platform != "win32",
                          reason="live stack fixtures are Windows-only here")


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------
def _record_conclusion(*, claim: str, produced_by: str,
                       evidence: Sequence[dict[str, Any]] = (),
                       uncertainty: float | None = None,
                       alternatives: Sequence[str] | None = None,
                       operation_id: str | None = None):
    return {}


@pytest.mark.parametrize("arguments, expected", [
    ({"evidence": "chronyc said so"},
     "argument 'evidence' must be a list of objects, got a string"),
    ({"evidence": [{"note": "a"}, "b"]},
     "argument 'evidence' must be a list of objects, got a list whose item 1 is a string"),
    ({"uncertainty": "high"},
     "argument 'uncertainty' must be a number or null, got a string"),
    ({"alternatives": "only one"},
     "argument 'alternatives' must be a list of strings or null, got a string"),
])
def test_the_live_mistakes_are_refused_in_words(arguments, expected):
    base = {"claim": "c", "produced_by": "ego"}
    assert argument_problem(_record_conclusion, {**base, **arguments}) == expected


@pytest.mark.parametrize("arguments", [
    {"evidence": [{"note": "a"}]}, {"evidence": []}, {"uncertainty": 0.2},
    {"uncertainty": 1}, {"uncertainty": None}, {"alternatives": ["x", "y"]},
])
def test_correct_calls_pass(arguments):
    assert argument_problem(_record_conclusion,
                            {"claim": "c", "produced_by": "ego", **arguments}) is None


def test_what_the_harness_binds_is_not_the_models_to_get_wrong():
    assert argument_problem(_record_conclusion,
                            {"claim": "c", "produced_by": "ego", "operation_id": 7},
                            skip=frozenset({"operation_id"})) is None


class _Unfamiliar:
    pass


def _takes_the_unfamiliar(*, target: _Unfamiliar, when: "datetime | None" = None):
    return {}


def test_an_annotation_the_checker_cannot_read_is_let_through():
    """Unsure is not wrong: a type added to a verb later must not start refusing."""
    assert argument_problem(_takes_the_unfamiliar,
                            {"target": "anything", "when": "2026-09-22"}) is None


def _sample(annotation: str):
    """A value an annotation plainly accepts, built from the annotation itself."""
    first = annotation.replace(" ", "").split("|")[0]
    base = first.split("[", 1)[0].replace("typing.", "")
    return {"str": "x", "int": 1, "float": 0.5, "bool": True, "None": None,
            "dict": {}, "Mapping": {}, "list": [], "Sequence": [], "tuple": [],
            "Iterable": [], "Any": "x"}.get(base, "x")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = start_stack(tmp_path_factory.mktemp("harden"),
                    scheduler={"id_startup_turn": False})
    s.wait_for_children(timeout=90)
    yield s
    s.stop()


@live
def test_no_correct_call_on_the_role_surface_is_refused(stack):
    """Over-strict validation would be worse than the crash it prevents."""
    checked = 0
    for role in ("ego", "id"):
        for cap in stack.call("role_environment", role=role)["manifest"]["capabilities"]:
            for arg in cap["arguments"]:
                annotation = arg.get("type")
                if not annotation:
                    continue
                value = _sample(annotation)
                if arg.get("allowed"):
                    value = ([arg["allowed"][0]] if arg.get("kind") == "list"
                             else arg["allowed"][0])
                assert _matches_any(annotation, value) is not False, \
                    (role, cap["verb"], arg["name"], annotation, value)
                checked += 1
    assert checked > 50


@live
def test_a_wrong_type_is_refused_before_the_verb_runs(stack):
    """The live AttributeError, replayed: now a sentence, and nothing written."""
    from amoeba.rpc import read_or_create_token

    cfg = stack.cfg
    inf = RpcClient(cfg.supervisor_host, cfg.inference_port,
                    read_or_create_token(cfg.token_path), timeout=60)
    inf.connect(retries=20, delay=0.5)
    call = json.dumps({"name": "record_conclusion", "arguments": {
        "claim": "gw-3 drifted", "produced_by": "ego", "evidence": "chronyc said so"}})
    inf.call("script_responses", responses=[
        {"role": "ego", "text": f"<tool_call>{call}</tool_call>",
         "finish_reason": "stop", "continues": True},
        {"role": "ego", "text": "Recorded nothing.", "finish_reason": "stop",
         "continues": True}])
    env = stack.call("ego_converse", message="conclude", wait=False)
    deadline = time.time() + 120
    while stack.call("role_answer", trigger_id=env["result"]["trigger_id"])[
            "status"] not in ("completed", "incomplete", "unanswerable"):
        assert time.time() < deadline
        time.sleep(0.3)
    invoked = [json.loads(e["payload_inline"]) for e in
               stack.call("history", kinds=["role.tool_invoked"], limit=20)
               if '"record_conclusion"' in (e.get("payload_inline") or "")]
    assert invoked and invoked[-1]["accepted"] is False
    reason = invoked[-1]["reason"]
    assert "argument 'evidence' must be a list of objects, got a string" in reason
    assert "AttributeError" not in reason
    assert stack.call("history", operation_id=env["operation_id"],
                      kinds=["conclusion.recorded"], limit=5) == []


# ---------------------------------------------------------------------------
# D8
# ---------------------------------------------------------------------------
class _SlowServer:
    """Answers each request with its own id; the first one late."""

    def __init__(self, first_delay: float):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.first_delay = first_delay
        self.served = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        f = conn.makefile("rwb")
        try:
            f.readline()                                   # the token handshake
            f.write(b'{"ok": true, "result": {}}\n')
            f.flush()
            while True:
                line = f.readline()
                if not line:
                    return
                req = json.loads(line)
                self.served += 1
                if self.served == 1:
                    time.sleep(self.first_delay)          # arrives after the timeout
                f.write((json.dumps({"id": req["id"], "ok": True,
                                     "result": {"answering": req["method"]}}) + "\n")
                        .encode())
                f.flush()
        except OSError:
            return

    def close(self):
        self.sock.close()


def test_a_late_reply_is_never_read_as_the_next_calls():
    """The stale-probe desync, reduced to a socket."""
    server = _SlowServer(first_delay=0.6)
    try:
        client = RpcClient("127.0.0.1", server.port, "t", timeout=0.25)
        client.connect()
        with pytest.raises(Exception):
            client.call("first")                          # times out
        time.sleep(0.8)                                   # its reply is now in flight
        assert client.call("second") == {"answering": "second"}, \
            "the second call read the first call's late reply"
    finally:
        server.close()


def test_a_reply_to_another_request_is_refused():
    """Belt and braces: an id that does not match is not believed."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _ = server.accept()
        f = conn.makefile("rwb")
        f.readline()
        f.write(b'{"ok": true, "result": {}}\n')
        f.flush()
        f.readline()
        f.write(b'{"id": 999, "ok": true, "result": "not yours"}\n')
        f.flush()
        time.sleep(0.5)

    threading.Thread(target=serve, daemon=True).start()
    try:
        client = RpcClient("127.0.0.1", port, "t", timeout=2)
        client.connect()
        with pytest.raises(RpcError, match="different request"):
            client.call("mine")
        assert not client.connected, "a desynchronised connection was kept"
    finally:
        server.close()
