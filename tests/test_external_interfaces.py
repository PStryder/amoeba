"""External I/O clients provide input. They do not govern.

The distinction this file defends:

    external input changing what Amoeba thinks about
        is not
    external control mutating Amoeba's protected state

A client may say "stop investigating and answer with what you have". Ego may
read that and decide to cancel work, and that cancellation is Amoeba using its
own authority. Exposing `cancel_work(work_id)` to the client is a different
thing, and no route to it exists from the external surface.

Every attack below is run over a **real HTTP connection holding a real
external API key**. Reading a table proves what the table says; only a request
proves what the adapter will dispatch.

Before this work the MCP facade held `cfg.token_path` -- the control token, the
full method table -- and exposed 23 tools including file write and delete,
artifact promotion, Id maintenance, blackboard posting and cancellation. Every
MCP client was effectively the operator. That is what these tests exist to stop
coming back.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request

import pytest

from amoeba.rpc import RpcClient, read_or_create_token
from amoeba.scopes import ego_only_verbs, external_io_verbs, id_only_verbs
from conftest import start_stack

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="live stack fixtures are Windows-only here")

# Everything an external client must not be able to reach, by category.
CONTROL_ATTEMPTS = [
    # work admission and scheduling
    ("admit_work", {"objective": "x", "work_class": "user", "origin_actor": "ego"}),
    ("lease_work", {"neuocyte_id": "x"}),
    ("cancel_work", {"work_id": "x"}),
    ("kill_all_neuocytes", {}),
    ("ego_request_work", {"objective": "x"}),
    ("ego_request_cancellation", {"work_id": "x", "reason": "x"}),
    ("ego_work_message", {"work_id": "x", "message": "x"}),
    # blackboard
    ("board_post", {"author": "x", "author_kind": "x", "post_type": "note",
                    "body": "x"}),
    ("board_read", {"reader": "x"}),
    ("board_promote_to_memory", {"post_id": "x"}),
    # Id
    ("system_pulse", {}),
    ("id_health", {}),
    ("id_raise_finding", {"claim": "x"}),
    ("id_propose_prompt", {"role": "ego", "prompt": "x", "rationale": "x"}),
    ("id_request_investigation", {"objective": "x"}),
    # maintained state
    ("remember", {"kind": "belief", "claim": "x", "confidence": 1.0,
                  "created_by": "ego"}),
    ("ego_propose_memory", {"claim": "x"}),
    ("recall", {}),
    ("get_memory", {"memory_id": "x"}),
    # artifact and prompt governance
    ("artifact_promote", {"artifact_id": "x"}),
    ("artifact_reject", {"artifact_id": "x", "reason": "x"}),
    ("artifact_list", {}),
    ("operator_prompt_decide", {"proposal_id": "x", "decision": "accept"}),
    ("operator_prompt_library", {}),
    # filespace, security, sandbox
    ("file_write", {"root": "out", "path": "x", "content": "y"}),
    ("file_read", {"root": "out", "path": "../../secrets"}),
    ("file_roots", {}),
    ("file_attach", {"path": "C:/Windows/win.ini", "work_id": "x"}),
    ("sandbox_run", {"sandbox_id": "x", "code": "print(1)"}),
    ("sandbox_files", {"sandbox_id": "x"}),
    ("tool_invoke", {"neuocyte_id": "x", "work_id": "y", "fencing_token": 1,
                     "name": "run_code"}),
    # homeostasis and lifecycle
    ("context_rejuvenate", {"role": "ego", "reason": "x"}),
    ("retire_session", {"session_id": "x", "reason": "x"}),
    ("shutdown", {}),
    # operator surface and history
    ("operator_overview", {}),
    ("operator_consult_id", {"question": "x"}),
    ("operator_backchannel", {}),
    ("history", {}),
    ("provenance", {"operation_id": "x"}),
    ("verify_integrity", {}),
]


@pytest.fixture()
def net(tmp_path):
    s = start_stack(tmp_path)
    cfg = s.cfg
    s.base = f"http://{cfg.api_host}:{cfg.api_port}"      # type: ignore[attr-defined]
    keys = json.loads(cfg.api_clients_path.read_text(encoding="utf-8"))
    s.api_key = next(iter(keys))                          # type: ignore[attr-defined]
    s.operator_session = cfg.operator_session_path.read_text(   # type: ignore[attr-defined]
        encoding="utf-8").strip()
    yield s
    s.stop()


def call(base, method, params=None, *, key=None, operator=None, origin=None,
         path="/rpc", timeout=120):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": params or {}}).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if operator:
        req.add_header("X-Amoeba-Operator", operator)
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"body": exc.read().decode("utf-8", "replace")}


# ===========================================================================
# Input in, output out
# ===========================================================================
def test_an_api_client_can_submit_input_and_collect_output(net):
    status, res = call(net.base, "io_submit",
                       {"text": "What is six times seven?"}, key=net.api_key)
    assert status == 200 and "result" in res, res
    interaction_id = res["result"]["interaction_id"]
    assert res["result"]["status"] == "accepted"
    assert len(res["result"]["input_sha256"]) == 64

    _s, out = call(net.base, "io_await",
                   {"interaction_id": interaction_id, "timeout_seconds": 90},
                   key=net.api_key)
    assert out["result"]["status"] == "complete", out
    assert out["result"]["output"] is not None

    _s, listed = call(net.base, "io_list", {}, key=net.api_key)
    assert listed["result"]["count"] == 1


def test_the_mcp_adapter_offers_only_the_io_surface(net):
    """MCP is a cognitive service interface, not a control plane.

    It used to expose 23 tools over the control token, including file write,
    file delete, artifact promotion and Id maintenance.
    """
    import asyncio
    import warnings

    from amoeba.mcp_api import build_server

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcp, facade = build_server(net.cfg)
        names = {t.name for t in asyncio.run(mcp.list_tools())}

    assert names == {"amoeba_capabilities", "amoeba_ask", "amoeba_submit",
                     "amoeba_status", "amoeba_output", "amoeba_list",
                     "amoeba_attach", "amoeba_result"}, names
    blob = json.dumps(sorted(names))
    for banned in ("file_write", "file_delete", "artifact_promote",
                   "id_maintenance", "board_post", "cancel"):
        assert banned not in blob

    # And its credential is the narrow one.
    external = read_or_create_token(net.cfg.scope_token_path("external_io"))
    control = read_or_create_token(net.cfg.token_path)
    assert facade.token == external
    assert facade.token != control, (
        "the MCP facade holds the control token; every MCP client would be an "
        "operator")


def test_mcp_and_api_reach_the_same_semantic_operations(net):
    """One core, two adapters. Not two implementations."""
    token = read_or_create_token(net.cfg.scope_token_path("external_io"))
    rpc = RpcClient(net.cfg.supervisor_host, net.cfg.supervisor_port, token,
                    timeout=120)
    rpc.connect(retries=10, delay=0.3)

    via_mcp = rpc.call("io_submit", client_id="mcp", surface="mcp",
                       text="via the mcp adapter")
    _s, via_api = call(net.base, "io_submit", {"text": "via the http adapter"},
                       key=net.api_key)
    assert via_mcp["status"] == "accepted"
    assert via_api["result"]["status"] == "accepted"

    # Each sees only its own.
    mine = rpc.call("io_list", client_id="mcp")
    assert all(i["interaction_id"] != via_api["result"]["interaction_id"]
               for i in mine["interactions"]), (
        "one client's interaction is visible to another")


def test_attached_bytes_enter_as_admitted_input_with_exact_provenance(net):
    raw = bytes([0x89]) + b"PNG\r\n" + bytes(range(200, 240))
    _s, res = call(net.base, "io_attach_input",
                   {"filename": "image.bin",
                    "content_base64": base64.b64encode(raw).decode()},
                   key=net.api_key)
    out = res["result"]
    assert out["bytes"] == len(raw)
    import hashlib
    assert out["sha256"] == hashlib.sha256(raw).hexdigest(), (
        "the bytes admitted are not the bytes sent")
    assert out["receipt_id"]

    kinds = [e["kind"] for e in net.call("history", limit=300)]
    assert "interaction.input_attached" in kinds


@pytest.mark.parametrize("bad", ["../escape.txt", "C:/Windows/win.ini",
                                 "sub/dir.txt", "..\\escape", ".hidden"])
def test_an_attachment_name_is_a_label_not_a_path(net, bad):
    _s, res = call(net.base, "io_attach_input",
                   {"filename": bad,
                    "content_base64": base64.b64encode(b"x").decode()},
                   key=net.api_key)
    assert "error" in res, f"{bad!r} was accepted as a filename"


def test_a_client_can_stream_its_own_progress(net):
    """SSE carries this client's interactions, not the event log."""
    import threading

    seen: list[str] = []

    def listen():
        req = urllib.request.Request(net.base + "/events?seconds=8")
        req.add_header("Authorization", f"Bearer {net.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                for line in r:
                    text = line.decode("utf-8", "replace").strip()
                    if text.startswith("data:"):
                        seen.append(text[5:].strip())
                        return
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    time.sleep(0.6)
    call(net.base, "io_submit", {"text": "stream me"}, key=net.api_key)
    t.join(timeout=15)
    assert seen, "no event reached the stream"
    assert "interaction" in seen[0]


# ===========================================================================
# Control attempts
# ===========================================================================
@pytest.mark.parametrize("method,params", CONTROL_ATTEMPTS,
                         ids=[m for m, _ in CONTROL_ATTEMPTS])
def test_an_external_client_cannot_reach_a_control_verb(net, method, params):
    """Absence, not refusal.

    The adapter dispatches nothing outside its own list, and the connection it
    holds could not name these verbs even if it did.
    """
    status, res = call(net.base, method, params, key=net.api_key)
    assert status == 200 and "error" in res, (
        f"{method} was reachable from the external surface: {res}")
    assert res["error"]["code"] == -32601, res["error"]


def test_spoofed_identity_fields_buy_nothing(net):
    """Identity is the credential; these are discarded, not honoured."""
    _s, mine = call(net.base, "io_submit", {"text": "mine"}, key=net.api_key)
    interaction_id = mine["result"]["interaction_id"]

    for forged in ({"client_id": "someone_else"}, {"role": "operator"},
                   {"actor": "id"}, {"caller": "ego"}, {"scope": "operator"},
                   {"from_role": "ego"}, {"operator": True}):
        _s, res = call(net.base, "io_status",
                       {"interaction_id": interaction_id, **forged},
                       key=net.api_key)
        assert "result" in res, (
            f"{forged} broke a legitimate call instead of being ignored")

    # And a forged client_id does not reach another client's interaction.
    token = read_or_create_token(net.cfg.scope_token_path("external_io"))
    rpc = RpcClient(net.cfg.supervisor_host, net.cfg.supervisor_port, token,
                    timeout=60)
    rpc.connect(retries=10, delay=0.3)
    other = rpc.call("io_submit", client_id="other_client", text="not yours")
    _s, res = call(net.base, "io_status",
                   {"interaction_id": other["interaction_id"]}, key=net.api_key)
    assert "error" in res, "a client read another client's interaction"


def test_discovery_does_not_reveal_privileged_methods(net):
    status, body = _get(net.base + "/rpc")
    assert status == 200
    advertised = set(body["methods"])
    assert advertised == set(external_io_verbs())
    blob = json.dumps(body)
    for verb in (list(id_only_verbs()) + list(ego_only_verbs()) +
                 ["admit_work", "artifact_promote", "operator_overview",
                  "system_pulse", "file_write"]):
        assert verb not in blob, f"discovery leaked {verb}"


def test_an_unknown_method_does_not_enumerate_the_surface(net):
    status, res = call(net.base, "definitely_not_real", {}, key=net.api_key)
    assert res["error"]["code"] == -32601
    data = res["error"].get("data") or {}
    assert not any(isinstance(v, list) for v in data.values()), data


def test_a_credential_is_required_even_on_loopback(net):
    """Binding to 127.0.0.1 is not authentication."""
    status, _ = call(net.base, "io_capabilities", {})
    assert status == 401
    status, _ = call(net.base, "io_capabilities", {}, key="not-a-real-key")
    assert status == 401


def test_a_cross_origin_browser_request_is_refused(net):
    status, _ = call(net.base, "io_capabilities", {}, key=net.api_key,
                     origin="http://evil.example")
    assert status == 403
    status, _ = call(net.base, "io_capabilities", {}, key=net.api_key,
                     origin=f"http://127.0.0.1:{net.cfg.api_port}")
    assert status == 200


def test_an_api_key_cannot_reach_the_operator_surface(net):
    status, _ = call(net.base, "operator_overview", {}, key=net.api_key,
                     path="/operator/rpc")
    assert status == 401, "an API key authenticated into the operator surface"


def test_disconnecting_does_not_cancel_anything(net):
    """A dropped client is not a control signal.

    Cancellation is a decision Amoeba makes; a socket closing is not one, and
    treating it as one would hand every client an implicit control verb.
    """
    _s, res = call(net.base, "io_submit",
                   {"text": "keep going after I leave"}, key=net.api_key)
    interaction_id = res["result"]["interaction_id"]
    time.sleep(0.3)
    _s, out = call(net.base, "io_await",
                   {"interaction_id": interaction_id, "timeout_seconds": 90},
                   key=net.api_key)
    assert out["result"]["status"] == "complete"
    kinds = [e["kind"] for e in net.call("history", limit=400)]
    assert "work.cancelled" not in kinds


# ===========================================================================
# Operator surface
# ===========================================================================
def test_the_operator_can_govern_through_the_harness(net):
    status, res = call(net.base, "operator_overview", {},
                       operator=net.operator_session, path="/operator/rpc")
    assert status == 200 and "result" in res, res
    overview = res["result"]
    assert set(overview) >= {"harness", "roles", "work", "scheduler",
                             "resources", "pending_decisions", "failures"}

    _s, lib = call(net.base, "operator_prompt_library", {},
                   operator=net.operator_session, path="/operator/rpc")
    assert "current" in lib["result"]


def test_operator_governance_actions_are_receipted(net):
    """Human authority, recorded like any other."""
    token = read_or_create_token(net.cfg.scope_token_path("id"))
    idc = RpcClient(net.cfg.supervisor_host, net.cfg.supervisor_port, token,
                    timeout=60)
    idc.connect(retries=10, delay=0.3)
    proposal = idc.call("id_propose_prompt", role="ego",
                        prompt="Be concise.", rationale="verbosity")

    _s, decided = call(net.base, "operator_prompt_decide",
                       {"proposal_id": proposal["proposal_id"],
                        "decision": "accept", "rationale": "agreed"},
                       operator=net.operator_session, path="/operator/rpc")
    assert decided["result"]["receipt_id"], decided

    events = net.call("history", limit=400)
    decisions = [e for e in events if e["kind"] == "prompt.decided"]
    assert decisions and decisions[-1]["actor_id"] == "operator"
    assert net.call("verify_integrity", deep=True)["hash_chain_ok"] is True


def test_accepting_a_prompt_does_not_silently_change_cognition(net):
    """A decision is recorded; what a running role thinks with is unchanged."""
    token = read_or_create_token(net.cfg.scope_token_path("id"))
    idc = RpcClient(net.cfg.supervisor_host, net.cfg.supervisor_port, token,
                    timeout=60)
    idc.connect(retries=10, delay=0.3)
    before = idc.call("system_pulse", max_age_seconds=0)["resources"]
    proposal = idc.call("id_propose_prompt", role="ego", prompt="Totally new.",
                        rationale="test")
    call(net.base, "operator_prompt_decide",
         {"proposal_id": proposal["proposal_id"], "decision": "accept"},
         operator=net.operator_session, path="/operator/rpc")

    after = idc.call("system_pulse", max_age_seconds=0)["resources"]
    assert (after["configured"]["prompt.ego"]["sha256"]
            == before["configured"]["prompt.ego"]["sha256"])
    assert (after["embodied"]["prompt.ego"]["sha256"]
            == before["embodied"]["prompt.ego"]["sha256"])


def test_the_dashboard_never_touches_the_database_or_filesystem(net):
    """A cockpit, not an authority. Running locally grants nothing.

    Checked as source, because this is a claim about what the code *can* do
    rather than what a particular request happened to do.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "amoeba"
    for name in ("dashboard.py", "http_api.py", "operator_api.py"):
        text = (src / name).read_text(encoding="utf-8")
        for banned in ("sqlite3", "executescript", "Database(", "shutil.rmtree",
                       "os.remove", "os.unlink"):
            assert banned not in text, (
                f"{name} reaches past the Harness using {banned}")

    status, body = _get(net.base + "/")
    assert status == 200 and "operator console" in body.lower()


def test_the_operator_surface_and_the_external_surface_are_separate_tables(net):
    """Same listener, different method tables, different credentials."""
    # The operator path does not serve external verbs.
    status, res = call(net.base, "io_submit", {"text": "x"},
                       operator=net.operator_session, path="/operator/rpc")
    assert res["error"]["code"] == -32601

    # And the external path does not serve operator verbs.
    status, res = call(net.base, "operator_overview", {}, key=net.api_key)
    assert res["error"]["code"] == -32601


def _get(url, key=None):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


# ===========================================================================
# The full walk: discover, believe the discovery, then ignore it entirely.
# ===========================================================================
def test_discovery_then_calling_the_exact_operator_verb_anyway(net):
    """Authenticate, discover, then POST the real names regardless.

    The point is the *shape* of the failure. Not:

        403 forbidden: role 'external' may not call operator_overview

    which would mean the operation exists here and something decided against
    it, and would be one refactor away from deciding differently. But:

        -32601 unknown method 'operator_overview'

    which means there is no such operation on this surface at all.

    The verb names are taken from the live surfaces rather than typed in, so
    this cannot quietly pass by guessing names that were never real. And the
    absence is checked twice: once at the adapter, and once against the
    credential the adapter actually holds -- so even a bypass of the adapter's
    own list reaches a connection that cannot name these either.
    """
    from amoeba.operator_api import OPERATOR_VERBS

    # 1. Authenticate and ask what exists here.
    status, discovery = _get(net.base + "/rpc")
    assert status == 200
    advertised = set(discovery["methods"])

    # 2. It is I/O and nothing else.
    assert advertised == set(external_io_verbs())
    assert all(v.startswith("io_") for v in advertised), advertised

    # 3. Take the *real* names of things that exist elsewhere in the system.
    elsewhere = sorted(
        (set(OPERATOR_VERBS) | set(id_only_verbs()) | set(ego_only_verbs()))
        - advertised)
    assert len(elsewhere) > 30, (
        f"only {len(elsewhere)} privileged verbs to try; this test would be "
        "proving very little")

    # 4. Call them anyway, with a valid credential, spelled correctly.
    denied_rather_than_absent = []
    for verb in elsewhere:
        http_status, res = call(net.base, verb, {}, key=net.api_key)
        assert http_status == 200, (verb, http_status, res)
        assert "result" not in res, f"{verb} EXECUTED on the external surface"
        code = res["error"]["code"]
        message = res["error"]["message"].lower()

        if code != -32601:
            denied_rather_than_absent.append((verb, code, message))
            continue
        assert "unknown method" in message, (verb, message)
        # Nothing in the refusal should read as an authorization decision.
        #
        # Scanned with the echoed verb name removed. The message quotes back
        # what was asked for, which is fine and useful, but a verb like
        # `role_environment` would otherwise trip the "role" check on its own
        # name rather than on anything the Harness said about authority.
        said = message.replace(f"'{verb.lower()}'", "").replace(verb.lower(), "")
        for word in ("forbidden", "denied", "not allowed", "permission",
                     "unauthorized", "insufficient", "role", "privilege"):
            assert word not in said, (
                f"{verb} was refused on authorization grounds ({word!r}); the "
                "operation should not exist here at all")

    assert not denied_rather_than_absent, (
        "these were refused rather than absent, which means the route exists "
        f"and a check rejected it: {denied_rather_than_absent[:5]}")

    # 5. The same names, against the credential the adapter itself holds.
    #    Even bypassing the adapter's own list reaches a connection that was
    #    never given these verbs.
    token = read_or_create_token(net.cfg.scope_token_path("external_io"))
    rpc = RpcClient(net.cfg.supervisor_host, net.cfg.supervisor_port, token,
                    timeout=60)
    rpc.connect(retries=10, delay=0.3)
    for verb in elsewhere:
        with pytest.raises(Exception) as exc:
            rpc.call(verb)
        assert "unknown method" in str(exc.value), (
            f"the external adapter's own credential can reach {verb}")

    # 6. And the surface still works for what it is actually for.
    _s, ok = call(net.base, "io_capabilities", {}, key=net.api_key)
    assert set(ok["result"]["verbs"]) == set(external_io_verbs())


def test_the_adapter_and_the_credential_agree_on_the_surface(net):
    """Two independent lists define the external surface, on purpose.

    The HTTP adapter dispatches only what `io_api.EXTERNAL_VERBS` names, and
    the credential it holds can only reach what `scopes.EXTERNAL_IO` grants.
    Either alone would be sufficient; having both means one mistake does not
    open the door.

    The risk of two lists is drift, so this pins the relationship: the adapter
    may never advertise or dispatch something the credential cannot reach, and
    neither may contain anything that is not an I/O verb.
    """
    from amoeba.io_api import EXTERNAL_VERBS

    adapter = set(EXTERNAL_VERBS)
    credential = set(external_io_verbs())

    assert adapter <= credential, (
        f"the adapter advertises verbs its credential cannot reach: "
        f"{sorted(adapter - credential)}")
    assert credential <= adapter, (
        f"the credential grants verbs the adapter never exposes -- capability "
        f"nobody asked for: {sorted(credential - adapter)}")
    for verb in adapter | credential:
        assert verb.startswith("io_"), (
            f"{verb} is on the external surface and is not an I/O verb")
