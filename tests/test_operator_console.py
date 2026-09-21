"""The operator console, which 1530 tests did not cover.

Two bugs reached a person driving the dashboard for the first time, and
neither could have been caught by anything already here. The suite proves the
operator *verbs* work; it never proved the *console* does, because every HTTP
test opens a fresh connection per request and none of them renders the page.

The two failures:

* the dashboard's JavaScript lives inside a Python string, and four places
  wrapped a long message with Python's adjacent-string-literal concatenation.
  That is a syntax error in JavaScript, so the whole script failed to parse:
  no nav, no panels, no request ever attempted, and a header left saying
  "connecting..." It looked like a connection problem and was a compile error.

* `/operator/rpc` authenticated before reading the request body, so a 401 left
  the POSTed JSON in the socket. On a keep-alive connection the *next* request
  was then parsed from the middle of it, producing

      501 Unsupported method ('{"jsonrpc":...,"method":"operator_overview"...}GET')

  -- the previous body with the next request's verb glued on. The failure
  surfaces one request after the mistake, on the innocent one, which is what
  made it read as a server that had lost its mind.

Both are cheap to prevent and were expensive to diagnose.
"""

from __future__ import annotations

import http.client
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

# The live stack, borrowed rather than rebuilt: the point of these tests is
# that the connection behaves, and that needs a real socket to a real server.
from test_external_interfaces import net  # noqa: F401

live = pytest.mark.skipif(
    sys.platform != "win32",
    reason="live stack fixtures are Windows-only here")


def _script() -> str:
    from amoeba.dashboard import DASHBOARD_HTML

    m = re.search(r"<script>(.*?)</script>", DASHBOARD_HTML, re.S)
    assert m, "the dashboard has no script block"
    return m.group(1)


# ---------------------------------------------------------------------------
# The embedded script has to be valid JavaScript
# ---------------------------------------------------------------------------
def test_the_dashboard_script_has_no_python_string_concatenation():
    """`"a"\\n"b"` concatenates in Python and is a syntax error in JavaScript.

    The script is written inside a Python string, so the habit is easy to
    reach for and the file still imports cleanly either way -- as Python it
    was always valid. This is the specific mistake that broke the console,
    checked without needing a JavaScript engine present.
    """
    lines = _script().splitlines()
    offenders = []
    for n, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped.endswith('"') or stripped.endswith('\\"'):
            continue
        nxt = lines[n + 1].strip() if n + 1 < len(lines) else ""
        if nxt.startswith('"'):
            offenders.append(f"line {n + 1}: {stripped.strip()[:60]}")
    assert not offenders, (
        "adjacent string literals in the dashboard script -- JavaScript needs "
        "`+` between them:\n  " + "\n  ".join(offenders))


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node is not available to parse the script")
def test_the_dashboard_script_parses_as_javascript():
    """The general case, when an engine is around to say so.

    The check above catches the mistake that actually happened; this catches
    the rest of the class. It is skipped rather than assumed when node is
    absent, because a guard that silently does nothing is worse than one that
    says it did nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "dashboard.js"
        js.write_text(_script(), encoding="utf-8")
        r = subprocess.run([shutil.which("node"), "--check", str(js)],
                           capture_output=True, text=True,
                           encoding="utf-8")
    assert r.returncode == 0, (
        "the dashboard script is not valid JavaScript:\n" + r.stderr.strip())


# ---------------------------------------------------------------------------
# Structured metrics are rendered, not stringified
# ---------------------------------------------------------------------------
HARNESS = Path(__file__).with_name("dashboard_render_harness.js")

# The real shape of `pending_decisions.disagreement_pressure`, from
# `pulse._disagreement_pressure`. It is a structured metric carrying a policy
# note, and `kv` used to JSON.stringify the whole thing into a column sized
# for a number.
PRESSURE = {
    "open": 3,
    "oldest_seconds": 187442.5,
    "over_a_day": 2,
    "over_a_week": 0,
    "note": ("open contradictions are never aged out; an unresolved "
             "dispute nobody has addressed is a fact about the organism"),
}


def _render(script: str, payload: dict) -> dict:
    """Run the dashboard's own kv() on a payload and return the tree."""
    node = shutil.which("node")
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "dashboard.js"
        js.write_text(script, encoding="utf-8")
        data = Path(tmp) / "payload.json"
        data.write_text(json.dumps(payload), encoding="utf-8")
        r = subprocess.run([node, str(HARNESS), str(js), str(data)],
                           capture_output=True, text=True,
                           encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr.strip()
    return json.loads(r.stdout)


def _walk(node):
    yield node
    for kid in node["kids"]:
        yield from _walk(kid)


def _pairs(tree):
    """Label -> value, for every leaf metric anywhere in the tree."""
    out = {}
    for node in _walk(tree):
        kids = node["kids"]
        for i, kid in enumerate(kids):
            if kid["tag"] == "dt" and i + 1 < len(kids) and kids[i + 1]["tag"] == "dd":
                out[kid["text"]] = kids[i + 1]["text"]
    return out


def _served_script(net) -> str:
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        conn.request("GET", "/")
        html = conn.getresponse().read().decode("utf-8")
    finally:
        conn.close()
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    assert m, "the served page has no script block"
    return m.group(1)


@live
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to render")
def test_a_structured_metric_is_not_stringified_into_the_value_column(net):
    """The bug, against the page the server actually sends.

    Reading the Python source proves what the string contains; only running
    the renderer proves what an operator sees, which is why this pulls the
    script back over HTTP rather than importing it.
    """
    tree = _render(_served_script(net), {"disagreement_pressure": PRESSURE})

    for node in _walk(tree):
        text = node["text"]
        assert "[object Object]" not in text, node
        assert not (text.startswith("{") or text.startswith("[")), (
            "a structured value was stringified into a cell: " + text[:120])
        assert '"open"' not in text and "oldest_seconds" not in text, text

    pairs = _pairs(tree)
    assert pairs["open"] == "3"
    assert pairs[">24h"] == "2"
    assert pairs[">7d"] == "0"
    # 187442.5s is a bit over two days, and says so.
    assert pairs["oldest"] == "2.2d", pairs["oldest"]

    groups = [n for n in _walk(tree) if n["cls"] == "group"]
    assert [g["text"] for g in groups] == ["disagreement_pressure"], groups


@live
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to render")
def test_an_absent_measurement_reads_as_an_absence(net):
    """`null` in a numeric column reads as a broken reading, not as nothing."""
    quiet = dict(PRESSURE, open=0, oldest_seconds=None, over_a_day=0)
    pairs = _pairs(_render(_served_script(net),
                           {"disagreement_pressure": quiet}))
    assert pairs["oldest"] == "—", pairs["oldest"]
    assert pairs["open"] == "0"


@live
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to render")
def test_the_policy_note_is_kept_but_out_of_the_measurement_column(net):
    """Prose is not a metric, and dropping it is not the alternative."""
    tree = _render(_served_script(net), {"disagreement_pressure": PRESSURE})
    notes = [n for n in _walk(tree) if n["cls"] == "note"]
    assert len(notes) == 1, notes
    assert notes[0]["text"] == PRESSURE["note"], "the note was altered"
    assert notes[0]["title"] == "note"
    # And it is not sitting in a label/value pair, where it would stretch the
    # column it shares with every number on the card.
    assert PRESSURE["note"] not in _pairs(tree).values()


@live
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to render")
def test_the_whole_pending_decisions_card_stays_scalar(net):
    """The card as the pulse actually reports it, scalars and all."""
    pairs = _pairs(_render(_served_script(net), {
        "artifact_proposals": 0,
        "disagreement_pressure": PRESSURE,
        "open_disagreements": 3,
        "unreviewed_conclusions": 7,
        "queued_maintenance": 0,
    }))
    assert pairs["artifact_proposals"] == "0"
    assert pairs["open_disagreements"] == "3"
    assert pairs["unreviewed_conclusions"] == "7"
    assert pairs["open"] == "3"


# ---------------------------------------------------------------------------
# A refused request must not desynchronise the connection
# ---------------------------------------------------------------------------
def _post(conn: http.client.HTTPConnection, token: str, method: str):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": {}})
    conn.request("POST", "/operator/rpc", body=body, headers={
        "Content-Type": "application/json",
        "X-Amoeba-Operator": token,
        "Content-Length": str(len(body)),
    })
    r = conn.getresponse()
    return r.status, r.read()


@live
def test_a_refused_request_does_not_poison_the_next_one(net):
    """The console's first two requests, in the order a browser makes them.

    A browser reuses the connection, so the unauthenticated probe and the
    authenticated retry travel down the same socket. Answering the first
    without reading its body left the JSON in the stream and the second was
    parsed from the middle of it.
    """
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        status, _ = _post(conn, "not-the-token", "operator_overview")
        assert status == 401, "the bad token should be refused"

        status, payload = _post(conn, net.operator_session, "operator_overview")
        assert status == 200, (
            f"the connection was poisoned by the refusal: {status} "
            f"{payload[:200]!r}")
        body = json.loads(payload)
        assert "result" in body, body
    finally:
        conn.close()


@live
def test_several_refusals_in_a_row_leave_the_connection_usable(net):
    """Somebody pasting a token wrongly twice is an ordinary thing to do."""
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        for _ in range(3):
            status, _ = _post(conn, "still-wrong", "operator_overview")
            assert status == 401
        status, payload = _post(conn, net.operator_session, "operator_overview")
        assert status == 200, f"{status} {payload[:200]!r}"
    finally:
        conn.close()


@live
def test_an_unknown_method_does_not_poison_the_next_one(net):
    """The other early-return path, which skips the body just as readily."""
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        status, payload = _post(conn, net.operator_session, "no_such_verb")
        assert status == 200 and b"error" in payload

        status, payload = _post(conn, net.operator_session, "operator_overview")
        assert status == 200, f"{status} {payload[:200]!r}"
        assert "result" in json.loads(payload)
    finally:
        conn.close()


@live
def test_the_dashboard_page_is_served_whole(net):
    """The page itself, which no test had ever fetched."""
    url = urlparse(net.base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
    try:
        conn.request("GET", "/")
        r = conn.getresponse()
        html = r.read().decode("utf-8")
        assert r.status == 200
    finally:
        conn.close()
    assert html.rstrip().endswith("</html>"), "the page was truncated"
    assert '<div id="main"' in html or 'id="main"' in html
    assert "<script>" in html and "</script>" in html
