"""A receipt naming a digest must be verifiable from inside the container.

The environment has to be trustworthy enough that a neuocyte can reason from it
as ground truth. If the Harness says "this is what you were given", that
statement cannot be approximate.

**These tests deliberately do not hash anything from the test process.** They
hash from *inside the sandbox*, using the neuocyte's own execution environment,
because that is the only view whose agreement means anything. A check the
Harness performs on itself proves the Harness is self-consistent; it does not
prove the neuocyte received those bytes. The distinction is the whole point:
the first version of `file_attach` passed every Harness-side check while
handing the sandbox different bytes, because it routed content through `str`
and replaced every non-UTF-8 byte with U+FFFD.

The invariant, stated once:

    If a receipt claims a neuocyte received bytes with digest D, then hashing
    the bytes actually available to that neuocyte must produce D.

`read_file` is the one surface where what a neuocyte *sees* legitimately
differs, because it returns text and a non-UTF-8 file cannot round-trip through
text. That is allowed, but it must be declared and the true digest must still
be reported, so the neuocyte can detect the discrepancy itself rather than
reasoning about U+FFFD soup as though it were the file.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from conftest import start_stack


@pytest.fixture()
def gstack(tmp_path):
    src = tmp_path / "project"
    out = tmp_path / "amoeba-out"
    src.mkdir()
    out.mkdir()
    extra = "\n".join([
        "[[filespace.roots]]",
        'name = "project"',
        f'path = "{src.as_posix()}"',
        'mode = "read_only"',
        "",
        "[[filespace.roots]]",
        'name = "out"',
        f'path = "{out.as_posix()}"',
        'mode = "read_write"',
    ])
    s = start_stack(tmp_path, extra_toml=extra)
    s.src = src        # type: ignore[attr-defined]
    s.out = out        # type: ignore[attr-defined]
    if not s.call("sandbox_capabilities").get("sandbox_available"):
        s.stop()
        pytest.skip("sandbox unavailable")
    yield s
    s.stop()


HASH_IN_SANDBOX = """import hashlib, pathlib
d = pathlib.Path(NAME).read_bytes()
print('SANDBOX_SHA256', hashlib.sha256(d).hexdigest())
print('SANDBOX_BYTES', len(d))
"""


def sandbox_digest(stack, work_id, token, neuocyte_id, name):
    """Hash a sandbox file *from inside the container*."""
    code = HASH_IN_SANDBOX.replace("NAME", repr(name))
    out = stack.call("tool_invoke", neuocyte_id=neuocyte_id, work_id=work_id,
                     fencing_token=token, name="run_code",
                     arguments={"code": code})
    assert out["accepted"], out
    assert out["result"]["exit_code"] == 0, out["result"]["stderr"]
    fields = {}
    for line in out["result"]["stdout"].splitlines():
        parts = line.split()
        if len(parts) == 2:
            fields[parts[0]] = parts[1]
    assert "SANDBOX_SHA256" in fields, out["result"]["stdout"]
    return fields["SANDBOX_SHA256"], int(fields["SANDBOX_BYTES"])


def admit_and_lease(stack, objective, neuocyte_id):
    work = stack.call("admit_work", objective=objective, work_class="user",
                      origin_actor="pete", sandbox_allowed=True)
    item = stack.call("lease_work", neuocyte_id=neuocyte_id,
                      work_id=work["work_id"])
    assert item, "could not lease the work item"
    return work["work_id"], item["fencing_token"]


PAYLOADS = [
    pytest.param(bytes([0x89]) + b"PNG\r\n\x1a\n" + bytes(range(200, 256)),
                 id="binary-png-header"),
    pytest.param(b"def f():\n    return 1\n", id="ascii-text"),
    pytest.param("naïve résumé — café\n".encode("utf-8"),
                 id="non-ascii-utf8"),
    pytest.param(b"\xff\xfe\x00\x00not valid in any encoding\x00\x80\x81",
                 id="invalid-utf8"),
    pytest.param(bytes(range(256)), id="every-byte-value"),
    pytest.param(b"", id="empty"),
    pytest.param(b"\r\n\r\n mixed \n line \r endings \r\n", id="line-endings"),
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_an_attached_files_digest_is_what_the_neuocyte_can_hash(gstack, payload):
    """The invariant, as a round trip through the container."""
    (gstack.src / "payload.bin").write_bytes(payload)
    work_id, token = admit_and_lease(gstack, "verify an attachment", "nc_gt")
    try:
        att = gstack.call("file_attach", path=str(gstack.src / "payload.bin"),
                          work_id=work_id, actor="pete")
        claimed = att["sha256"]
        assert claimed == hashlib.sha256(payload).hexdigest()

        observed, n = sandbox_digest(gstack, work_id, token, "nc_gt",
                                     "payload.bin")
        assert observed == claimed, (
            f"the receipt claims digest {claimed} but the bytes the neuocyte "
            f"can actually read hash to {observed}")
        assert n == att["bytes"] == len(payload)
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")


@pytest.mark.parametrize("payload", PAYLOADS)
def test_the_durable_event_carries_the_same_digest(gstack, payload):
    """Not just the return value: the claim that outlives the call.

    A return value nobody rereads is not provenance. The event log is what an
    audit sees months later, so it has to carry the digest that was actually
    delivered.
    """
    (gstack.src / "payload.bin").write_bytes(payload)
    work_id, token = admit_and_lease(gstack, "verify the event", "nc_ev")
    try:
        gstack.call("file_attach", path=str(gstack.src / "payload.bin"),
                    work_id=work_id, actor="pete")
        observed, _n = sandbox_digest(gstack, work_id, token, "nc_ev",
                                      "payload.bin")

        events = [e for e in gstack.call("history", limit=600)
                  if e["kind"] == "file.attached"]
        assert events, "no file.attached event was recorded"
        recorded = json.loads(events[-1]["payload_inline"])
        assert recorded["sha256"] == observed, (
            f"the event log claims {recorded['sha256']}; the sandbox holds "
            f"{observed}. An audit months from now would read a false claim.")
        assert recorded["sandbox_sha256"] == observed
        assert recorded["bytes"] == len(payload)
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")


def test_a_digest_a_neuocyte_was_told_it_wrote_is_what_is_on_disk(gstack):
    """Same invariant, the other direction.

    `write_file` hands a digest back to the model. If that is not what the
    sandbox now holds, the model is reasoning from a false statement about its
    own actions -- which is worse than no statement.
    """
    work_id, token = admit_and_lease(gstack, "write and verify", "nc_w")
    try:
        content = "café — naïve\n"
        w = gstack.call("tool_invoke", neuocyte_id="nc_w", work_id=work_id,
                        fencing_token=token, name="write_file",
                        arguments={"path": "work/note.txt", "content": content})
        assert w["accepted"], w
        claimed = w["result"]["sha256"]
        observed, _n = sandbox_digest(gstack, work_id, token, "nc_w", "note.txt")
        assert observed == claimed, (
            f"write_file reported {claimed}; the file hashes to {observed}")
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")


def test_a_proposed_artifacts_digest_matches_what_the_sandbox_holds(gstack):
    """At the point where bytes leave the sandbox.

    Promotion re-hashes and refuses a mismatch (I34). That check is only
    meaningful if the digest recorded at proposal described the real file.
    """
    work_id, token = admit_and_lease(gstack, "propose and verify", "nc_pr")
    try:
        gstack.call("tool_invoke", neuocyte_id="nc_pr", work_id=work_id,
                    fencing_token=token, name="write_file",
                    arguments={"path": "work/out.py", "content": "x = 1\n"})
        prop = gstack.call("tool_invoke", neuocyte_id="nc_pr", work_id=work_id,
                           fencing_token=token, name="propose_artifact",
                           arguments={"path": "work/out.py", "rationale": "r"})
        assert prop["accepted"], prop
        claimed = prop["result"]["sha256"]
        observed, _n = sandbox_digest(gstack, work_id, token, "nc_pr", "out.py")
        assert observed == claimed, (
            f"the proposal claims {claimed}; the sandbox holds {observed}")
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")


def test_a_lossy_read_declares_itself_and_still_reports_the_true_digest(gstack):
    """The one surface where what a neuocyte sees legitimately differs.

    `read_file` returns text, so a non-UTF-8 file cannot round-trip through it.
    Allowed -- but it must say so, and the digest it reports must still be the
    real one, so the neuocyte can notice the discrepancy rather than reasoning
    about U+FFFD soup as though it were the file.
    """
    payload = bytes([0x89]) + b"PNG" + bytes(range(200, 240))
    (gstack.src / "img.bin").write_bytes(payload)
    work_id, token = admit_and_lease(gstack, "read a binary", "nc_lr")
    try:
        att = gstack.call("file_attach", path=str(gstack.src / "img.bin"),
                          work_id=work_id, actor="pete")
        seen = gstack.call("tool_invoke", neuocyte_id="nc_lr", work_id=work_id,
                           fencing_token=token, name="read_file",
                           arguments={"path": "work/img.bin"})
        assert seen["accepted"], seen
        res = seen["result"]

        assert res["lossy_decode"] is True
        assert "not what the file contains" in res["note"]
        assert res["content"].encode("utf-8") != payload, (
            "premise: this payload should not survive a text round trip")

        # The digest is the real one, so the discrepancy is detectable.
        observed, _n = sandbox_digest(gstack, work_id, token, "nc_lr", "img.bin")
        assert res["sha256"] == att["sha256"] == observed
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")


def test_a_text_read_is_not_marked_lossy(gstack):
    """Control: the flag must discriminate, not always fire."""
    (gstack.src / "plain.txt").write_bytes(b"hello\n")
    work_id, token = admit_and_lease(gstack, "read text", "nc_tr")
    try:
        gstack.call("file_attach", path=str(gstack.src / "plain.txt"),
                    work_id=work_id, actor="pete")
        seen = gstack.call("tool_invoke", neuocyte_id="nc_tr", work_id=work_id,
                           fencing_token=token, name="read_file",
                           arguments={"path": "work/plain.txt"})
        assert seen["result"]["lossy_decode"] is False
        assert seen["result"]["note"] == ""
        assert seen["result"]["content"] == "hello\n"
    finally:
        gstack.call("cancel_work", work_id=work_id, reason="test")
