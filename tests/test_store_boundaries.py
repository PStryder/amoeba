"""Four stores, four lifetimes, and the boundary between them.

Amoeba keeps bytes in four distinct places, and conflating any two of them is
how a "safe to destroy" claim quietly becomes false:

| Name | Where | Lifetime | Who writes |
|---|---|---|---|
| **Filespace** | configured host roots | yours; outlives Amoeba | Harness only |
| **Blob store** | `state_dir/blobs` | durable, content-addressed | Harness only |
| **Compute sandbox** | `state_dir/sandbox/<id>` | one work item, then gone | code inside it |
| **Artifact proposal** | blob store, by digest | durable; the promotable copy | Harness, at propose time |
| **Accepted artifact** | a filespace root, or `state_dir/artifacts` | durable, authoritative | Harness, on promotion |

Two invariants follow, and both are asserted here rather than described:

    Destroying a compute sandbox cannot destroy authoritative input, durable
    evidence, or accepted work product.

    Code executing inside a compute sandbox cannot directly mutate Filespace,
    the blob store, or another work item's durable state. Movement across that
    boundary is performed only by the Harness.

The first makes the sandbox a disposable laboratory. The second is what makes
the first true: if code inside could reach out, "destroying the sandbox" would
not bound what it had already changed.

Everything about what sandboxed code can reach is measured **from inside the
container**. A test that checks from outside is checking the Harness's opinion
of the boundary, not the boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conftest import start_stack

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="AppContainer isolation is Windows-only")


@pytest.fixture()
def bstack(tmp_path):
    fsroot = tmp_path / "myfiles"
    fsroot.mkdir()
    (fsroot / "source.txt").write_bytes(b"AUTHORITATIVE INPUT\n")
    extra = "\n".join([
        "[[filespace.roots]]", 'name = "mine"',
        f'path = "{fsroot.as_posix()}"', 'mode = "read_write"',
    ])
    s = start_stack(tmp_path, extra_toml=extra)
    s.fsroot = fsroot          # type: ignore[attr-defined]
    if not s.call("sandbox_capabilities").get("sandbox_available"):
        s.stop()
        pytest.skip("sandbox unavailable")
    yield s
    s.stop()


REACH_PROBE = r'''
import json, pathlib
out = {}
for label, p in json.loads(TARGETS).items():
    path = pathlib.Path(p)
    entry = {}
    try:
        entry["read"] = ("dir:%d" % len(list(path.iterdir())) if path.is_dir()
                         else "file:%r" % path.read_bytes()[:16])
    except OSError as e:
        entry["read"] = "DENIED:" + type(e).__name__
    try:
        if path.is_dir():
            (path / "amoeba_breach.txt").write_text("breach")
            entry["write"] = "WROTE"
        else:
            with open(path, "ab") as f:
                f.write(b"breach")
            entry["write"] = "APPENDED"
    except OSError as e:
        entry["write"] = "DENIED:" + type(e).__name__
    out[label] = entry
print("REACH " + json.dumps(out))
'''


def admit_lease(stack, objective, nc):
    w = stack.call("admit_work", objective=objective, work_class="user",
                   origin_actor="pete", sandbox_allowed=True)
    item = stack.call("lease_work", neuocyte_id=nc, work_id=w["work_id"])
    assert item, "could not lease"
    return w["work_id"], item["fencing_token"]


def reach(stack, work_id, token, nc, targets):
    code = "TARGETS = " + repr(json.dumps(targets)) + "\n" + REACH_PROBE
    r = stack.call("tool_invoke", neuocyte_id=nc, work_id=work_id,
                   fencing_token=token, name="run_code",
                   arguments={"code": code})
    assert r["accepted"], r
    assert r["result"]["exit_code"] == 0, r["result"]["stderr"]
    for line in r["result"]["stdout"].splitlines():
        if line.startswith("REACH "):
            return json.loads(line[6:])
    raise AssertionError(r["result"]["stdout"])


def scratch_of(stack, sandbox_id):
    return Path([s for s in stack.call("sandbox_list")
                 if s["sandbox_id"] == sandbox_id][0]["root"])


# ===========================================================================
# Code inside a compute sandbox cannot reach the other three stores.
# ===========================================================================
def test_sandboxed_code_cannot_reach_filespace_blobs_or_state(bstack):
    """The inverse invariant, measured from inside the container.

    Note this holds even though the parent of the state tree grants `Everyone`
    full control: an AppContainer token is not satisfied by `Everyone`, so the
    denial here is the container, not the ACLs. Both are in place; this test
    is about the container.
    """
    cfg = bstack.cfg
    work_id, token = admit_lease(bstack, "reach out", "nc_reach")
    att = bstack.call("file_attach", path=str(bstack.fsroot / "source.txt"),
                      work_id=work_id, actor="pete")
    mine = scratch_of(bstack, att["sandbox_id"])
    try:
        targets = {
            "filespace_dir": str(bstack.fsroot),
            "filespace_file": str(bstack.fsroot / "source.txt"),
            "blob_store": str(cfg.blob_dir),
            "state_db": str(cfg.db_path),
            "artifact_store": str(cfg.artifact_dir),
            "log_dir": str(cfg.log_dir),
            "own_scratch": str(mine),
        }
        got = reach(bstack, work_id, token, "nc_reach", targets)

        for label in ("filespace_dir", "filespace_file", "blob_store",
                      "state_db", "artifact_store", "log_dir"):
            assert got[label]["read"].startswith("DENIED"), (
                f"sandboxed code READ {label}: {got[label]['read']}")
            assert got[label]["write"].startswith("DENIED"), (
                f"sandboxed code WROTE {label}: {got[label]['write']}")

        # Control: it can use its own scratch, or the test proves nothing.
        assert got["own_scratch"]["write"] == "WROTE"

        # And the host file really is untouched.
        assert (bstack.fsroot / "source.txt").read_bytes() == b"AUTHORITATIVE INPUT\n"
    finally:
        bstack.call("cancel_work", work_id=work_id, reason="test")


def test_sandboxed_code_cannot_reach_another_work_items_scratch(bstack):
    """Each work item's laboratory is its own."""
    a_id, a_tok = admit_lease(bstack, "item a", "nc_a")
    bstack.call("tool_invoke", neuocyte_id="nc_a", work_id=a_id,
                fencing_token=a_tok, name="write_file",
                arguments={"path": "work/a_secret.txt", "content": "A ONLY"})
    a_sandbox = bstack.call("sandbox_list")[0]["sandbox_id"]
    a_root = scratch_of(bstack, a_sandbox)

    b_id, b_tok = admit_lease(bstack, "item b", "nc_b")
    try:
        got = reach(bstack, b_id, b_tok, "nc_b",
                    {"other": str(a_root / "work" / "a_secret.txt")})
        assert got["other"]["read"].startswith("DENIED"), got["other"]
        assert got["other"]["write"].startswith("DENIED"), got["other"]
        assert (a_root / "work" / "a_secret.txt").read_bytes() == b"A ONLY"
    finally:
        bstack.call("cancel_work", work_id=a_id, reason="test")
        bstack.call("cancel_work", work_id=b_id, reason="test")


def test_movement_across_the_boundary_is_only_ever_the_harness(bstack):
    """A neuocyte has no verb that crosses it, in either direction.

    Inputs arrive because the Harness materialised them; outputs leave because
    the Harness promoted them. Nothing in a neuocyte's vocabulary names a
    filespace root, a blob digest, or another sandbox.
    """
    work_id, _token = admit_lease(bstack, "vocabulary", "nc_v")
    try:
        schemas = bstack.call("tool_schemas", work_id=work_id, role="neuocyte")
        names = {t["name"] for t in schemas["tools"]}
        assert not (names & {"file_write", "file_read", "file_attach",
                             "file_delete", "file_restore",
                             "artifact_promote", "sandbox_create",
                             "sandbox_destroy"})
        for tool in schemas["tools"]:
            props = set(tool["parameters"]["properties"])
            assert not (props & {"root", "sandbox_id", "sha256", "work_id"}), (
                f"{tool['name']} lets a model address another store: {props}")
    finally:
        bstack.call("cancel_work", work_id=work_id, reason="test")


# ===========================================================================
# Destroying a compute sandbox destroys only the laboratory.
# ===========================================================================
def test_destroying_a_sandbox_preserves_input_evidence_and_work_product(bstack):
    """The disposable-laboratory claim, end to end.

    Everything authoritative must already live outside the sandbox before it
    dies: the input in Filespace and the blob store, the evidence in the hash
    chain, the accepted work product in a filespace root.
    """
    work_id, token = admit_lease(bstack, "produce something", "nc_d")
    att = bstack.call("file_attach", path=str(bstack.fsroot / "source.txt"),
                      work_id=work_id, actor="pete")
    scratch = scratch_of(bstack, att["sandbox_id"])

    bstack.call("tool_invoke", neuocyte_id="nc_d", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/product.py",
                           "content": "def accepted():\n    return 1\n"})
    proposed = bstack.call("tool_invoke", neuocyte_id="nc_d", work_id=work_id,
                           fencing_token=token, name="propose_artifact",
                           arguments={"path": "work/product.py",
                                      "rationale": "keep"})
    bstack.call("artifact_promote",
                artifact_id=proposed["result"]["artifact_id"],
                decided_by="pete", root="mine", path="product.py")

    assert scratch.exists()
    bstack.call("complete_work", work_id=work_id, neuocyte_id="nc_d",
                fencing_token=token, result={"finding": "done"})

    # The laboratory is gone.
    assert not scratch.exists(), "the compute sandbox outlived its work item"

    # Authoritative input: untouched, in Filespace.
    assert (bstack.fsroot / "source.txt").read_bytes() == b"AUTHORITATIVE INPUT\n"

    # Durable evidence: the chain still verifies and nothing it references
    # has gone missing.
    integrity = bstack.call("verify_integrity", deep=True)
    assert integrity["hash_chain_ok"] is True
    assert not integrity.get("missing_content"), integrity.get("missing_content")

    # The attached bytes are still recoverable by digest, from the blob store
    # rather than from the scratch that is now gone.
    restored = bstack.call("file_write", root="mine", path="roundtrip.txt",
                           content="placeholder", actor="pete")
    assert restored["receipt_id"]
    bstack.call("file_restore", root="mine", path="roundtrip.txt",
                sha256=att["sha256"], actor="pete")
    assert (bstack.fsroot / "roundtrip.txt").read_bytes() == b"AUTHORITATIVE INPUT\n"

    # Accepted work product: outside the sandbox, on disk.
    assert (bstack.fsroot / "product.py").read_bytes() == (
        b"def accepted():\n    return 1\n")


def test_a_proposal_stays_promotable_after_its_sandbox_is_destroyed(bstack):
    """Sandbox lifetime is absent from the proposal state machine.

    A proposal's bytes are content-addressed when it is made, so the compute
    sandbox is where the file was *made*, not where the promotable copy lives.
    Destroying scratch removes a copy, not *the* copy.

    An earlier version lapsed proposals on destruction. That was a workaround
    for a constraint that stopped existing the moment proposals became durable
    evidence, and keeping it would have coupled a decision to an unrelated
    lifetime.
    """
    work_id, token = admit_lease(bstack, "propose then finish", "nc_l")
    bstack.call("tool_invoke", neuocyte_id="nc_l", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/draft.py",
                           "content": "# decided later\n"})
    pending = bstack.call("tool_invoke", neuocyte_id="nc_l", work_id=work_id,
                          fencing_token=token, name="propose_artifact",
                          arguments={"path": "work/draft.py",
                                     "rationale": "worth keeping"})
    art_id = pending["result"]["artifact_id"]
    digest = pending["result"]["sha256"]
    sandbox_id = bstack.call("artifact_list", limit=50)[0]["sandbox_id"]
    scratch = scratch_of(bstack, sandbox_id)

    bstack.call("complete_work", work_id=work_id, neuocyte_id="nc_l",
                fencing_token=token, result={"finding": "done"})

    # The laboratory is gone; the decision is still open.
    assert not scratch.exists()
    rows = {a["artifact_id"]: a for a in bstack.call("artifact_list", limit=50)}
    assert rows[art_id]["status"] == "proposed", (
        f"the proposal became {rows[art_id]['status']!r} because its sandbox "
        "was destroyed; sandbox lifetime must not touch this state machine")

    # And it can still be promoted, from the blob rather than from scratch.
    out = bstack.call("artifact_promote", artifact_id=art_id,
                      decided_by="pete", root="mine", path="late.py")
    assert out["status"] == "promoted"
    assert out["sha256"] == digest
    assert out["source"] == "proposal blob"
    assert (bstack.fsroot / "late.py").read_bytes() == b"# decided later\n"


def test_promotion_materialises_the_reviewed_bytes_not_whatever_scratch_holds(bstack):
    """Substitution is impossible rather than detected.

    Promotion used to re-read the scratch file and refuse if it had changed --
    fail-closed detection of a time-of-check/time-of-use swap. Sourcing from
    the immutable proposal blob is stronger: the reviewed bytes are what land,
    and a later change to scratch cannot influence the result at all.

    The divergence is still *recorded*, because a neuocyte rewriting a file
    after proposing it is a fact worth having, just not a reason to refuse.
    """
    work_id, token = admit_lease(bstack, "propose then tamper", "nc_t")
    bstack.call("tool_invoke", neuocyte_id="nc_t", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/checker.py",
                           "content": "def check(x):\n    return x > 0\n"})
    prop = bstack.call("tool_invoke", neuocyte_id="nc_t", work_id=work_id,
                       fencing_token=token, name="propose_artifact",
                       arguments={"path": "work/checker.py",
                                  "rationale": "reviewed and sound"})
    art_id = prop["result"]["artifact_id"]
    reviewed = prop["result"]["sha256"]

    # The swap a compromised or over-eager neuocyte would make.
    bstack.call("tool_invoke", neuocyte_id="nc_t", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/checker.py",
                           "content": "import os\nos.system('calc')\n"})

    out = bstack.call("artifact_promote", artifact_id=art_id,
                      decided_by="pete", root="mine", path="checker.py")
    assert out["status"] == "promoted"
    assert out["sha256"] == reviewed
    assert out["scratch_diverged"] is True
    landed = (bstack.fsroot / "checker.py").read_bytes()
    assert landed == b"def check(x):\n    return x > 0\n", (
        f"the substituted bytes were promoted: {landed!r}")
    assert b"os.system" not in landed

    kinds = [e["kind"] for e in bstack.call("history", limit=600)]
    assert "artifact.scratch_diverged" in kinds, (
        "the divergence was not recorded; it is a fact about the neuocyte "
        "even though it changed nothing about the promotion")
    bstack.call("cancel_work", work_id=work_id, reason="test")


def test_an_undecided_proposal_keeps_its_evidence_and_its_pending_status(bstack):
    """The three truths that must hold together after destruction.

        scratch copy       GONE
        proposal record    still PENDING, still promotable
        proposal bytes     PRESERVED, and are what promotion would use

    Losing any one of these makes the record say something untrue: that the
    proposal was decided, that its content is unknowable, or that it is
    promotable when it is not.
    """
    work_id, token = admit_lease(bstack, "evidence and status", "nc_e")
    bstack.call("tool_invoke", neuocyte_id="nc_e", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/p.py", "content": "proposed = True\n"})
    prop = bstack.call("tool_invoke", neuocyte_id="nc_e", work_id=work_id,
                       fencing_token=token, name="propose_artifact",
                       arguments={"path": "work/p.py", "rationale": "r"})
    assert prop["result"]["evidence_preserved"] is True
    digest = prop["result"]["sha256"]

    bstack.call("complete_work", work_id=work_id, neuocyte_id="nc_e",
                fencing_token=token, result={"finding": "done"})

    rows = {a["artifact_id"]: a for a in bstack.call("artifact_list", limit=50)}
    assert rows[prop["result"]["artifact_id"]]["status"] == "proposed"

    # The bytes are retrievable by digest independently of any decision.
    bstack.call("file_write", root="mine", path="evidence.py",
                content="x", actor="pete")
    bstack.call("file_restore", root="mine", path="evidence.py",
                sha256=digest, actor="pete")
    assert (bstack.fsroot / "evidence.py").read_bytes() == b"proposed = True\n"

    integrity = bstack.call("verify_integrity", deep=True)
    assert integrity["hash_chain_ok"] is True
    assert not integrity.get("missing_content"), integrity.get("missing_content")


def test_rejecting_a_proposal_still_closes_it(bstack):
    """Control: a decision still decides.

    Removing sandbox-driven lapsing must not remove the ways a proposal is
    genuinely resolved, or "pending" would become a state nothing ever leaves.
    """
    work_id, token = admit_lease(bstack, "propose then reject", "nc_r")
    bstack.call("tool_invoke", neuocyte_id="nc_r", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/no.py", "content": "no = 1\n"})
    prop = bstack.call("tool_invoke", neuocyte_id="nc_r", work_id=work_id,
                       fencing_token=token, name="propose_artifact",
                       arguments={"path": "work/no.py", "rationale": "meh"})
    art_id = prop["result"]["artifact_id"]

    bstack.call("artifact_reject", artifact_id=art_id, reason="not reproducible")
    rows = {a["artifact_id"]: a for a in bstack.call("artifact_list", limit=50)}
    assert rows[art_id]["status"] == "rejected"
    with pytest.raises(Exception) as exc:
        bstack.call("artifact_promote", artifact_id=art_id, decided_by="pete")
    assert "not awaiting a decision" in str(exc.value)
    bstack.call("cancel_work", work_id=work_id, reason="test")


def test_destroying_a_sandbox_decides_nothing(bstack):
    """Stated directly: teardown is not a verdict.

    Destruction records how many proposals are still pending, and changes none
    of them.
    """
    work_id, token = admit_lease(bstack, "teardown decides nothing", "nc_n")
    bstack.call("tool_invoke", neuocyte_id="nc_n", work_id=work_id,
                fencing_token=token, name="write_file",
                arguments={"path": "work/a.py", "content": "a = 1\n"})
    bstack.call("tool_invoke", neuocyte_id="nc_n", work_id=work_id,
                fencing_token=token, name="propose_artifact",
                arguments={"path": "work/a.py", "rationale": "one"})

    before = [a["status"] for a in bstack.call("artifact_list", limit=50)]
    bstack.call("complete_work", work_id=work_id, neuocyte_id="nc_n",
                fencing_token=token, result={"finding": "done"})
    after = [a["status"] for a in bstack.call("artifact_list", limit=50)]
    assert before == after, f"teardown changed artifact statuses: {before} -> {after}"

    kinds = [e["kind"] for e in bstack.call("history", limit=600)]
    assert "artifact.lapsed" not in kinds, (
        "lapsing is retired; sandbox lifetime must not touch proposal state")

    events = [e for e in bstack.call("history", limit=600)
              if e["kind"] == "sandbox.destroyed"]
    payload = json.loads(events[-1]["payload_inline"])
    assert payload["proposals_still_pending"] >= 1


def test_the_four_stores_are_in_different_places(bstack):
    """The taxonomy, asserted as paths rather than described in prose.

    If any two of these nested inside one another, a claim about one's lifetime
    would silently become a claim about the other's.
    """
    cfg = bstack.cfg
    fs = Path(bstack.fsroot).resolve()
    blobs = Path(cfg.blob_dir).resolve()
    sandboxes = Path(cfg.sandbox_dir).resolve()
    artifacts = Path(cfg.artifact_dir).resolve()

    assert len({blobs, sandboxes, artifacts}) == 3
    for a, b in ((blobs, sandboxes), (artifacts, sandboxes),
                 (blobs, artifacts)):
        assert a != b and b not in a.parents and a not in b.parents, (
            f"{a} and {b} are nested; their lifetimes would be entangled")
    assert sandboxes not in fs.parents and fs not in sandboxes.parents, (
        "a compute sandbox lives inside Filespace; destroying one would touch "
        "the other")
    assert artifacts.name == "artifacts", (
        "the durable artifact store must not be called a 'workspace'; that "
        "word was doing double duty for the ephemeral compute sandbox")
