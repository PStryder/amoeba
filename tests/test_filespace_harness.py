"""File verbs through the Harness: receipts, supersession, and restoration.

`test_filespace.py` covers resolution -- what may be reached. This covers what
happens when it is: every destructive act preserves what was there, every act
is receipted, and a refusal is recorded rather than only returned.

The central claim is that Amoeba cannot break a file. It can supersede one, and
the version it replaced stays recoverable by digest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amoeba.errors import InvalidInput
from conftest import LiveStack, start_stack


@pytest.fixture()
def fstack(tmp_path):
    """A live stack with two filespace roots configured."""
    out = tmp_path / "amoeba-out"
    src = tmp_path / "project"
    out.mkdir()
    src.mkdir()
    # Bytes, not text: write_text translates \n to \r\n on Windows, and the
    # assertions below would then be about newline handling rather than about
    # supersession.
    (src / "module.py").write_bytes(b"def f():\n    return 1\n")
    (out / "report.md").write_bytes(b"# v1\n")

    extra = "\n".join([
        "[filespace]",
        "max_read_bytes = 1048576",
        "",
        "[[filespace.roots]]",
        'name = "out"',
        f'path = "{out.as_posix()}"',
        'mode = "read_write"',
        "",
        "[[filespace.roots]]",
        'name = "project"',
        f'path = "{src.as_posix()}"',
        'mode = "read_only"',
    ])
    s = start_stack(tmp_path, extra_toml=extra)
    s.out = out          # type: ignore[attr-defined]
    s.src = src          # type: ignore[attr-defined]
    yield s
    s.stop()


def test_roots_are_reported_with_their_modes(fstack):
    caps = fstack.call("file_roots")
    assert caps["filespace_available"] is True
    by_name = {r["name"]: r for r in caps["roots"]}
    assert by_name["out"]["writable"] is True
    assert by_name["project"]["writable"] is False


def test_write_read_and_list_round_trip(fstack):
    w = fstack.call("file_write", root="out", path="notes/today.md",
                    content="# hello\n", actor="ego", rationale="test")
    assert w["bytes"] == 8 and w["receipt_id"]
    assert w["overwrote"] is False

    r = fstack.call("file_read", root="out", path="notes/today.md")
    assert r["content"] == "# hello\n"
    assert r["sha256"] == w["sha256"]

    listing = {e["path"] for e in fstack.call("file_list", root="out")}
    assert "notes/today.md" in listing
    assert (fstack.out / "notes" / "today.md").read_text(encoding="utf-8") == "# hello\n"


def test_an_overwrite_supersedes_and_the_prior_version_is_restorable(fstack):
    """The claim that makes writing to disk safe at all."""
    first = fstack.call("file_read", root="out", path="report.md")
    assert first["content"] == "# v1\n"

    second = fstack.call("file_write", root="out", path="report.md",
                         content="# v2 (clobbered)\n", actor="nc_1")
    assert second["overwrote"] is True
    assert second["prior_sha256"], "the previous content was not snapshotted"
    assert (fstack.out / "report.md").read_text(encoding="utf-8") == "# v2 (clobbered)\n"

    versions = fstack.call("file_versions", root="out", path="report.md")
    assert any(v["sha256"] == second["prior_sha256"] and v["restorable"]
               for v in versions), versions

    back = fstack.call("file_restore", root="out", path="report.md",
                       sha256=second["prior_sha256"], actor="pete")
    assert back["receipt_id"]
    assert (fstack.out / "report.md").read_text(encoding="utf-8") == "# v1\n"

    kinds = [e["kind"] for e in fstack.call("history", limit=500)]
    assert "file.superseded" in kinds and "file.restored" in kinds


def test_restoring_does_not_lose_the_version_it_replaces(fstack):
    """Undo must not be its own way to destroy something."""
    v1 = fstack.call("file_read", root="out", path="report.md")["sha256"]
    w2 = fstack.call("file_write", root="out", path="report.md",
                     content="# v2\n", actor="nc_1")
    fstack.call("file_restore", root="out", path="report.md", sha256=v1)

    versions = {v["sha256"] for v in
                fstack.call("file_versions", root="out", path="report.md")}
    assert w2["sha256"] in versions, (
        "restoring v1 discarded v2 with no way back")
    fstack.call("file_restore", root="out", path="report.md", sha256=w2["sha256"])
    assert (fstack.out / "report.md").read_text(encoding="utf-8") == "# v2\n"


def test_a_delete_keeps_the_content_recoverable(fstack):
    d = fstack.call("file_delete", root="out", path="report.md", actor="nc_1",
                    reason="superseded")
    assert d["prior_sha256"]
    assert not (fstack.out / "report.md").exists()

    fstack.call("file_restore", root="out", path="report.md",
                sha256=d["prior_sha256"])
    assert (fstack.out / "report.md").read_text(encoding="utf-8") == "# v1\n"


def test_a_read_only_root_refuses_writes_through_the_harness(fstack):
    with pytest.raises(Exception) as exc:
        fstack.call("file_write", root="project", path="module.py",
                    content="# clobbered\n", actor="nc_1")
    assert "read-only" in str(exc.value)
    assert (fstack.src / "module.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_an_escape_attempt_is_refused_and_recorded(fstack):
    with pytest.raises(Exception):
        fstack.call("file_write", root="out", path="../project/module.py",
                    content="pwned", actor="nc_1")
    kinds = [e["kind"] for e in fstack.call("history", limit=500)]
    assert "file.denied" in kinds, (
        "a refused path must be recorded, not just returned")
    assert (fstack.src / "module.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_restoring_an_unknown_digest_is_refused(fstack):
    with pytest.raises(Exception) as exc:
        fstack.call("file_restore", root="out", path="report.md",
                    sha256="0" * 64)
    assert "digest" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Handing a file in
# ---------------------------------------------------------------------------
def test_attaching_a_file_puts_it_in_the_work_items_sandbox(fstack):
    if not fstack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work = fstack.call("admit_work", objective="review this module",
                       work_class="user", origin_actor="pete",
                       sandbox_allowed=True)
    work_id = work["work_id"]
    # Lease first: the supervisor dispatches queued work on its own, and
    # attaching is slow enough (it creates a sandbox) for its scheduler to take
    # the item in between.
    item = fstack.call("lease_work", neuocyte_id="nc_att", work_id=work_id)
    assert item, "could not lease the work item"
    try:
        att = fstack.call("file_attach", path=str(fstack.src / "module.py"),
                          work_id=work_id, actor="pete")
        assert att["sandbox_path"] == "work/module.py"
        assert att["sha256"] and att["receipt_id"]

        seen = fstack.call("tool_invoke", neuocyte_id="nc_att", work_id=work_id,
                           fencing_token=item["fencing_token"],
                           name="read_file", arguments={"path": "work/module.py"})
        assert seen["accepted"], seen
        assert "def f()" in seen["result"]["content"]

        kinds = [e["kind"] for e in fstack.call("history", limit=500)]
        assert "file.attached" in kinds
    finally:
        fstack.call("cancel_work", work_id=work_id, reason="test")


def test_attaching_a_file_outside_every_root_is_refused(fstack, tmp_path):
    outside = tmp_path / "not-allowed.txt"
    outside.write_text("secret", encoding="utf-8")
    work = fstack.call("admit_work", objective="x", work_class="user",
                       origin_actor="pete", sandbox_allowed=True)
    try:
        with pytest.raises(Exception) as exc:
            fstack.call("file_attach", path=str(outside),
                        work_id=work["work_id"], actor="pete")
        assert "not inside any configured filespace root" in str(exc.value)
        kinds = [e["kind"] for e in fstack.call("history", limit=500)]
        assert "file.denied" in kinds
    finally:
        fstack.call("cancel_work", work_id=work["work_id"], reason="test")


# ---------------------------------------------------------------------------
# Promotion to a real location: how Amoeba actually produces a file
# ---------------------------------------------------------------------------
def test_promotion_can_target_a_filespace_root(fstack):
    if not fstack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work = fstack.call("admit_work", objective="write a checker",
                       work_class="user", origin_actor="pete",
                       sandbox_allowed=True)
    work_id = work["work_id"]
    item = fstack.call("lease_work", neuocyte_id="nc_p", work_id=work_id)
    try:
        fstack.call("tool_invoke", neuocyte_id="nc_p", work_id=work_id,
                    fencing_token=item["fencing_token"], name="write_file",
                    arguments={"path": "work/checker.py",
                               "content": "def check(x):\n    return x > 0\n"})
        proposed = fstack.call(
            "tool_invoke", neuocyte_id="nc_p", work_id=work_id,
            fencing_token=item["fencing_token"], name="propose_artifact",
            arguments={"path": "work/checker.py", "rationale": "reusable"})
        assert proposed["accepted"], proposed

        out = fstack.call("artifact_promote",
                          artifact_id=proposed["result"]["artifact_id"],
                          decided_by="pete", root="out", path="checker.py")
        assert out["status"] == "promoted"
        assert out["root"] == "out" and out["path"] == "checker.py"
        landed = fstack.out / "checker.py"
        assert landed.read_text(encoding="utf-8") == "def check(x):\n    return x > 0\n"
    finally:
        fstack.call("cancel_work", work_id=work_id, reason="test")


def test_promotion_cannot_target_a_read_only_root(fstack):
    if not fstack.call("sandbox_capabilities").get("sandbox_available"):
        pytest.skip("sandbox unavailable")
    work = fstack.call("admit_work", objective="x", work_class="user",
                       origin_actor="pete", sandbox_allowed=True)
    work_id = work["work_id"]
    item = fstack.call("lease_work", neuocyte_id="nc_ro", work_id=work_id)
    try:
        fstack.call("tool_invoke", neuocyte_id="nc_ro", work_id=work_id,
                    fencing_token=item["fencing_token"], name="write_file",
                    arguments={"path": "work/module.py", "content": "# replaced\n"})
        proposed = fstack.call(
            "tool_invoke", neuocyte_id="nc_ro", work_id=work_id,
            fencing_token=item["fencing_token"], name="propose_artifact",
            arguments={"path": "work/module.py", "rationale": "improvement"})
        with pytest.raises(Exception) as exc:
            fstack.call("artifact_promote",
                        artifact_id=proposed["result"]["artifact_id"],
                        decided_by="pete", root="project", path="module.py")
        assert "read-only" in str(exc.value)
        assert (fstack.src / "module.py").read_text(encoding="utf-8") == (
            "def f():\n    return 1\n")
    finally:
        fstack.call("cancel_work", work_id=work_id, reason="test")


def test_a_neuocyte_has_no_tool_that_reaches_the_host_filesystem(fstack):
    """Files go out by proposal, never by a neuocyte writing directly.

    The sandbox tools address the sandbox. Nothing in a neuocyte's vocabulary
    names a filespace root, so the decision to put bytes on disk stays with
    whoever promotes the artifact.
    """
    work = fstack.call("admit_work", objective="x", work_class="user",
                       origin_actor="pete", sandbox_allowed=True)
    try:
        schemas = fstack.call("tool_schemas", work_id=work["work_id"],
                              role="neuocyte")
        names = {t["name"] for t in schemas["tools"]}
        assert not (names & {"file_write", "file_read", "file_delete",
                             "file_restore", "file_attach", "artifact_promote"})
        for tool in schemas["tools"]:
            props = set(tool["parameters"]["properties"])
            assert "root" not in props, f"{tool['name']} lets a model name a root"
    finally:
        fstack.call("cancel_work", work_id=work["work_id"], reason="test")
