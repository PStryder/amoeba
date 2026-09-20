"""Host filesystem access: the allowlist is the boundary.

Two claims carry everything else here:

1. **Nothing outside a configured root is reachable.** Not clamped into the
   root, not sanitised -- refused. The caller is ultimately a language model,
   so a path that tries to escape is a signal, not a typo.
2. **No write destroys.** Prior content is content-addressed before any
   overwrite or delete, so every version is recoverable by digest. This is what
   makes it safe to let a mind write to disk at all.

The escape tests below are parametrised over the specific tricks that work on
Windows, not just ``..``. Several of them -- alternate data streams, trailing
dots, reserved device names -- are silent on Windows: the write succeeds and
goes somewhere other than where the string appears to point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from amoeba.config import FilespaceConfig, FilespaceRoot
from amoeba.errors import InvalidInput, NotFound, ResourceExhausted
from amoeba.filespace import Filespace, FilespaceDenied


@pytest.fixture()
def fs(tmp_path):
    rw = tmp_path / "out"
    ro = tmp_path / "src"
    outside = tmp_path / "private"
    for d in (rw, ro, outside):
        d.mkdir()
    (ro / "readable.txt").write_text("source content", encoding="utf-8")
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (rw / "existing.txt").write_text("v1", encoding="utf-8")

    cfg = FilespaceConfig(roots=[
        FilespaceRoot(name="out", path=str(rw), mode="read_write"),
        FilespaceRoot(name="src", path=str(ro), mode="read_only"),
    ])
    space = Filespace(cfg)
    space.tmp = tmp_path          # type: ignore[attr-defined]
    space.rw = rw                 # type: ignore[attr-defined]
    space.ro = ro                 # type: ignore[attr-defined]
    space.outside = outside       # type: ignore[attr-defined]
    return space


# ---------------------------------------------------------------------------
# It has to work, or refusing everything would be a trivial way to pass.
# ---------------------------------------------------------------------------
def test_a_permitted_path_resolves(fs):
    r = fs.resolve("out", "notes.txt", need_write=True)
    assert r.writable is True
    assert r.exists is False
    assert r.path.parent == fs.rw


def test_reading_and_writing_inside_a_writable_root(fs):
    r = fs.resolve("out", "sub/dir/file.txt", need_write=True)
    out = fs.write_bytes(r, b"hello")
    assert out["bytes"] == 5
    back, truncated = fs.read_bytes(fs.resolve("out", "sub/dir/file.txt"))
    assert back == b"hello" and truncated is False


def test_listing_reports_relative_paths(fs):
    fs.write_bytes(fs.resolve("out", "a/b.txt", need_write=True), b"x")
    listing = {e["path"] for e in fs.list("out")}
    assert "a/b.txt" in listing and "existing.txt" in listing


# ---------------------------------------------------------------------------
# Nothing outside a root is reachable.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "../private/secret.txt",
    "../../private/secret.txt",
    "sub/../../private/secret.txt",
    "./../private/secret.txt",
    "..\\private\\secret.txt",
    "/etc/passwd",
    "C:/Windows/System32/drivers/etc/hosts",
    "C:\\Windows\\win.ini",
    "\\\\server\\share\\file.txt",
    "//server/share/file.txt",
    "\\\\?\\C:\\Windows\\win.ini",
    "notes.txt:hidden",                 # alternate data stream
    "notes.txt:hidden:$DATA",
    "CON",
    "NUL.txt",
    "COM1",
    "lpt1.log",
    "trailing.",
    "trailing ",
    "sub/trailing./file.txt",
    "",
    "   ",
    ".",
    "..",
])
def test_paths_that_leave_the_root_or_name_a_device_are_refused(fs, bad):
    with pytest.raises(InvalidInput):
        fs.resolve("out", bad, need_write=True)


def test_an_unknown_root_is_refused(fs):
    with pytest.raises(FilespaceDenied) as exc:
        fs.resolve("nope", "file.txt")
    assert "unknown filespace root" in exc.value.message


def test_a_refused_path_is_never_silently_clamped(fs):
    """Refusal, not sanitisation.

    Stripping `..` and carrying on would put the write *somewhere* -- probably
    inside the root, which looks safe and is wrong. The caller asked for a file
    it may not have; the answer is no, not a different file.
    """
    before = {p.name for p in fs.rw.rglob("*")}
    for bad in ("../private/secret.txt", "..\\..\\private\\secret.txt"):
        with pytest.raises(InvalidInput):
            fs.resolve("out", bad, need_write=True)
    assert {p.name for p in fs.rw.rglob("*")} == before
    assert (fs.outside / "secret.txt").read_text(encoding="utf-8") == "SECRET"


def test_a_read_only_root_refuses_writes(fs):
    assert fs.resolve("src", "readable.txt").writable is False
    with pytest.raises(FilespaceDenied) as exc:
        fs.resolve("src", "readable.txt", need_write=True)
    assert "read-only" in exc.value.message

    r = fs.resolve("src", "readable.txt")
    with pytest.raises(FilespaceDenied):
        fs.write_bytes(r, b"overwritten")
    assert (fs.ro / "readable.txt").read_text(encoding="utf-8") == "source content"


def test_an_absolute_host_path_outside_every_root_is_refused(fs):
    with pytest.raises(FilespaceDenied) as exc:
        fs.resolve_host_path(str(fs.outside / "secret.txt"))
    assert "not inside any configured filespace root" in exc.value.message


def test_an_absolute_host_path_inside_a_root_is_accepted(fs):
    r = fs.resolve_host_path(str(fs.ro / "readable.txt"))
    assert r.root_name == "src" and r.relpath == "readable.txt"


# ---------------------------------------------------------------------------
# Links: the interesting case, because the string looks fine.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform != "win32", reason="junction test is Windows-only")
def test_a_junction_pointing_out_of_the_root_is_refused(fs):
    """The path has no `..` in it at all.

    A directory junction inside the root is an ordinary-looking name that
    resolves elsewhere. String checks cannot catch this; only resolving fully
    and re-checking containment can.
    """
    link = fs.rw / "escape"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(fs.outside)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"could not create a junction: {r.stdout}{r.stderr}")

    with pytest.raises(FilespaceDenied) as exc:
        fs.resolve("out", "escape/secret.txt", need_write=True)
    assert "escapes its filespace root" in exc.value.message

    # And it must not be reachable for reading either.
    with pytest.raises(FilespaceDenied):
        fs.resolve("out", "escape/secret.txt")


@pytest.mark.skipif(sys.platform != "win32", reason="junction test is Windows-only")
def test_listing_does_not_walk_through_a_junction(fs):
    link = fs.rw / "escape"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(fs.outside)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("could not create a junction")
    names = [e["path"] for e in fs.list("out")]
    assert not any("secret" in n for n in names), (
        f"listing leaked names from outside the root: {names}")


# ---------------------------------------------------------------------------
# Writes are atomic and bounded.
# ---------------------------------------------------------------------------
def test_a_write_over_an_existing_file_replaces_it_completely(fs):
    r = fs.resolve("out", "existing.txt", need_write=True)
    fs.write_bytes(r, b"shorter")
    assert (fs.rw / "existing.txt").read_bytes() == b"shorter"


def test_a_write_larger_than_the_limit_is_refused(fs):
    fs.cfg.max_write_bytes = 16
    r = fs.resolve("out", "big.txt", need_write=True)
    with pytest.raises(ResourceExhausted):
        fs.write_bytes(r, b"x" * 100)
    assert not (fs.rw / "big.txt").exists()


def test_no_temporary_file_is_left_behind(fs):
    r = fs.resolve("out", "atomic.txt", need_write=True)
    fs.write_bytes(r, b"content")
    leftovers = [p.name for p in fs.rw.iterdir() if "amoeba-tmp" in p.name]
    assert leftovers == [], leftovers


def test_deleting_a_directory_is_refused(fs):
    (fs.rw / "adir").mkdir()
    r = fs.resolve("out", "adir", need_write=True)
    with pytest.raises(InvalidInput):
        fs.delete(r)
    assert (fs.rw / "adir").is_dir()


def test_reading_a_missing_file_is_not_found(fs):
    with pytest.raises(NotFound):
        fs.read_bytes(fs.resolve("out", "nope.txt"))


def test_with_no_roots_configured_nothing_is_reachable(tmp_path):
    space = Filespace(FilespaceConfig(roots=[]))
    assert space.available() is False
    with pytest.raises(FilespaceDenied):
        space.resolve("out", "anything.txt")
    with pytest.raises(FilespaceDenied):
        space.resolve_host_path(str(tmp_path / "anything.txt"))


def test_a_root_that_does_not_exist_is_dropped_not_invented(tmp_path):
    """A typo in config must not create a directory somewhere."""
    missing = tmp_path / "does-not-exist"
    space = Filespace(FilespaceConfig(roots=[
        FilespaceRoot(name="ghost", path=str(missing), mode="read_write")]))
    assert space.available() is False
    assert not missing.exists()


def test_capabilities_state_the_residual_risk(fs):
    caps = fs.capabilities()
    assert caps["filespace_available"] is True
    assert "refused" in caps["outside_roots"]
    assert "not atomic" in caps["residual_risk"]
    assert "reversible" in caps["overwrite"]


# ---------------------------------------------------------------------------
# Hard links: the trap where nothing about the path is unusual.
#
# A symlink or junction points at something, so resolving it and re-checking
# containment catches it. A hard link is not a pointer: it is a second
# directory entry for the same file record. `resolve()` has nothing to resolve
# and `is_symlink()` is False, so path containment says "inside the root" and
# is telling the truth about the path while being wrong about the file.
#
# Measured before it was fixed: reading through a planted hard link returned
# content from outside the root.
# ---------------------------------------------------------------------------
def _hardlink(link: Path, target: Path) -> bool:
    r = subprocess.run(["cmd", "/c", "mklink", "/H", str(link), str(target)],
                       capture_output=True, text=True)
    return r.returncode == 0


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS hard links are Windows-only")
def test_a_hard_link_into_the_root_cannot_be_used_to_read_outside_it(fs):
    """The leak this was found by.

    Path containment passes and is correct about the path. The file is still
    reachable under another name that the root does not cover.
    """
    link = fs.rw / "innocent.txt"
    if not _hardlink(link, fs.outside / "secret.txt"):
        pytest.skip("could not create a hard link (same volume required)")

    # The premise: nothing about this path looks wrong.
    assert link.is_symlink() is False
    assert os.stat(link).st_nlink == 2
    assert link.resolve().parent == fs.rw

    with pytest.raises(FilespaceDenied) as exc:
        fs.resolve("out", "innocent.txt")
    assert "hard link" in exc.value.message

    with pytest.raises(FilespaceDenied):
        fs.resolve("out", "innocent.txt", need_write=True)

    assert (fs.outside / "secret.txt").read_bytes() == b"SECRET"


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS hard links are Windows-only")
def test_a_hard_linked_file_is_listed_but_marked_inaccessible(fs):
    """Shown, not hidden.

    The entry exists and pretending otherwise would be its own kind of lie.
    Whether it may be opened is a separate answer, and the listing carries the
    reason so it is visible rather than mysterious.
    """
    link = fs.rw / "innocent.txt"
    if not _hardlink(link, fs.outside / "secret.txt"):
        pytest.skip("could not create a hard link")

    entry = next(e for e in fs.list("out") if e["path"] == "innocent.txt")
    assert entry["multiply_linked"] is True
    assert entry["accessible"] is False
    ordinary = next(e for e in fs.list("out") if e["path"] == "existing.txt")
    assert ordinary["multiply_linked"] is False
    assert ordinary["accessible"] is True


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS hard links are Windows-only")
def test_allowing_multiply_linked_files_is_a_deliberate_choice(fs):
    """The escape hatch exists, and turning it on really does open the door.

    Pinned so the default cannot quietly become permissive: if this stops
    reading through the link, the config flag has stopped meaning anything.
    """
    link = fs.rw / "innocent.txt"
    if not _hardlink(link, fs.outside / "secret.txt"):
        pytest.skip("could not create a hard link")

    fs.cfg.allow_multiply_linked = True
    data, _ = fs.read_bytes(fs.resolve("out", "innocent.txt"))
    assert data == b"SECRET", (
        "with the flag on this should read through; if it does not, the flag "
        "is not the thing controlling this behaviour")


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS hard links are Windows-only")
def test_a_write_replaces_the_directory_entry_rather_than_the_file_record(fs):
    """Pinning behaviour that is load-bearing and was found by experiment.

    `write_bytes` writes a temporary file and renames it over the target, which
    replaces the *directory entry*. An existing file record is therefore never
    modified in place, so a write cannot reach a file's other names even if one
    slipped past resolution.

    This is a second line of defence -- multiply-linked files are refused at
    resolution -- but it is real, and rewriting this as `path.write_bytes(data)`
    would silently make write-through live again. Hence a test rather than a
    comment.
    """
    link = fs.rw / "innocent.txt"
    if not _hardlink(link, fs.outside / "secret.txt"):
        pytest.skip("could not create a hard link")

    fs.cfg.allow_multiply_linked = True     # get past resolution deliberately
    resolved = fs.resolve("out", "innocent.txt", need_write=True)
    fs.write_bytes(resolved, b"OVERWRITTEN")

    assert (fs.outside / "secret.txt").read_bytes() == b"SECRET", (
        "the write reached the file's other name; write_bytes is no longer "
        "replacing the directory entry")
    assert (fs.rw / "innocent.txt").read_bytes() == b"OVERWRITTEN"
    assert os.stat(fs.outside / "secret.txt").st_nlink == 1, (
        "the link should have been broken by the rename")


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS hard links are Windows-only")
def test_deleting_a_hard_link_does_not_remove_the_other_name(fs):
    """NTFS semantics, asserted rather than assumed."""
    link = fs.rw / "innocent.txt"
    if not _hardlink(link, fs.outside / "secret.txt"):
        pytest.skip("could not create a hard link")

    fs.cfg.allow_multiply_linked = True
    fs.delete(fs.resolve("out", "innocent.txt", need_write=True))
    assert (fs.outside / "secret.txt").read_bytes() == b"SECRET"


def test_capabilities_state_the_hard_link_policy(fs):
    assert "refused" in fs.capabilities()["hard_links"]
    fs.cfg.allow_multiply_linked = True
    assert "ALLOWED" in fs.capabilities()["hard_links"]
