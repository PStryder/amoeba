"""The inward boundary, asserted at the layer the guarantee lives on.

``sandbox.py`` protects the host from code running inside a container. This
file is about the opposite direction: what the rest of the machine can do to
Amoeba's files. The threat is concrete rather than theoretical -- a writable
sandbox runtime is a code-injection path straight into the container, and a
readable scratch tree is exfiltration of whatever a neuocyte was working on.

Every assertion here reads the DACL **the OS reports**, not the return value of
our own hardening call. A ``HardenResult`` saying ``hardened=True`` is
bookkeeping; ``icacls`` saying ``Everyone`` is absent is the guarantee. The
first test deliberately proves the exposure is real before the rest prove it is
closed, because a suite that only ever sees hardened directories cannot tell
the difference between protection and a no-op.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from amoeba import security
from amoeba.security import (audit_path, audit_paths, harden, harden_state_tree,
                             read_dacl, revoke)

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="DACL hardening is Windows-only")


def grant_everyone(path: Path) -> None:
    """Make a directory world-writable, the way the real parent directory is."""
    r = subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:(OI)(CI)(F)", "/Q"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def dacl_text(path: Path) -> str:
    return " ".join(read_dacl(path))


def mentions_everyone(path: Path) -> bool:
    t = dacl_text(path).lower()
    return "everyone" in t or "s-1-1-0" in t


@pytest.fixture()
def exposed_tree(tmp_path):
    """A directory tree that inherits world-writable permissions, as found."""
    parent = tmp_path / "permissive"
    parent.mkdir()
    grant_everyone(parent)
    child = parent / "amoeba-state"
    child.mkdir()
    (child / "existing.txt").write_text("secret", encoding="utf-8")
    nested = child / "blobs"
    nested.mkdir()
    (nested / "deep.txt").write_text("also secret", encoding="utf-8")
    return SimpleNamespace(parent=parent, root=child, nested=nested)


# ---------------------------------------------------------------------------
# The exposure is real. Without this, every test below could pass on a machine
# where nothing was ever exposed, and prove nothing.
# ---------------------------------------------------------------------------
def test_an_unhardened_directory_really_does_inherit_world_access(exposed_tree):
    assert mentions_everyone(exposed_tree.root), (
        "a directory under a world-writable parent should inherit that access; "
        "if it does not, this suite is not testing the condition it claims to")
    assert audit_path(exposed_tree.root)["exposed"] is True
    assert audit_path(exposed_tree.root)["has_inherited_aces"] is True


# ---------------------------------------------------------------------------
# Nothing outside Amoeba can write to it.
# ---------------------------------------------------------------------------
def test_hardening_removes_every_ace_for_everyone(exposed_tree):
    harden(exposed_tree.root)
    assert not mentions_everyone(exposed_tree.root), (
        f"Everyone still on the DACL after hardening: {dacl_text(exposed_tree.root)}")
    assert audit_path(exposed_tree.root)["exposed_to"] == []


def test_hardening_removes_inheritance_so_the_parent_cannot_regrant(exposed_tree):
    """The strongest form of the claim.

    Removing the ACEs is not enough on its own: while inheritance is live, any
    later change to the parent flows straight back down. This re-grants
    Everyone on the parent *after* hardening and requires the child to be
    unaffected.
    """
    harden(exposed_tree.root)
    grant_everyone(exposed_tree.parent)

    assert not mentions_everyone(exposed_tree.root), (
        "a grant on the parent propagated into a hardened directory; "
        "inheritance was not actually severed")
    assert audit_path(exposed_tree.root)["has_inherited_aces"] is False


def test_children_lose_the_permissive_access_they_inherited(exposed_tree):
    """Existing files are covered, without touching them individually.

    They re-inherit from the hardened parent. This is asserted rather than
    assumed because the obvious alternative -- applying the grant with ``/T`` --
    silently produces files with an ACE that conveys nothing.
    """
    assert mentions_everyone(exposed_tree.nested / "deep.txt")
    harden(exposed_tree.root)
    assert not mentions_everyone(exposed_tree.nested)
    assert not mentions_everyone(exposed_tree.nested / "deep.txt")
    assert not mentions_everyone(exposed_tree.root / "existing.txt")


def test_files_created_after_hardening_are_not_world_accessible(exposed_tree):
    harden(exposed_tree.root)
    later = exposed_tree.root / "written-later.txt"
    later.write_text("new secret", encoding="utf-8")
    assert not mentions_everyone(later)


# ---------------------------------------------------------------------------
# Hardening must not become a new failure mode of its own.
# ---------------------------------------------------------------------------
def test_hardened_directories_remain_usable_by_amoeba(exposed_tree):
    harden(exposed_tree.root)
    (exposed_tree.root / "rw.txt").write_text("x", encoding="utf-8")
    assert (exposed_tree.root / "rw.txt").read_text(encoding="utf-8") == "x"
    assert (exposed_tree.nested / "deep.txt").read_text(encoding="utf-8") == "also secret"


def test_hardening_is_one_atomic_icacls_call(exposed_tree, monkeypatch):
    """Expressed as a call count, because that *is* the guarantee.

    Splitting this into ``/inheritance:r`` then ``/grant`` leaves a window where
    a failure between the two strips the DACL and locks out the owner. That is
    not hypothetical: it happened, and recovering needed ``/reset /T``. One call
    has no window, so the test that defends it counts calls.
    """
    calls: list[list[str]] = []
    real = security._icacls

    def counting(args, **kw):
        calls.append(list(args))
        return real(args, **kw)

    monkeypatch.setattr(security, "_icacls", counting)
    result = harden(exposed_tree.root)
    assert result.hardened

    mutating = [c for c in calls
                if any(a.startswith("/") and a != "/Q" for a in c)]
    assert len(mutating) == 1, (
        f"hardening took {len(mutating)} mutating icacls calls: {mutating}. "
        "More than one reintroduces a window in which a failure leaves the "
        "directory with no usable DACL.")
    assert "/inheritance:r" in mutating[0] and "/grant" in mutating[0]
    assert "/T" not in mutating[0], (
        "/T applies (OI)(CI) inheritance flags to files, where they convey no "
        "access; children must re-inherit instead")


def test_hardening_rolls_back_rather_than_locking_the_account_out(exposed_tree,
                                                                 monkeypatch):
    """Fail safe, not fail locked."""
    monkeypatch.setattr(security, "_usable", lambda p: False)
    result = harden(exposed_tree.root)

    assert result.hardened is False
    assert "locked out" in result.detail
    # The rollback has to have really happened, at the OS layer.
    (exposed_tree.root / "still-works.txt").write_text("ok", encoding="utf-8")
    assert audit_path(exposed_tree.root)["has_inherited_aces"] is True


def test_hardening_is_idempotent(exposed_tree):
    first = harden(exposed_tree.root)
    second = harden(exposed_tree.root)
    assert first.hardened and second.hardened
    assert not mentions_everyone(exposed_tree.root)
    (exposed_tree.root / "after-twice.txt").write_text("ok", encoding="utf-8")


def test_hardening_a_missing_path_fails_without_raising(tmp_path):
    result = harden(tmp_path / "nope")
    assert result.hardened is False
    assert "does not exist" in result.detail


# ---------------------------------------------------------------------------
# The whole tree, not just the root.
# ---------------------------------------------------------------------------
def test_every_state_directory_is_hardened_including_ones_outside_the_root(tmp_path):
    """Inheritance from ``state_dir`` is configuration, not a guarantee.

    ``blob_dir`` and its siblings are separately configurable. A version that
    hardened only the root and relied on propagation would leave a relocated
    blob store with its original permissive DACL -- which is exactly where
    content-addressed artifacts live.
    """
    home = tmp_path / "permissive-home"
    away = tmp_path / "permissive-away"
    for d in (home, away):
        d.mkdir()
        grant_everyone(d)

    cfg = SimpleNamespace(
        state_dir=home / "state",
        log_dir=home / "state" / "logs",
        sandbox_dir=home / "state" / "sandbox",
        artifact_dir=home / "state" / "artifacts",
        blob_dir=away / "blobs",          # deliberately outside state_dir
    )
    report = harden_state_tree(cfg)

    assert report["audit"]["exposed_count"] == 0, report["audit"]["exposed_paths"]
    assert not mentions_everyone(cfg.blob_dir), (
        "a blob directory outside state_dir kept its inherited world access")
    assert audit_path(cfg.blob_dir)["has_inherited_aces"] is False


def test_audit_reports_exposure_instead_of_asserting_safety(exposed_tree):
    before = audit_paths([exposed_tree.root])
    assert before["exposed_count"] == 1
    assert before["all_protected"] is False
    harden(exposed_tree.root)
    after = audit_paths([exposed_tree.root])
    assert after["exposed_count"] == 0
    assert after["all_protected"] is True


def test_the_audit_does_not_claim_protection_it_does_not_have(exposed_tree):
    """The owning account can rewrite the DACL. Say so, every time.

    This is the difference between a boundary and a speed bump, and it is the
    kind of claim that quietly inflates once the code starts working.
    """
    harden(exposed_tree.root)
    report = audit_path(exposed_tree.root)
    assert report["owner_can_restore_access"] is True
    assert "service account" in report["caveat"]


# ---------------------------------------------------------------------------
# Per-container grants are removed when the container goes away.
# ---------------------------------------------------------------------------
def test_revoke_removes_a_named_sid_from_the_dacl(tmp_path):
    d = tmp_path / "runtime"
    d.mkdir()
    grant_everyone(d)
    assert mentions_everyone(d)
    assert revoke(d, "S-1-1-0") is True
    assert not mentions_everyone(d)
