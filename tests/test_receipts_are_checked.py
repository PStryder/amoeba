"""If the sandbox does not land what we hashed, the receipt is refused.

I41 says a receipt's digest is what the neuocyte can actually hash. The round
trip through a real container proves the two agree when everything works. What
nothing asserted is the guard that makes them agree: `file_attach` compares
what the sandbox reports it wrote against the digest it recorded, and refuses
if they differ.

That guard cannot fire while the sandbox is honest, so no live test could ever
reach it -- and the full mutation sweep on 2026-09-24 duly found that removing
it changed nothing any test could see. A sandbox that lands something else is
the case it exists for, so that is what this hands it.
"""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace

import pytest

from amoeba.errors import IntegrityError


def _harness(mind, tmp_path, *, lands):
    """`file_attach`, with a sandbox that reports whatever `lands` says."""
    from amoeba import harness_api
    from amoeba.config import FilespaceRoot
    from amoeba.filespace import Filespace

    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    mind.cfg.filespace.roots = [
        FilespaceRoot(name="src", path=str(root), mode="read_only")]

    sandboxes = SimpleNamespace(
        write_bytes=lambda sandbox_id, dest, data: {
            "sha256": lands(data), "bytes": len(data), "path": dest})

    sup = SimpleNamespace(
        mind=mind, cfg=mind.cfg, log=logging.getLogger("t"),
        filespace=Filespace(mind.cfg.filespace),
        sandboxes=sandboxes,
        sandbox_for_work=lambda work_id, owner=None: "sbx_1",
        methods=lambda: {}, note_trigger=lambda role: None,
        arbiter=SimpleNamespace())
    return harness_api.build(sup)["file_attach"], root


def test_a_sandbox_that_lands_something_else_is_refused(mind, tmp_path):
    payload = b"\x89PNG\r\n" + bytes(range(200, 240))
    attach, root = _harness(
        mind, tmp_path, lands=lambda data: hashlib.sha256(b"not this").hexdigest())
    (root / "payload.bin").write_bytes(payload)

    with pytest.raises(IntegrityError) as caught:
        attach(path=str(root / "payload.bin"), work_id="work_1", actor="pete")

    assert "did not land in the sandbox intact" in caught.value.message
    assert caught.value.details["source_sha256"] == hashlib.sha256(payload).hexdigest()


def test_a_sandbox_that_lands_the_bytes_is_receipted(mind, tmp_path):
    """The other side, so the test is not merely asserting that things break."""
    payload = b"the exact bytes"
    attach, root = _harness(
        mind, tmp_path, lands=lambda data: hashlib.sha256(data).hexdigest())
    (root / "payload.bin").write_bytes(payload)

    out = attach(path=str(root / "payload.bin"), work_id="work_1", actor="pete")

    assert out["sha256"] == hashlib.sha256(payload).hexdigest()
    assert out["bytes"] == len(payload)
