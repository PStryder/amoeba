"""Content-addressed blob store.

Blob identity is SHA-256 of the exact bytes, recorded together with an encoding
label.  Bytes are written and fsynced *before* any event referencing them
commits, so a crash can leave orphan blobs (recoverable garbage) but never a
committed reference to missing content (an integrity failure).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..errors import IntegrityError
from ..ids import sha256_hex


def _fsync_dir(path: Path) -> None:
    """Flush the directory entry the rename just created.

    I4 says bytes are durable before anything references them, and fsyncing
    the file is only half of that: after a crash the file can be intact and
    its name absent, because the rename lived in an unflushed directory
    entry. An audit on 2026-09-24 noted the missing step and, correctly,
    declined to accept "it is fine on NTFS" from a code reading.

    POSIX exposes this directly. Windows does not -- a directory cannot be
    opened with `os.open` for fsync, and flushing the volume is both
    privileged and far heavier than this warrants -- so there it is a no-op
    and the guarantee rests on NTFS metadata journalling, which orders the
    rename but does not promise it is on the platter. That is a real
    difference between the platforms and is written down rather than assumed
    away.
    """
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / f"{digest}.blob"

    # -- writing -------------------------------------------------------
    def put(self, data: bytes) -> str:
        """Store bytes durably, returning the digest. Idempotent."""
        digest = sha256_hex(data)
        target = self.path_for(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            _fsync_dir(target.parent)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return digest

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def put_json(self, obj: Any) -> str:
        return self.put(canonical_json(obj))

    # -- reading -------------------------------------------------------
    def exists(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def get(self, digest: str, *, verify: bool = True) -> bytes:
        p = self.path_for(digest)
        if not p.exists():
            raise IntegrityError("blob missing from content store", digest=digest)
        data = p.read_bytes()
        if verify and sha256_hex(data) != digest:
            raise IntegrityError("blob content does not match its digest", digest=digest)
        return data

    def get_text(self, digest: str) -> str:
        return self.get(digest).decode("utf-8")

    def get_json(self, digest: str) -> Any:
        return json.loads(self.get(digest).decode("utf-8"))

    def verify(self, digest: str) -> bool:
        try:
            self.get(digest, verify=True)
        except IntegrityError:
            return False
        return True


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding used for hashing and for blob identity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
