"""Identifier and hashing helpers.

Identifiers are prefixed ULID-ish strings: sortable by creation time and
visually self-describing in the event log.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_B32[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def ulid() -> str:
    """A lexicographically sortable 26-char identifier (48-bit ms + 80-bit random)."""
    ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return _encode(ms, 10) + _encode(rand, 16)


def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid()}"


def uuid4() -> str:
    return str(uuid.uuid4())


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chain_hash(prev_hash: str, canonical_payload: bytes) -> str:
    """Hash-chain link: H(prev || payload). Detects ordinary mutation, not an
    administrator who rewrites both rows and hashes."""
    h = hashlib.sha256()
    h.update(prev_hash.encode("ascii"))
    h.update(b"\x00")
    h.update(canonical_payload)
    return h.hexdigest()


GENESIS_HASH = "0" * 64
