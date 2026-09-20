"""Host files: an allowlist, and nothing outside it.

Amoeba can produce files you actually use, and can be handed files to work on.
Both go through here, and here is the only place that turns a caller-supplied
string into a host path.

**The allowlist is the boundary.** Configured roots name specific directories
and whether each is writable. A path that does not resolve inside one of them
is refused -- not sanitised, not clamped, refused. There is no fallback root
and no "default" location, so a bug that loses the root name produces an error
rather than a write somewhere arbitrary.

**Every destructive act is reversible.** Before an overwrite or a delete the
existing bytes are content-addressed into the blob store and the digest is put
in the event log, so any prior version can be restored by digest. That is the
answer to a neuocyte writing something wrong: it cannot destroy, only
supersede.

## Why the resolution is fussy

The caller is ultimately a language model, and Windows has more ways to name a
path than POSIX does. Each of these is refused explicitly rather than left to
whatever ``Path`` happens to do:

| Trap | Why it matters |
|---|---|
| ``..`` traversal | the obvious one |
| absolute paths, drive letters, ``\\\\server\\share`` | ignore the root entirely |
| ``\\\\?\\`` long-path prefix | bypasses normalisation Win32 would otherwise do |
| alternate data streams (``notes.txt:hidden``) | writes content that does not show up in a listing |
| device names (``CON``, ``NUL``, ``COM1``) | open a device, not a file, wherever they appear |
| trailing dots and spaces | Windows silently strips them, so ``secret.txt.`` and ``secret.txt`` are the same file but compare differently |
| symlinks and junctions | a link inside the root can point anywhere |
| **hard links** | a second name for the same file record; nothing about the path is unusual, so only the link count reveals it |

Symlinks and junctions are handled by resolving fully and re-checking
containment, and writes refuse to go *through* one at all.

Hard links are a different problem and were missed on the first pass. A
hard link does not *point* at a file, it **is** the file: a second
directory entry for the same MFT record. ``resolve()`` has nothing to
resolve, ``is_symlink()`` is False, and containment correctly reports the
path as inside the root while the record is also reachable under a name
outside it. Measured before the fix: a planted hard link read content from
outside the root. The only signal is the file's link count, so a file with
more than one name is refused.

## The honest gap

Resolution and the subsequent open are not atomic. A link swapped in between
the two would be followed -- a TOCTOU race. Closing it properly needs
``O_NOFOLLOW`` semantics Windows does not offer through ``pathlib``, so it is
stated here rather than papered over. It requires an attacker already running
as this account and racing a specific operation; the same account can rewrite
the roots' ACLs anyway (see ``security.py``), so this is not the weakest link.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import FilespaceConfig, FilespaceRoot
from .errors import InvalidInput, NotFound, ResourceExhausted
from .ids import sha256_hex
from .logging_setup import get_logger

IS_WINDOWS = sys.platform == "win32"

# Windows treats these as devices in *any* directory, with or without a suffix.
RESERVED_STEMS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
MAX_RELPATH = 512
MAX_COMPONENT = 200


class FilespaceDenied(InvalidInput):
    """A path was refused. Always says which rule refused it."""


@dataclass(slots=True)
class ResolvedPath:
    root_name: str
    relpath: str
    path: Path
    writable: bool
    exists: bool

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root_name, "path": self.relpath,
                "host_path": str(self.path), "writable": self.writable,
                "exists": self.exists}


def _reject_component(part: str) -> None:
    if not part or part in (".", ".."):
        raise FilespaceDenied("path traversal is not permitted", component=part)
    if len(part) > MAX_COMPONENT:
        raise FilespaceDenied("path component too long", component=part[:40])
    if ":" in part:
        # On Windows this is an alternate data stream: content written to
        # `notes.txt:hidden` does not appear in any listing of the directory.
        raise FilespaceDenied(
            "':' is not permitted in a path (alternate data stream)",
            component=part)
    if part != part.rstrip(". "):
        # Windows strips these when opening, so `secret.txt.` resolves to
        # `secret.txt` while comparing as a different string.
        raise FilespaceDenied(
            "path components may not end in a dot or space", component=part)
    if part.split(".")[0].lower() in RESERVED_STEMS:
        raise FilespaceDenied("reserved device name", component=part)
    if any(ord(c) < 32 for c in part):
        raise FilespaceDenied("control characters are not permitted in a path")
    if any(c in part for c in '<>"|?*'):
        raise FilespaceDenied("invalid character in path", component=part)


def _link_count(path: Path) -> int:
    """How many names this file record has.

    Windows populates ``st_nlink`` from the file's link count, so a hard link
    is detectable even though nothing about the *path* is unusual. Anything
    that cannot be stat'd is reported as 1 rather than raising: the callers
    treat a count above 1 as a refusal, and failing to stat should not become
    an accidental grant.
    """
    try:
        return int(os.stat(path).st_nlink)
    except OSError:
        return 1


class Filespace:
    """Resolution and raw IO. Receipts and events belong to the Harness."""

    def __init__(self, cfg: FilespaceConfig, *, log_name: str = "filespace") -> None:
        self.cfg = cfg
        self.log = get_logger(log_name)
        self._roots: dict[str, tuple[FilespaceRoot, Path]] = {}
        for root in cfg.roots:
            resolved = Path(root.path).expanduser()
            try:
                resolved = resolved.resolve(strict=True)
            except OSError:
                self.log.warning(
                    "filespace root %r does not exist: %s", root.name, root.path)
                continue
            if not resolved.is_dir():
                self.log.warning("filespace root %r is not a directory", root.name)
                continue
            self._roots[root.name] = (root, resolved)
            self.log.info("filespace root %r -> %s (%s)",
                          root.name, resolved, root.mode)

    # -- description ----------------------------------------------------
    def available(self) -> bool:
        return bool(self._roots)

    def roots(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "host_path": str(path), "mode": root.mode,
             "writable": root.mode == "read_write",
             "description": root.description}
            for name, (root, path) in sorted(self._roots.items())
        ]

    def capabilities(self) -> dict[str, Any]:
        return {
            "filespace_available": self.available(),
            "roots": self.roots(),
            "outside_roots": "refused; there is no default or fallback location",
            "overwrite": ("prior content is content-addressed before any "
                          "overwrite or delete, so every write is reversible"
                          if self.cfg.snapshot_before_overwrite else
                          "SNAPSHOTTING IS DISABLED: writes can destroy content"),
            "max_read_bytes": self.cfg.max_read_bytes,
            "max_write_bytes": self.cfg.max_write_bytes,
            "hard_links": ("files with more than one name are refused: a hard "
                           "link is the same file record under another name, so "
                           "path containment does not bound it"
                           if not self.cfg.allow_multiply_linked else
                           "ALLOWED: a hard link can reach a file outside a root"),
            "residual_risk": ("resolution and open are not atomic; a link "
                              "swapped between them would be followed"),
        }

    # -- resolution -----------------------------------------------------
    def resolve(self, root_name: str, relpath: str, *, need_write: bool = False
                ) -> ResolvedPath:
        """Turn (root, relative path) into a host path, or refuse.

        The only way a host path is produced. A caller that has a
        ``ResolvedPath`` has already passed every check.
        """
        if not isinstance(root_name, str) or root_name not in self._roots:
            raise FilespaceDenied(
                "unknown filespace root", root=str(root_name)[:64],
                known=sorted(self._roots))
        root, root_path = self._roots[root_name]

        if need_write and root.mode != "read_write":
            raise FilespaceDenied("this root is read-only", root=root_name,
                                  mode=root.mode)

        if not isinstance(relpath, str):
            raise FilespaceDenied("path must be a string")
        # Separators are normalised; whitespace deliberately is not. Stripping
        # it would turn "notes.txt " into "notes.txt" silently -- resolving the
        # ambiguity instead of refusing it, which is the opposite of the job
        # here. Windows strips trailing spaces on open, so the two names are
        # the same file while comparing as different strings.
        relpath = relpath.replace("\\", "/")
        if not relpath or relpath == ".":
            raise FilespaceDenied("path must not be empty")
        if len(relpath) > MAX_RELPATH:
            raise FilespaceDenied("path too long", length=len(relpath))
        if relpath.startswith("//") or relpath.startswith("\\\\"):
            raise FilespaceDenied("UNC paths are not permitted", path=relpath)
        if relpath.startswith("/"):
            raise FilespaceDenied("path must be relative to the root", path=relpath)
        if re.match(r"^[A-Za-z]:", relpath):
            raise FilespaceDenied("drive-qualified paths are not permitted",
                                  path=relpath)
        if "?" in relpath or relpath.startswith("\\\\?\\"):
            raise FilespaceDenied("extended-length path syntax is not permitted",
                                  path=relpath)

        parts = [p for p in relpath.split("/") if p != ""]
        for part in parts:
            _reject_component(part)

        candidate = root_path.joinpath(*parts)

        # Resolve fully: this follows links, so a junction inside the root that
        # points elsewhere lands outside and is caught by the check below.
        try:
            final = candidate.resolve()
        except OSError as exc:
            raise FilespaceDenied("path could not be resolved",
                                  path=relpath, error=str(exc)) from exc

        if final != root_path and root_path not in final.parents:
            raise FilespaceDenied(
                "path escapes its filespace root", root=root_name, path=relpath)

        exists = final.exists()
        if exists and need_write and final.is_symlink():
            raise FilespaceDenied(
                "refusing to write through a symbolic link or junction",
                root=root_name, path=relpath)
        if exists and not self.cfg.allow_multiply_linked:
            links = _link_count(final)
            if links > 1:
                # Containment passed and was honest about the path. A hard link
                # is a second name for the same file record, so the file is
                # also reachable somewhere this root does not cover -- possibly
                # outside it entirely. `resolve()` has nothing to resolve and
                # `is_symlink()` is False, so this is the only signal there is.
                raise FilespaceDenied(
                    "refusing a file with more than one name (hard link); it is "
                    "reachable outside this root",
                    root=root_name, path=relpath, link_count=links)

        return ResolvedPath(root_name=root_name, relpath="/".join(parts),
                            path=final, writable=root.mode == "read_write",
                            exists=exists)

    def resolve_host_path(self, host_path: str, *, need_write: bool = False
                          ) -> ResolvedPath:
        """Accept an absolute host path, but only inside an allowlisted root.

        This is what ``file_attach`` uses: a person naming a real file is the
        natural way to hand one in, and it still has to land in a root.
        """
        if not isinstance(host_path, str) or not host_path.strip():
            raise FilespaceDenied("path must be a non-empty string")
        try:
            target = Path(host_path.strip()).expanduser().resolve()
        except OSError as exc:
            raise FilespaceDenied("path could not be resolved",
                                  path=host_path[:120], error=str(exc)) from exc
        for name, (_root, root_path) in sorted(self._roots.items()):
            if target == root_path or root_path in target.parents:
                rel = target.relative_to(root_path).as_posix()
                return self.resolve(name, rel, need_write=need_write)
        raise FilespaceDenied(
            "path is not inside any configured filespace root",
            path=str(target), known_roots=sorted(self._roots))

    # -- reading --------------------------------------------------------
    def list(self, root_name: str, subpath: str = "", *, limit: int | None = None
             ) -> list[dict[str, Any]]:
        limit = min(limit or self.cfg.max_listing, self.cfg.max_listing)
        if subpath:
            base = self.resolve(root_name, subpath).path
        else:
            if root_name not in self._roots:
                raise FilespaceDenied("unknown filespace root", root=root_name,
                                      known=sorted(self._roots))
            base = self._roots[root_name][1]
        if not base.is_dir():
            raise InvalidInput("not a directory", root=root_name, path=subpath)

        root_path = self._roots[root_name][1]
        out: list[dict[str, Any]] = []
        for entry in self._walk(base, root_path, limit):
            out.append(entry)
        return out

    def _walk(self, base: Path, root_path: Path, limit: int) -> Iterator[dict[str, Any]]:
        count = 0
        for path in sorted(base.rglob("*")):
            if count >= limit:
                return
            # A link is skipped rather than followed: listing through one would
            # report names and sizes from outside the root.
            if path.is_symlink():
                continue
            try:
                resolved = path.resolve()
                if root_path not in resolved.parents and resolved != root_path:
                    continue
                stat = path.stat()
            except OSError:
                continue
            links = int(getattr(stat, "st_nlink", 1) or 1)
            yield {
                "path": path.relative_to(root_path).as_posix(),
                "is_dir": path.is_dir(),
                "bytes": stat.st_size if path.is_file() else None,
                "modified": stat.st_mtime,
                # Surfaced rather than hidden: the entry exists, and whether it
                # may be opened is a separate answer the caller should be able
                # to see the reason for.
                "multiply_linked": links > 1 and path.is_file(),
                "accessible": not (links > 1 and path.is_file()
                                   and not self.cfg.allow_multiply_linked),
            }
            count += 1

    def read_bytes(self, resolved: ResolvedPath, *, max_bytes: int | None = None
                   ) -> tuple[bytes, bool]:
        cap = min(max_bytes or self.cfg.max_read_bytes, self.cfg.max_read_bytes)
        if not resolved.path.is_file():
            raise NotFound("no such file", root=resolved.root_name,
                           path=resolved.relpath)
        data = resolved.path.read_bytes()
        return data[:cap], len(data) > cap

    # -- writing --------------------------------------------------------
    def write_bytes(self, resolved: ResolvedPath, data: bytes) -> dict[str, Any]:
        """Write, atomically, having already been told the path is permitted.

        Written to a temporary file in the same directory and then moved, so a
        failure part-way leaves the previous file intact rather than a truncated
        one. Snapshotting of the prior content is the Harness's job -- it owns
        the blob store and the event log.

        The rename has a second effect that was found by experiment rather than
        designed: it replaces the *directory entry*, so a write never modifies
        an existing file record in place. A hard link in the root therefore
        cannot be used to write through to the file's other names -- the link
        is broken and the new bytes land in a new record. That is a real
        property and it is pinned by a test, because rewriting this as
        ``path.write_bytes(data)`` would silently make write-through live
        again. It is a second line of defence: multiply-linked files are
        refused at resolution.
        """
        if len(data) > self.cfg.max_write_bytes:
            raise ResourceExhausted("file exceeds the filespace write limit",
                                    size=len(data), limit=self.cfg.max_write_bytes)
        if not resolved.writable:
            raise FilespaceDenied("this root is read-only", root=resolved.root_name)
        resolved.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = resolved.path.with_name(f".{resolved.path.name}.amoeba-tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, resolved.path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return {"root": resolved.root_name, "path": resolved.relpath,
                "host_path": str(resolved.path), "bytes": len(data),
                "sha256": sha256_hex(data)}

    def delete(self, resolved: ResolvedPath) -> dict[str, Any]:
        if not resolved.writable:
            raise FilespaceDenied("this root is read-only", root=resolved.root_name)
        if resolved.path.is_dir():
            raise InvalidInput(
                "refusing to delete a directory; delete files individually",
                root=resolved.root_name, path=resolved.relpath)
        if not resolved.path.exists():
            raise NotFound("no such file", root=resolved.root_name,
                           path=resolved.relpath)
        resolved.path.unlink()
        return {"root": resolved.root_name, "path": resolved.relpath,
                "host_path": str(resolved.path), "deleted": True}

    def copy_into(self, resolved: ResolvedPath, destination: Path) -> int:
        """Copy a permitted host file somewhere Amoeba owns (a sandbox)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved.path, destination)
        return destination.stat().st_size
