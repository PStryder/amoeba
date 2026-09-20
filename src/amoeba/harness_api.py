"""Harness RPC surface for the blackboard, sandboxed compute and homeostasis.

Three rules hold across everything in this file:

1. **A neuocyte proposes; the Harness disposes.** Promotion of a board post
   into memory, promotion of a sandbox artifact into the durable workspace, and
   rejuvenation of a context are all acts of the Harness. The verbs a neuocyte
   can reach create *proposals*.
2. **Every consequential act returns a receipt.** Refusals are recorded too --
   a refused request is a fact about how the mind governed itself.
3. **Model-generated strings never become host paths or host commands.** Every
   sandbox path is resolved inside its scratch root; every promotion target is
   named by the Harness, not the caller.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .errors import IntegrityError, InvalidInput, NotFound, ResourceExhausted
from .ids import new_id, sha256_hex
from .sandbox import SandboxLimits
from .store.events import EventKind
from .filespace import decode_exact_text
from .tools import ToolCallRequest, build_neuocyte_registry
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

SCHEMA_VERSION = "1.0.0"
PROMOTABLE_SUFFIXES = {".py", ".txt", ".md", ".json", ".csv", ".yaml", ".yml",
                       ".toml", ".sql", ".log", ".tsv", ".ini", ".cfg"}
MAX_PROMOTED_BYTES = 16 * 1024 * 1024
# Tool arguments are model-generated and can carry a whole file. The event
# log keeps the shape and a digest, not the payload; the payload lives in
# the sandbox, which is where it belongs.
MAX_LOGGED_ARG_CHARS = 500


def _redact(arguments: dict) -> dict:
    out = {}
    for k, v in arguments.items():
        if isinstance(v, str) and len(v) > MAX_LOGGED_ARG_CHARS:
            out[k] = {"truncated": True, "chars": len(v),
                      "sha256": sha256_hex(v.encode("utf-8", "replace")),
                      "head": v[:MAX_LOGGED_ARG_CHARS]}
        else:
            out[k] = v
    return out


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None

    # ==================================================================
    # Cognitive blackboard
    # ==================================================================
    def board_post(*, author: str, author_kind: str, post_type: str, body: str,
                   title: str | None = None, thread_id: str | None = None,
                   work_id: str | None = None, operation_id: str | None = None,
                   confidence: float | None = None,
                   evidence: Sequence[dict[str, Any]] = (),
                   relations: Sequence[dict[str, str]] = (),
                   snapshot_id: str | None = None,
                   model_generation: str | None = None,
                   supersedes: str | None = None) -> dict[str, Any]:
        post_id, receipt = mind.board.post(
            author=author, author_kind=author_kind, post_type=post_type, body=body,
            title=title, thread_id=thread_id, work_id=work_id,
            operation_id=operation_id, confidence=confidence, evidence=evidence,
            relations=relations, snapshot_id=snapshot_id,
            model_generation=model_generation, supersedes=supersedes)
        post = mind.board.get_post(post_id)
        return {"post_id": post_id, "thread_id": post["thread_id"], "seq": post["seq"],
                "receipt_id": receipt.receipt_id,
                "state_version": receipt.result_version,
                "board_naive": post["board_naive"],
                "informed_by_count": post["read_count_before"]}

    def board_read(*, reader: str, thread_id: str | None = None,
                   post_types: Sequence[str] | None = None, query: str | None = None,
                   since_seq: int | None = None, limit: int = 20,
                   work_id: str | None = None, record: bool = True) -> dict[str, Any]:
        posts = mind.board.read(
            reader=reader, thread_id=thread_id, post_types=post_types, query=query,
            since_seq=since_seq, limit=limit, work_id=work_id, record=record)
        return {"posts": posts, "count": len(posts),
                "cursor": mind.board.latest_seq(),
                "reads_recorded": record,
                "note": ("this read was recorded against the reader, so any finding "
                         "they publish afterwards is marked as socially informed"
                         if record else
                         "read without recording; use only for Harness and audit")}

    def board_get_post(*, post_id: str, reader: str | None = None,
                       work_id: str | None = None) -> dict[str, Any]:
        return mind.board.get_post(post_id, reader=reader, work_id=work_id)

    def board_thread(*, thread_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return mind.board.thread(thread_id, limit=limit)

    def board_relate(*, from_post: str, to_post: str, relation: str, actor: str
                     ) -> dict[str, Any]:
        r = mind.board.relate(from_post=from_post, to_post=to_post,
                              relation=relation, actor=actor)
        return {"receipt_id": r.receipt_id}

    def board_set_status(*, post_id: str, status: str, actor: str,
                         reason: str | None = None) -> dict[str, Any]:
        r = mind.board.set_status(post_id=post_id, status=status, actor=actor,
                                  reason=reason)
        return {"receipt_id": r.receipt_id}

    def board_independence(*, post_a: str, post_b: str) -> dict[str, Any]:
        return mind.board.independence(post_a, post_b)

    def board_corroboration(*, post_id: str) -> dict[str, Any]:
        return mind.board.corroboration(post_id)

    def board_stats() -> dict[str, Any]:
        return mind.board.stats()

    def board_promote_to_memory(*, post_id: str, kind: str = "belief",
                                confidence: float | None = None,
                                actor: str = "supervisor",
                                operation_id: str | None = None) -> dict[str, Any]:
        """Turn something a neuocyte *said* into something the organism *believes*.

        Deliberately an explicit Harness act with its own receipt. The post is
        not consumed: it keeps its identity on the board, and the new memory
        item carries it as supporting evidence.

        Independent corroboration is attached to the memory item, and merely
        social agreement is attached as well -- clearly labelled -- so that a
        later reader can see how much of the support was real.
        """
        post = mind.board.get_post(post_id)
        corr = mind.board.corroboration(post_id)
        supporting = [{"note": f"board post {post_id} by {post['author']}",
                       "blob_sha256": None}]
        for pid in corr["independent_support"]:
            supporting.append({"note": f"independent replication: board post {pid}"})
        opposing = [{"note": f"challenge on the board: post {pid}"}
                    for pid in corr["challenges"]]
        if corr["socially_informed_support"]:
            opposing.append({
                "note": ("support from "
                         f"{len(corr['socially_informed_support'])} post(s) that had "
                         "already read the original; not independent evidence")})
        conf = confidence if confidence is not None else (post.get("confidence") or 0.5)
        memory_id, receipt = mind.memory.remember(
            kind=kind, claim=post["body"][:2000], confidence=float(conf),
            created_by=actor, supporting=supporting, opposing=opposing,
            tags=["from_board", post["post_type"]], operation_id=operation_id)

        def body(m: Mutation) -> None:
            m.emit(EventKind.BOARD_PROMOTED, {
                "post_id": post_id, "memory_id": memory_id, "actor": actor,
                "independent_support": corr["independent_support_count"],
                "socially_informed_support": len(corr["socially_informed_support"]),
            })

        mind.writer.apply(body, actor=actor, bump_version=False,
                          mutation_id=f"board-promote:{post_id}")
        return {"post_id": post_id, "memory_id": memory_id,
                "receipt_id": receipt.receipt_id,
                "independent_support": corr["independent_support_count"],
                "socially_informed_support": len(corr["socially_informed_support"]),
                "note": ("the post remains on the board; promotion created a separate "
                         "maintained belief citing it")}

    # ==================================================================
    # Sandboxed compute
    # ==================================================================
    def _sandbox_manager():
        if sup.sandboxes is None:
            raise ResourceExhausted("sandboxing is disabled in this configuration")
        return sup.sandboxes

    def sandbox_capabilities() -> dict[str, Any]:
        if sup.sandboxes is None:
            return {"sandbox_available": False, "detail": "disabled in configuration"}
        return sup.sandboxes.capabilities()

    def sandbox_create(*, owner: str, work_id: str | None = None,
                       wall_seconds: float | None = None,
                       operation_id: str | None = None) -> dict[str, Any]:
        mgr = _sandbox_manager()
        active = [s for s in mgr.list()]
        if len(active) >= sup.cfg.sandbox.max_concurrent:
            mind.writer.record_rejection(
                actor=owner, reason="sandbox concurrency limit reached",
                kind=EventKind.SANDBOX_DENIED,
                payload={"owner": owner, "active": len(active)},
                operation_id=operation_id)
            raise ResourceExhausted("too many live sandboxes",
                                    active=len(active),
                                    limit=sup.cfg.sandbox.max_concurrent)
        c = sup.cfg.sandbox
        limits = SandboxLimits(
            wall_seconds=min(wall_seconds or c.wall_seconds, c.wall_seconds),
            cpu_seconds=c.cpu_seconds, memory_bytes=c.memory_bytes,
            max_processes=c.max_processes, max_output_bytes=c.max_output_bytes,
            max_scratch_bytes=c.max_scratch_bytes,
            max_artifact_bytes=c.max_artifact_bytes)
        sb = mgr.create(owner=owner, limits=limits)

        def body(m: Mutation) -> None:
            m.sql("INSERT INTO sandboxes(sandbox_id, owner, work_id, container_sid,"
                  " root, limits, status, created_at, state_version)"
                  " VALUES (?,?,?,?,?,?,?,?,?)",
                  (sb.sandbox_id, owner, work_id, sb.container_sid, str(sb.root),
                   __import__("json").dumps(limits.to_dict()), "active",
                   sb.created_at, m.prior_version + 1))
            m.emit(EventKind.SANDBOX_CREATED, {
                "sandbox_id": sb.sandbox_id, "owner": owner, "work_id": work_id,
                "limits": limits.to_dict(), "isolation": "windows_appcontainer"})

        receipt, _ = mind.writer.apply(body, actor=owner, operation_id=operation_id,
                                       mutation_id=f"sbx-create:{sb.sandbox_id}")
        return {"sandbox_id": sb.sandbox_id, "limits": limits.to_dict(),
                "receipt_id": receipt.receipt_id,
                "capabilities": mgr.capabilities()}

    def sandbox_run(*, sandbox_id: str, code: str | None = None,
                    script: str | None = None, argv: Sequence[str] = (),
                    timeout: float | None = None, actor: str = "neuocyte",
                    operation_id: str | None = None) -> dict[str, Any]:
        mgr = _sandbox_manager()
        t0 = time.perf_counter()
        result = mgr.run_python(sandbox_id, code=code, script=script, argv=argv,
                                timeout=timeout)

        def body(m: Mutation) -> None:
            m.emit(EventKind.SANDBOX_RUN, {
                "sandbox_id": sandbox_id, "actor": actor,
                "exit_code": result.exit_code, "timed_out": result.timed_out,
                "seconds": round(result.seconds, 3),
                "killed_reason": result.killed_reason,
                "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
                "code_sha256": sha256_hex((code or script or "").encode()),
            })

        receipt, _ = mind.writer.apply(body, actor=actor, operation_id=operation_id,
                                       bump_version=False)
        out = result.to_dict()
        out["receipt_id"] = receipt.receipt_id
        out["harness_seconds"] = time.perf_counter() - t0
        return out

    def sandbox_write(*, sandbox_id: str, path: str, content: str) -> dict[str, Any]:
        return _sandbox_manager().write_file(sandbox_id, path, content)

    def sandbox_read(*, sandbox_id: str, path: str,
                     max_bytes: int = 262144) -> dict[str, Any]:
        return _sandbox_manager().read_file(sandbox_id, path, max_bytes=max_bytes)

    def sandbox_files(*, sandbox_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return _sandbox_manager().list_files(sandbox_id, limit=limit)

    def sandbox_list() -> list[dict[str, Any]]:
        return _sandbox_manager().list()

    def sandbox_destroy(*, sandbox_id: str, actor: str = "supervisor",
                        reason: str = "", operation_id: str | None = None
                        ) -> dict[str, Any]:
        mgr = _sandbox_manager()
        out = mgr.destroy(sandbox_id)

        def body(m: Mutation) -> None:
            m.sql("UPDATE sandboxes SET status = 'destroyed', destroyed_at = ?"
                  " WHERE sandbox_id = ?", (time.time(), sandbox_id))
            m.emit(EventKind.SANDBOX_DESTROYED,
                   {"sandbox_id": sandbox_id, "actor": actor,
                    "reason": reason})

        receipt, _ = mind.writer.apply(body, actor=actor, operation_id=operation_id,
                                       bump_version=False,
                                       mutation_id=f"sbx-destroy:{sandbox_id}")
        out["receipt_id"] = receipt.receipt_id
        return out

    # -- promotion: the only way out of a sandbox ----------------------
    def artifact_propose(*, sandbox_id: str, path: str, rationale: str,
                         proposed_by: str, work_id: str | None = None,
                         media_type: str | None = None,
                         operation_id: str | None = None) -> dict[str, Any]:
        """A neuocyte proposes that a scratch file become durable.

        Proposing copies nothing and grants nothing. It records an intention
        that the Harness can later act on.
        """
        mgr = _sandbox_manager()
        # resolve_inside is the path check; read_file would additionally slurp
        # the whole file just to validate, and this one is read again below.
        target = mgr.resolve_inside(mgr.get(sandbox_id), path)
        if not target.is_file():
            raise NotFound("no such file in sandbox", path=path)
        size = target.stat().st_size
        if size > MAX_PROMOTED_BYTES:
            raise ResourceExhausted("artifact too large to promote", bytes=size,
                                    limit=MAX_PROMOTED_BYTES)
        artifact_id = new_id("art")
        digest = sha256_hex(target.read_bytes())

        def body(m: Mutation) -> None:
            m.sql("INSERT INTO artifacts(artifact_id, sandbox_id, proposed_by, work_id,"
                  " path, sha256, bytes, media_type, rationale, status, created_at,"
                  " state_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (artifact_id, sandbox_id, proposed_by, work_id, path, digest, size,
                   media_type, rationale, "proposed", time.time(), m.prior_version + 1))
            m.emit(EventKind.ARTIFACT_PROPOSED, {
                "artifact_id": artifact_id, "sandbox_id": sandbox_id,
                "proposed_by": proposed_by, "path": path, "bytes": size,
                "sha256": digest, "rationale": rationale})

        receipt, _ = mind.writer.apply(body, actor=proposed_by,
                                       operation_id=operation_id,
                                       mutation_id=f"art-propose:{artifact_id}")
        return {"artifact_id": artifact_id, "status": "proposed", "bytes": size,
                "sha256": digest, "receipt_id": receipt.receipt_id,
                "note": "proposed only; nothing has been copied out of the sandbox"}

    def artifact_promote(*, artifact_id: str, decided_by: str = "supervisor",
                         root: str | None = None, path: str | None = None,
                         operation_id: str | None = None) -> dict[str, Any]:
        """The Harness copies a proposed artifact out of the sandbox.

        Without ``root`` it lands in the internal workspace under a name the
        Harness picks. With ``root`` it lands in a configured filespace root at
        ``path`` -- this is how Amoeba produces a file you actually use, and it
        is a *decision*, made here, on a destination the person deciding named.
        The neuocyte that proposed the artifact never sees either.

        The bytes are re-hashed on arrival so the receipt describes what landed
        rather than what was promised, and anything already at the destination
        is content-addressed first, so promoting over a file supersedes it
        rather than destroying it.
        """
        row = mind.db.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise NotFound("unknown artifact", artifact_id=artifact_id)
        if row["status"] != "proposed":
            raise InvalidInput("artifact is not awaiting a decision",
                               artifact_id=artifact_id, status=row["status"])
        mgr = _sandbox_manager()
        src = mgr.resolve_inside(mgr.get(row["sandbox_id"]), row["path"])
        if not src.is_file():
            raise NotFound("proposed file no longer exists", path=row["path"])
        suffix = src.suffix.lower()
        if suffix and suffix not in PROMOTABLE_SUFFIXES:
            raise InvalidInput("file type is not promotable",
                               suffix=suffix, allowed=sorted(PROMOTABLE_SUFFIXES))

        data = src.read_bytes()
        digest = sha256_hex(data)
        if digest != row["sha256"]:
            # Fail closed. The rationale the Harness is acting on describes the
            # bytes that were proposed; promoting different bytes under that
            # approval is the whole shape of a time-of-check/time-of-use
            # attack. Re-propose instead.
            def rejected(m: Mutation) -> None:
                m.sql("UPDATE artifacts SET status = 'rejected', decided_by = ?,"
                      " decided_at = ?, reason = ? WHERE artifact_id = ?",
                      (decided_by, time.time(),
                       "content changed after proposal", artifact_id))
                m.emit(EventKind.ARTIFACT_REJECTED, {
                    "artifact_id": artifact_id, "decided_by": decided_by,
                    "reason": "content changed between proposal and promotion",
                    "sha256_at_proposal": row["sha256"],
                    "sha256_on_arrival": digest})

            mind.writer.apply(rejected, actor=decided_by, operation_id=operation_id,
                              mutation_id=f"art-tamper:{artifact_id}")
            raise IntegrityError(
                "artifact content changed between proposal and promotion; "
                "refusing to promote bytes the Harness did not review",
                artifact_id=artifact_id, sha256_at_proposal=row["sha256"],
                sha256_on_arrival=digest)

        prior_sha = prior_bytes = None
        resolved = None
        if root is not None:
            # An explicit destination: still resolved through the allowlist,
            # and still refused if it escapes or the root is read-only.
            fs = _filespace()
            try:
                resolved = fs.resolve(root, path or src.name, need_write=True)
            except InvalidInput as exc:
                _deny(decided_by, exc.message,
                      {"root": root, "path": (path or src.name)[:200],
                       "operation": "promote", "artifact_id": artifact_id})
                raise
            if resolved.exists and resolved.path.is_file() \
                    and sup.cfg.filespace.snapshot_before_overwrite:
                existing = resolved.path.read_bytes()
                prior_sha = mind.blobs.put(existing)
                prior_bytes = len(existing)
            fs.write_bytes(resolved, data)
            dest = resolved.path
        else:
            workspace = sup.cfg.workspace_dir
            workspace.mkdir(parents=True, exist_ok=True)
            # The Harness names the destination: artifact id + original basename.
            dest = workspace / f"{artifact_id}_{src.name}"
            dest.write_bytes(data)
        blob = mind.blobs.put(data)

        def body(m: Mutation) -> None:
            m.register_blob(blob, len(data), "application/octet-stream", "artifact")
            m.sql("UPDATE artifacts SET status = 'promoted', decided_by = ?,"
                  " decided_at = ?, sha256 = ? WHERE artifact_id = ?",
                  (decided_by, time.time(), digest, artifact_id))
            if prior_sha:
                m.register_blob(prior_sha, prior_bytes or 0,
                                "application/octet-stream", "file_version")
                m.emit(EventKind.FILE_SUPERSEDED, {
                    "root": root, "path": resolved.relpath,
                    "host_path": str(dest), "prior_sha256": prior_sha,
                    "prior_bytes": prior_bytes, "restorable": True,
                    "superseded_by": artifact_id})
            m.emit(EventKind.ARTIFACT_PROMOTED, {
                "artifact_id": artifact_id, "decided_by": decided_by,
                "workspace_path": dest.name, "bytes": len(data),
                "sha256_at_proposal": row["sha256"], "sha256_on_arrival": digest,
                "content_changed_since_proposal": digest != row["sha256"],
                "blob": blob, "root": root,
                "path": resolved.relpath if resolved else None,
                "host_path": str(dest)})
            if root is not None:
                m.emit(EventKind.FILE_WRITTEN, {
                    "root": root, "path": resolved.relpath,
                    "host_path": str(dest), "actor": decided_by,
                    "bytes": len(data), "sha256": digest,
                    "rationale": f"promotion of artifact {artifact_id}",
                    "overwrote": bool(prior_sha)})

        receipt, _ = mind.writer.apply(body, actor=decided_by,
                                       operation_id=operation_id,
                                       mutation_id=f"art-promote:{artifact_id}")
        return {"artifact_id": artifact_id, "status": "promoted",
                "workspace_path": dest.name, "host_path": str(dest),
                "root": root, "path": resolved.relpath if resolved else None,
                "bytes": len(data), "sha256": digest,
                "content_verified": True, "blob": blob,
                "overwrote": bool(prior_sha), "prior_sha256": prior_sha,
                "receipt_id": receipt.receipt_id}

    def artifact_reject(*, artifact_id: str, reason: str,
                        decided_by: str = "supervisor",
                        operation_id: str | None = None) -> dict[str, Any]:
        def body(m: Mutation) -> None:
            cur = m.sql("UPDATE artifacts SET status = 'rejected', decided_by = ?,"
                        " decided_at = ?, reason = ? WHERE artifact_id = ?"
                        " AND status = 'proposed'",
                        (decided_by, time.time(), reason, artifact_id))
            if cur.rowcount == 0:
                raise NotFound("no proposed artifact with that id",
                               artifact_id=artifact_id)
            m.emit(EventKind.ARTIFACT_REJECTED, {
                "artifact_id": artifact_id, "decided_by": decided_by,
                "reason": reason})

        receipt, _ = mind.writer.apply(body, actor=decided_by,
                                       operation_id=operation_id,
                                       mutation_id=f"art-reject:{artifact_id}")
        return {"artifact_id": artifact_id, "status": "rejected",
                "receipt_id": receipt.receipt_id}

    def artifact_list(*, status: str | None = None, limit: int = 50
                      ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in mind.db.conn.execute(sql, params)]

    # ==================================================================
    # Context homeostasis
    # ==================================================================
    # ==================================================================
    # Host filesystem
    # ==================================================================
    def _filespace():
        if sup.filespace is None or not sup.filespace.available():
            raise ResourceExhausted(
                "no filespace roots are configured; Amoeba has nowhere on "
                "disk it is permitted to read or write")
        return sup.filespace

    def _deny(actor: str, reason: str, detail: dict[str, Any]) -> None:
        """A refused path is a fact about how the mind governed itself."""
        def body(m: Mutation) -> None:
            m.emit(EventKind.FILE_DENIED,
                   {"actor": actor, "reason": reason, **detail})

        mind.writer.apply(body, actor=actor, bump_version=False)

    def file_roots() -> dict[str, Any]:
        if sup.filespace is None:
            return {"filespace_available": False, "roots": [],
                    "detail": "filespace not initialised"}
        return sup.filespace.capabilities()

    def file_list(*, root: str, path: str = "", limit: int = 200
                  ) -> list[dict[str, Any]]:
        return _filespace().list(root, path, limit=limit)

    def file_read(*, root: str, path: str, max_bytes: int | None = None,
                  actor: str = "supervisor") -> dict[str, Any]:
        fs = _filespace()
        try:
            resolved = fs.resolve(root, path)
        except InvalidInput as exc:
            _deny(actor, exc.message, {"root": root, "path": path[:200],
                                       "operation": "read"})
            raise
        data, truncated = fs.read_bytes(resolved, max_bytes=max_bytes)
        digest = sha256_hex(data)
        text = decode_exact_text(data, truncated=truncated,
                                 where=resolved.relpath, digest=digest,
                                 total_bytes=len(data))
        return {**resolved.to_dict(), "bytes": len(data), "truncated": truncated,
                "sha256": digest, "content": text}

    def file_write(*, root: str, path: str, content: str,
                   actor: str = "supervisor", rationale: str = "",
                   operation_id: str | None = None) -> dict[str, Any]:
        """Write a file, preserving whatever was there before.

        The prior bytes are content-addressed into the blob store *before* the
        write and the digest goes into the event log, so a write is always a
        supersession rather than a destruction. `file_restore` puts any earlier
        version back by digest.

        This is a Harness verb. A neuocyte reaches the host filesystem only by
        proposing an artifact, which is decided separately.
        """
        fs = _filespace()
        try:
            resolved = fs.resolve(root, path, need_write=True)
        except InvalidInput as exc:
            _deny(actor, exc.message, {"root": root, "path": path[:200],
                                       "operation": "write"})
            raise

        data = content.encode("utf-8")
        prior_sha = prior_bytes = None
        if resolved.exists and resolved.path.is_file():
            if sup.cfg.filespace.snapshot_before_overwrite:
                existing = resolved.path.read_bytes()
                prior_sha = mind.blobs.put(existing)
                prior_bytes = len(existing)
            else:
                prior_sha = None

        out = fs.write_bytes(resolved, data)

        def body(m: Mutation) -> None:
            if prior_sha:
                m.register_blob(prior_sha, prior_bytes or 0,
                                "application/octet-stream", "file_version")
                m.emit(EventKind.FILE_SUPERSEDED, {
                    "root": root, "path": resolved.relpath,
                    "host_path": str(resolved.path),
                    "prior_sha256": prior_sha, "prior_bytes": prior_bytes,
                    "restorable": True,
                    "note": "previous content is recoverable by this digest"})
            m.emit(EventKind.FILE_WRITTEN, {
                "root": root, "path": resolved.relpath,
                "host_path": str(resolved.path), "actor": actor,
                "bytes": out["bytes"], "sha256": out["sha256"],
                "rationale": rationale[:500],
                "overwrote": bool(prior_sha)})

        receipt, _ = mind.writer.apply(body, actor=actor,
                                       operation_id=operation_id)
        return {**out, "overwrote": bool(prior_sha),
                "prior_sha256": prior_sha,
                "receipt_id": receipt.receipt_id,
                "note": ("the previous content is preserved and restorable"
                         if prior_sha else "no previous file at this path")}

    def file_delete(*, root: str, path: str, actor: str = "supervisor",
                    reason: str = "", operation_id: str | None = None
                    ) -> dict[str, Any]:
        """Delete a file, keeping its content recoverable."""
        fs = _filespace()
        try:
            resolved = fs.resolve(root, path, need_write=True)
        except InvalidInput as exc:
            _deny(actor, exc.message, {"root": root, "path": path[:200],
                                       "operation": "delete"})
            raise
        prior_sha = prior_bytes = None
        if resolved.path.is_file() and sup.cfg.filespace.snapshot_before_overwrite:
            existing = resolved.path.read_bytes()
            prior_sha = mind.blobs.put(existing)
            prior_bytes = len(existing)
        out = fs.delete(resolved)

        def body(m: Mutation) -> None:
            if prior_sha:
                m.register_blob(prior_sha, prior_bytes or 0,
                                "application/octet-stream", "file_version")
            m.emit(EventKind.FILE_DELETED, {
                "root": root, "path": resolved.relpath,
                "host_path": str(resolved.path), "actor": actor,
                "reason": reason[:500], "prior_sha256": prior_sha,
                "prior_bytes": prior_bytes, "restorable": bool(prior_sha)})

        receipt, _ = mind.writer.apply(body, actor=actor,
                                       operation_id=operation_id)
        return {**out, "prior_sha256": prior_sha,
                "receipt_id": receipt.receipt_id,
                "note": ("content is recoverable by digest via file_restore"
                         if prior_sha else "content was NOT snapshotted")}

    def file_restore(*, root: str, path: str, sha256: str,
                     actor: str = "supervisor", operation_id: str | None = None
                     ) -> dict[str, Any]:
        """Put an earlier version back, by digest.

        The counterpart that makes supersession meaningful. Restoring is itself
        a write, so the content being replaced is snapshotted too -- undo is
        not a way to lose the current version.
        """
        fs = _filespace()
        resolved = fs.resolve(root, path, need_write=True)
        if not mind.blobs.exists(sha256):
            raise NotFound("no stored content with that digest", sha256=sha256,
                           hint="file_versions lists the digests for a path")
        data = mind.blobs.get(sha256)

        prior_sha = prior_bytes = None
        if resolved.exists and resolved.path.is_file():
            existing = resolved.path.read_bytes()
            prior_sha = mind.blobs.put(existing)
            prior_bytes = len(existing)
        out = fs.write_bytes(resolved, data)

        def body(m: Mutation) -> None:
            if prior_sha:
                m.register_blob(prior_sha, prior_bytes or 0,
                                "application/octet-stream", "file_version")
            m.emit(EventKind.FILE_RESTORED, {
                "root": root, "path": resolved.relpath,
                "host_path": str(resolved.path), "actor": actor,
                "restored_sha256": sha256, "replaced_sha256": prior_sha,
                "bytes": out["bytes"]})

        receipt, _ = mind.writer.apply(body, actor=actor,
                                       operation_id=operation_id)
        return {**out, "restored_sha256": sha256, "replaced_sha256": prior_sha,
                "receipt_id": receipt.receipt_id}

    def file_versions(*, root: str, path: str, limit: int = 50
                      ) -> list[dict[str, Any]]:
        """Every recorded version of one path, newest first.

        Reconstructed from the event log rather than a separate index, so it
        cannot drift from what actually happened.
        """
        from .store.events import read_events

        resolved = _filespace().resolve(root, path)
        wanted = (EventKind.FILE_WRITTEN, EventKind.FILE_SUPERSEDED,
                  EventKind.FILE_DELETED, EventKind.FILE_RESTORED)
        out: list[dict[str, Any]] = []
        for ev in read_events(mind.db.conn, kinds=list(wanted), limit=2000):
            payload = ev.payload(mind.blobs) or {}
            if payload.get("root") != root or payload.get("path") != resolved.relpath:
                continue
            digest = (payload.get("prior_sha256") if ev.kind in
                      (EventKind.FILE_SUPERSEDED, EventKind.FILE_DELETED)
                      else payload.get("sha256"))
            if not digest:
                continue
            out.append({"sha256": digest, "kind": ev.kind, "seq": ev.seq,
                        "ts": ev.ts, "actor": ev.actor_id,
                        "bytes": payload.get("bytes") or payload.get("prior_bytes"),
                        "restorable": mind.blobs.exists(digest)})
        out.sort(key=lambda r: r["seq"], reverse=True)
        return out[:limit]

    def file_attach(*, path: str, work_id: str, actor: str = "supervisor",
                    as_name: str | None = None, operation_id: str | None = None
                    ) -> dict[str, Any]:
        """Hand a host file to a work item so a neuocyte can work on it.

        Nothing is read that was not named here, and the named file still has
        to be inside a configured root. The content is content-addressed on the
        way in, so what a neuocyte saw is recoverable later by digest -- a
        finding about a file can be checked against the exact bytes that
        produced it.
        """
        fs = _filespace()
        try:
            resolved = fs.resolve_host_path(path)
        except InvalidInput as exc:
            _deny(actor, exc.message, {"path": path[:200], "operation": "attach",
                                       "work_id": work_id})
            raise
        data, truncated = fs.read_bytes(resolved)
        if truncated:
            raise ResourceExhausted(
                "file exceeds the filespace read limit",
                limit=sup.cfg.filespace.max_read_bytes, path=resolved.relpath)

        digest = mind.blobs.put(data)
        sandbox_id = sup.sandbox_for_work(work_id, owner=actor)
        name = as_name or Path(resolved.path).name
        dest = f"work/{Path(name).name}"
        # Bytes, not text. Routing this through str replaces every non-UTF-8
        # byte with U+FFFD, which silently corrupts any binary file *and*
        # leaves a receipt describing the source's digest while the sandbox
        # holds something else -- a claim that the neuocyte saw content it
        # never saw.
        landed = _sandbox_manager().write_bytes(sandbox_id, dest, data)
        if landed["sha256"] != digest:
            raise IntegrityError(
                "the attached file did not land in the sandbox intact",
                path=resolved.relpath, source_sha256=digest,
                sandbox_sha256=landed["sha256"])

        def body(m: Mutation) -> None:
            m.register_blob(digest, len(data), "application/octet-stream",
                            "attachment")
            m.emit(EventKind.FILE_ATTACHED, {
                "root": resolved.root_name, "path": resolved.relpath,
                "host_path": str(resolved.path), "work_id": work_id,
                "sandbox_id": sandbox_id, "sandbox_path": dest,
                "bytes": len(data), "sha256": digest, "actor": actor,
                # Stated separately and checked above: the receipt describes
                # what the neuocyte can actually read, not merely what was read
                # from disk.
                "sandbox_sha256": landed["sha256"]})

        receipt, _ = mind.writer.apply(body, actor=actor,
                                       operation_id=operation_id)
        return {"root": resolved.root_name, "path": resolved.relpath,
                "work_id": work_id, "sandbox_id": sandbox_id,
                "sandbox_path": dest, "bytes": len(data), "sha256": digest,
                "sandbox_sha256": landed["sha256"], "content_verified": True,
                "receipt_id": receipt.receipt_id}

    # ==================================================================
    # Tool execution
    # ==================================================================
    def tool_invoke(*, neuocyte_id: str, work_id: str, fencing_token: int,
                    name: str, arguments: dict[str, Any] | None = None,
                    turn: int = 0, operation_id: str | None = None
                    ) -> dict[str, Any]:
        """Execute one tool call on behalf of a neuocyte.

        The neuocyte parses a request out of generated text and sends it here.
        It does not execute it: this is the only path, and it runs in the
        Harness rather than in the process holding the model's output.

        Three things are established before any handler runs, all from durable
        state rather than from the request:

        * the work item is still leased to this neuocyte with this fencing
          token, so a killed or superseded neuocyte cannot still be running code;
        * the capabilities come from the work row, so asking for the sandbox is
          not a way to be granted it;
        * the sandbox, if any, is resolved from ``work_id``, so a model cannot
          name one.

        Both outcomes are recorded. A refusal is a fact about how the mind
        governed itself, so it gets an event too, not just a return value.
        """
        arguments = arguments or {}
        perms = mind.work.authorise_tool_call(
            work_id=work_id, neuocyte_id=neuocyte_id, fencing_token=fencing_token)

        registry = build_neuocyte_registry(
            sup, work_id=work_id, neuocyte_id=neuocyte_id,
            sandbox_allowed=perms["sandbox_allowed"])
        request = ToolCallRequest(name=name, arguments=arguments, raw="")

        def requested(m: Mutation) -> None:
            m.emit(EventKind.TOOL_REQUESTED, {
                "neuocyte_id": neuocyte_id, "work_id": work_id, "tool": name,
                "arguments": _redact(arguments), "turn": turn,
                "sandbox_allowed": perms["sandbox_allowed"]})

        mind.writer.apply(requested, actor=neuocyte_id, operation_id=operation_id,
                          bump_version=False)

        outcome = registry.execute(request, role="neuocyte",
                                   context={"work_id": work_id,
                                            "neuocyte_id": neuocyte_id})

        def recorded(m: Mutation) -> None:
            if not outcome.accepted:
                m.emit(EventKind.TOOL_REJECTED, {
                    "neuocyte_id": neuocyte_id, "work_id": work_id, "tool": name,
                    "reason": outcome.reason, "turn": turn,
                    "arguments": _redact(arguments)})
            else:
                m.emit(EventKind.TOOL_RESULT, {
                    "neuocyte_id": neuocyte_id, "work_id": work_id, "tool": name,
                    "turn": turn, "error": outcome.error,
                    "duration_seconds": round(outcome.duration_seconds, 4),
                    "result_digest": sha256_hex(
                        repr(outcome.result).encode("utf-8", "replace"))})

        receipt, _ = mind.writer.apply(recorded, actor=neuocyte_id,
                                       operation_id=operation_id,
                                       bump_version=outcome.accepted
                                       and not outcome.error)
        return {**outcome.to_dict(), "receipt_id": receipt.receipt_id,
                "turn": turn,
                "available_tools": registry.names(role="neuocyte")}

    def tool_schemas(*, work_id: str | None = None, role: str = "neuocyte"
                     ) -> dict[str, Any]:
        """What a role may call. Reflects the work item's real permissions."""
        sandbox_allowed = False
        if work_id:
            row = mind.db.conn.execute(
                "SELECT sandbox_allowed FROM work_items WHERE work_id = ?",
                (work_id,)).fetchone()
            sandbox_allowed = bool(row["sandbox_allowed"]) if row else False
        reg = build_neuocyte_registry(sup, work_id=work_id or "", neuocyte_id="",
                                      sandbox_allowed=sandbox_allowed)
        return {"role": role, "sandbox_allowed": sandbox_allowed,
                "tools": reg.schemas(role=role),
                "prompt_block": reg.prompt_block(role=role)}

    def context_report() -> dict[str, Any]:
        return sup.homeostasis.measure().to_dict()

    def context_assess() -> dict[str, Any]:
        return sup.homeostasis.assess()

    def context_rejuvenate(*, role: str, reason: str, mode: str = "trim",
                           operation_id: str | None = None) -> dict[str, Any]:
        """Harness-initiated. Ego and Id cannot call this; they request."""
        return sup.homeostasis.rejuvenate(role=role, reason=reason, mode=mode,
                                          requested_by="supervisor",
                                          operation_id=operation_id)

    def request_rejuvenation(*, role: str, reason: str, requested_by: str = "id",
                             mode: str = "trim", operation_id: str | None = None
                             ) -> dict[str, Any]:
        return sup.homeostasis.request_rejuvenation(
            role=role, reason=reason, requested_by=requested_by, mode=mode,
            operation_id=operation_id)

    def retire_session(*, session_id: str, reason: str,
                       operation_id: str | None = None) -> dict[str, Any]:
        return sup.homeostasis.retire_session(session_id=session_id, reason=reason,
                                              operation_id=operation_id)

    return {
        # blackboard
        "board_post": board_post, "board_read": board_read,
        "board_get_post": board_get_post, "board_thread": board_thread,
        "board_relate": board_relate, "board_set_status": board_set_status,
        "board_independence": board_independence,
        "board_corroboration": board_corroboration, "board_stats": board_stats,
        "board_promote_to_memory": board_promote_to_memory,
        # sandbox
        "sandbox_capabilities": sandbox_capabilities,
        "sandbox_create": sandbox_create, "sandbox_run": sandbox_run,
        "sandbox_write": sandbox_write, "sandbox_read": sandbox_read,
        "sandbox_files": sandbox_files, "sandbox_list": sandbox_list,
        "sandbox_destroy": sandbox_destroy,
        "artifact_propose": artifact_propose, "artifact_promote": artifact_promote,
        "artifact_reject": artifact_reject, "artifact_list": artifact_list,
        # tool execution
        "tool_invoke": tool_invoke, "tool_schemas": tool_schemas,
        # host filesystem
        "file_roots": file_roots, "file_list": file_list,
        "file_read": file_read, "file_write": file_write,
        "file_delete": file_delete, "file_restore": file_restore,
        "file_versions": file_versions, "file_attach": file_attach,
        # homeostasis
        "context_report": context_report, "context_assess": context_assess,
        "context_rejuvenate": context_rejuvenate,
        "request_rejuvenation": request_rejuvenation,
        "retire_session": retire_session,
    }
