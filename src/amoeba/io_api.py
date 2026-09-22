"""The external I/O surface: input in, output out, and nothing else.

MCP-connected and API-connected systems are **unprivileged cognitive clients**.
They may hand Amoeba something to think about and collect what it produced.
They may not admit work, schedule it, cancel it, post to the blackboard, read
maintained state, promote an artifact, touch a prompt, or reach any role's
internal effectors.

## The distinction this file exists to hold

    external input changing what Amoeba thinks about
        is not
    external control mutating Amoeba's protected state

A client saying *"stop investigating and answer with what you have"* is valid
input. Ego may read it and decide to cancel work, and that cancellation is
Amoeba exercising its own authority over its own state. Exposing
`work.cancel(work_id)` to that client is a different thing entirely, and this
surface does not contain it.

So input here may legitimately cause a great deal: Ego may request workers,
workers may use tools, the blackboard may fill up, artifacts may be proposed,
maintained cognition may change. None of that makes the caller a control-plane
actor, because none of the verbs that did it are reachable from here.

## Identity is the credential

`client_id` comes from the authenticated credential, never from the request.
A client's interactions are its own: "show me mine" is a fact about who is
asking, not a filter that could be widened by asking differently. There is no
parameter anywhere in this module that names another client.

## Knowing an identifier is not authority

An artifact is readable only once it has been deliberately surfaced as a result
of *that client's* interaction. Knowing an artifact id, a blob digest or a work
id buys nothing: there is no route from an identifier to the blob store here.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import TYPE_CHECKING, Any, Sequence

from .errors import (DeadlineExceeded, InvalidInput, NotFound,
                     ResourceExhausted)
from .ids import new_id, sha256_hex
from .store.events import EventKind
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

PROTOCOL_VERSION = "1.0.0"
MAX_INPUT_CHARS = 32_000
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENTS = 8
KINDS = ("converse", "investigate")

# The entire external surface, named once. Discovery advertises exactly this;
# anything absent from it is absent from the adapter, not merely undocumented.
EXTERNAL_VERBS = (
    "io_capabilities", "io_submit", "io_status", "io_await", "io_output",
    "io_attach_input", "io_result", "io_list",
)


def _text(value: str, field: str, *, limit: int = MAX_INPUT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInput(f"{field} must be a non-empty string")
    if len(value) > limit:
        raise ResourceExhausted(f"{field} exceeds the input limit",
                                length=len(value), limit=limit)
    return value.strip()


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None
    running: dict[str, threading.Thread] = {}
    done = threading.Condition()

    def _own(interaction_id: str, client_id: str) -> dict[str, Any]:
        """Fetch an interaction, or behave as though it does not exist.

        Deliberately `NotFound` rather than "forbidden": telling a client that
        an interaction exists but belongs to someone else is itself a
        disclosure, and there is nothing useful it could do with the fact.
        """
        row = mind.db.conn.execute(
            "SELECT * FROM interactions WHERE interaction_id = ? AND client_id = ?",
            (interaction_id, client_id)).fetchone()
        if row is None:
            raise NotFound("no such interaction", interaction_id=interaction_id)
        return dict(row)

    # ==================================================================
    def io_capabilities(*, client_id: str = "") -> dict[str, Any]:
        """What this surface can do. Everything here is I/O.

        Discovery advertises the external surface only. It does not enumerate
        the Harness, and there is no verb it could name that this adapter
        holds a credential for.
        """
        return {
            "protocol_version": PROTOCOL_VERSION,
            "surface": "external_io",
            "verbs": list(EXTERNAL_VERBS),
            "kinds": list(KINDS),
            "limits": {"max_input_chars": MAX_INPUT_CHARS,
                       "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
                       "max_attachments": MAX_ATTACHMENTS},
            "authority": ("input and output only; this surface holds no verb "
                          "that admits work, cancels it, edits state, governs "
                          "artifacts or prompts, or reaches a role's internal "
                          "effectors"),
            "note": ("input may cause Amoeba to do a great deal using its own "
                     "authority; that is cognition responding to input, not "
                     "the caller controlling anything"),
        }

    def io_attach_input(*, filename: str, content_base64: str,
                        client_id: str, media_type: str | None = None,
                        interaction_id: str | None = None) -> dict[str, Any]:
        """Admit bytes from an external client.

        Strictly input. The bytes are content-addressed exactly as an
        operator-attached file is, so what the client sent is recoverable by
        digest -- but no host path is accepted, no Filespace root is touched,
        and materialising this into a compute sandbox remains a Harness act
        that happens later, if at all.
        """
        name = _text(filename, "filename", limit=255)
        if "/" in name or "\\" in name or name.startswith("."):
            raise InvalidInput(
                "attachment names are labels, not paths",
                filename=name,
                hint="send a bare filename; this surface cannot address the "
                     "host filesystem")
        try:
            data = base64.b64decode(content_base64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise InvalidInput("content_base64 is not valid base64") from exc
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise ResourceExhausted("attachment too large", bytes=len(data),
                                    limit=MAX_ATTACHMENT_BYTES)
        if interaction_id:
            _own(interaction_id, client_id)

        digest = mind.blobs.put(data)
        input_id = new_id("inp")

        def body(m: Mutation) -> None:
            m.register_blob(digest, len(data),
                            media_type or "application/octet-stream",
                            "external_input")
            m.sql("INSERT INTO interaction_inputs(input_id, interaction_id,"
                  " client_id, filename, sha256, bytes, media_type, created_at,"
                  " state_version) VALUES (?,?,?,?,?,?,?,?,?)",
                  (input_id, interaction_id, client_id, name, digest, len(data),
                   media_type, time.time(), m.prior_version + 1))
            m.emit(EventKind.INTERACTION_INPUT_ATTACHED, {
                "input_id": input_id, "interaction_id": interaction_id,
                "client_id": client_id, "filename": name, "sha256": digest,
                "bytes": len(data), "source": "external_client",
                "note": "admitted as input; not written to any filespace root"})

        receipt, _ = mind.writer.apply(body, actor=f"client:{client_id}")
        return {"input_id": input_id, "filename": name, "sha256": digest,
                "bytes": len(data), "receipt_id": receipt.receipt_id,
                "note": "admitted as input with exact-byte provenance"}

    def io_submit(*, text: str, client_id: str, surface: str = "api",
                  kind: str = "converse", conversation_id: str | None = None,
                  input_ids: Sequence[str] = ()) -> dict[str, Any]:
        """Give Amoeba something to think about.

        Returns immediately with an interaction id. Cognition runs on its own
        thread using Amoeba's *internal* authority -- Ego may request work,
        neuocytes may run, the blackboard may fill. The caller observes the
        outcome; it never named any of the verbs that produced it.
        """
        if kind not in KINDS:
            raise InvalidInput("unknown interaction kind", kind=kind,
                               allowed=list(KINDS))
        body_text = _text(text, "text")
        if len(input_ids) > MAX_ATTACHMENTS:
            raise ResourceExhausted("too many attachments",
                                    count=len(input_ids), limit=MAX_ATTACHMENTS)
        for input_id in input_ids:
            row = mind.db.conn.execute(
                "SELECT client_id FROM interaction_inputs WHERE input_id = ?",
                (input_id,)).fetchone()
            if row is None or row["client_id"] != client_id:
                raise NotFound("no such input", input_id=input_id)

        interaction_id = new_id("ixn")
        digest = mind.blobs.put(body_text.encode("utf-8"))

        def body(m: Mutation) -> None:
            m.register_blob(digest, len(body_text.encode("utf-8")),
                            "text/plain", "external_input")
            m.sql("INSERT INTO interactions(interaction_id, client_id, surface,"
                  " kind, conversation_id, input_sha256, input_preview, status,"
                  " created_at, state_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (interaction_id, client_id, surface, kind, conversation_id,
                   digest, body_text[:500], "accepted", time.time(),
                   m.prior_version + 1))
            for input_id in input_ids:
                m.sql("UPDATE interaction_inputs SET interaction_id = ?"
                      " WHERE input_id = ? AND client_id = ?",
                      (interaction_id, input_id, client_id))
            m.emit(EventKind.INTERACTION_ACCEPTED, {
                "interaction_id": interaction_id, "client_id": client_id,
                "surface": surface, "kind": kind,
                "input_sha256": digest, "attachments": list(input_ids),
                "note": ("external input admitted; any work that follows is "
                         "Amoeba acting on its own authority")})

        receipt, _ = mind.writer.apply(body, actor=f"client:{client_id}")

        thread = threading.Thread(
            target=_run, args=(interaction_id, client_id, kind, body_text,
                               conversation_id),
            name=f"interaction-{interaction_id}", daemon=True)
        running[interaction_id] = thread
        thread.start()
        return {"interaction_id": interaction_id, "status": "accepted",
                "kind": kind, "input_sha256": digest,
                "receipt_id": receipt.receipt_id}

    def _run(interaction_id: str, client_id: str, kind: str, text: str,
             conversation_id: str | None) -> None:
        """Cognition, on Amoeba's authority rather than the caller's."""
        _set_status(interaction_id, "running")
        try:
            # `submit_wait_seconds` bounds how long a *synchronous caller*
            # blocks. This is not one: the external surface is asynchronous by
            # construction -- `io_submit` returns an id and the client polls.
            # So it waits for the answer to actually exist, across however
            # many continuation turns the thought needs, rather than giving up
            # on the caller's behalf and reporting an empty result.
            sched = sup.cfg.scheduler
            patience = (sched.turn_wall_seconds
                        * (max(1, sched.max_continuations) + 2))
            # What the client sent with this request, read back from the
            # rows `io_submit` already verified they own. Built here rather
            # than passed in, so what Ego is told about is what the database
            # says arrived.
            attachments = [
                {"input_id": r["input_id"], "filename": r["filename"],
                 "media_type": r["media_type"], "bytes": r["bytes"],
                 "sha256": r["sha256"]}
                for r in mind.db.conn.execute(
                    "SELECT input_id, filename, media_type, bytes, sha256"
                    " FROM interaction_inputs WHERE interaction_id = ?"
                    " ORDER BY created_at", (interaction_id,))]
            if kind == "investigate":
                out = sup.methods()["ego_investigate"](
                    question=text, wait_seconds=patience)
            else:
                out = sup.methods()["ego_converse"](
                    message=text, conversation_id=conversation_id,
                    interaction_id=interaction_id, attachments=attachments,
                    wait_seconds=patience)
            result = out.get("result") if isinstance(out, dict) else None
            answer = ""
            if isinstance(result, dict):
                answer = (result.get("answer") or result.get("claim")
                          or result.get("plan") or "")
            # "Complete" has to mean answered. Reporting an empty answer as a
            # completed interaction tells the client, permanently, that
            # nothing was the organism's reply.
            state = result.get("status") if isinstance(result, dict) else None
            # Terminal means Amoeba has said all it is going to about this
            # input. Only "completed" is a finished answer; "incomplete" is
            # everything Ego said before it was stopped, and the client gets
            # both the text and that fact rather than one without the other.
            settled = state in ("completed", "incomplete")
            final = "complete" if state == "completed" else "incomplete"
            if not settled:
                raise DeadlineExceeded(
                    "Ego did not answer within this interaction's patience; "
                    "the input remains queued and will still be processed, "
                    "but this interaction carries no answer",
                    interaction_id=interaction_id,
                    trigger_id=(result or {}).get("trigger_id"))
            payload = {"result": result, "operation_id": out.get("operation_id")
                       if isinstance(out, dict) else None}
            digest = mind.blobs.put_json(payload)

            def body(m: Mutation) -> None:
                m.register_blob(digest, 0, "application/json",
                                "external_output")
                m.sql("UPDATE interactions SET status = ?,"
                      " output_sha256 = ?, output_preview = ?, operation_id = ?,"
                      " completed_at = ? WHERE interaction_id = ?",
                      (final, digest, str(answer)[:1000],
                       payload.get("operation_id"), time.time(),
                       interaction_id))
                m.emit(EventKind.INTERACTION_COMPLETED, {
                    "interaction_id": interaction_id, "client_id": client_id,
                    "output_sha256": digest, "status": final})

            mind.writer.apply(body, actor=f"client:{client_id}")
        except Exception as exc:  # noqa: BLE001
            def failed(m: Mutation) -> None:
                m.sql("UPDATE interactions SET status = 'failed', error = ?,"
                      " completed_at = ? WHERE interaction_id = ?",
                      (f"{type(exc).__name__}: {exc}"[:2000], time.time(),
                       interaction_id))
                m.emit(EventKind.INTERACTION_FAILED, {
                    "interaction_id": interaction_id, "client_id": client_id,
                    "error": type(exc).__name__})

            try:
                mind.writer.apply(failed, actor=f"client:{client_id}")
            except Exception:  # noqa: BLE001
                sup.log.exception("could not record interaction failure")
        finally:
            running.pop(interaction_id, None)
            with done:
                done.notify_all()

    def _set_status(interaction_id: str, status: str) -> None:
        def body(m: Mutation) -> None:
            m.sql("UPDATE interactions SET status = ? WHERE interaction_id = ?",
                  (status, interaction_id))

        try:
            mind.writer.apply(body, actor="supervisor", bump_version=False)
        except Exception:  # noqa: BLE001
            pass

    def io_status(*, interaction_id: str, client_id: str) -> dict[str, Any]:
        row = _own(interaction_id, client_id)
        return {"interaction_id": interaction_id, "status": row["status"],
                "kind": row["kind"], "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "has_output": bool(row["output_sha256"]),
                "error": row["error"]}

    def io_await(*, interaction_id: str, client_id: str,
                 timeout_seconds: float = 30.0) -> dict[str, Any]:
        """Block until the interaction settles, or the timeout expires."""
        deadline = time.monotonic() + max(0.0, min(float(timeout_seconds), 300.0))
        while True:
            row = _own(interaction_id, client_id)
            if row["status"] in ("complete", "incomplete", "failed"):
                return io_output(interaction_id=interaction_id,
                                 client_id=client_id)
            if time.monotonic() >= deadline:
                return {"interaction_id": interaction_id,
                        "status": row["status"], "timed_out": True}
            with done:
                done.wait(0.25)

    def io_output(*, interaction_id: str, client_id: str) -> dict[str, Any]:
        row = _own(interaction_id, client_id)
        payload = None
        if row["output_sha256"] and mind.blobs.exists(row["output_sha256"]):
            payload = mind.blobs.get_json(row["output_sha256"])
        results = [dict(r) for r in mind.db.conn.execute(
            "SELECT result_id, artifact_id, sha256, filename, media_type, bytes"
            " FROM interaction_results WHERE interaction_id = ? AND client_id = ?",
            (interaction_id, client_id))]
        return {"interaction_id": interaction_id, "status": row["status"],
                "kind": row["kind"], "output": payload, "error": row["error"],
                "results": results, "timed_out": False}

    def io_list(*, client_id: str, limit: int = 20) -> dict[str, Any]:
        """This client's own interactions. There is no parameter for anyone else's."""
        rows = [dict(r) for r in mind.db.conn.execute(
            "SELECT interaction_id, kind, status, created_at, completed_at,"
            " input_preview FROM interactions WHERE client_id = ?"
            " ORDER BY created_at DESC LIMIT ?",
            (client_id, max(1, min(int(limit), 100))))]
        return {"interactions": rows, "count": len(rows)}

    def io_result(*, result_id: str, client_id: str) -> dict[str, Any]:
        """Fetch a result that was deliberately surfaced to this client.

        The only route from this surface to stored bytes, and it goes through
        a row that says those bytes were meant to leave. Knowing an artifact id
        or a digest is not authority to fetch anything.
        """
        row = mind.db.conn.execute(
            "SELECT * FROM interaction_results WHERE result_id = ? AND client_id = ?",
            (result_id, client_id)).fetchone()
        if row is None:
            raise NotFound("no such result", result_id=result_id)
        digest = row["sha256"]
        if not mind.blobs.exists(digest):
            raise NotFound("the result content is no longer available",
                           result_id=result_id)
        data = mind.blobs.get(digest)
        return {"result_id": result_id, "artifact_id": row["artifact_id"],
                "filename": row["filename"], "media_type": row["media_type"],
                "bytes": len(data), "sha256": digest,
                "content_base64": base64.b64encode(data).decode("ascii")}

    return {
        "io_capabilities": io_capabilities,
        "io_attach_input": io_attach_input,
        "io_submit": io_submit,
        "io_status": io_status,
        "io_await": io_await,
        "io_output": io_output,
        "io_list": io_list,
        "io_result": io_result,
    }


def surface_result(sup: "Supervisor", *, interaction_id: str, sha256: str,
                   filename: str | None = None, artifact_id: str | None = None,
                   media_type: str | None = None,
                   surfaced_by: str = "ego") -> dict[str, Any]:
    """Deliberately make something fetchable by the client who asked.

    Called from *inside* -- by Ego or the operator deciding this belongs in the
    answer. Not reachable from the external surface, which is the point: an
    external client cannot surface a result to itself.
    """
    mind = sup.mind
    assert mind is not None
    row = mind.db.conn.execute(
        "SELECT client_id FROM interactions WHERE interaction_id = ?",
        (interaction_id,)).fetchone()
    if row is None:
        raise NotFound("no such interaction", interaction_id=interaction_id)
    if not mind.blobs.exists(sha256):
        raise NotFound("no stored content with that digest", sha256=sha256)
    size = len(mind.blobs.get(sha256))
    result_id = new_id("res")

    def body(m: Mutation) -> None:
        m.sql("INSERT INTO interaction_results(result_id, interaction_id,"
              " client_id, artifact_id, sha256, filename, media_type, bytes,"
              " surfaced_by, created_at, state_version)"
              " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (result_id, interaction_id, row["client_id"], artifact_id,
               sha256, filename, media_type, size, surfaced_by, time.time(),
               m.prior_version + 1))
        m.emit(EventKind.INTERACTION_RESULT_SURFACED, {
            "result_id": result_id, "interaction_id": interaction_id,
            "artifact_id": artifact_id, "sha256": sha256,
            "surfaced_by": surfaced_by,
            "note": "deliberately made fetchable by the requesting client"})

    receipt, _ = mind.writer.apply(body, actor=surfaced_by)
    return {"result_id": result_id, "sha256": sha256, "bytes": size,
            "receipt_id": receipt.receipt_id}
