"""The MCP facade: cognitive verbs, not neuocyte or cache controls.

This process is a thin, disposable stdio adapter. It holds no state and owns no
part of the mind: it forwards to the long-lived supervisor over the loopback
control plane. When the client hangs up, this process dies and the mind keeps
running.

Every response carries ``schema_version``, ``operation_id``, ``status``,
``receipt_id``, ``state_version``, ``result`` and ``limitations``. Long
operations return a durable handle immediately and are polled through
``ego_status``.

Nothing here exposes neuocyte topology, sequence ids, KV handles or snapshot
mechanics. Those are implementation, and a cognitive client has no business
driving them.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import sys
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field

from .config import Config, load_config
from .errors import MindError
from .logging_setup import get_logger, setup_logging
from .rpc import RpcClient, RpcError, read_or_create_token

SCHEMA_VERSION = "1.0.0"

SERVER_INSTRUCTIONS = """Amoeba: a persistent local cognitive system.

Two halves are directly callable. Ego does outward cognition (conversation,
investigation, recall). Id does inward homeostasis (introspection, health,
audit, disagreements, maintenance).

You are a client and a cognitive peer, not a component of this mind. Its state
outlives your connection.

Three invariants worth knowing before you read results:
  * Raw history is evidence, not memory. `ego_recall` searches MAINTAINED
    interpretations with confidence and supporting/opposing evidence. Raw events
    are reached through `id_audit`.
  * `id_audit` resolves an Ego conclusion through the recorded evidence. It does
    not ask Ego to defend itself, so its verdict is independent of Ego's account.
  * The blackboard is communication between neuocytes, not belief. Agreement on it
    only counts as corroboration when the agreeing parties had not read each
    other -- `board_corroboration` tells you which kind you are looking at. Your
    own reads are recorded too.

Check `limitations` on every response. If the backend is simulated, every
response says so explicitly."""


class Facade:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = get_logger("mcp")
        self.token = read_or_create_token(cfg.token_path)
        self.client = RpcClient(cfg.supervisor_host, cfg.supervisor_port, self.token,
                                name="mcp->supervisor")

    def call(self, method: str, **params: Any) -> Any:
        try:
            return self.client.call(method, **params)
        except RpcError as exc:
            # Reconnect once: the supervisor may have been restarted under us.
            try:
                self.client.close()
                self.client.connect(retries=3)
                return self.client.call(method, **params)
            except Exception:  # noqa: BLE001
                raise exc

    async def call_cancellable(self, method: str, *, idempotency_key: str,
                               **params: Any) -> Any:
        """Run a long call so that MCP cancellation actually stops work.

        The SDK cancels the task awaiting the tool, but a sync handler running
        in a worker thread cannot be interrupted mid-call. So the call is
        awaited here, and when that await is cancelled the facade tells the
        supervisor to cancel the operation: queued work is dropped, a running
        neuocyte is killed, and the in-flight generation is asked to stop
        between tokens.

        The operation is addressed by idempotency key, because a cancellation
        can arrive before the operation id has come back to the client.
        """
        import anyio

        try:
            # abandon_on_cancel=True is load-bearing. The default is False,
            # which defers cancellation until the worker thread finishes -- so
            # the await never raises, the handler below never runs, and the
            # whole automatic-cancellation path is silently inert. Abandoning
            # the thread is the right trade here: the blocking RPC is left to
            # finish in the background while we tell the supervisor to cancel
            # the operation it belongs to.
            return await anyio.to_thread.run_sync(
                lambda: self.call(method, idempotency_key=idempotency_key, **params),
                abandon_on_cancel=True)
        except anyio.get_cancelled_exc_class():
            # Shielded: the cancel itself must survive the cancellation that
            # triggered it, or the work carries on unattended.
            with anyio.CancelScope(shield=True):
                try:
                    await anyio.to_thread.run_sync(
                        lambda: self.call("cancel_operation",
                                          idempotency_key=idempotency_key,
                                          reason="client cancelled the MCP call",
                                          actor="mcp_client"))
                    self.log.info("cancelled operation for key %s", idempotency_key)
                except Exception:  # noqa: BLE001
                    self.log.warning("could not cancel %s after client abort",
                                     idempotency_key, exc_info=True)
            raise

    def envelope(self, raw: Any, *, fallback_status: str = "completed") -> dict[str, Any]:
        """Normalise any supervisor reply into the versioned response shape."""
        if isinstance(raw, dict) and "schema_version" in raw:
            out = dict(raw)
        else:
            out = {
                "schema_version": SCHEMA_VERSION,
                "operation_id": None,
                "status": fallback_status,
                "receipt_id": None,
                "state_version": None,
                "result": raw,
                "limitations": [],
            }
        out.setdefault("operation_id", None)
        out.setdefault("receipt_id", None)
        out.setdefault("state_version", None)
        out.setdefault("status", fallback_status)
        out.setdefault("limitations", [])
        return out

    def error(self, exc: Exception, *, operation_id: str | None = None) -> dict[str, Any]:
        if isinstance(exc, MindError):
            err = exc.to_dict()
        else:
            err = {"code": "internal_error",
                   "message": f"{type(exc).__name__}: {exc}", "details": {}}
        return {
            "schema_version": SCHEMA_VERSION,
            "operation_id": operation_id,
            "status": "failed",
            "receipt_id": None,
            "state_version": None,
            "result": None,
            "error": err,
            "limitations": [f"call failed: {err['code']}"],
        }


def build_server(cfg: Config):  # noqa: C901
    from mcp.server.fastmcp import FastMCP

    facade = Facade(cfg)
    mcp = FastMCP(name="amoeba", instructions=SERVER_INSTRUCTIONS)

    def guarded(fn):
        """Turn an unexpected exception into a well-formed error envelope.

        FastMCP builds each tool's input schema from the callable's signature,
        so the wrapper must expose the wrapped function's signature exactly.
        A bare ``*args, **kwargs`` wrapper would publish a schema demanding
        fields named ``a`` and ``kw``, and every real call would be rejected by
        argument validation before it ever ran.
        """
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*a: Any, **kw: Any) -> dict[str, Any]:
                try:
                    return await fn(*a, **kw)
                except BaseException as exc:
                    # A cancellation must propagate rather than be flattened
                    # into an error envelope: the SDK needs to see the task
                    # actually end, and the facade has already told the
                    # supervisor to stop the work.
                    if type(exc).__name__ == "CancelledError" or isinstance(
                            exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    facade.log.exception("tool %s failed", fn.__name__)
                    return facade.error(exc)
        else:
            @functools.wraps(fn)
            def wrapper(*a: Any, **kw: Any) -> dict[str, Any]:
                try:
                    return fn(*a, **kw)
                except Exception as exc:  # noqa: BLE001
                    facade.log.exception("tool %s failed", fn.__name__)
                    return facade.error(exc)
        # `from __future__ import annotations` makes every annotation a
        # string; eval_str resolves them here so the schema builder does not
        # have to re-resolve names like Literal out of a wrapper's namespace.
        wrapper.__signature__ = inspect.signature(fn, eval_str=True)
        wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}))
        return wrapper

    # ---------------- Ego ----------------
    @mcp.tool(title="Ego: converse")
    @guarded
    async def ego_converse(
        message: Annotated[str, Field(description="What to say to Ego.", max_length=8000)],
        conversation_id: Annotated[str | None, Field(
            description="Group turns into one conversation.")] = None,
        idempotency_key: Annotated[str | None, Field(
            description="Replaying the same key returns the original result instead of "
                        "re-running the turn.")] = None,
        max_tokens: Annotated[int, Field(ge=1, le=2048)] = 384,
    ) -> dict[str, Any]:
        """One conversational turn with Ego, grounded in maintained memory.

        Returns the answer, the memories it drew on, and the id of the
        conclusion recorded for it -- which is what `id_audit` can later
        examine.

        Cancellable: aborting this call stops the generation between tokens
        rather than letting it run to its budget.
        """
        key = idempotency_key or f"mcp-converse-{uuid.uuid4()}"
        return facade.envelope(await facade.call_cancellable(
            "ego_converse", idempotency_key=key, message=message,
            conversation_id=conversation_id, max_tokens=max_tokens,
        ))

    @mcp.tool(title="Ego: investigate")
    @guarded
    def ego_investigate(
        question: Annotated[str, Field(description="The question to investigate.",
                                       max_length=4000)],
        constraints: Annotated[str, Field(
            description="Scope limits Ego should respect.", max_length=2000)] = "",
        budget_tokens: Annotated[int | None, Field(
            ge=1, le=8192,
            description="Requested token budget. The arbiter may reduce it.")] = None,
        idempotency_key: Annotated[str | None, Field()] = None,
    ) -> dict[str, Any]:
        """Start a bounded investigation.

        Returns promptly with a durable operation id and the scope actually
        accepted by the arbiter -- which may be smaller than requested. Poll
        `ego_status` with that operation id for progress and findings.
        """
        return facade.envelope(facade.call(
            "ego_investigate", question=question, constraints=constraints,
            budget_tokens=budget_tokens, idempotency_key=idempotency_key,
        ))

    @mcp.tool(title="Ego: recall")
    @guarded
    def ego_recall(
        query: Annotated[str, Field(description="Substring to match against claims.",
                                    max_length=1000)] = "",
        scope: Annotated[Literal["active", "all", "superseded", "retracted"], Field(
            description="Which maintained memories to search.")] = "active",
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
    ) -> dict[str, Any]:
        """Search MAINTAINED memory with provenance, confidence and versions.

        This is interpretation, not raw history: each item carries supporting
        AND opposing evidence and a supersession chain. Raw events are reached
        through `id_audit`.
        """
        return facade.envelope(facade.call("ego_recall", query=query, scope=scope,
                                           limit=limit))

    @mcp.tool(title="Ego: status")
    @guarded
    def ego_status(
        operation_id: Annotated[str | None, Field(
            description="Poll one durable operation; omit for overall Ego state.")] = None,
    ) -> dict[str, Any]:
        """Progress, outcome, limitations and current state version."""
        return facade.envelope(facade.call("ego_status", operation_id=operation_id))

    # ---------------- Id ----------------
    @mcp.tool(title="Id: introspect")
    @guarded
    async def id_introspect(
        question: Annotated[str, Field(description="What to ask Id about this mind.",
                                       max_length=4000)],
        scope: Annotated[str, Field(max_length=200)] = "all",
        idempotency_key: Annotated[str | None, Field()] = None,
    ) -> dict[str, Any]:
        """Id's observed account of the mind's own operation.

        The response separates what was measured from durable state from what
        Id inferred. Do not read the inferred part as evidence.
        """
        key = idempotency_key or f"mcp-introspect-{uuid.uuid4()}"
        return facade.envelope(await facade.call_cancellable(
            "id_introspect", idempotency_key=key, question=question, scope=scope))

    @mcp.tool(title="Id: health")
    @guarded
    def id_health(
        scope: Annotated[str, Field(max_length=200)] = "all",
    ) -> dict[str, Any]:
        """Measured health, queue and resource status, and capability flags.

        Stays answerable while inference is failing or saturated. Capability
        flags distinguish one resident weight set, serialized execution,
        continuous batching and verified physical overlap -- which are not the
        same thing.
        """
        return facade.envelope(facade.call("id_health", scope=scope))

    @mcp.tool(title="Id: audit")
    @guarded
    async def id_audit(
        conclusion_id: Annotated[str | None, Field(
            description="Conclusion to audit, e.g. from an ego_converse result.")] = None,
        operation_id: Annotated[str | None, Field(
            description="Audit a whole operation instead.")] = None,
        focus: Annotated[str, Field(max_length=1000)] = "",
        idempotency_key: Annotated[str | None, Field()] = None,
    ) -> dict[str, Any]:
        """Audit an Ego conclusion through its recorded evidence.

        Id resolves the claim to its original inputs, model configuration and
        evidence chain without asking Ego to defend itself. A contested verdict
        opens a recorded disagreement rather than overwriting Ego's claim.
        """
        key = idempotency_key or f"mcp-audit-{uuid.uuid4()}"
        return facade.envelope(await facade.call_cancellable(
            "id_audit", idempotency_key=key, conclusion_id=conclusion_id,
            operation_id=operation_id, focus=focus,
        ))

    @mcp.tool(title="Id: disagreements")
    @guarded
    def id_disagreements(
        scope: Annotated[Literal["open", "resolved", "stale", "all"], Field()] = "open",
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Competing claims and the evidence on each side.

        This is not a majority-truth score and neither side is marked correct.
        """
        return facade.envelope(facade.call("id_disagreements", scope=scope, limit=limit))

    @mcp.tool(title="Id: maintenance")
    @guarded
    def id_maintenance(
        objective: Annotated[str, Field(description="Maintenance objective.",
                                        max_length=2000)],
        scope: Annotated[str, Field(max_length=500)] = "",
        budget_tokens: Annotated[int | None, Field(ge=1, le=8192)] = None,
    ) -> dict[str, Any]:
        """Propose bounded maintenance work.

        The arbiter accepts or refuses. A refusal gives the reason: recursion
        depth, hourly rate limit, or queue saturation.
        """
        return facade.envelope(facade.call("id_maintenance", objective=objective,
                                           scope=scope, budget_tokens=budget_tokens))

    @mcp.tool(title="Mind: cancel an operation")
    @guarded
    def mind_cancel(
        operation_id: Annotated[str, Field(
            description="Operation to stop, e.g. from an ego_investigate result.",
            max_length=64)],
        reason: Annotated[str, Field(max_length=500)] = "client cancelled",
    ) -> dict[str, Any]:
        """Stop an operation and everything downstream of it.

        Queued work is cancelled so no neuocyte picks it up, a neuocyte already
        running is killed, and an in-flight generation is asked to stop between
        tokens. Durable state already committed is untouched: cancelling is not
        undoing.

        Cancelling an operation that has already finished is not an error. You
        get `already_terminal: true` and its final status, because a client
        that cancels just as the work lands deserves the truth rather than a
        failure.
        """
        return facade.envelope(facade.call("cancel_operation",
                                           operation_id=operation_id,
                                           reason=reason, actor="mcp_client"))

    # ---------------- cognitive blackboard ----------------
    @mcp.tool(title="Board: read")
    @guarded
    def board_read(
        query: Annotated[str, Field(description="Substring to match in posts.",
                                    max_length=1000)] = "",
        post_types: Annotated[list[str] | None, Field(
            description="finding | question | hypothesis | challenge | request | "
                        "answer | note | retraction")] = None,
        since_seq: Annotated[int | None, Field(
            description="Cursor from a previous read; returns only newer posts.")] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Read the swarm's working discussion.

        This is communication between neuocytes, NOT the mind's beliefs. A post
        is something a neuocyte said; `ego_recall` is what the organism holds to
        be true.

        Reading is recorded against you. If you then post something that agrees
        with what you read, that agreement is marked socially informed rather
        than independent -- which is the point, not a side effect.

        Keep the returned `cursor` and pass it as `since_seq` next time.
        """
        return facade.envelope(facade.call(
            "board_read", reader="mcp_client", query=query, post_types=post_types,
            since_seq=since_seq, limit=limit, record=True))

    @mcp.tool(title="Board: post")
    @guarded
    def board_post(
        body: Annotated[str, Field(description="What you want to contribute.",
                                   max_length=8000)],
        post_type: Annotated[
            Literal["finding", "question", "hypothesis", "challenge", "note"],
            Field(description="What kind of contribution this is.")] = "note",
        title: Annotated[str | None, Field(max_length=200)] = None,
        thread_id: Annotated[str | None, Field(
            description="Reply into an existing thread.")] = None,
        replies_to: Annotated[str | None, Field(
            description="Post id this responds to.")] = None,
        relation: Annotated[
            Literal["reply_to", "challenges", "supports", "refines", "answers"],
            Field(description="How it relates to replies_to.")] = "reply_to",
        confidence: Annotated[float | None, Field(ge=0.0, le=1.0)] = None,
    ) -> dict[str, Any]:
        """Contribute to the swarm's discussion as a cognitive peer.

        You are a participant here, not an authority: posting does not change
        what the mind believes. Promotion of a post into maintained memory is a
        separate act performed by the Harness.

        Whatever you had already read is recorded against this post, so a later
        reader can tell whether you reached this independently.
        """
        relations = ([{"to_post": replies_to, "relation": relation}]
                     if replies_to else [])
        return facade.envelope(facade.call(
            "board_post", author="mcp_client", author_kind="operator",
            post_type=post_type, body=body, title=title, thread_id=thread_id,
            relations=relations, confidence=confidence))

    @mcp.tool(title="Board: how real is this agreement?")
    @guarded
    def board_corroboration(
        post_id: Annotated[str, Field(description="Post to examine.", max_length=64)],
    ) -> dict[str, Any]:
        """Separate independent replication from socially propagated agreement.

        Several neuocytes agreeing means very different things depending on
        whether they had read each other. This splits the support into
        `independent_support` (separate routes to the same answer) and
        `socially_informed_support` (one observation restated), and lists any
        challenges. Only the first is corroboration.
        """
        return facade.envelope(facade.call("board_corroboration", post_id=post_id))

    # ---------------- provenance ----------------
    @mcp.tool(title="Provenance: resolve an operation")
    @guarded
    def mind_provenance(
        operation_id: Annotated[str, Field(description="Operation to resolve.",
                                           max_length=64)],
    ) -> dict[str, Any]:
        """Resolve the input -> work -> inference -> conclusion -> mutation chain.

        Reports the hash-chain verification result and names any referenced
        content that cannot be produced. Missing committed content is an
        integrity failure, not a gap to be glossed over.
        """
        return facade.envelope(facade.call("provenance", operation_id=operation_id))

    return mcp, facade


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amoeba.mcp_api")
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    ap.add_argument("--transport", default="stdio", choices=["stdio"])
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    # stdout belongs to the MCP protocol; diagnostics go to file and stderr only.
    setup_logging(cfg, "mcp", stderr=True)
    mcp, facade = build_server(cfg)
    try:
        facade.client.connect(retries=10, delay=0.5)
    except Exception as exc:  # noqa: BLE001
        facade.log.error("supervisor not reachable at %s:%s (%s); "
                         "tools will return errors until it starts",
                         cfg.supervisor_host, cfg.supervisor_port, exc)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
