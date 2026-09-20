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

import anyio
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
        # The external I/O credential, deliberately not the control token.
        #
        # This facade used to hold `cfg.token_path`, which resolves to operator
        # scope -- the entire method table. Every MCP client could therefore
        # write and delete host files, promote artifacts, run Id maintenance,
        # post to the blackboard and cancel operations. MCP is a cognitive
        # service interface, not a control plane, so it now connects with the
        # narrowest scope in the system and simply cannot name those verbs.
        self.token = read_or_create_token(cfg.scope_token_path("external_io"))
        self.client = RpcClient(cfg.supervisor_host, cfg.supervisor_port, self.token,
                                name="mcp->supervisor")
        self.client_id = "mcp"

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

    # =================================================================
    # The external I/O surface, and nothing else.
    #
    # This block used to expose 23 tools including file write and delete,
    # artifact promotion, Id maintenance, blackboard posting and operation
    # cancellation -- over a facade holding the control token. That made every
    # MCP client an operator.
    #
    # MCP is a cognitive service interface: input in, output out. The verbs
    # below are the entire surface, the facade holds only the external_io
    # credential, and the supervisor put nothing else in that scope's table.
    # There is no tool here to misuse and no method name that would reach one.
    # =================================================================
    @mcp.tool(title="Amoeba: what this interface can do")
    @guarded
    def amoeba_capabilities() -> dict[str, Any]:
        """Describe the external interface.

        Discovery covers this surface only. It does not enumerate Amoeba's
        internals, and there is no verb it could name that this adapter holds
        a credential for.
        """
        return facade.envelope(facade.call("io_capabilities",
                                           client_id=facade.client_id))

    @mcp.tool(title="Amoeba: ask")
    @guarded
    async def amoeba_ask(
        text: Annotated[str, Field(description="What you want Amoeba to think "
                                               "about.", max_length=32000)],
        kind: Annotated[Literal["converse", "investigate"],
                        Field(description="converse for dialogue, investigate "
                                          "for a bounded question.")] = "converse",
        conversation_id: Annotated[str, Field(description="Continue an earlier "
                                                          "exchange.",
                                              max_length=64)] = "",
        wait_seconds: Annotated[float, Field(description="How long to wait for "
                                                         "the answer.",
                                             ge=0, le=300)] = 60.0,
    ) -> dict[str, Any]:
        """Submit input and wait for Amoeba's answer.

        The input may cause a great deal: Ego may request workers, neuocytes
        may run tools, the blackboard may fill with findings, artifacts may be
        proposed, maintained cognition may change. All of that is Amoeba acting
        on its **own** authority in response to what you said -- none of those
        verbs are callable from here, and none of them become callable because
        input caused them.

        If you want something stopped, say so in the text. Amoeba decides what
        follows.
        """
        submitted = facade.call("io_submit", client_id=facade.client_id,
                                surface="mcp", text=text, kind=kind,
                                conversation_id=conversation_id or None)
        if wait_seconds <= 0:
            return facade.envelope(submitted)
        out = await anyio.to_thread.run_sync(
            functools.partial(facade.call, "io_await",
                              client_id=facade.client_id,
                              interaction_id=submitted["interaction_id"],
                              timeout_seconds=wait_seconds),
            abandon_on_cancel=True)
        return facade.envelope({**out,
                                "interaction_id": submitted["interaction_id"]})

    @mcp.tool(title="Amoeba: submit without waiting")
    @guarded
    def amoeba_submit(
        text: Annotated[str, Field(description="Input for Amoeba.",
                                   max_length=32000)],
        kind: Annotated[Literal["converse", "investigate"],
                        Field(description="Interaction kind.")] = "converse",
        conversation_id: Annotated[str, Field(description="Continue an earlier "
                                                          "exchange.",
                                              max_length=64)] = "",
    ) -> dict[str, Any]:
        """Hand Amoeba something to think about and return immediately."""
        return facade.envelope(facade.call(
            "io_submit", client_id=facade.client_id, surface="mcp", text=text,
            kind=kind, conversation_id=conversation_id or None))

    @mcp.tool(title="Amoeba: status of your interaction")
    @guarded
    def amoeba_status(
        interaction_id: Annotated[str, Field(description="One of your own "
                                                         "interactions.",
                                             max_length=64)],
    ) -> dict[str, Any]:
        """Progress on something you submitted. Yours only."""
        return facade.envelope(facade.call("io_status",
                                           client_id=facade.client_id,
                                           interaction_id=interaction_id))

    @mcp.tool(title="Amoeba: collect your output")
    @guarded
    def amoeba_output(
        interaction_id: Annotated[str, Field(description="One of your own "
                                                         "interactions.",
                                             max_length=64)],
    ) -> dict[str, Any]:
        """The answer, plus any results deliberately surfaced to you."""
        return facade.envelope(facade.call("io_output",
                                           client_id=facade.client_id,
                                           interaction_id=interaction_id))

    @mcp.tool(title="Amoeba: your interactions")
    @guarded
    def amoeba_list(
        limit: Annotated[int, Field(description="How many.", ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Your own interactions. There is no parameter for anyone else's."""
        return facade.envelope(facade.call("io_list",
                                           client_id=facade.client_id,
                                           limit=limit))

    @mcp.tool(title="Amoeba: attach input bytes")
    @guarded
    def amoeba_attach(
        filename: Annotated[str, Field(description="A label, not a path.",
                                       max_length=255)],
        content_base64: Annotated[str, Field(description="Base64 of the bytes.")],
        media_type: Annotated[str, Field(description="Optional media type.",
                                         max_length=128)] = "",
    ) -> dict[str, Any]:
        """Send bytes as admitted input.

        Content-addressed with exact-byte provenance, exactly as an
        operator-attached file is. No host path is accepted and no filespace is
        written; whether this is ever materialised into a compute sandbox is a
        Harness decision that happens later, if at all.
        """
        return facade.envelope(facade.call(
            "io_attach_input", client_id=facade.client_id, filename=filename,
            content_base64=content_base64, media_type=media_type or None))

    @mcp.tool(title="Amoeba: fetch a surfaced result")
    @guarded
    def amoeba_result(
        result_id: Annotated[str, Field(description="A result surfaced to you.",
                                        max_length=64)],
    ) -> dict[str, Any]:
        """Fetch a result that was deliberately surfaced to you.

        Knowing an artifact id or a digest is not authority to fetch anything;
        the only route out is through a result that Amoeba decided belongs in
        your answer.
        """
        return facade.envelope(facade.call("io_result",
                                           client_id=facade.client_id,
                                           result_id=result_id))


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
