"""HTTP front door: one listener, two authority surfaces that never merge.

Two kinds of caller reach this process and they are not equivalent:

* **External I/O clients** (`/rpc`, `/events`) -- MCP servers, ordinary
  software. They submit input and collect output.
* **The Operator** (`/operator/*`, `/`) -- the trusted human at the console.

They share a socket. They do not share a method table, a credential, or a
connection to the Harness. Each surface holds its own RPC client bound to its
own scope token, so "which verbs exist" is decided when the process starts, by
which credential each adapter was given -- not per request by inspecting a role
field.

That is the whole design. A dispatcher that looked up a verb in one namespace
and *then* checked whether the caller was allowed would be one forgotten check
away from handing an API client the Harness. Here there is nothing to forget:
the external adapter's connection cannot name an operator verb, because the
supervisor never put one in its table.

## Localhost is not authentication

Binding to loopback keeps other machines out. It does nothing about other
processes on this machine, or a web page in the user's browser making requests
to `127.0.0.1`. So:

* every request needs a credential, loopback or not;
* browser-shaped requests are checked for `Origin`, and a cross-origin one is
  refused before dispatch;
* operator mutations require a session token in a header -- never a cookie
  alone, because a cookie is exactly what a hostile page can make the browser
  send for you.
"""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import parse_qs, urlparse

from .errors import MindError
from .ids import new_id
from .logging_setup import get_logger
from .rpc import RpcClient, read_or_create_token

if TYPE_CHECKING:
    from .config import Config

PROTOCOL_VERSION = "1.0.0"
MAX_BODY_BYTES = 12 * 1024 * 1024


class BodyTooLarge(ValueError):
    """A request body past the limit. Distinct, because "too large" and
    "not JSON" are different things to be told, and the client that sent
    twelve megabytes of valid JSON was told the second one."""

    def __init__(self, length: int) -> None:
        super().__init__("request body too large")
        self.length = length
JSONRPC = "2.0"

# JSON-RPC 2.0 reserved codes, plus ours.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32000
FORBIDDEN_ORIGIN = -32001


class EventHub:
    """Fan-out of client-visible interaction events.

    Deliberately *not* a window onto the event log. A subscriber sees progress
    on its own interactions and nothing else -- publishing the organism's
    history because a stream happens to exist would turn a convenience into a
    disclosure channel.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[queue.Queue]] = {}

    def subscribe(self, client_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._subs.setdefault(client_id, []).append(q)
        return q

    def unsubscribe(self, client_id: str, q: queue.Queue) -> None:
        with self._lock:
            subs = self._subs.get(client_id) or []
            if q in subs:
                subs.remove(q)
            if not subs:
                self._subs.pop(client_id, None)

    def publish(self, client_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            for q in list(self._subs.get(client_id) or []):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass


class Surface:
    """One authority surface: a credential, a scope, and its own connection."""

    def __init__(self, cfg: "Config", *, scope: str, name: str) -> None:
        self.name = name
        self.scope = scope
        self.log = get_logger(f"http.{name}")
        token_path = (cfg.token_path if scope == "operator"
                      else cfg.scope_token_path(scope))
        self._token = read_or_create_token(token_path)
        self._cfg = cfg
        self._lock = threading.RLock()
        self._client: RpcClient | None = None

    def call(self, method: str, **params: Any) -> Any:
        with self._lock:
            if self._client is None:
                self._client = RpcClient(
                    self._cfg.supervisor_host, self._cfg.supervisor_port,
                    self._token, name=f"http-{self.name}->supervisor",
                    timeout=300)
                self._client.connect(retries=5, delay=0.3)
            try:
                return self._client.call(method, **params)
            except Exception:
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    pass
                self._client = None
                raise


class ApiServer:
    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.log = get_logger("http")
        # Two adapters, two credentials, two tables. The external one cannot
        # name an operator verb because its connection was never given one.
        self.external = Surface(cfg, scope="external_io", name="external")
        self.operator = Surface(cfg, scope="operator", name="operator")
        self.events = EventHub()
        self.clients = ClientRegistry(cfg)
        self.operator_session = secrets.token_urlsafe(32)
        # Written where the operator can find it, beside the other
        # credentials. A console nobody can log into is not a console, and
        # scraping it out of a log file is worse than a file with the same
        # protection as every other secret here.
        session_path = cfg.state_dir / "operator.session"
        session_path.write_text(self.operator_session, encoding="utf-8")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> tuple[str, int]:
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(
            (self.cfg.api_host, self.cfg.api_port), handler)
        self._server.daemon_threads = True
        host, port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="http-api", daemon=True)
        self._thread.start()
        self.log.info("http api on http://%s:%s (external /rpc, operator "
                      "/operator/rpc)", host, port)
        return str(host), int(port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


class ClientRegistry:
    """API keys, and the identity each one carries.

    A key maps to a `client_id`. The client never sends its own identity: it
    sends a secret, and the identity is what that secret *is*. That is what
    makes "my interactions" a fact rather than a filter.
    """

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self._keys: dict[str, str] = {}
        self._load()

    def _path(self):
        return self.cfg.state_dir / "api_clients.json"

    def _load(self) -> None:
        path = self._path()
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                self._keys = {str(k): str(v) for k, v in raw.items()}
                return
            except Exception:  # noqa: BLE001
                pass
        # A default client so the surface is usable out of the box. It is a
        # random secret, not an absence of one: "we only listen on loopback"
        # is not authentication.
        key = secrets.token_urlsafe(32)
        self._keys = {key: "default"}
        self._save()

    def _save(self) -> None:
        self._path().write_text(json.dumps(self._keys, indent=2),
                                encoding="utf-8")

    def identify(self, presented: str) -> str | None:
        with self._lock:
            for key, client_id in self._keys.items():
                if secrets.compare_digest(presented, key):
                    return client_id
        return None

    def issue(self, client_id: str) -> str:
        with self._lock:
            key = secrets.token_urlsafe(32)
            self._keys[key] = client_id
            self._save()
            return key

    def any_key(self) -> str:
        with self._lock:
            return next(iter(self._keys))


def _make_handler(api: ApiServer) -> type[BaseHTTPRequestHandler]:  # noqa: C901
    from .io_api import EXTERNAL_VERBS
    from .operator_api import OPERATOR_VERBS

    external_allowed = frozenset(EXTERNAL_VERBS)
    operator_allowed = frozenset(OPERATOR_VERBS)

    class Handler(BaseHTTPRequestHandler):
        server_version = "Amoeba/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
            api.log.debug("%s - %s", self.address_string(), fmt % args)

        # -- helpers ---------------------------------------------------
        def handle_one_request(self) -> None:
            # One handler instance serves every request on a keep-alive
            # connection, so per-request state has to be cleared here. Leaving
            # `_body_consumed` set from the previous request meant the second
            # refusal on a connection skipped its drain and desynchronised the
            # stream -- the same failure the drain exists to prevent, one
            # request later.
            self._body_consumed = False
            super().handle_one_request()

        def _drain(self) -> None:
            """Consume any request body the handler did not read.

            A keep-alive connection carries the next request immediately after
            this one's body. Answering without reading that body leaves it in
            the socket, and the next request is parsed starting from the
            middle of it -- so the failure surfaces on the *following*
            request, which is invariably an innocent one.

            An oversized body is not drained. Reading an attacker-chosen
            number of bytes in order to discard them is exactly what the size
            limit exists to prevent, so the connection is closed instead.
            """
            if getattr(self, "_body_consumed", False):
                return
            self._body_consumed = True
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                return
            try:
                self.rfile.read(length)
            except Exception:  # noqa: BLE001
                self.close_connection = True

        def handle_expect_100(self) -> bool:  # noqa: N802
            """Refuse an oversized body before the client sends it.

            A body past the limit cannot be answered readably once it is in
            flight: the refusal is written while the client is still writing,
            and the client sees the connection abort rather than the reason.
            A client that asks first is told first -- which is what
            `Expect: 100-continue` is for.
            """
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > MAX_BODY_BYTES:
                self._body_consumed = True
                self.close_connection = True
                self._send(413, {"error": "request body too large",
                                 "bytes": length, "limit": MAX_BODY_BYTES})
                return False
            return super().handle_expect_100()

        def _send(self, code: int, payload: Any, *, ctype: str = "application/json"
                  ) -> None:
            # Before the response, not after: the body has to leave the socket
            # while the connection is still ours to keep in step.
            self._drain()
            data = (payload if isinstance(payload, bytes)
                    else json.dumps(payload, default=str).encode("utf-8"))
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                # Not read at all, so the connection cannot be reused: the
                # bytes are still in the socket and draining them is the very
                # thing this limit refuses to do.
                self.close_connection = True
                self._body_consumed = True
                raise BodyTooLarge(length)
            self._body_consumed = True
            return json.loads(self.rfile.read(length) or b"null")

        def _origin_ok(self) -> bool:
            """Refuse cross-origin browser requests before anything else.

            A page on another origin can make the browser POST to
            127.0.0.1 with whatever body it likes. Requiring a credential
            handles most of it; refusing a foreign Origin closes the case
            where a credential is somehow already present.
            """
            origin = self.headers.get("Origin")
            if not origin:
                return True          # not a browser-initiated request
            host, port = api.cfg.api_host, api.cfg.api_port
            allowed = {f"http://{host}:{port}", f"http://127.0.0.1:{port}",
                       f"http://localhost:{port}"}
            return origin in allowed

        def _external_client(self) -> str | None:
            auth = self.headers.get("Authorization") or ""
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            token = token or (self.headers.get("X-Amoeba-Key") or "").strip()
            return api.clients.identify(token) if token else None

        def _is_operator(self) -> bool:
            presented = (self.headers.get("X-Amoeba-Operator") or "").strip()
            return bool(presented) and secrets.compare_digest(
                presented, api.operator_session)

        def _rpc_error(self, code: int, message: str, req_id: Any = None,
                       http: int = 200, **data: Any) -> None:
            self._send(http, {"jsonrpc": JSONRPC, "id": req_id,
                              "error": {"code": code, "message": message,
                                        "data": data or None}})

        # -- routing ---------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not self._origin_ok():
                return self._send(403, {"error": "cross-origin request refused"})
            if path == "/health":
                return self._send(200, {"status": "ok",
                                        "protocol_version": PROTOCOL_VERSION})
            if path == "/rpc":
                return self._send(200, {
                    "protocol": "json-rpc-2.0", "endpoint": "/rpc",
                    "protocol_version": PROTOCOL_VERSION,
                    "methods": sorted(external_allowed),
                    "note": ("the external I/O surface; this adapter holds no "
                             "credential for anything else")})
            if path == "/events":
                return self._events()
            if path == "/" or path.startswith("/static"):
                return self._dashboard()
            return self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not self._origin_ok():
                return self._send(403, {"error": "cross-origin request refused"})
            if path == "/rpc":
                return self._external_rpc()
            if path == "/operator/rpc":
                return self._operator_rpc()
            return self._send(404, {"error": "not found"})

        # -- external surface ------------------------------------------
        def _external_rpc(self) -> None:
            client_id = self._external_client()
            if client_id is None:
                return self._rpc_error(UNAUTHORIZED, "a valid API key is required",
                                       http=401)
            try:
                req = self._body()
            except BodyTooLarge as big:
                return self._rpc_error(INVALID_PARAMS, "request body too large",
                                       http=413, bytes=big.length,
                                       limit=MAX_BODY_BYTES)
            except Exception:  # noqa: BLE001
                return self._rpc_error(PARSE_ERROR, "invalid JSON")
            if not isinstance(req, dict) or req.get("jsonrpc") != JSONRPC:
                return self._rpc_error(INVALID_REQUEST, "expected JSON-RPC 2.0")
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                return self._rpc_error(INVALID_PARAMS,
                                       "params must be an object", req_id)
            if method not in external_allowed:
                # Absent, not refused. This adapter holds no credential that
                # names an operator, Ego, Id or Harness verb, so guessing one
                # reaches nothing regardless of what the caller claims to be.
                return self._rpc_error(
                    METHOD_NOT_FOUND, f"unknown method {method!r}", req_id,
                    surface="external_io")

            # Identity is bound here, from the credential. Anything the caller
            # sent under these names is discarded rather than honoured.
            for forged in ("client_id", "actor", "role", "caller", "scope",
                           "from_role", "origin_actor", "operator"):
                params.pop(forged, None)
            try:
                result = api.external.call(method, client_id=client_id, **params)
            except MindError as exc:
                return self._rpc_error(INVALID_PARAMS, exc.message, req_id,
                                       **(exc.details or {}))
            except Exception as exc:  # noqa: BLE001
                api.log.exception("external rpc %s failed", method)
                return self._rpc_error(INTERNAL_ERROR, type(exc).__name__, req_id)

            if method == "io_submit" and isinstance(result, dict):
                api.events.publish(client_id, {
                    "event": "interaction.accepted",
                    "interaction_id": result.get("interaction_id")})
            self._send(200, {"jsonrpc": JSONRPC, "id": req_id, "result": result})

        def _events(self) -> None:
            """SSE of this client's own interaction progress."""
            client_id = self._external_client()
            if client_id is None:
                return self._send(401, {"error": "a valid API key is required"})
            params = parse_qs(urlparse(self.path).query)
            limit = float((params.get("seconds") or ["30"])[0])
            q = api.events.subscribe(client_id)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            deadline = time.monotonic() + max(1.0, min(limit, 600.0))
            try:
                while time.monotonic() < deadline:
                    try:
                        event = q.get(timeout=1.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(event, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                api.events.unsubscribe(client_id, q)

        # -- operator surface ------------------------------------------
        def _operator_rpc(self) -> None:
            if not self._is_operator():
                return self._rpc_error(
                    UNAUTHORIZED,
                    "operator session required in X-Amoeba-Operator", http=401)
            try:
                req = self._body()
            except BodyTooLarge as big:
                return self._rpc_error(INVALID_PARAMS, "request body too large",
                                       http=413, bytes=big.length,
                                       limit=MAX_BODY_BYTES)
            except Exception:  # noqa: BLE001
                return self._rpc_error(PARSE_ERROR, "invalid JSON")
            req_id = req.get("id") if isinstance(req, dict) else None
            method = req.get("method") if isinstance(req, dict) else None
            params = req.get("params") or {} if isinstance(req, dict) else {}
            if method not in operator_allowed:
                return self._rpc_error(
                    METHOD_NOT_FOUND, f"unknown method {method!r}", req_id,
                    surface="operator")
            try:
                result = api.operator.call(method, **params)
            except MindError as exc:
                return self._rpc_error(INVALID_PARAMS, exc.message, req_id,
                                       **(exc.details or {}))
            except Exception as exc:  # noqa: BLE001
                api.log.exception("operator rpc %s failed", method)
                return self._rpc_error(INTERNAL_ERROR, type(exc).__name__, req_id)
            self._send(200, {"jsonrpc": JSONRPC, "id": req_id, "result": result})

        def _dashboard(self) -> None:
            from .dashboard import DASHBOARD_HTML

            self._send(200, DASHBOARD_HTML.encode("utf-8"),
                       ctype="text/html; charset=utf-8")

    return Handler
