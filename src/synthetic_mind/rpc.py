"""Loopback JSON line protocol for the internal control plane.

The supervisor and the inference service each listen on a loopback TCP port.
Ego, Id, workers and the MCP facade are clients. Binding the MCP facade to a
long-lived supervisor -- rather than hosting the mind inside the facade -- is
what makes a client disconnect harmless: the stdio process dies, the mind does
not.

Framing is one JSON object per line, UTF-8. The first line a client sends is
``{"token": "..."}`` carrying the shared secret written to the state directory
at supervisor start; the socket is loopback-only and the token stops another
local user's process from driving the mind.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .errors import MindError

MAX_LINE = 32 * 1024 * 1024


class RpcError(MindError):
    code = "rpc_error"


def read_or_create_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        tok = path.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(tok, encoding="utf-8")
    os.replace(tmp, path)
    return tok


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
class _Handler(socketserver.StreamRequestHandler):
    server: "RpcServer"

    def handle(self) -> None:  # noqa: D102
        authed = False
        while True:
            try:
                line = self.rfile.readline(MAX_LINE)
            except (ConnectionError, OSError):
                return
            if not line:
                return
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send({"ok": False, "error": {"code": "invalid_input",
                                                   "message": f"bad frame: {exc}"}})
                return
            if not authed:
                if not secrets.compare_digest(str(msg.get("token", "")), self.server.token):
                    self._send({"ok": False, "error": {"code": "unauthorized",
                                                       "message": "bad control token"}})
                    return
                authed = True
                self._send({"ok": True, "result": {"service": self.server.service_name}})
                continue
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        handler = self.server.methods.get(method)
        if handler is None:
            self._send({"id": req_id, "ok": False,
                        "error": {"code": "not_found",
                                  "message": f"unknown method {method!r}",
                                  "details": {"known": sorted(self.server.methods)}}})
            return
        try:
            result = handler(**params) if params else handler()
            self._send({"id": req_id, "ok": True, "result": result})
        except MindError as exc:
            self._send({"id": req_id, "ok": False, "error": exc.to_dict()})
        except TypeError as exc:
            self._send({"id": req_id, "ok": False,
                        "error": {"code": "invalid_input", "message": str(exc),
                                  "details": {"method": method}}})
        except Exception as exc:  # noqa: BLE001
            self.server.on_error(method, exc)
            self._send({"id": req_id, "ok": False,
                        "error": {"code": "internal_error",
                                  "message": f"{type(exc).__name__}: {exc}",
                                  "details": {"method": method}}})

    def _send(self, obj: dict[str, Any]) -> None:
        try:
            self.wfile.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (ConnectionError, OSError):
            pass


class RpcServer(socketserver.ThreadingTCPServer):
    """Threaded loopback server.

    Handlers must be internally thread safe. The inference service serialises
    everything behind one lock because llama.cpp contexts are not thread safe;
    the supervisor serialises writes behind the single state writer.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, *, token: str, service_name: str,
                 on_error: Callable[[str, Exception], None] | None = None) -> None:
        self.token = token
        self.service_name = service_name
        self.methods: dict[str, Callable[..., Any]] = {}
        self.on_error = on_error or (lambda method, exc: None)
        super().__init__((host, port), _Handler)

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.methods[name] = fn

    def register_all(self, mapping: dict[str, Callable[..., Any]]) -> None:
        self.methods.update(mapping)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def serve_in_thread(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, name=f"rpc-{self.service_name}",
                             daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class RpcClient:
    """Synchronous client. One connection per thread; not shared concurrently."""

    def __init__(self, host: str, port: int, token: str, *, timeout: float = 600.0,
                 name: str = "client") -> None:
        self.host = host
        self.port = port
        self.token = token
        self.timeout = timeout
        self.name = name
        self._sock: socket.socket | None = None
        self._fh: Any = None
        self._next_id = 0
        self._lock = threading.Lock()

    def connect(self, *, retries: int = 1, delay: float = 0.25) -> dict[str, Any]:
        last: Exception | None = None
        for _ in range(max(1, retries)):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                sock.settimeout(self.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = sock
                self._fh = sock.makefile("rwb")
                self._write({"token": self.token})
                resp = self._read()
                if not resp.get("ok"):
                    raise RpcError("control handshake rejected", **(resp.get("error") or {}))
                return resp["result"]
            except (ConnectionError, OSError) as exc:
                last = exc
                self.close()
                time.sleep(delay)
        raise RpcError(f"cannot reach {self.host}:{self.port}: {last}")

    def close(self) -> None:
        for obj in (self._fh, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except OSError:
                pass
        self._fh = None
        self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _write(self, obj: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))
        self._fh.flush()

    def _read(self) -> dict[str, Any]:
        assert self._fh is not None
        line = self._fh.readline()
        if not line:
            raise RpcError("control connection closed by peer")
        return json.loads(line.decode("utf-8"))

    def call(self, method: str, **params: Any) -> Any:
        with self._lock:
            if self._sock is None:
                self.connect(retries=3)
            self._next_id += 1
            req_id = self._next_id
            self._write({"id": req_id, "method": method, "params": params})
            resp = self._read()
            if not resp.get("ok"):
                err = resp.get("error") or {}
                raise RpcError(err.get("message", "rpc failed"),
                               remote_code=err.get("code"), **(err.get("details") or {}))
            return resp.get("result")

    def __enter__(self) -> "RpcClient":
        self.connect(retries=3)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def wait_for_port(host: str, port: int, *, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.1)
    return False
