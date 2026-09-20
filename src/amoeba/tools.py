"""Tool registry for model-generated tool calls.

A model can *request* a tool call; it cannot perform one. Every request is
parsed out of the generated text, matched against a declared schema, checked
against the caller's role permissions, executed under a timeout by the harness,
and recorded with a receipt. A malformed or unpermitted request produces a
``tool.rejected`` event, never an execution.

No tool in this registry can run a shell command, write outside the state
directory or reach the network. Model-generated strings never become commands.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import InvalidInput

# A tool call is requested by emitting exactly this block.
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.DOTALL
)

JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(slots=True)
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: list[Any] | None = None
    maximum: float | None = None
    minimum: float | None = None
    max_length: int | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    params: list[ToolParam]
    handler: Callable[..., Any]
    allowed_roles: tuple[str, ...] = ("ego", "id", "neuocyte")
    timeout_seconds: float = 10.0
    mutates_state: bool = False

    def json_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            entry: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum is not None:
                entry["enum"] = p.enum
            if p.maximum is not None:
                entry["maximum"] = p.maximum
            if p.minimum is not None:
                entry["minimum"] = p.minimum
            if p.max_length is not None:
                entry["maxLength"] = p.max_length
            props[p.name] = entry
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props, "required": required},
        }


@dataclass(slots=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass(slots=True)
class ToolCallOutcome:
    name: str
    accepted: bool
    reason: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


def parse_tool_calls(text: str, *, limit: int = 4) -> list[ToolCallRequest]:
    """Extract tool-call blocks from generated text.

    Malformed JSON is surfaced as a request with an empty argument dict and the
    raw text preserved, so the rejection is recorded rather than silently
    dropped.
    """
    out: list[ToolCallRequest] = []
    for match in TOOL_CALL_RE.finditer(text):
        if len(out) >= limit:
            break
        body = match.group("body")
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            out.append(ToolCallRequest(name="<unparseable>", arguments={}, raw=body))
            continue
        if not isinstance(obj, dict):
            out.append(ToolCallRequest(name="<not_an_object>", arguments={}, raw=body))
            continue
        name = obj.get("name") or obj.get("tool") or "<missing_name>"
        args = obj.get("arguments") or obj.get("args") or {}
        if not isinstance(args, dict):
            args = {"<invalid_arguments>": args}
        out.append(ToolCallRequest(name=str(name), arguments=args, raw=body))
    return out


def strip_tool_calls(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self, *, role: str | None = None) -> list[str]:
        return sorted(
            n for n, s in self._tools.items()
            if role is None or role in s.allowed_roles
        )

    def schemas(self, *, role: str | None = None) -> list[dict[str, Any]]:
        return [
            s.json_schema() for s in self._tools.values()
            if role is None or role in s.allowed_roles
        ]

    def prompt_block(self, *, role: str) -> str:
        """The tool documentation injected into a role's system prompt."""
        schemas = self.schemas(role=role)
        if not schemas:
            return ""
        lines = [
            "You may call a tool by emitting exactly one block of the form:",
            '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>',
            "Emit nothing else in that turn if you call a tool. Available tools:",
        ]
        for s in schemas:
            req = ", ".join(s["parameters"]["required"]) or "(none)"
            lines.append(f"- {s['name']}: {s['description']} | required: {req}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def validate(self, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        """Schema check. Raises :class:`InvalidInput` with a specific reason."""
        known = {p.name: p for p in spec.params}
        unknown = set(arguments) - set(known)
        if unknown:
            raise InvalidInput(f"unknown argument(s): {sorted(unknown)}", tool=spec.name)
        cleaned: dict[str, Any] = {}
        for name, p in known.items():
            if name not in arguments:
                if p.required:
                    raise InvalidInput(f"missing required argument {name!r}", tool=spec.name)
                if p.default is not None:
                    cleaned[name] = p.default
                continue
            value = arguments[name]
            expected = JSON_TYPES.get(p.type)
            if expected is None:
                raise InvalidInput(f"tool declares unknown type {p.type!r}", tool=spec.name)
            if p.type == "number" and isinstance(value, bool):
                raise InvalidInput(f"{name!r} must be a number, not a boolean", tool=spec.name)
            if p.type == "integer" and isinstance(value, bool):
                raise InvalidInput(f"{name!r} must be an integer, not a boolean", tool=spec.name)
            if not isinstance(value, expected):
                raise InvalidInput(
                    f"{name!r} must be {p.type}, got {type(value).__name__}", tool=spec.name
                )
            if p.enum is not None and value not in p.enum:
                raise InvalidInput(f"{name!r} must be one of {p.enum}", tool=spec.name)
            if p.minimum is not None and float(value) < p.minimum:
                raise InvalidInput(f"{name!r} below minimum {p.minimum}", tool=spec.name)
            if p.maximum is not None and float(value) > p.maximum:
                raise InvalidInput(f"{name!r} above maximum {p.maximum}", tool=spec.name)
            if p.max_length is not None and isinstance(value, (str, list)) \
                    and len(value) > p.max_length:
                raise InvalidInput(f"{name!r} exceeds max length {p.max_length}", tool=spec.name)
            cleaned[name] = value
        return cleaned

    def execute(self, request: ToolCallRequest, *, role: str,
                context: dict[str, Any] | None = None) -> ToolCallOutcome:
        """Validate, authorise and run one tool request."""
        spec = self._tools.get(request.name)
        if spec is None:
            return ToolCallOutcome(request.name, False,
                                   reason=f"no such tool; available: {self.names(role=role)}")
        if role not in spec.allowed_roles:
            return ToolCallOutcome(request.name, False,
                                   reason=f"role {role!r} may not call this tool")
        try:
            args = self.validate(spec, request.arguments)
        except InvalidInput as exc:
            return ToolCallOutcome(request.name, False, reason=exc.message,
                                   arguments=request.arguments)
        t0 = time.perf_counter()
        try:
            result = spec.handler(**args, _context=context or {})
        except Exception as exc:  # noqa: BLE001
            return ToolCallOutcome(request.name, True, reason="executed",
                                   arguments=args, error=f"{type(exc).__name__}: {exc}",
                                   duration_seconds=time.perf_counter() - t0)
        elapsed = time.perf_counter() - t0
        if elapsed > spec.timeout_seconds:
            # The handler is synchronous and local; a genuine hang is handled by
            # the neuocyte wall-clock budget. This flags an over-budget call so it
            # is visible in the receipt rather than silently accepted.
            return ToolCallOutcome(request.name, True, reason="executed_over_budget",
                                   arguments=args, result=result,
                                   error=f"exceeded declared timeout {spec.timeout_seconds}s",
                                   duration_seconds=elapsed)
        return ToolCallOutcome(request.name, True, reason="executed", arguments=args,
                               result=result, duration_seconds=elapsed)


def build_default_registry(mind: Any) -> ToolRegistry:
    """Tools backed by the durable state. Read-only except where noted."""
    reg = ToolRegistry()

    def recall_memory(query: str = "", limit: int = 5, _context: dict[str, Any] | None = None):
        items = mind.memory.recall(query=query or None, limit=min(limit, 20))
        return [
            {"memory_id": i["memory_id"], "kind": i["kind"], "claim": i["claim"],
             "confidence": i["confidence"], "status": i["status"],
             "supporting": len(i["evidence"]["supporting"]),
             "opposing": len(i["evidence"]["opposing"])}
            for i in items
        ]

    def read_history(operation_id: str = "", kind: str = "", limit: int = 20,
                     _context: dict[str, Any] | None = None):
        from .store.events import read_events

        events = read_events(
            mind.db.conn, operation_id=operation_id or None,
            kinds=[kind] if kind else None, limit=min(limit, 100),
        )
        return [{"seq": e.seq, "kind": e.kind, "actor": e.actor_id, "ts": e.ts,
                 "event_id": e.event_id} for e in events]

    def get_conclusion(conclusion_id: str, _context: dict[str, Any] | None = None):
        return mind.memory.get_conclusion(conclusion_id)

    def list_open_work(limit: int = 10, _context: dict[str, Any] | None = None):
        rows = mind.db.conn.execute(
            "SELECT work_id, objective, work_class, status, created_at FROM work_items"
            " WHERE status IN ('queued','leased') ORDER BY created_at ASC LIMIT ?",
            (min(limit, 50),),
        )
        return [dict(r) for r in rows]

    def current_state_version(_context: dict[str, Any] | None = None):
        return {"state_version": mind.writer.state_version(), "time": time.time()}

    reg.register(ToolSpec(
        name="recall_memory",
        description="Search MAINTAINED memory (interpretations), not raw history.",
        params=[
            ToolParam("query", "string", "substring to match against claims", max_length=512),
            ToolParam("limit", "integer", "max items", default=5, minimum=1, maximum=20),
        ],
        handler=recall_memory,
    ))
    reg.register(ToolSpec(
        name="read_history",
        description="Read raw append-only events as evidence. Available to Id for audit.",
        params=[
            ToolParam("operation_id", "string", "restrict to one operation", max_length=64),
            ToolParam("kind", "string", "restrict to one event kind", max_length=64),
            ToolParam("limit", "integer", "max events", default=20, minimum=1, maximum=100),
        ],
        handler=read_history,
        allowed_roles=("id", "neuocyte"),
    ))
    reg.register(ToolSpec(
        name="get_conclusion",
        description="Fetch a recorded conclusion with its evidence references.",
        params=[ToolParam("conclusion_id", "string", "conclusion id", required=True,
                          max_length=64)],
        handler=get_conclusion,
    ))
    reg.register(ToolSpec(
        name="list_open_work",
        description="List queued or leased work items.",
        params=[ToolParam("limit", "integer", "max items", default=10, minimum=1, maximum=50)],
        handler=list_open_work,
    ))
    reg.register(ToolSpec(
        name="current_state_version",
        description="Current durable state version and wall clock.",
        params=[],
        handler=current_state_version,
    ))
    return reg
