"""Check a model's arguments against the verb's real annotations, before dispatch.

A role's tool call used to reach the verb with whatever the model wrote. Live,
Ego passed `evidence` as a string where a list of objects was wanted, and the
Harness answered with `AttributeError: 'str' object has no attribute 'get'`
-- an internal failure, told to the model as the reason. The call did no
harm, but only because the crash happened before anything was written.

Checked against the same annotations the declaration renders its kinds from,
so what a role is told and what it is held to are one thing. An annotation
this cannot read is let through rather than blocked: refusing a correct call
because the checker was unsure would be worse than the crash it prevents.
"""

from __future__ import annotations

import inspect
from typing import Any


def _options(annotation: str) -> list[str]:
    """Split a union at its top level: 'a | b[c | d]' -> ['a', 'b[c|d]']."""
    out, depth, cur = [], 0, ""
    for ch in annotation.replace(" ", ""):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "|" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [o for o in out if o]


def _inner(option: str) -> str | None:
    if "[" not in option or not option.endswith("]"):
        return None
    return option[option.index("[") + 1:-1]


def _matches(option: str, value: Any) -> bool | None:
    """True or False when the option is understood; None when it is not."""
    base = option.split("[", 1)[0].replace("typing.", "")
    if base == "None":
        return value is None
    if base in ("Any", "object"):
        return True
    if base == "str":
        return isinstance(value, str)
    if base == "bool":
        return isinstance(value, bool)
    if base == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if base == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if base in ("dict", "Mapping"):
        return isinstance(value, dict)
    if base in ("list", "Sequence", "tuple", "Iterable"):
        if not isinstance(value, list):
            return False
        inner = _inner(option)
        if inner:
            first = inner.split(",", 1)[0] if base == "tuple" else inner
            for item in value:
                ok = _matches_any(first, item)
                if ok is False:
                    return False
        return True
    return None


def _matches_any(annotation: str, value: Any) -> bool | None:
    verdicts = [_matches(o, value) for o in _options(annotation)]
    if any(v is True for v in verdicts):
        return True
    if any(v is None for v in verdicts):
        return None
    return False


_NAMES = {"str": "a string", "int": "an integer", "float": "a number",
          "bool": "true or false", "dict": "an object", "Mapping": "an object",
          "list": "a list", "Sequence": "a list", "tuple": "a list",
          "Iterable": "a list", "None": "null"}


_PLURAL = {"str": "strings", "int": "integers", "float": "numbers",
           "bool": "booleans", "dict": "objects", "Mapping": "objects"}


def _describe(annotation: str) -> str:
    parts = []
    for option in _options(annotation):
        base = option.split("[", 1)[0]
        text = _NAMES.get(base, option)
        inner = _inner(option)
        if inner and base in ("list", "Sequence", "tuple", "Iterable"):
            item = inner.split(",", 1)[0].split("[", 1)[0]
            text += " of " + _PLURAL.get(item, item)
        parts.append(text)
    return " or ".join(parts)


def _got(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    return {str: "a string", int: "an integer", float: "a number",
            dict: "an object", list: "a list"}.get(type(value), type(value).__name__)


def argument_problem(handler: Any, arguments: dict[str, Any],
                     *, skip: frozenset[str] = frozenset()) -> str | None:
    """The first argument that plainly cannot be what the verb takes, if any."""
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return None
    for name, value in arguments.items():
        if name in skip or name not in params:
            continue
        annotation = params[name].annotation
        if annotation is inspect.Parameter.empty:
            continue
        text = str(annotation)
        if _matches_any(text, value) is False:
            got = _got(value)
            if isinstance(value, list):
                # The list may be right and one item wrong; say which, or the
                # model is sent to fix the part that was fine.
                for option in _options(text):
                    inner = _inner(option)
                    if inner and option.split("[", 1)[0] in ("list", "Sequence",
                                                              "tuple", "Iterable"):
                        for i, item in enumerate(value):
                            if _matches_any(inner.split(",", 1)[0], item) is False:
                                got = f"a list whose item {i} is {_got(item)}"
                                break
                        break
            return f"argument {name!r} must be {_describe(text)}, got {got}"
    return None
