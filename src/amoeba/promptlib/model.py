"""The vocabulary of the cognitive family tree.

Namespaces, lineage references, prompt composition modes, and the model
variables the backend can actually apply. Everything else in ``promptlib``
builds on the rules stated here.

## Namespaces are ancestry

``ego.neuocyte.research`` means *ego* → *ego.neuocyte* → *ego.neuocyte.research*.
Parent first, specialisation afterwards, so the name itself carries the family
tree and a node can never be reparented by renaming.

Only ``ego`` and ``id`` are roots, and only the bootstrap path may establish
them. That is enforced structurally rather than by a flag a caller could set:
the runtime creation function has no parameter that would permit a new root.

## Lineage references

A resolved profile is identified by one component per namespace level, ordered
**leaf to root**:

    ego.neuocyte.research@3.7.5
      research@3  ->  neuocyte@7  ->  ego@5

Depth and component count must always match, so a reference is self-checking:
a three-level namespace with two components is malformed, not merely wrong.

Leaf-first ordering is deliberate. The part that changes most often is the part
you read first, and it matches how the namespace is written.

## Model variables

Exactly the six the inference backend applies, and no more. Inventing
``repetition_penalty`` because it is a familiar knob would mean recording a
setting the backend silently ignores -- a profile claiming to have shaped
cognition that did not. If the backend gains a sampler, this list grows with
it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..errors import InvalidInput

# Only bootstrap may introduce these; see `bootstrap.py`.
ROOTS = ("ego", "id")

_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_DEPTH = 8
MAX_COMPONENT_CHARS = 48

# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------
PROMPT_MODES = ("inherit", "append", "prepend", "replace")

SEPARATOR = "\n\n"
"""The exact bytes inserted between a parent's effective prompt and a local
fragment.

Stated once, here, because "some whitespace" is not a specification. A blank
line, never trailing or leading, and never inserted when one side is empty --
so a root establishing the base prompt is byte-identical to its own text.
"""


def compose(parent_text: str, local_text: str, mode: str) -> str:
    """The canonical composition rule. Deterministic, byte-exact.

    ``inherit`` ignores any local text; ``replace`` ignores the parent's. The
    two additive modes join with exactly one :data:`SEPARATOR`, and only when
    both sides have content.
    """
    if mode == "inherit":
        return parent_text
    if mode == "replace":
        return local_text
    if mode == "append":
        parts = (parent_text, local_text)
    elif mode == "prepend":
        parts = (local_text, parent_text)
    else:
        raise ValueError(f"unknown prompt mode {mode!r}")
    if not parts[0]:
        return parts[1]
    if not parts[1]:
        return parts[0]
    return parts[0] + SEPARATOR + parts[1]


# ---------------------------------------------------------------------------
# Model-generation variables
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VarSpec:
    name: str
    kind: str                      # number | integer | string_list
    minimum: float | None = None
    maximum: float | None = None
    note: str = ""


MODEL_VARS: dict[str, VarSpec] = {
    v.name: v for v in (
        VarSpec("temperature", "number", 0.0, 2.0,
                "0 makes the backend greedy"),
        VarSpec("top_p", "number", 0.0, 1.0, "nucleus cutoff; 1.0 disables"),
        VarSpec("top_k", "integer", 0, 100_000, "0 means the whole vocabulary"),
        VarSpec("max_output_tokens", "integer", 1, 1_000_000,
                "the profile's ceiling; the Harness may impose a lower one"),
        VarSpec("seed", "integer", 0, 2 ** 31 - 1, "sampler seed"),
        VarSpec("stop_sequences", "string_list", note="generation halts on any"),
    )
}

BACKEND_ARGUMENT = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_output_tokens": "max_tokens",
    "seed": "seed",
    "stop_sequences": "stop_strings",
}
"""How each profile variable reaches the backend.

A variable with no entry here would be one the backend cannot apply, which is
why there are none: the map is total over :data:`MODEL_VARS` and a test says so.
"""


FALLBACK_OUTPUT_CEILINGS: dict[str, int] = {
    "ego": 3072,
    "id": 1024,
    "ego.neuocyte": 512,
    "id.neuocyte": 384,
}
"""The generation ceiling a mind gets when its governed profile states none.

Ceilings, not target lengths: a model that has finished stops long before
reaching one. Ego is the persistent synthesiser the operator talks to and
gets the most room; Id reports and audits; neuocytes are bounded workers.

The governed values live in the shipped prompt headers and reach a mind
through its bound profile. This table is only what applies when a profile is
silent -- which is not hypothetical: the Ego root shipped without a ceiling,
and a hardcoded 384 decided how much Ego could say to anyone. It is a code
constant rather than a read of the shipped file on purpose. Editing a shipped
file makes a governed *candidate*; if the fallback read that file, the edit
would reach a running mind without anyone approving it.

A guard test holds this table equal to the shipped headers, so the two
cannot drift.
"""


def fallback_output_ceiling(namespace: str) -> int:
    """The nearest ancestor's fallback ceiling, as the resolver inherits it.

    A specialist `ego.neuocyte.reviewer` that states nothing inherits the
    worker ceiling, not Ego's -- the same answer resolution gives, because
    model variables are inherited root-first with the nearest one winning.
    """
    parts = namespace.split(".")
    for size in range(len(parts), 0, -1):
        found = FALLBACK_OUTPUT_CEILINGS.get(".".join(parts[:size]))
        if found is not None:
            return found
    raise InvalidInput("no output ceiling for that namespace", namespace=namespace)


class NamespaceError(InvalidInput, ValueError):
    """A namespace or lineage reference that cannot mean anything.

    An :class:`InvalidInput` so it crosses the RPC boundary as a typed,
    machine-readable rejection carrying its details, and a ``ValueError`` so
    the pure-vocabulary functions here remain usable outside the Harness.
    """


def validate_namespace(namespace: str, *, require_root: bool = True) -> tuple[str, ...]:
    """Split and check a namespace, returning its components root-first."""
    if not isinstance(namespace, str) or not namespace.strip():
        raise NamespaceError("namespace must be a non-empty string")
    parts = tuple(namespace.split("."))
    if len(parts) > MAX_DEPTH:
        raise NamespaceError(f"namespace deeper than {MAX_DEPTH} levels",)
    for part in parts:
        if len(part) > MAX_COMPONENT_CHARS:
            raise NamespaceError(f"namespace component too long: {part[:20]!r}")
        if not _COMPONENT.match(part):
            raise NamespaceError(
                f"invalid namespace component {part!r}; use lowercase letters, "
                "digits and underscores, starting with a letter")
    if require_root and parts[0] not in ROOTS:
        raise NamespaceError(
            f"unknown root {parts[0]!r}; roots are established only by "
            "bootstrap", known_roots=list(ROOTS))
    return parts


def parent_of(namespace: str) -> str | None:
    """The namespace one level up, or None for a root."""
    parts = namespace.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


def depth(namespace: str) -> int:
    return len(namespace.split("."))


@dataclass(frozen=True, slots=True)
class ProfileRef:
    """A namespace plus its leaf-to-root lineage vector."""

    namespace: str
    versions: tuple[int, ...]      # leaf first

    def __str__(self) -> str:
        return f"{self.namespace}@{'.'.join(str(v) for v in self.versions)}"

    @property
    def local_version(self) -> int:
        return self.versions[0]

    def root_first(self) -> tuple[int, ...]:
        """Versions ordered root-first, which is how ancestry walks."""
        return tuple(reversed(self.versions))

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace, "local_version": self.local_version,
                "lineage_version": ".".join(str(v) for v in self.versions),
                "profile_ref": str(self)}


def parse_ref(text: str, *, require_root: bool = True) -> ProfileRef:
    """Parse ``ego.neuocyte.research@3.7.5``.

    The component count must equal the namespace depth. A reference that does
    not describe a complete ancestry is rejected here rather than resolved
    against today's parents -- guessing which levels were omitted is exactly
    how an explicit historical selection would quietly become a current one.
    """
    if not isinstance(text, str) or "@" not in text:
        raise NamespaceError(
            "a profile reference looks like namespace@leaf...root, one component per namespace level",
            given=str(text)[:80])
    namespace, _, versions = text.partition("@")
    parts = validate_namespace(namespace, require_root=require_root)
    if not versions:
        raise NamespaceError("a profile reference needs a lineage version",
                             given=text[:80])
    try:
        nums = tuple(int(v) for v in versions.split("."))
    except ValueError as exc:
        raise NamespaceError("lineage components must be integers",
                             given=text[:80]) from exc
    if any(n < 1 for n in nums):
        raise NamespaceError("lineage components start at 1", given=text[:80])
    if len(nums) != len(parts):
        raise NamespaceError(
            f"lineage has {len(nums)} component(s) but {namespace!r} has "
            f"{len(parts)} level(s); one component per level, leaf first",
            given=text[:80], expected_components=len(parts))
    return ProfileRef(namespace=namespace, versions=nums)


def validate_model_vars(values: dict[str, Any]) -> dict[str, Any]:
    """Check and normalise locally defined model variables.

    Unknown names are refused rather than dropped: a profile that silently
    discarded ``repetition_penalty`` would look like it configured something.
    """
    if not isinstance(values, dict):
        raise NamespaceError("model variables must be an object")
    out: dict[str, Any] = {}
    for name, value in values.items():
        spec = MODEL_VARS.get(name)
        if spec is None:
            raise NamespaceError(
                f"unsupported model variable {name!r}; this backend applies "
                f"{sorted(MODEL_VARS)}",
                hint="a variable the backend ignores would be a profile "
                     "claiming to have shaped cognition that it did not")
        if spec.kind == "string_list":
            if not isinstance(value, (list, tuple)) or not all(
                    isinstance(v, str) for v in value):
                raise NamespaceError(f"{name} must be a list of strings")
            out[name] = [str(v) for v in value][:8]
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NamespaceError(f"{name} must be a number, not "
                                 f"{type(value).__name__}")
        if spec.kind == "integer":
            if float(value) != int(value):
                raise NamespaceError(f"{name} must be a whole number")
            value = int(value)
        else:
            value = float(value)
        if spec.minimum is not None and value < spec.minimum:
            raise NamespaceError(f"{name} below minimum {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise NamespaceError(f"{name} above maximum {spec.maximum}")
        out[name] = value
    return out


def validate_prompt(mode: str, text: str | None) -> str:
    """Mode and text must agree.

    ``inherit`` with local text would silently discard it; the additive and
    replacing modes with none would be a no-op wearing a mode's name. Both are
    refused so a node's declared intent matches what it does.
    """
    if mode not in PROMPT_MODES:
        raise NamespaceError(f"unknown prompt mode {mode!r}",
                             allowed=list(PROMPT_MODES))
    body = text or ""
    if mode == "inherit" and body.strip():
        raise NamespaceError(
            "prompt_mode=inherit takes no local prompt text",
            hint="use append, prepend or replace to contribute text")
    if mode != "inherit" and not body.strip():
        raise NamespaceError(f"prompt_mode={mode} requires local prompt text")
    return body


UNRESOLVED_PLACEHOLDER = re.compile(r"\{\{\s*[A-Za-z_][\w.]*\s*\}\}")


def check_no_unresolved_placeholders(text: str) -> None:
    """Refuse ``{{ like_this }}`` left in a prompt.

    A placeholder nobody substitutes reaches the model verbatim, which is a
    silent way for a prompt to be subtly wrong forever.
    """
    found = UNRESOLVED_PLACEHOLDER.findall(text or "")
    if found:
        raise NamespaceError("prompt text contains unresolved placeholders",
                             placeholders=found[:5])
