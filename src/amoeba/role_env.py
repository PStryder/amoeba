"""The role environment: what exists and what a role can do, right now.

Three things reach a long-lived mind, and conflating any two of them is how a
system ends up rewriting its constitution because a tool was added:

``PROFILE``
    Who the role is and how it should reason. Governed, versioned, immutable,
    and bound once at incarnation -- see ``promptlib``.

``ENVIRONMENT``
    What currently exists and what this role may currently do. Authored by the
    Harness, rebuilt per turn, and *not* doctrine. This module.

``TURN INPUT``
    What demands cognition right now: a user message, a dossier, a trigger.

Neuocytes already worked this way -- their tool block is built by the Harness
per work item, so their prompt cannot advertise a capability the work row does
not carry. Ego and Id had a profile and a turn input and nothing in between,
which meant a newly approved ``ego.neuocyte.research`` was invisible to Ego
unless somebody rewrote Ego's root prompt. Constitutional doctrine was the
only channel for environmental fact.

Two properties this module exists to guarantee:

**A role is never told it has a capability it cannot invoke.** The verb list
is ``scopes.model_facing_verbs(role)``, every entry is resolved against the
supervisor's real method table, and the description and argument schema are
read off the actual function. There is no second hand-maintained list to drift.

**The manifest is stable within a turn and fresh between turns.** It is built
once at turn start, content-addressed, and recorded; the same state produces
byte-identical bytes, so a repeated environment deduplicates to one blob.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from .errors import InvalidInput
from .ids import sha256_hex
from .scopes import model_facing_verbs
from .vocabularies import allowed

if TYPE_CHECKING:
    from .supervisor import Supervisor

ROLES = ("ego", "id")

# A manifest is injected into a context on every turn, so it has a budget.
MAX_PROFILES = 40
MAX_DESCRIPTION_CHARS = 240


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _summary(fn: Any) -> str:
    """The first sentence of the real function's docstring.

    Read from the function the dispatcher will actually call, so a verb cannot
    be described as one thing and implemented as another.
    """
    doc = inspect.getdoc(fn) or ""
    first = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    return first[:MAX_DESCRIPTION_CHARS] or "(no description)"


def _kind_of(annotation: str) -> str | None:
    """A compact JSON kind for a non-string argument, from its annotation.

    Shown because the model otherwise guesses: live, Ego passed `evidence` as
    a string where a list was wanted and `confidence` as "high" where a number
    was, and both calls were refused. Plain strings stay unmarked, which is
    most arguments, so the declaration grows only where it was wrong.
    """
    a = annotation.replace("typing.", "").replace(" ", "")
    for token, kind in (("Sequence", "list"), ("list", "list"), ("tuple", "list"),
                        ("dict", "object"), ("Mapping", "object"),
                        ("bool", "boolean"), ("float", "number"), ("int", "integer")):
        if a.startswith(token) or f"|{token}" in a or a.startswith(f"{token}|"):
            return kind
    return None


def _schema(fn: Any, verb: str = "") -> dict[str, Any]:
    """Argument names, requiredness, defaults, kinds and accepted values."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):       # builtins and partials
        return {"arguments": [], "note": "signature unavailable"}
    args = []
    for name, param in sig.parameters.items():
        if name.startswith("_") or param.kind in (param.VAR_POSITIONAL,
                                                  param.VAR_KEYWORD):
            continue
        # Never advertised: the Harness sets these itself, and a model that
        # believed it could choose them would be wrong about who it is.
        if name in ("operation_id", "mutation_id", "client_id", "actor",
                    "caller", "role", "scope", "origin_actor", "from_role"):
            continue
        entry: dict[str, Any] = {"name": name,
                                 "required": param.default is param.empty}
        annotation = param.annotation
        if annotation is not param.empty:
            entry["type"] = (getattr(annotation, "__name__", None)
                             or str(annotation).replace("typing.", ""))[:60]
        if param.default is not param.empty and param.default is not None:
            if isinstance(param.default, (str, int, float, bool)):
                entry["default"] = param.default
        values = allowed(verb, name)
        if values:
            entry["allowed"] = list(values)
        kind = _kind_of(str(entry.get("type", "")))
        if kind:
            entry["kind"] = kind
        args.append(entry)
    return {"arguments": args}


def capability_manifest(sup: "Supervisor", role: str) -> list[dict[str, Any]]:
    """The verbs this role's model may call, described from the real functions.

    Every name is looked up in the supervisor's dispatch table. A model-facing
    verb that does not resolve is a bug that surfaces here rather than at the
    moment the model tries to use it.
    """
    methods = sup.methods()
    out = []
    for name in model_facing_verbs(role):
        fn = methods.get(name)
        if fn is None:
            raise InvalidInput(
                "a model-facing verb is not in the dispatch table",
                verb=name, role=role,
                hint="scopes.MODEL_FACING and the supervisor method table "
                     "have drifted apart")
        out.append({"verb": name, "summary": _summary(fn), **_schema(fn, name)})
    return out


def _profile_inventory(sup: "Supervisor", role: str) -> list[dict[str, Any]]:
    """Productive cognitive profiles this role may currently draw on.

    For Ego, the ``ego.*`` descendants -- the neuocyte specialisations it can
    ask for. For Id, the ``id.*`` ones. A root is excluded: a role does not
    "use" its own doctrine as a resource, it *is* it.

    Only approved-and-selected lineages appear. An approved version nobody
    selected is not something Ego can be given, and advertising it would be
    the manifest describing the library's possibilities rather than the
    organism's current capability.
    """
    from .promptlib.store import PromptStore

    store = PromptStore(sup.mind)
    out = []
    for namespace in store.namespaces():
        if not namespace.startswith(role + "."):
            continue
        selected = store.selected(namespace)
        if selected is None:
            continue
        try:
            ref = store.ref_for(namespace, int(selected["local_version"]))
        except Exception:                 # a broken ancestry is not offered
            continue
        out.append({"namespace": namespace, "profile_ref": str(ref),
                    "prompt_mode": selected["prompt_mode"],
                    "state": selected["state"]})
    return out[:MAX_PROFILES]


def _resource_identities(sup: "Supervisor", role: str) -> dict[str, Any]:
    """Version identities needed to interpret this turn's cognition.

    Identities, not configuration: a role learns *which* prompt surface and
    tool surface produced a result, not the security policy or the filespace
    layout, neither of which is its business.
    """
    from .resources import prompt_version, schema_version, tool_surface_version

    # The same source `ego_resource_identities` reads, so the manifest and
    # that sense cannot disagree about which weights produced a turn. The
    # pulse is cached, so this costs nothing per turn; only the identity is
    # taken from it, never the telemetry.
    generation = ""
    pulse = getattr(sup, "pulse", None)
    if pulse is not None:
        try:
            generation = pulse.capture(max_age_seconds=60.0).get(
                "model_generation") or ""
        except Exception:                 # a manifest is not worth failing a turn
            generation = ""

    return {
        "model_generation": generation,
        "prompt": prompt_version(role, sup.cfg, sup.mind).to_dict(),
        "tool_surface": tool_surface_version().to_dict(),
        "schema": schema_version().to_dict(),
    }


def build(sup: "Supervisor", role: str, *, incarnation: int | None = None,
          profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one role's environment manifest.

    Deterministic: the same authoritative state produces byte-identical bytes,
    which is what lets a repeated environment resolve to one content-addressed
    blob instead of a new one per turn.
    """
    if role not in ROLES:
        raise InvalidInput("unknown role", role=role, allowed=list(ROLES))
    manifest: dict[str, Any] = {
        "schema": "amoeba.role_environment/1",
        "role": role,
        "incarnation": incarnation,
        "bound_profile": {
            "profile_ref": (profile or {}).get("profile_ref"),
            "prompt_sha256": (profile or {}).get("prompt_sha256"),
            "config_sha256": (profile or {}).get("config_sha256"),
        },
        "available_profiles": _profile_inventory(sup, role),
        "capabilities": capability_manifest(sup, role),
        "resources": _resource_identities(sup, role),
        "contract": (
            "This declaration is authoritative for this turn and is rebuilt "
            "for the next one. A capability absent from it cannot be invoked, "
            "and one present in a previous turn may be gone from this one."
        ),
    }
    manifest["environment_sha256"] = sha256_hex(_canon(
        {k: v for k, v in manifest.items() if k != "incarnation"}))
    return manifest


def _render_argument(a: dict[str, Any]) -> str:
    """`name`, `name?`, `name:list?`, or `name:{a|b|c}` -- what it takes, tersely."""
    text = a["name"]
    if a.get("allowed"):
        text += ":{" + "|".join(a["allowed"]) + "}"
    elif a.get("kind"):
        text += ":" + a["kind"]
    return text + ("" if a["required"] else "?")


def render(manifest: dict[str, Any]) -> str:
    """The manifest as the compact text a role actually reads.

    Structured and declarative. It carries names, lineages, identities and
    argument schemas from authoritative state, and no free prose from anyone
    -- this is an environment declaration, not another conversational channel
    into the mind.
    """
    lines = [
        "<role_environment>",
        f"role: {manifest['role']}  incarnation: {manifest['incarnation']}",
        f"profile: {manifest['bound_profile']['profile_ref']}",
    ]
    res = manifest["resources"]
    lines.append(f"model_generation: {res.get('model_generation') or 'unknown'}")
    lines.append(f"tool_surface: {res['tool_surface']['sha256'][:12]}")

    profiles = manifest["available_profiles"]
    lines.append("")
    lines.append(f"available cognitive profiles ({len(profiles)}):")
    if profiles:
        for p in profiles:
            lines.append(f"  {p['profile_ref']}")
    else:
        lines.append("  (none currently selectable)")

    lines.append("")
    lines.append(f"capabilities you may invoke now ({len(manifest['capabilities'])}):")
    for cap in manifest["capabilities"]:
        args = ", ".join(_render_argument(a) for a in cap.get("arguments", []))
        lines.append(f"  {cap['verb']}({args}) - {cap['summary']}")

    lines.append("")
    lines.append(manifest["contract"])
    lines.append(f"environment: {manifest['environment_sha256'][:12]}")
    lines.append("</role_environment>")
    return "\n".join(lines)
