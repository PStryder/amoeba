"""Versioned cognitive resources.

A resource is anything Harness-managed that can materially change how the
organism thinks: a role's system prompt, the tool surface a model is offered,
the security policy, the filespace configuration, the store schema. Changing
one changes cognition, so each gets a digest, and the digest goes in the pulse.

Two things this makes possible that were not:

* **Noticing that cognition changed.** A behavioural shift with no resource
  change is a different problem from one that follows a prompt edit, and Id
  cannot tell them apart without seeing the versions.
* **Distinguishing configured from embodied.** A running role primed its
  context with the prompt that existed when it started. Editing the config
  changes what a *new* incarnation would run, not what the live one is
  running. Reporting only the configured version would assert the running mind
  is something it is not, so both are reported and compared.

Digests are over canonical bytes, so they are stable across processes and
restarts, and cheap enough to recompute on every pulse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .ids import sha256_hex

RESOURCE_KINDS = ("prompt.ego", "prompt.id", "tools.neuocyte", "security.policy",
                  "filespace.config", "store.schema")


@dataclass(slots=True)
class ResourceVersion:
    kind: str
    sha256: str
    detail: dict[str, Any]

    @property
    def short(self) -> str:
        return self.sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sha256": self.sha256, "short": self.short,
                "detail": self.detail}


def _digest(obj: Any) -> str:
    return sha256_hex(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                 default=str).encode("utf-8"))


def prompt_text(role: str, cfg: Any, mind: Any = None) -> str:
    """Exactly what a role primes its context with.

    Mirrors ``RoleProcess._system_text`` deliberately: a digest computed from a
    different composition would be a version number for something nobody runs.
    That mirroring is why this reads the Prompt Library when one is available.
    A role now takes its base text from the selected profile, so hashing the
    module constant would report a version of text the organism has not used
    since bootstrap.

    The constant remains the fallback for exactly the case where a role also
    falls back to it: no library, or nothing selected for that namespace.
    """
    from .roles import EGO_SYSTEM, ID_SYSTEM

    base = EGO_SYSTEM if role == "ego" else ID_SYSTEM
    source = "builtin"
    if mind is not None:
        try:
            from .promptlib.resolver import Resolver
            from .promptlib.store import PromptStore

            resolved = Resolver(PromptStore(mind)).resolve_selected(role)
            base, source = resolved.prompt_text, str(resolved.ref)
        except Exception:                      # no library, or nothing selected
            pass
    return base


def prompt_source(role: str, cfg: Any, mind: Any = None) -> str:
    """Where the base text came from: a profile reference, or ``builtin``."""
    if mind is None:
        return "builtin"
    try:
        from .promptlib.resolver import Resolver
        from .promptlib.store import PromptStore

        return str(Resolver(PromptStore(mind)).resolve_selected(role).ref)
    except Exception:
        return "builtin"


def prompt_version(role: str, cfg: Any, mind: Any = None) -> ResourceVersion:
    text = prompt_text(role, cfg, mind)
    return ResourceVersion(
        kind=f"prompt.{role}", sha256=sha256_hex(text.encode("utf-8")),
        detail={"chars": len(text), "source": prompt_source(role, cfg, mind),
                "governed_by": "prompt_library"})


def tool_surface_version(*, sandbox_allowed: bool = True) -> ResourceVersion:
    """A digest of the tool surface a neuocyte model is offered.

    Over the declared schemas rather than the implementation, because that is
    what the model sees and reasons about. Handlers are never called here.
    """
    from .tools import build_neuocyte_registry

    class _NoMind:
        def __getattr__(self, name: str) -> Any:  # pragma: no cover - never used
            raise AssertionError("digesting the tool surface must not touch state")

    class _NoSup:
        mind = _NoMind()

        def methods(self) -> dict[str, Any]:
            return {}

        def sandbox_for_work(self, work_id: str, *, owner: str) -> str:
            raise AssertionError("digesting the tool surface must not touch state")

    reg = build_neuocyte_registry(_NoSup(), work_id="", neuocyte_id="",
                                  sandbox_allowed=sandbox_allowed)
    schemas = reg.schemas(role="neuocyte")
    return ResourceVersion(
        kind="tools.neuocyte", sha256=_digest(schemas),
        detail={"tool_count": len(schemas),
                "names": sorted(s["name"] for s in schemas),
                "sandbox_allowed": sandbox_allowed})


def security_policy_version(cfg: Any) -> ResourceVersion:
    from .filespace import RESERVED_STEMS
    from .security import FORBIDDEN_SIDS

    policy = {
        "forbidden_sids": sorted(FORBIDDEN_SIDS),
        "reserved_stems": sorted(RESERVED_STEMS),
        "allow_multiply_linked": cfg.filespace.allow_multiply_linked,
        "snapshot_before_overwrite": cfg.filespace.snapshot_before_overwrite,
        "sandbox_enabled": cfg.sandbox.enabled,
        "sandbox_max_concurrent": cfg.sandbox.max_concurrent,
        "max_tool_turns": cfg.arbiter.max_tool_turns,
    }
    return ResourceVersion(kind="security.policy", sha256=_digest(policy),
                           detail=policy)


def filespace_version(cfg: Any) -> ResourceVersion:
    roots = [{"name": r.name, "path": r.path, "mode": r.mode}
             for r in cfg.filespace.roots]
    roots.sort(key=lambda r: r["name"])
    return ResourceVersion(
        kind="filespace.config", sha256=_digest(roots),
        detail={"root_count": len(roots),
                "writable": [r["name"] for r in roots if r["mode"] == "read_write"],
                "read_only": [r["name"] for r in roots if r["mode"] == "read_only"]})


def schema_version() -> ResourceVersion:
    from .store.db import SCHEMA_VERSION

    return ResourceVersion(kind="store.schema", sha256=_digest(SCHEMA_VERSION),
                           detail={"schema_version": SCHEMA_VERSION})


def all_versions(cfg: Any, mind: Any = None) -> dict[str, dict[str, Any]]:
    """Every versioned resource, as configured right now.

    Cheap: string hashing and one registry construction with no handler calls.
    """
    out: dict[str, dict[str, Any]] = {}
    for rv in (prompt_version("ego", cfg, mind), prompt_version("id", cfg, mind),
               tool_surface_version(), security_policy_version(cfg),
               filespace_version(cfg), schema_version()):
        out[rv.kind] = rv.to_dict()
    return out
