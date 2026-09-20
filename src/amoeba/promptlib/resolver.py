"""Turning a lineage into the exact cognitive configuration a mind receives.

Resolution is a pure function of the **pinned** ancestry. It reads no
selection table, consults no "current" parent, and has no clock: the same
lineage vector resolves to byte-identical text and settings today and in a year,
which is what makes a frozen incarnation binding meaningful rather than
decorative.

Two rules, applied root-first down the chain:

* **Prompt text** composes through each level's declared mode -- the parent's
  *effective* text (not just its local fragment) is what a child inherits, so
  contribution accumulates down the whole lineage.
* **Ordinary properties** take the nearest ancestor that defines them. A leaf
  setting ``temperature`` shadows its grandparent's; a leaf that is silent
  inherits, and the resolved value records *which* level supplied it.

Nothing here is advisory. :func:`resolve` produces the digests that the
incarnation binding freezes and that `explain_profile` narrates, so "where did
this instruction come from" has one answer computed one way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..errors import InvalidInput
from ..ids import sha256_hex
from .model import (BACKEND_ARGUMENT, MODEL_VARS, ProfileRef, compose,
                    check_no_unresolved_placeholders)


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(slots=True)
class Contribution:
    """What one level of the lineage actually did."""

    namespace: str
    local_version: int
    version_id: str
    prompt_mode: str
    local_prompt: str
    text_before: str
    text_after: str
    vars_set: dict[str, Any] = field(default_factory=dict)
    vars_shadowed: dict[str, str] = field(default_factory=dict)

    @property
    def changed_prompt(self) -> bool:
        return self.text_after != self.text_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "local_version": self.local_version,
            "version_id": self.version_id,
            "prompt_mode": self.prompt_mode,
            "local_prompt_chars": len(self.local_prompt),
            "local_prompt": self.local_prompt,
            "changed_prompt": self.changed_prompt,
            "resulting_prompt_chars": len(self.text_after),
            "model_vars_set": self.vars_set,
            "model_vars_shadowed_by": self.vars_shadowed,
        }


@dataclass(slots=True)
class ResolvedProfile:
    ref: ProfileRef
    prompt_text: str
    model_vars: dict[str, Any]
    var_source: dict[str, str]           # variable -> "namespace@local_version"
    contributions: list[Contribution]    # root first
    prompt_sha256: str
    config_sha256: str
    profile_sha256: str

    # -- what the backend is actually handed ---------------------------
    def backend_kwargs(self, constraints: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
        """Map resolved variables onto the engine's argument names.

        Unset variables are simply absent, so the engine's own defaults apply
        rather than a second set of defaults invented here.
        """
        effective = self.effective_settings(constraints)
        return {BACKEND_ARGUMENT[name]: value
                for name, value in effective.items() if name in BACKEND_ARGUMENT}

    def effective_settings(self, constraints: dict[str, Any] | None = None
                           ) -> dict[str, Any]:
        """Resolved variables after the Harness narrows them.

        A profile states a ceiling; the Harness may impose a lower one and
        never a higher one. The result is recorded on the incarnation binding,
        so a transcript shows the settings that were *used*, not the ones the
        profile asked for.
        """
        out = dict(self.model_vars)
        for name, limit in (constraints or {}).items():
            if name not in MODEL_VARS:
                raise InvalidInput("unknown harness constraint", name=name)
            if name == "max_output_tokens":
                out[name] = min(int(limit), int(out.get(name, limit)))
            else:
                out[name] = limit
        return out

    def suffix_after(self, namespace: str) -> str:
        """The part of the resolved prompt contributed *below* ``namespace``.

        A neuocyte forked from an Ego snapshot already physically holds the
        text its ancestors primed. Injecting the whole resolved profile would
        duplicate it; injecting only this suffix reproduces the resolved
        profile in the forked session.

        The distinction is recorded rather than glossed over: the binding
        stores both digests, so "this mind received profile P" and "these were
        the bytes actually injected" are separate, checkable statements.
        """
        names = [c.namespace for c in self.contributions]
        if namespace not in names:
            raise InvalidInput(
                "that namespace is not in this lineage",
                namespace=namespace, lineage=names)
        text = ""
        for contribution in self.contributions[names.index(namespace) + 1:]:
            text = compose(text, contribution.local_prompt,
                           contribution.prompt_mode)
        return text

    def to_dict(self) -> dict[str, Any]:
        d = self.ref.to_dict()
        d.update({
            "prompt_text": self.prompt_text,
            "prompt_chars": len(self.prompt_text),
            "model_vars": self.model_vars,
            "model_var_source": self.var_source,
            "prompt_sha256": self.prompt_sha256,
            "config_sha256": self.config_sha256,
            "profile_sha256": self.profile_sha256,
        })
        return d


def resolve_chain(chain: list[dict[str, Any]]) -> ResolvedProfile:
    """Resolve an already-verified pinned chain (leaf first, as stored)."""
    if not chain:
        raise InvalidInput("cannot resolve an empty lineage")
    leaf = chain[0]
    root_first = list(reversed(chain))

    text = ""
    variables: dict[str, Any] = {}
    source: dict[str, str] = {}
    contributions: list[Contribution] = []

    for node in root_first:
        label = f"{node['namespace']}@{node['local_version']}"
        before = text
        text = compose(before, node["prompt_text"] or "", node["prompt_mode"])

        local_vars = node["model_vars"] or {}
        shadowed: dict[str, str] = {}
        for name, value in sorted(local_vars.items()):
            if name in source:
                # A nearer ancestor has not spoken yet -- we walk root-first, so
                # anything already present came from *further* up and is being
                # overridden here. Record it on that earlier contribution.
                previous = source[name]
                for earlier in contributions:
                    if f"{earlier.namespace}@{earlier.local_version}" == previous:
                        earlier.vars_shadowed[name] = label
                        break
            variables[name] = value
            source[name] = label

        contributions.append(Contribution(
            namespace=node["namespace"], local_version=int(node["local_version"]),
            version_id=node["version_id"], prompt_mode=node["prompt_mode"],
            local_prompt=node["prompt_text"] or "", text_before=before,
            text_after=text, vars_set=dict(sorted(local_vars.items())),
            vars_shadowed=shadowed))

    # A placeholder that survived composition reaches the model verbatim.
    check_no_unresolved_placeholders(text)

    ref = ProfileRef(namespace=leaf["namespace"],
                     versions=tuple(int(n["local_version"]) for n in chain))
    prompt_sha = sha256_hex(text.encode("utf-8"))
    config_sha = sha256_hex(_canon(variables))
    profile_sha = sha256_hex(_canon({
        "profile_ref": str(ref),
        "prompt_sha256": prompt_sha,
        "config_sha256": config_sha,
        "lineage": [{"namespace": n["namespace"],
                     "local_version": int(n["local_version"]),
                     "version_id": n["version_id"],
                     "local_sha256": n["local_sha256"]} for n in chain],
    }))

    return ResolvedProfile(ref=ref, prompt_text=text, model_vars=variables,
                           var_source=source, contributions=contributions,
                           prompt_sha256=prompt_sha, config_sha256=config_sha,
                           profile_sha256=profile_sha)


class Resolver:
    """Resolution against a :class:`~amoeba.promptlib.store.PromptStore`."""

    def __init__(self, store) -> None:
        self.store = store

    def resolve_version(self, namespace: str, local_version: int) -> ResolvedProfile:
        return resolve_chain(self.store.pinned_chain(namespace, local_version))

    def resolve_ref(self, ref: ProfileRef) -> ResolvedProfile:
        """Resolve an explicitly requested historical lineage."""
        return resolve_chain(self.store.resolve_ref(ref))

    def resolve_selected(self, namespace: str, purpose: str = "production"
                         ) -> ResolvedProfile:
        """Resolve whatever is selected for a namespace *right now*.

        The only time-dependent entry point, and the only one an incarnation
        uses at birth. Once bound, the incarnation carries the resolved
        lineage vector; it never re-reads this.
        """
        selected = self.store.selected(namespace, purpose)
        if selected is None:
            raise InvalidInput(
                f"no {purpose} profile selected for {namespace!r}",
                hint="approve and select a version, or resolve an explicit "
                     "lineage reference")
        return self.resolve_version(namespace, int(selected["local_version"]))

    # -- narration -------------------------------------------------------
    def explain(self, ref: ProfileRef) -> dict[str, Any]:
        """Answer "where did this instruction come from" level by level.

        Returns the resolved text and, for every level, the mode it used, the
        fragment it contributed, whether that fragment changed anything, the
        variables it set, and which descendant later shadowed them.
        """
        resolved = self.resolve_ref(ref)
        return {
            "profile_ref": str(resolved.ref),
            "prompt_text": resolved.prompt_text,
            "prompt_sha256": resolved.prompt_sha256,
            "config_sha256": resolved.config_sha256,
            "profile_sha256": resolved.profile_sha256,
            "model_vars": resolved.model_vars,
            "model_var_source": resolved.var_source,
            "unset_model_vars": sorted(
                name for name in MODEL_VARS if name not in resolved.model_vars),
            "lineage": [c.to_dict() for c in resolved.contributions],
            "note": ("levels are ordered root first, which is the order they "
                     "compose in; the lineage vector is written leaf first"),
        }
