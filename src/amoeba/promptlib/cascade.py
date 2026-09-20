"""Carrying a parent change down the family tree, or deliberately not.

Approving a new version of ``ego`` does **not** change what ``ego.neuocyte``
inherits. Every child pins an exact parent version, so an existing descendant
keeps resolving to exactly the text it always did. That is the property the
pinning exists to buy, and cascade is how you give it up on purpose.

Three propagation modes, chosen by the Operator at approval time:

``none``
    Nothing. Descendants keep their pins. The new parent version applies only
    to nodes that later pin it.

``queue``
    For each descendant, create a **candidate** rebased onto the new parent,
    copying the descendant's local definition unchanged. Each one still needs
    approval. The subtree is offered the change; nobody is moved.

``approve``
    The same rebase, approved and selected in one act. The Operator is saying
    "carry the whole subtree forward", and the receipt records that they did.

A rebase never edits a local definition. The new version's ``local_sha256`` is
byte-identical to the one it was rebased from, so "only the parent binding
changed" is a checkable fact rather than a promise -- and a test checks it.

Cascade descends level by level. Rebasing ``ego.neuocyte`` onto ``ego@2``
produces ``ego.neuocyte@2``; ``ego.neuocyte.research`` sees nothing until it is
in turn rebased onto *that*, which is why a plan is computed breadth-first and
applied in the same order.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidInput
from ..store.events import EventKind
from ..store.writer import Mutation
from .model import depth
from .store import APPROVED, PromptStore

MODES = ("none", "queue", "approve")


def plan(store: PromptStore, namespace: str, local_version: int, *,
         mode: str) -> dict[str, Any]:
    """Work out exactly which descendants would be rebased, and which not.

    Computed before anything is written so the Operator approves a concrete
    list rather than a verb. Descendants with nothing selected are reported as
    skipped with a reason rather than silently omitted -- an empty-looking
    cascade should never be indistinguishable from a complete one.
    """
    if mode not in MODES:
        raise InvalidInput("unknown propagation mode", mode=mode,
                           allowed=list(MODES))
    root = store.get_version(namespace, local_version)
    if mode == "none":
        return {"mode": mode, "namespace": namespace,
                "parent_version": local_version, "steps": [], "skipped": [],
                "note": "descendants keep their pinned parent versions"}
    if root["state"] not in APPROVED:
        raise InvalidInput(
            "only an approved parent version may be cascaded",
            namespace=namespace, local_version=local_version,
            state=root["state"])

    steps: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # (namespace, the parent local version its children will pin)
    frontier: list[tuple[str, int]] = [(namespace, local_version)]

    while frontier:
        parent_ns, parent_version = frontier.pop(0)
        for child_ns in store.children_of(parent_ns):
            selected = store.selected(child_ns)
            if selected is None:
                skipped.append({"namespace": child_ns,
                                "reason": "no production version is selected, "
                                          "so there is nothing running to "
                                          "carry forward"})
                continue
            if (selected["parent_namespace"] == parent_ns
                    and int(selected["parent_version"]) == parent_version):
                skipped.append({"namespace": child_ns,
                                "reason": f"already pins {parent_ns}@"
                                          f"{parent_version}"})
                # It already sees the new parent, but its own children may not.
                frontier.append((child_ns, int(selected["local_version"])))
                continue
            new_version = store.next_local_version(child_ns)
            steps.append({
                "namespace": child_ns,
                "from_version": int(selected["local_version"]),
                "from_version_id": selected["version_id"],
                "new_local_version": new_version,
                "old_parent_version": selected["parent_version"],
                "new_parent_namespace": parent_ns,
                "new_parent_version": parent_version,
                "local_sha256": selected["local_sha256"],
                "depth": depth(child_ns),
            })
            frontier.append((child_ns, new_version))

    steps.sort(key=lambda s: (s["depth"], s["namespace"]))
    return {"mode": mode, "namespace": namespace,
            "parent_version": local_version, "steps": steps,
            "skipped": skipped,
            "resulting_state": ("candidate" if mode == "queue"
                                else "production_approved"),
            "selection_moves": mode == "approve"}


def apply_plan(m: Mutation, store: PromptStore, plan_result: dict[str, Any], *,
               actor: str, rationale: str = "") -> dict[str, Any]:
    """Execute a plan, re-deriving each pin from what was actually created.

    The plan's predicted version numbers are not trusted here: each rebase
    pins the version that the previous step really produced. A plan computed a
    moment ago against a library that has since moved would otherwise write a
    pin to a version that does not exist.
    """
    mode = plan_result["mode"]
    if mode == "none" or not plan_result["steps"]:
        m.emit(EventKind.PROMPT_CASCADE_PLANNED, {
            "namespace": plan_result["namespace"],
            "parent_version": plan_result["parent_version"],
            "mode": mode, "steps": 0,
            "skipped": plan_result.get("skipped", []), "actor": actor})
        return {"mode": mode, "cascaded": [], "skipped": plan_result.get("skipped", [])}

    state = "production_approved" if mode == "approve" else "candidate"
    # namespace -> the local version its children should now pin.
    produced: dict[str, int] = {
        plan_result["namespace"]: int(plan_result["parent_version"])}
    cascaded: list[dict[str, Any]] = []

    for step in plan_result["steps"]:
        parent_ns = step["new_parent_namespace"]
        parent_version = produced.get(parent_ns, step["new_parent_version"])
        source = store.by_id(step["from_version_id"])
        created = store.create_version(
            m, namespace=step["namespace"],
            prompt_mode=source["prompt_mode"],
            prompt_text=source["prompt_text"],
            model_vars=source["model_vars"],
            parent_version=parent_version,
            origin="cascade", created_by=actor, state=state,
            rationale=(rationale or
                       f"rebased from {step['namespace']}@{step['from_version']} "
                       f"onto {parent_ns}@{parent_version}; local definition "
                       f"unchanged"))
        if created["local_sha256"] != source["local_sha256"]:
            # Unreachable unless a rebase changed a definition, which is the
            # one thing cascade must never do.
            raise InvalidInput(
                "rebase altered a local definition",
                namespace=step["namespace"],
                expected=source["local_sha256"], actual=created["local_sha256"])
        if mode == "approve":
            store.select(m, namespace=step["namespace"],
                         version_id=created["version_id"], purpose="production",
                         selected_by=actor)
        produced[step["namespace"]] = int(created["local_version"])
        cascaded.append({
            "namespace": step["namespace"],
            "from_version": step["from_version"],
            "new_local_version": created["local_version"],
            "version_id": created["version_id"],
            "pinned_parent": f"{parent_ns}@{parent_version}",
            "local_sha256": created["local_sha256"],
            "state": state, "selected": mode == "approve"})

    m.emit(EventKind.PROMPT_CASCADED, {
        "namespace": plan_result["namespace"],
        "parent_version": plan_result["parent_version"], "mode": mode,
        "actor": actor, "cascaded": cascaded,
        "skipped": plan_result.get("skipped", []),
        "note": ("local definitions were copied unchanged; only the pinned "
                 "parent differs" if cascaded else "")})
    return {"mode": mode, "cascaded": cascaded,
            "skipped": plan_result.get("skipped", [])}
