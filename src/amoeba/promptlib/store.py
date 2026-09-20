"""Persistence for the cognitive family tree.

Every row is immutable once written. There is no update path for a node
version's definition, and there is no function that takes a `version_id` and
changes what it means -- a "change" always produces a new local version.

Two rules this module exists to enforce structurally rather than by checking a
flag:

* **Roots come only from bootstrap.** :func:`create_version` has no parameter
  that would permit a new root or a new version of one. The bootstrap path
  calls a separate function that is not exported to any RPC surface.
* **A child pins its parent's exact version.** The parent version is stored on
  the child row, so promoting a new parent cannot reach backwards and change
  what an existing child inherits.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from ..errors import InvalidInput, NotFound
from ..ids import new_id, sha256_hex
from ..store.events import EventKind
from ..store.writer import Mutation
from .model import (MODEL_VARS, NamespaceError, ProfileRef, ROOTS,
                    check_no_unresolved_placeholders, depth, parent_of,
                    validate_model_vars, validate_namespace, validate_prompt)

if TYPE_CHECKING:
    from ..mind import Mind

# candidate -> validated -> evaluated -> proposed -> {production|experimental}
# with rejected and retired as terminal outcomes.
STATES = ("candidate", "validated", "evaluated", "proposed",
          "production_approved", "experimental_approved", "rejected", "retired")

APPROVED = ("production_approved", "experimental_approved")

LEGAL_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "candidate": ("validated", "rejected"),
    "validated": ("evaluated", "proposed", "rejected"),
    "evaluated": ("proposed", "rejected"),
    "proposed": ("production_approved", "experimental_approved", "rejected"),
    "production_approved": ("retired", "rejected"),
    "experimental_approved": ("production_approved", "retired", "rejected"),
    "rejected": (),
    "retired": (),
}

ORIGINS = ("bootstrap", "id", "operator", "cascade")


def canonical_local(prompt_mode: str, prompt_text: str,
                    model_vars: dict[str, Any]) -> str:
    """Digest of a node's *local* definition only.

    Deliberately excludes the parent binding: two versions that differ only by
    rebase have the same local digest, which is exactly what makes "the local
    definitions were copied unchanged" a checkable claim after a cascade.
    """
    return sha256_hex(json.dumps(
        {"prompt_mode": prompt_mode, "prompt_text": prompt_text,
         "model_vars": model_vars},
        sort_keys=True, separators=(",", ":")).encode("utf-8"))


class PromptStore:
    def __init__(self, mind: "Mind") -> None:
        self.mind = mind

    # -- reading ---------------------------------------------------------
    @property
    def conn(self):
        return self.mind.db.conn

    def get_version(self, namespace: str, local_version: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM prompt_versions WHERE namespace = ? AND local_version = ?",
            (namespace, int(local_version))).fetchone()
        if row is None:
            raise NotFound("no such prompt version", namespace=namespace,
                           local_version=local_version)
        out = dict(row)
        out["model_vars"] = json.loads(out["model_vars"] or "{}")
        return out

    def by_id(self, version_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM prompt_versions WHERE version_id = ?",
            (version_id,)).fetchone()
        if row is None:
            raise NotFound("no such prompt version", version_id=version_id)
        out = dict(row)
        out["model_vars"] = json.loads(out["model_vars"] or "{}")
        return out

    def versions(self, namespace: str) -> list[dict[str, Any]]:
        return [self.by_id(r["version_id"]) for r in self.conn.execute(
            "SELECT version_id FROM prompt_versions WHERE namespace = ?"
            " ORDER BY local_version ASC", (namespace,))]

    def namespaces(self) -> list[str]:
        return [r["namespace"] for r in self.conn.execute(
            "SELECT DISTINCT namespace FROM prompt_versions ORDER BY namespace")]

    def next_local_version(self, namespace: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(local_version) AS v FROM prompt_versions WHERE namespace = ?",
            (namespace,)).fetchone()
        return int((row["v"] or 0)) + 1

    def selected(self, namespace: str, purpose: str = "production"
                 ) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT version_id FROM prompt_selections"
            " WHERE namespace = ? AND purpose = ?", (namespace, purpose)).fetchone()
        return self.by_id(row["version_id"]) if row else None

    def children_of(self, namespace: str) -> list[str]:
        """Direct child namespaces, by name rather than by stored edge.

        Ancestry is the name, so this cannot disagree with the hierarchy.
        """
        prefix = namespace + "."
        seen = set()
        for ns in self.namespaces():
            if ns.startswith(prefix) and depth(ns) == depth(namespace) + 1:
                seen.add(ns)
        return sorted(seen)

    # -- writing ---------------------------------------------------------
    def create_version(self, m: Mutation, *, namespace: str, prompt_mode: str,
                       prompt_text: str = "", model_vars: dict[str, Any] | None = None,
                       parent_version: int | None = None, origin: str,
                       created_by: str, rationale: str = "",
                       state: str = "candidate") -> dict[str, Any]:
        """Create an immutable version of a namespace that already exists.

        This has **no power to establish a top-level namespace**, and no
        parameter that would grant it one. The previous design gated that on a
        private boolean, which meant the guarantee rested on every caller
        declining to pass a flag. The permission is now *which function you
        called*: establishing lives in :meth:`establish_root`, which no scope
        table names and no RPC verb calls.

        Creating a new version of an **existing** root is ordinary governance
        and is allowed. Ego's and Id's doctrine has to be able to change, and
        forbidding it conflated "no new top-level namespace" with "no new
        version of a root". What comes back is still a candidate, and still
        has to be validated, evaluated, proposed and approved like any other.
        """
        parts = validate_namespace(namespace)
        if len(parts) == 1 and not self.versions(namespace):
            # A top-level name with no versions would be *established* here,
            # which is bootstrap's alone. `validate_namespace` has already
            # refused any root outside ROOTS, so a brand-new top-level
            # namespace is blocked twice, independently.
            raise NamespaceError(
                f"{namespace!r} is a top-level namespace that does not exist "
                "yet; only bootstrap establishes one",
                hint="bootstrap ships the roots; runtime may version them "
                     "once they exist")
        return self._write_version(
            m, namespace=namespace, prompt_mode=prompt_mode,
            prompt_text=prompt_text, model_vars=model_vars,
            parent_version=parent_version, origin=origin,
            created_by=created_by, rationale=rationale, state=state)

    def establish_root(self, m: Mutation, *, namespace: str, prompt_mode: str,
                       prompt_text: str = "",
                       model_vars: dict[str, Any] | None = None,
                       created_by: str = "bootstrap", rationale: str = "",
                       state: str = "production_approved") -> dict[str, Any]:
        """Bring a top-level namespace into existence. Bootstrap's alone.

        Unreachable from any RPC surface: no scope table names it, no tool
        schema describes it, and `prompt_api` never calls it. That is the
        protection, and it is stronger than the boolean it replaced, because
        there is no argument that turns :meth:`create_version` into this.

        It refuses a namespace that already exists, so it cannot be used to
        slip a second `ego` past governance -- that is `create_version`'s job
        and goes through approval.
        """
        parts = validate_namespace(namespace)
        if len(parts) != 1:
            raise NamespaceError(
                f"{namespace!r} is not top-level; descendants go through "
                "create_version", namespace=namespace)
        if self.versions(namespace):
            raise NamespaceError(
                f"{namespace!r} already exists; a further version is ordinary "
                "governance, not establishment", namespace=namespace)
        return self._write_version(
            m, namespace=namespace, prompt_mode=prompt_mode,
            prompt_text=prompt_text, model_vars=model_vars,
            parent_version=None, origin="bootstrap", created_by=created_by,
            rationale=rationale, state=state)

    def _write_version(self, m: Mutation, *, namespace: str, prompt_mode: str,
                       prompt_text: str, model_vars: dict[str, Any] | None,
                       parent_version: int | None, origin: str,
                       created_by: str, rationale: str,
                       state: str) -> dict[str, Any]:
        """The mechanics, with no root policy of its own.

        Private because it is the one function here that would create
        anything: the policy that decides *what* may be created lives in the
        two public callers above.
        """
        validate_namespace(namespace)
        if origin not in ORIGINS:
            raise InvalidInput("unknown origin", origin=origin,
                               allowed=list(ORIGINS))
        if state not in STATES:
            raise InvalidInput("unknown state", state=state)

        text = validate_prompt(prompt_mode, prompt_text)
        check_no_unresolved_placeholders(text)
        variables = validate_model_vars(model_vars or {})

        parent_ns = parent_of(namespace)
        if parent_ns is None:
            if parent_version is not None:
                raise NamespaceError("a root has no parent to pin")
        else:
            if parent_version is None:
                selected = self.selected(parent_ns)
                if selected is None:
                    raise NamespaceError(
                        f"parent {parent_ns!r} has no production version to "
                        "inherit from; pin one explicitly or establish the "
                        "parent first")
                parent_version = int(selected["local_version"])
            # Pinning a version that does not exist would make the lineage a
            # claim rather than a fact.
            self.get_version(parent_ns, parent_version)

        local_version = self.next_local_version(namespace)
        version_id = new_id("pv")
        local_sha = canonical_local(prompt_mode, text, variables)

        m.sql("INSERT INTO prompt_versions(version_id, namespace, local_version,"
              " parent_namespace, parent_version, prompt_mode, prompt_text,"
              " model_vars, local_sha256, state, origin, created_by, created_at,"
              " rationale, state_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (version_id, namespace, local_version, parent_ns, parent_version,
               prompt_mode, text,
               json.dumps(variables, sort_keys=True, separators=(",", ":")),
               local_sha, state, origin, created_by, time.time(), rationale,
               m.prior_version + 1))
        m.emit(EventKind.PROMPT_NODE_INGESTED, {
            "version_id": version_id, "namespace": namespace,
            "local_version": local_version, "parent_namespace": parent_ns,
            "parent_version": parent_version, "prompt_mode": prompt_mode,
            "model_vars": variables, "local_sha256": local_sha,
            "origin": origin, "state": state, "created_by": created_by,
            "rationale": rationale[:500]})
        return {"version_id": version_id, "namespace": namespace,
                "local_version": local_version, "parent_version": parent_version,
                "local_sha256": local_sha, "state": state, "origin": origin}

    def create_runtime_version(self, m: Mutation, **kwargs: Any) -> dict[str, Any]:
        """The creation path reachable from a runtime caller.

        Identical to :meth:`create_version`, kept as a separate name so call
        sites read honestly. There is nothing left to strip: `create_version`
        cannot establish a top-level namespace under any arguments, so a
        runtime caller has no expression for it rather than a flag it is
        trusted not to set.
        """
        return self.create_version(m, **kwargs)

    def set_state(self, m: Mutation, version_id: str, new_state: str, *,
                  actor: str) -> dict[str, Any]:
        row = self.by_id(version_id)
        current = row["state"]
        if new_state not in LEGAL_TRANSITIONS.get(current, ()):
            raise InvalidInput(
                f"cannot move a prompt version from {current!r} to {new_state!r}",
                version_id=version_id, allowed=list(LEGAL_TRANSITIONS.get(current, ())))
        m.sql("UPDATE prompt_versions SET state = ? WHERE version_id = ?",
              (new_state, version_id))
        return {"version_id": version_id, "from": current, "to": new_state,
                "actor": actor}

    def select(self, m: Mutation, *, namespace: str, version_id: str,
               purpose: str, selected_by: str) -> dict[str, Any]:
        if purpose not in ("production", "experimental"):
            raise InvalidInput("purpose must be production or experimental",
                               purpose=purpose)
        row = self.by_id(version_id)
        if row["namespace"] != namespace:
            raise InvalidInput("that version belongs to another namespace",
                               namespace=namespace, version_namespace=row["namespace"])
        if row["state"] not in APPROVED:
            raise InvalidInput("only an approved version may be selected",
                               state=row["state"])
        previous = self.selected(namespace, purpose)
        m.sql("INSERT INTO prompt_selections(namespace, purpose, version_id,"
              " selected_by, selected_at, state_version) VALUES (?,?,?,?,?,?)"
              " ON CONFLICT(namespace, purpose) DO UPDATE SET version_id = excluded"
              ".version_id, selected_by = excluded.selected_by,"
              " selected_at = excluded.selected_at,"
              " state_version = excluded.state_version",
              (namespace, purpose, version_id, selected_by, time.time(),
               m.prior_version + 1))
        m.emit(EventKind.PROMPT_SELECTED, {
            "namespace": namespace, "purpose": purpose, "version_id": version_id,
            "local_version": row["local_version"],
            "previous_version_id": previous["version_id"] if previous else None,
            "selected_by": selected_by,
            "note": ("selection changes what a *new* incarnation receives; a "
                     "running role keeps the profile it was born with")})
        return {"namespace": namespace, "purpose": purpose,
                "version_id": version_id, "local_version": row["local_version"]}

    # -- ancestry --------------------------------------------------------
    def pinned_chain(self, namespace: str, local_version: int
                     ) -> list[dict[str, Any]]:
        """The exact immutable ancestry, leaf first.

        Walks the stored parent bindings rather than today's selections, which
        is the whole reason the binding is stored.
        """
        chain: list[dict[str, Any]] = []
        ns, version = namespace, int(local_version)
        seen: set[tuple[str, int]] = set()
        while True:
            if (ns, version) in seen:
                raise InvalidInput("inheritance cycle in the pinned ancestry",
                                   namespace=ns, local_version=version)
            seen.add((ns, version))
            node = self.get_version(ns, version)
            chain.append(node)
            if node["parent_namespace"] is None:
                break
            ns, version = node["parent_namespace"], int(node["parent_version"])
            if len(chain) > 16:
                raise InvalidInput("ancestry longer than any legal namespace",
                                   namespace=namespace)
        if len(chain) != depth(namespace):
            raise InvalidInput(
                "pinned ancestry does not match namespace depth",
                namespace=namespace, depth=depth(namespace), chain=len(chain))
        return chain

    def ref_for(self, namespace: str, local_version: int) -> ProfileRef:
        """Compute the lineage vector from the real pinned chain.

        Never from anything a caller supplied: a lineage string is a claim
        until the Harness derives it from the stored bindings.
        """
        chain = self.pinned_chain(namespace, local_version)
        return ProfileRef(namespace=namespace,
                          versions=tuple(int(n["local_version"]) for n in chain))

    def resolve_ref(self, ref: ProfileRef) -> list[dict[str, Any]]:
        """Verify that an explicitly requested lineage really exists.

        Every component is checked against the stored ancestry, so
        ``ego.neuocyte.research@3.7.5`` resolves to exactly that chain and is
        never reinterpreted against today's parents.
        """
        chain = self.pinned_chain(ref.namespace, ref.versions[0])
        actual = tuple(int(n["local_version"]) for n in chain)
        if actual != ref.versions:
            raise NotFound(
                "no such lineage; that namespace version has a different ancestry",
                requested=str(ref),
                actual=str(ProfileRef(ref.namespace, actual)))
        return chain
