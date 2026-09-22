"""The Prompt Library's RPC surface: read, evaluate, propose, govern.

Authority is split along the same line as everywhere else in Amoeba. Id can
see the whole library, compare lineages, evaluate a version and propose a new
one. Id cannot approve, select, or cascade. Those are the Operator's, and the
split is structural: the two sets are different verbs in different scope
tables, not one verb with a caller check inside it.

Nothing here reaches `external_io`. An external client cannot read the prompt
library, let alone change it -- a client that could see the organism's
cognitive configuration would be reading privileged internals, and one that
could propose to it would be writing the organism's mind through a public
door.

**Selection is not installation.** Approving and selecting a version changes
what the *next* incarnation is born with. A running Ego, Id or neuocyte keeps
the profile it was bound to, because its context was primed with that text and
pretending otherwise would make the incarnation binding a lie.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Sequence

from .errors import InvalidInput
from .ids import new_id, sha256_hex
from .promptlib import cascade
from .promptlib.model import (ProfileRef, fallback_output_ceiling, parse_ref,
                              validate_namespace)
from .promptlib.resolver import Resolver
from .promptlib.store import APPROVED, PromptStore
from .store.events import EventKind
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

# Visible to Id and to the Operator. Reading the family tree is how a
# proposal gets made on evidence rather than on vibes.
PROMPT_READ = (
    "prompt_tree", "prompt_versions", "prompt_resolve", "prompt_diff",
    "explain_profile", "prompt_incarnations",
)

# Id's half: judge and suggest.
ID_PROMPT = ("id_evaluate_prompt", "id_propose_profile")

# The Operator's half: decide, select, propagate. Absent from every other
# scope table, which is what makes "Id may not promote" a fact about the
# dispatcher rather than a rule Id is asked to respect.
OPERATOR_PROMPT = (
    "operator_prompt_author", "operator_prompt_state", "operator_prompt_select",
    "operator_prompt_cascade_plan", "operator_prompt_cascade",
    "operator_prompt_bootstrap_report",
)

VERDICTS = ("endorse", "concern", "oppose")


def _ref(store: PromptStore, *, namespace: str | None, profile_ref: str | None,
         local_version: int | None, purpose: str) -> ProfileRef:
    """Resolve the three ways a caller can name a profile, in priority order.

    An explicit ``profile_ref`` is a historical lineage and is honoured exactly.
    A namespace plus a version is that version's real ancestry. A bare
    namespace means "whatever is selected now", which is the only
    time-dependent reading and is labelled as such in every response.
    """
    if profile_ref:
        return parse_ref(profile_ref)
    if not namespace:
        raise InvalidInput("name a namespace or a profile_ref")
    validate_namespace(namespace)
    if local_version is not None:
        return store.ref_for(namespace, int(local_version))
    selected = store.selected(namespace, purpose)
    if selected is None:
        raise InvalidInput(f"no {purpose} version selected for {namespace!r}",
                           namespace=namespace)
    return store.ref_for(namespace, int(selected["local_version"]))


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None
    store = PromptStore(mind)
    resolver = Resolver(store)

    # ==================================================================
    # Reading
    # ==================================================================
    def prompt_tree(*, include_versions: bool = False) -> dict[str, Any]:
        """The family tree: every namespace, what is selected, what is pending."""
        nodes = []
        for namespace in store.namespaces():
            versions = store.versions(namespace)
            selected = store.selected(namespace)
            experimental = store.selected(namespace, "experimental")
            node: dict[str, Any] = {
                "namespace": namespace,
                "parent": namespace.rpartition(".")[0] or None,
                "children": store.children_of(namespace),
                "version_count": len(versions),
                "latest_version": versions[-1]["local_version"] if versions else None,
                "selected": (store.ref_for(namespace,
                                           int(selected["local_version"])).to_dict()
                             if selected else None),
                "experimental": (str(store.ref_for(
                    namespace, int(experimental["local_version"])))
                    if experimental else None),
                "pending": [{"local_version": v["local_version"],
                             "version_id": v["version_id"], "state": v["state"],
                             "origin": v["origin"], "created_by": v["created_by"]}
                            for v in versions
                            if v["state"] not in APPROVED
                            and v["state"] not in ("rejected", "retired")],
            }
            if include_versions:
                node["versions"] = [
                    {k: v[k] for k in ("version_id", "local_version",
                                       "parent_version", "prompt_mode", "state",
                                       "origin", "created_by", "created_at",
                                       "local_sha256")}
                    for v in versions]
            nodes.append(node)
        return {"roots": [n for n in nodes if n["parent"] is None],
                "nodes": nodes,
                "note": ("a child pins an exact parent version, so approving a "
                         "new parent changes no existing descendant until it "
                         "is cascaded")}

    def prompt_versions(*, namespace: str) -> dict[str, Any]:
        """Every version of one namespace, with its lineage and evaluations."""
        validate_namespace(namespace)
        out = []
        for v in store.versions(namespace):
            row = dict(v)
            row["prompt_chars"] = len(row.pop("prompt_text", "") or "")
            try:
                row["profile_ref"] = str(store.ref_for(namespace,
                                                       int(v["local_version"])))
            except Exception as exc:               # broken ancestry is reportable
                row["profile_ref"] = None
                row["lineage_error"] = str(exc)
            out.append(row)
        selected = store.selected(namespace)
        return {"namespace": namespace, "versions": out,
                "selected_version_id": selected["version_id"] if selected else None,
                "evaluations": _evaluations(namespace)}

    def _evaluations(namespace: str) -> list[dict[str, Any]]:
        rows = mind.db.conn.execute(
            "SELECT e.* FROM prompt_evaluations e JOIN prompt_versions v"
            " ON v.version_id = e.version_id WHERE v.namespace = ?"
            " ORDER BY e.created_at DESC LIMIT 50", (namespace,))
        return [dict(r) for r in rows]

    def prompt_resolve(*, namespace: str | None = None,
                       profile_ref: str | None = None,
                       local_version: int | None = None,
                       purpose: str = "production") -> dict[str, Any]:
        """What a new incarnation of this profile would actually receive."""
        ref = _ref(store, namespace=namespace, profile_ref=profile_ref,
                   local_version=local_version, purpose=purpose)
        resolved = resolver.resolve_ref(ref)
        return {**resolved.to_dict(),
                "backend_arguments": resolved.backend_kwargs(),
                "selected_now": bool(
                    namespace and not profile_ref and local_version is None),
                "note": ("this is what the next incarnation would be born "
                         "with; running minds keep what they were bound to")}

    def explain_profile(*, namespace: str | None = None,
                        profile_ref: str | None = None,
                        local_version: int | None = None,
                        purpose: str = "production") -> dict[str, Any]:
        """Where every instruction and every setting came from, level by level."""
        ref = _ref(store, namespace=namespace, profile_ref=profile_ref,
                   local_version=local_version, purpose=purpose)
        return resolver.explain(ref)

    def prompt_diff(*, left: str, right: str) -> dict[str, Any]:
        """Compare two lineages -- what actually differs in the resolved result.

        Version numbers differing is not a difference in cognition; a rebase
        changes the lineage vector and may change nothing a mind receives.
        This answers the question that matters.
        """
        a = resolver.resolve_ref(parse_ref(left))
        b = resolver.resolve_ref(parse_ref(right))
        changed_vars = {}
        for name in sorted(set(a.model_vars) | set(b.model_vars)):
            if a.model_vars.get(name) != b.model_vars.get(name):
                changed_vars[name] = {"left": a.model_vars.get(name),
                                      "right": b.model_vars.get(name),
                                      "left_from": a.var_source.get(name),
                                      "right_from": b.var_source.get(name)}
        return {
            "left": str(a.ref), "right": str(b.ref),
            "prompt_identical": a.prompt_sha256 == b.prompt_sha256,
            "config_identical": a.config_sha256 == b.config_sha256,
            "profile_identical": a.profile_sha256 == b.profile_sha256,
            "prompt_chars": {"left": len(a.prompt_text), "right": len(b.prompt_text)},
            "model_vars_changed": changed_vars,
            "lineage": {"left": [str(c.namespace) + "@" + str(c.local_version)
                                 for c in a.contributions],
                        "right": [str(c.namespace) + "@" + str(c.local_version)
                                  for c in b.contributions]},
            "note": ("identical prompt and config digests mean the two "
                     "lineages produce the same cognition despite different "
                     "version numbers"),
        }

    def prompt_incarnations(*, namespace: str | None = None, limit: int = 50
                            ) -> dict[str, Any]:
        """Which minds were actually born with which profile.

        Read from the frozen bindings, not recomputed. A binding records the
        resolved digests as they were at birth, so this stays true even after
        the library moves on.
        """
        sql = ("SELECT binding_id, actor_id, actor_kind, incarnation, work_id,"
               " namespace, profile_ref, prompt_sha256, config_sha256,"
               " profile_sha256, model_generation, effective_settings, created_at"
               " FROM incarnation_profiles")
        params: tuple[Any, ...] = ()
        if namespace:
            validate_namespace(namespace)
            sql += " WHERE namespace = ?"
            params = (namespace,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        rows = mind.db.conn.execute(sql, (*params, max(1, min(int(limit), 500))))
        return {"bindings": [dict(r) for r in rows]}

    # ==================================================================
    # Birth: freezing what a mind actually received
    # ==================================================================
    def _harness_constraints(namespace: str, resolved: Any) -> dict[str, Any]:
        """What the Harness supplies on top of the profile, on the record.

        It never narrows a ceiling a profile states: the governed value is the
        one canonical ceiling, and a Harness default quietly lowering it is
        exactly how Ego was granted 3072 and given 512. It only fills in a
        ceiling a profile does not state -- an approved root that predates
        the setting -- with the shipped value for that namespace. That lands
        here, in the binding's `harness_constraints`, so the record says the
        number was supplied rather than chosen, and startup says so aloud.
        """
        if "max_output_tokens" in (resolved.model_vars or {}):
            return {}
        return {"max_output_tokens": fallback_output_ceiling(namespace)}

    def bind_profile(*, namespace: str, actor_id: str, actor_kind: str,
                     work_id: str | None = None, purpose: str = "production",
                     model_generation: str = "",
                     inherited_namespace: str | None = None,
                     operation_id: str | None = None) -> dict[str, Any]:
        """Resolve a profile and freeze exactly what this incarnation got.

        Called at birth, before the mind primes its context, because the
        digest reported at registration has to be the digest of the text
        actually about to be used.

        The binding stores the resolved text's digest, the effective settings
        and the full lineage vector, rather than a pointer to be re-resolved
        later. Cognition that happened must stay explicable from what the
        organism held at the time, and re-resolving against a library that has
        since moved would quietly rewrite history.

        Note what is *not* a parameter: harness constraints. A caller that
        could widen its own ceiling by asking would not be constrained.
        """
        if actor_kind not in ("ego", "id", "neuocyte"):
            raise InvalidInput("unknown actor kind", actor_kind=actor_kind,
                               allowed=["ego", "id", "neuocyte"])
        validate_namespace(namespace)
        selected = store.selected(namespace, purpose)
        if selected is None:
            raise InvalidInput(
                f"no {purpose} profile selected for {namespace!r}",
                namespace=namespace,
                hint="bootstrap establishes a baseline; approve and select a "
                     "version for anything beyond it")
        resolved = resolver.resolve_version(namespace,
                                            int(selected["local_version"]))
        constraints = _harness_constraints(namespace, resolved)
        effective = resolved.effective_settings(constraints)
        # A ceiling above the platform cap is a contradiction in the
        # configuration, and it is refused here rather than clamped somewhere
        # later: a clamp is precisely how a governed number becomes a lie.
        ceiling = effective.get("max_output_tokens")
        platform_cap = int(sup.cfg.arbiter.max_completion_tokens)
        if ceiling is not None and int(ceiling) > platform_cap:
            raise InvalidInput(
                f"{namespace} states an output ceiling of {ceiling}, above the "
                f"platform cap of {platform_cap}",
                namespace=namespace, profile_ref=str(resolved.ref),
                max_output_tokens=int(ceiling), platform_cap=platform_cap,
                hint="lower the profile's ceiling or raise "
                     "[arbiter] max_completion_tokens")
        # A neuocyte forked from an Ego snapshot already holds its ancestors'
        # text. It injects only the suffix, and the binding records both facts
        # separately rather than claiming the whole profile was handed over.
        inject_text = resolved.prompt_text
        inherited_sha = None
        if inherited_namespace:
            inject_text = resolved.suffix_after(inherited_namespace)
            inherited_sha = sha256_hex(
                resolved.prompt_text[:len(resolved.prompt_text)
                                     - len(inject_text)].encode("utf-8"))
        inject_sha = sha256_hex(inject_text.encode("utf-8"))
        binding_id = new_id("bnd")
        lineage = [{"namespace": c.namespace, "local_version": c.local_version,
                    "version_id": c.version_id, "prompt_mode": c.prompt_mode}
                   for c in resolved.contributions]

        def body(m: Mutation) -> None:
            m.sql("INSERT INTO incarnation_profiles(binding_id, actor_id,"
                  " actor_kind, incarnation, work_id, namespace, profile_ref,"
                  " lineage, prompt_sha256, config_sha256, profile_sha256,"
                  " model_generation, effective_settings, harness_constraints,"
                  " created_at, state_version)"
                  " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (binding_id, actor_id, actor_kind, None, work_id, namespace,
                   str(resolved.ref), json.dumps(lineage),
                   resolved.prompt_sha256, resolved.config_sha256,
                   resolved.profile_sha256, model_generation,
                   json.dumps(effective, sort_keys=True),
                   json.dumps(constraints, sort_keys=True), time.time(),
                   m.prior_version + 1))
            m.emit(EventKind.INCARNATION_BOUND, {
                "binding_id": binding_id, "actor_id": actor_id,
                "actor_kind": actor_kind, "work_id": work_id,
                "profile_ref": str(resolved.ref), "lineage": lineage,
                "prompt_sha256": resolved.prompt_sha256,
                "config_sha256": resolved.config_sha256,
                "profile_sha256": resolved.profile_sha256,
                "effective_settings": effective,
                "harness_constraints": constraints,
                "injected_sha256": inject_sha,
                "inherited_from": inherited_namespace,
                "inherited_prefix_sha256": inherited_sha,
                "note": ("frozen at birth; this incarnation keeps these bytes "
                         "even if the library moves on"
                         + (f"; the {inherited_namespace} prefix was inherited "
                            "physically from the forked context rather than "
                            "injected" if inherited_namespace else ""))})

        receipt, _ = mind.writer.apply(body, actor=actor_id,
                                       operation_id=operation_id)
        return {"binding_id": binding_id, "profile_ref": str(resolved.ref),
                "namespace": namespace, "prompt_text": resolved.prompt_text,
                "inject_text": inject_text, "injected_sha256": inject_sha,
                "inherited_from": inherited_namespace,
                "prompt_sha256": resolved.prompt_sha256,
                "config_sha256": resolved.config_sha256,
                "profile_sha256": resolved.profile_sha256,
                "model_vars": resolved.model_vars,
                "effective_settings": effective,
                "backend_arguments": resolved.backend_kwargs(constraints),
                "lineage": lineage, "receipt_id": receipt.receipt_id}

    # ==================================================================
    # Id: evaluate and propose
    # ==================================================================
    def id_evaluate_prompt(*, version_id: str, verdict: str, notes: str = "",
                           evidence: Sequence[dict[str, Any]] = (),
                           operation_id: str | None = None) -> dict[str, Any]:
        """Id's judgement on a candidate. Advisory, durable, attributed.

        It does not move the version's state on its own. An endorsement that
        promoted would make Id the approver by a longer route.
        """
        if verdict not in VERDICTS:
            raise InvalidInput("unknown verdict", verdict=verdict,
                               allowed=list(VERDICTS))
        row = store.by_id(version_id)
        evaluation_id = new_id("pev")

        def body(m: Mutation) -> dict[str, Any]:
            m.sql("INSERT INTO prompt_evaluations(evaluation_id, version_id,"
                  " evaluator, verdict, notes, evidence, created_at, state_version)"
                  " VALUES (?,?,?,?,?,?,?,?)",
                  (evaluation_id, version_id, "id", verdict, notes[:4000],
                   json.dumps(list(evidence)[:20]), time.time(),
                   m.prior_version + 1))
            m.emit(EventKind.PROMPT_EVALUATED, {
                "evaluation_id": evaluation_id, "version_id": version_id,
                "namespace": row["namespace"],
                "local_version": row["local_version"], "verdict": verdict,
                "notes": notes[:2000], "evaluator": "id",
                "note": "advisory; the Operator decides"})
            return {}

        receipt, _ = mind.writer.apply(body, actor="id", operation_id=operation_id)
        return {"evaluation_id": evaluation_id, "version_id": version_id,
                "verdict": verdict, "receipt_id": receipt.receipt_id,
                "status": "recorded",
                "note": "advisory only; this did not change the version's state"}

    def id_propose_profile(*, namespace: str, prompt_mode: str,
                           prompt_text: str = "",
                           model_vars: dict[str, Any] | None = None,
                           parent_version: int | None = None,
                           rationale: str = "",
                           operation_id: str | None = None) -> dict[str, Any]:
        """Id proposes a new version of a descendant profile.

        Created as a ``candidate``: it is written down, it is not adopted. Id
        cannot author a root -- ``create_runtime_version`` has no parameter
        that would let it, so proposing a new ``ego`` is not a request that
        gets refused, it is a sentence Id cannot say.
        """
        if not rationale.strip():
            raise InvalidInput(
                "a prompt proposal needs a rationale",
                hint="the reasoning is what the Operator is actually judging")

        def body(m: Mutation) -> dict[str, Any]:
            return store.create_runtime_version(
                m, namespace=namespace, prompt_mode=prompt_mode,
                prompt_text=prompt_text, model_vars=model_vars or {},
                parent_version=parent_version, origin="id", created_by="id",
                rationale=rationale[:4000], state="candidate")

        receipt, created = mind.writer.apply(body, actor="id",
                                             operation_id=operation_id)
        return {**created, "receipt_id": receipt.receipt_id,
                "profile_ref": str(store.ref_for(namespace,
                                                 created["local_version"])),
                "status": "candidate",
                "note": ("recorded as a candidate; nothing running changed and "
                         "nothing will until the Operator approves and selects "
                         "it")}

    # ==================================================================
    # Operator: decide, select, propagate
    # ==================================================================
    def operator_prompt_author(*, namespace: str, prompt_mode: str,
                               prompt_text: str = "",
                               model_vars: dict[str, Any] | None = None,
                               parent_version: int | None = None,
                               rationale: str = "",
                               operation_id: str | None = None) -> dict[str, Any]:
        """The Operator writes a candidate directly.

        Still a candidate, and still not a root: the Operator governs the
        library through this surface, while the *origin* of a root stays the
        shipped bootstrap files, which are reviewable in the repository rather
        than typed into a running system.
        """
        def body(m: Mutation) -> dict[str, Any]:
            return store.create_runtime_version(
                m, namespace=namespace, prompt_mode=prompt_mode,
                prompt_text=prompt_text, model_vars=model_vars or {},
                parent_version=parent_version, origin="operator",
                created_by="operator", rationale=rationale[:4000],
                state="candidate")

        receipt, created = mind.writer.apply(body, actor="operator",
                                             operation_id=operation_id)
        return {**created, "receipt_id": receipt.receipt_id,
                "profile_ref": str(store.ref_for(namespace,
                                                 created["local_version"]))}

    def operator_prompt_state(*, version_id: str, state: str, rationale: str = "",
                              operation_id: str | None = None) -> dict[str, Any]:
        """Advance a version through the governance state machine."""
        row = store.by_id(version_id)
        decision_id = new_id("pdc")

        def body(m: Mutation) -> dict[str, Any]:
            moved = store.set_state(m, version_id, state, actor="operator")
            terminal = state in (*APPROVED, "rejected", "retired")
            if terminal:
                m.sql("INSERT INTO prompt_decisions(decision_id, version_id,"
                      " decision, decided_by, rationale, propagation, created_at,"
                      " state_version) VALUES (?,?,?,?,?,?,?,?)",
                      (decision_id, version_id, state, "operator",
                       rationale[:4000], None, time.time(), m.prior_version + 1))
            kind = (EventKind.PROMPT_REJECTED if state == "rejected"
                    else EventKind.PROMPT_DECIDED)
            m.emit(kind, {
                "decision_id": decision_id if terminal else None,
                "version_id": version_id, "namespace": row["namespace"],
                "local_version": row["local_version"], "decision": state,
                "from_state": moved["from"], "decided_by": "operator",
                "rationale": rationale[:2000],
                "note": ("approval makes a version selectable; it does not "
                         "select it, and it changes no running mind")})
            return moved

        receipt, moved = mind.writer.apply(body, actor="operator",
                                           operation_id=operation_id)
        return {**moved, "receipt_id": receipt.receipt_id,
                "selected": False,
                "note": ("approved versions still have to be selected; nothing "
                         "running changed")}

    def operator_prompt_select(*, namespace: str, version_id: str,
                               purpose: str = "production",
                               operation_id: str | None = None) -> dict[str, Any]:
        """Choose which approved version new incarnations are born with.

        Any approved version may be chosen, including an older one: rolling
        back is selecting a historical lineage, not reconstructing it.
        """
        def body(m: Mutation) -> dict[str, Any]:
            return store.select(m, namespace=namespace, version_id=version_id,
                                purpose=purpose, selected_by="operator")

        receipt, out = mind.writer.apply(body, actor="operator",
                                         operation_id=operation_id)
        return {**out, "receipt_id": receipt.receipt_id,
                "profile_ref": str(store.ref_for(namespace, out["local_version"])),
                "affects": "incarnations born from now on",
                "note": ("running Ego, Id and neuocytes keep the profile they "
                         "were bound to until they are reborn")}

    def operator_prompt_cascade_plan(*, namespace: str, local_version: int,
                                     mode: str = "queue") -> dict[str, Any]:
        """Preview exactly which descendants a cascade would rebase."""
        return cascade.plan(store, namespace, int(local_version), mode=mode)

    def operator_prompt_cascade(*, namespace: str, local_version: int,
                                mode: str = "queue", rationale: str = "",
                                operation_id: str | None = None) -> dict[str, Any]:
        """Propagate a parent version down the subtree.

        The plan is recomputed here rather than taken from the caller. A plan
        the console rendered a minute ago describes a library that may have
        moved, and executing a stale description of the tree is how a cascade
        would pin a version that no longer means what it did.
        """
        plan_result = cascade.plan(store, namespace, int(local_version), mode=mode)

        def body(m: Mutation) -> dict[str, Any]:
            return cascade.apply_plan(m, store, plan_result, actor="operator",
                                      rationale=rationale)

        receipt, out = mind.writer.apply(body, actor="operator",
                                         operation_id=operation_id)
        return {**out, "receipt_id": receipt.receipt_id,
                "plan": plan_result,
                "note": ("local definitions were copied unchanged; only each "
                         "node's pinned parent differs")}

    def operator_prompt_bootstrap_report() -> dict[str, Any]:
        """What the shipped files did at the last startup, and what they are now.

        Shows deltas still sitting as candidates, which is the answer to "I
        edited a prompt file and restarted -- why is nothing different".
        """
        from .promptlib import bootstrap as bs
        from .store.events import read_events

        events = []
        for ev in read_events(mind.db.conn,
                              kinds=[EventKind.PROMPT_BOOTSTRAP_BASELINE,
                                     EventKind.PROMPT_BOOTSTRAP_MATCHED,
                                     EventKind.PROMPT_BOOTSTRAP_DELTA],
                              limit=200):
            events.append({"kind": ev.kind, "seq": ev.seq, "ts": ev.ts,
                           **(ev.payload(mind.blobs) or {})})
        pending = []
        for ev in events:
            if ev["kind"] != EventKind.PROMPT_BOOTSTRAP_DELTA:
                continue
            try:
                row = store.by_id(ev["version_id"])
            except Exception:
                continue
            if row["state"] == "candidate":
                pending.append({"namespace": row["namespace"],
                                "local_version": row["local_version"],
                                "version_id": row["version_id"],
                                "source": ev.get("source")})
        return {"prompt_dir": str(bs.PROMPT_DIR), "events": events[:100],
                "pending_file_deltas": pending,
                "note": ("an edited prompt file becomes a candidate; it does "
                         "not take effect until it is approved and selected")}

    return {
        "prompt_tree": prompt_tree,
        "prompt_versions": prompt_versions,
        "prompt_resolve": prompt_resolve,
        "prompt_diff": prompt_diff,
        "explain_profile": explain_profile,
        "prompt_incarnations": prompt_incarnations,
        "bind_profile": bind_profile,
        "id_evaluate_prompt": id_evaluate_prompt,
        "id_propose_profile": id_propose_profile,
        "operator_prompt_author": operator_prompt_author,
        "operator_prompt_state": operator_prompt_state,
        "operator_prompt_select": operator_prompt_select,
        "operator_prompt_cascade_plan": operator_prompt_cascade_plan,
        "operator_prompt_cascade": operator_prompt_cascade,
        "operator_prompt_bootstrap_report": operator_prompt_bootstrap_report,
    }
