"""The supervisor's RPC surface.

Two layers live here:

* **State operations** -- the only write path into durable state, used by Ego,
  Id and neuocytes. Every mutation returns a receipt.
* **Cognitive verbs** -- ``ego_converse``, ``id_audit`` and friends. These
  proxy to the long-lived role processes and then commit the result. Because
  they run here rather than in the caller, an MCP client hanging up mid-call
  cannot abandon an operation half-committed.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from .arbiter import ResourceSnapshot
from .errors import (
    BackendUnavailable, InvalidInput, MindError, NotFound, ResourceExhausted,
)
from .ids import new_id
from .store.events import EventKind, read_events
from .store.work_repo import TERMINAL_WORK
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

SCHEMA_VERSION = "1.0.0"


VERDICTS = ("supported", "contested", "unsupported", "inconclusive")
_VERDICT_RE = re.compile(
    r"\b(unsupported|inconclusive|contested|supported)\b")

def _evidence_basis_digest(dossier: Any) -> str:
    """A digest of what an audit was actually performed against.

    Taken from the measured dossier rather than from anything Id says it
    looked at, because the point is to tell a changed world from a changed
    mind.
    """
    from .ids import sha256_hex

    return sha256_hex(json.dumps(dossier, sort_keys=True, default=str)
                      .encode("utf-8", "replace"))

def _settle_if_the_ground_moved(mind: Any, conclusion_id: str, basis: str,
                                audit_id: str, op_id: str | None = None
                                ) -> dict[str, Any]:
    """A supporting audit closes a dispute only if the evidence moved.

    Id reversing itself about the same evidence is not a resolution -- it
    is the adjudicator changing its mind, and letting it close the dispute
    would make the audit self-certifying. So the basis digest has to
    differ from the one recorded when the dispute opened.

    When it does not differ, the contradiction is recorded rather than
    discarded. Two opposite verdicts against identical evidence is a fact
    about the organism's own reasoning, and it is exactly the sort of
    thing it should be able to notice about itself.
    """
    live = mind.memory.open_disagreement_for(
        subject_kind="conclusion", subject_id=conclusion_id)
    if live is None:
        return {}
    opened_on = live.get("evidence_basis_digest")
    if opened_on and opened_on == basis:
        mind.writer.apply(
            lambda m: m.emit(EventKind.AUDIT_SELF_CONTRADICTED, {
                "disagreement_id": live["disagreement_id"],
                "conclusion_id": conclusion_id, "audit_id": audit_id,
                "evidence_basis_digest": basis,
                "note": ("a later audit reached the opposite verdict "
                         "against an identical evidence basis; the "
                         "dispute stays open, because the adjudicator "
                         "changing its mind is not the ground moving")}),
            actor="id", operation_id=op_id, bump_version=False)
        return {"disagreement_id": live["disagreement_id"],
                "disagreement_resolved": False,
                "self_contradicted": True,
                "note": ("the evidence basis is unchanged, so this "
                         "reversal does not settle the dispute")}
    mind.memory.resolve_disagreement(
        disagreement_id=live["disagreement_id"],
        resolution="resolved_supported", actor="harness",
        detail=(f"a later audit ({audit_id}) supported the claim against a "
                f"changed evidence basis"),
        operation_id=op_id)
    return {"disagreement_id": live["disagreement_id"],
            "disagreement_resolved": True,
            "resolution": "resolved_supported"}

def _parse_audit(text: str) -> dict[str, Any]:
    """Pull VERDICT / FINDING / UNRESOLVED out of an audit turn.

    Matched on **whole words**. Substring matching read `unsupported` as
    `supported` -- the word contains it -- so an adverse audit became a
    favourable durable verdict, and the disagreement it should have opened
    never was, because that branch tests for `unsupported`. An audit that
    silently inverts is worse than no audit at all.

    An answer naming more than one verdict is treated as unstated rather
    than resolved by ordering: a model echoing the menu back
    ("supported | contested | ...") has not reached a judgement, and
    picking the first one would invent a finding out of formatting.

    Whether a verdict was actually stated is reported separately, so an
    unparsed reply stays distinguishable from a judged one.
    """
    verdict, finding, unresolved, stated = "inconclusive", "", "", False
    for line in (text or "").splitlines():
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            candidate = line.split(":", 1)[1].strip().lower()
            found = {m.group(1) for m in _VERDICT_RE.finditer(candidate)}
            if len(found) == 1:
                verdict, stated = found.pop(), True
        elif upper.startswith("FINDING:"):
            finding = line.split(":", 1)[1].strip()
        elif upper.startswith("UNRESOLVED:"):
            unresolved = line.split(":", 1)[1].strip()
    return {"verdict": verdict, "finding": finding,
            "unresolved": unresolved, "verdict_stated": stated}


AUDIT_FRAMING = ("Use audit_dossier to resolve the recorded evidence, and judge "
                 "only from what the record shows. Finish with exactly three "
                 "lines:\n"
                 "VERDICT: supported | contested | unsupported | inconclusive\n"
                 "FINDING: <one sentence>\n"
                 "UNRESOLVED: <what the record does not establish>")


def commit_audit(mind: Any, dossier: dict[str, Any], *, conclusion_id: str | None,
                 operation_id: str | None, focus: str, text: str,
                 op_id: str) -> dict[str, Any]:
    """Record what Id concluded about a conclusion, and what follows from it.

    The one place an audit verdict becomes durable: the audit itself, a
    disagreement when it is contested, and settlement when a supporting audit
    stands on changed ground (I102). It used to live inside `id_audit`'s
    waiter, which meant an audit nobody waited for -- the only kind a wake
    can produce -- had its verdict produced and then discarded. Now the
    waiter and the completion hook both call this, so there is one
    definition of what an audit does.
    """
    target = conclusion_id or operation_id
    parsed = _parse_audit(text)
    findings = [parsed["finding"]] if parsed["finding"] else []
    out: dict[str, Any] = {
        "verdict": parsed["verdict"], "findings": findings,
        "unresolved": parsed["unresolved"],
        "finding": parsed["finding"] or text,
        "verdict_stated": parsed["verdict_stated"],
    }
    audit_id, _ = mind.memory.record_audit(
        target_kind="conclusion" if conclusion_id else "operation",
        target_id=target, verdict=out["verdict"], focus=focus or None,
        findings=findings, unresolved=out["unresolved"],
        evidence={"hash_chain_ok": dossier.get("hash_chain_ok"),
                  "event_count": len(dossier.get("events", [])),
                  "unresolved_content": dossier.get("unresolved_content", [])},
        operation_id=op_id)
    out["audit_id"] = audit_id
    basis = _evidence_basis_digest(dossier)
    out["evidence_basis_digest"] = basis
    # A contested audit of an Ego conclusion is a real disagreement, and it is
    # recorded as one rather than quietly overwriting the claim.
    if conclusion_id and out["verdict"] in ("contested", "unsupported"):
        concl = mind.memory.get_conclusion(conclusion_id)
        did, _ = mind.memory.open_disagreement(
            subject_kind="conclusion", subject_id=conclusion_id,
            claim_a=concl["claim"], actor_a=concl["produced_by"],
            claim_b="; ".join(findings or ["contested"]), actor_b="id",
            evidence_a={"conclusion_evidence": concl.get("evidence", [])},
            evidence_b={"audit_id": audit_id},
            evidence_basis_digest=basis, operation_id=op_id)
        out["disagreement_id"] = did
    elif conclusion_id and out["verdict"] == "supported":
        out.update(_settle_if_the_ground_moved(
            mind, conclusion_id, basis, audit_id, op_id))
    return out


def build(sup: "Supervisor") -> dict[str, Any]:
    mind = sup.mind
    assert mind is not None

    # ------------------------------------------------------------------
    # health and status
    # ------------------------------------------------------------------
    def health() -> dict[str, Any]:
        """Stays answerable while inference is failing or saturated."""
        children: dict[str, Any] = {}
        for name in ("inference", "ego", "id"):
            proc = sup.procs.get(name)
            entry: dict[str, Any] = {
                "pid": proc.pid if proc else None,
                "running": bool(proc and proc.poll() is None),
            }
            try:
                # probe=True: a down child must not make this call block for
                # the patient reconnect window.
                entry["reported"] = sup.client(name, probe=True).call("health")
            except Exception as exc:  # noqa: BLE001
                entry["reported"] = None
                entry["error"] = repr(exc)
            children[name] = entry
        snap = sup.resource_snapshot()
        return {
            "schema_version": SCHEMA_VERSION,
            "service": "supervisor",
            "status": "alive",
            "uptime_seconds": time.time() - sup.started_at,
            "run_id": mind.run_id,
            "state_version": mind.state_version(),
            "children": children,
            "resources": {
                "active_neuocytes": snap.active_neuocytes,
                "active_user_neuocytes": snap.active_user_neuocytes,
                "active_maintenance_neuocytes": snap.active_maintenance_neuocytes,
                "queued_user": snap.queued_user,
                "queued_maintenance": snap.queued_maintenance,
                "outstanding_work": snap.outstanding_work,
                "oldest_queued_age": snap.oldest_queued_age,
                "inference_sessions": snap.inference_sessions,
                "max_inference_sessions": snap.max_inference_sessions,
                "vram_free_bytes": snap.vram_free_bytes,
            },
            "limits": {
                "max_neuocytes": sup.cfg.arbiter.max_neuocytes,
                "max_outstanding_work": sup.cfg.arbiter.max_outstanding_work,
                "max_prompt_tokens": sup.cfg.arbiter.max_prompt_tokens,
                "max_completion_tokens": sup.cfg.arbiter.max_completion_tokens,
                "max_maintenance_depth": sup.cfg.arbiter.max_maintenance_depth,
                "max_maintenance_per_hour": sup.cfg.arbiter.max_maintenance_per_hour,
            },
            "fairness": sup.arbiter.fairness_state(),
            "supervision": {
                "passes": sup._supervision_passes,
                "seconds_since_last_pass": (time.time() - sup._supervision_last
                                            if sup._supervision_last else None),
            },
        }

    def debug_threads() -> dict[str, Any]:
        """Stack of every live thread in the supervisor.

        A diagnostic for exactly the situation it was written for: a loop that
        stops making progress and gives no clue why. Cheap, read-only, and far
        better than inferring a deadlock from the outside.
        """
        import sys as _sys
        import threading as _th
        import traceback as _tb

        names = {t.ident: t.name for t in _th.enumerate()}
        out = {}
        for ident, frame in _sys._current_frames().items():
            out[names.get(ident, str(ident))] = [
                f"{fn}:{ln} {func}" for fn, ln, func, _ in
                _tb.extract_stack(frame)[-6:]]
        return {"threads": out, "count": len(out)}

    def capabilities() -> dict[str, Any]:
        try:
            caps = sup.client("inference").call("capabilities")
        except Exception as exc:  # noqa: BLE001
            caps = {"error": repr(exc), "backend_kind": "unavailable"}
        caps["schema_version"] = SCHEMA_VERSION
        caps["supervisor_run_id"] = mind.run_id
        return caps

    def status() -> dict[str, Any]:
        return mind.status()

    # ------------------------------------------------------------------
    # agents
    # ------------------------------------------------------------------
    def register_agent(*, agent_id: str, role: str, pid: int | None = None,
                       session_handle: str | None = None, snapshot_id: str | None = None,
                       model_generation: str | None = None, work_id: str | None = None,
                       prompt_sha256: str | None = None,
                       profile_binding_id: str | None = None) -> dict[str, Any]:
        incarnation, receipt = mind.work.register_agent(
            agent_id=agent_id, role=role, pid=pid, session_handle=session_handle,
            snapshot_id=snapshot_id, model_generation=model_generation, work_id=work_id,
        )
        if profile_binding_id:
            # The profile was bound a moment ago, before this incarnation had
            # a number. Stamping it here is what lets a transcript be traced
            # back to the exact prompt bytes that produced it.
            #
            # Through the writer, not a raw execute. The writer owns this
            # connection: a bare `conn.commit()` from outside it can land in
            # the middle of another mutation's transaction and commit half of
            # it. The first version of this did exactly that, and the visible
            # symptom was bindings left with a NULL incarnation -- the stamp
            # itself being lost was the *harmless* half of the bug.
            def _stamp(m: Mutation) -> None:
                m.sql("UPDATE incarnation_profiles SET incarnation = ?,"
                      " model_generation = COALESCE(NULLIF(model_generation, ''), ?)"
                      " WHERE binding_id = ? AND actor_id = ?",
                      (incarnation, model_generation or "", profile_binding_id,
                       agent_id))

            mind.writer.apply(_stamp, actor=agent_id, bump_version=False)
        if prompt_sha256 and role in ("ego", "id"):
            # What this incarnation is actually running, which is not
            # necessarily what the configuration now says.
            sup.role_prompt_digest[role] = prompt_sha256
        return {"agent_id": agent_id, "incarnation": incarnation,
                "receipt_id": receipt.receipt_id,
                "profile_binding_id": profile_binding_id}

    def role_environment(*, role: str, incarnation: int | None = None,
                         profile_ref: str | None = None,
                         prompt_sha256: str | None = None,
                         config_sha256: str | None = None,
                         trigger: str = "",
                         operation_id: str | None = None) -> dict[str, Any]:
        """Build, record and return this turn's authoritative environment.

        The role asks for it at the start of a bounded turn and uses the
        answer for the whole turn. It is deliberately *not* `system_pulse`:
        the pulse is Id's live physiological telemetry, this is the cognitive
        operating environment -- what profiles exist, what this role may
        invoke, and which resource identities produced the result.

        The manifest's exact bytes are content-addressed before they are
        returned, and the turn event references that digest. A digest whose
        content cannot be recovered is not provenance, so the bytes go to the
        blob store rather than being recomputed later from state that has
        since moved.

        Identical environments across turns resolve to the same blob, which is
        the point of content addressing: a quiet organism does not accumulate
        one copy of an unchanged world per turn.
        """
        from . import role_env

        manifest = role_env.build(
            sup, role, incarnation=incarnation,
            profile={"profile_ref": profile_ref, "prompt_sha256": prompt_sha256,
                     "config_sha256": config_sha256})
        text = role_env.render(manifest)
        body_bytes = role_env._canon(manifest)

        def _record(m: Mutation) -> dict[str, Any]:
            blob = m.put_blob(body_bytes, encoding="application/json",
                              schema="amoeba.role_environment/1")
            m.emit(EventKind.ROLE_ENVIRONMENT_BUILT, {
                "role": role, "incarnation": incarnation,
                "environment_sha256": manifest["environment_sha256"],
                "environment_blob": blob,
                "capability_count": len(manifest["capabilities"]),
                "profile_count": len(manifest["available_profiles"])})
            m.emit(EventKind.ROLE_TURN_BEGAN, {
                "role": role, "incarnation": incarnation,
                "profile_ref": profile_ref, "prompt_sha256": prompt_sha256,
                "config_sha256": config_sha256,
                "environment_sha256": manifest["environment_sha256"],
                "environment_blob": blob,
                "trigger": trigger[:500],
                "model_generation": manifest["resources"].get("model_generation"),
                "note": ("profile + environment + trigger is everything this "
                         "turn's cognition was produced from")})
            return {"environment_blob": blob}

        receipt, recorded = mind.writer.apply(
            _record, actor=role, operation_id=operation_id, bump_version=False)
        return {"manifest": manifest, "text": text,
                "environment_sha256": manifest["environment_sha256"],
                "environment_blob": recorded["environment_blob"],
                "receipt_id": receipt.receipt_id}

    def heartbeat(*, agent_id: str) -> dict[str, Any]:
        """Liveness, and the session the record says this agent is using.

        Carried on every beat so a role that missed a handover heals itself
        within seconds instead of calling a closed session forever. The
        record is the authority here: it is what the Harness checkpoints and
        rebuilds from, and a role holding something else is the one that is
        wrong.
        """
        mind.work.heartbeat(agent_id)
        row = mind.db.conn.execute(
            "SELECT session_handle FROM agents WHERE agent_id = ?",
            (agent_id,)).fetchone()
        return {"ok": True, "at": time.time(),
                "session_handle": row["session_handle"] if row else None}

    def retire_agent(*, agent_id: str, reason: str = "", crashed: bool = False
                     ) -> dict[str, Any]:
        receipt = mind.work.retire_agent(agent_id=agent_id, reason=reason, crashed=crashed)
        return {"agent_id": agent_id, "receipt_id": receipt.receipt_id}

    # ------------------------------------------------------------------
    # memory and conclusions
    # ------------------------------------------------------------------
    def recall(*, query: str | None = None, scope: str = "active",
               kinds: Sequence[str] | None = None, limit: int = 20,
               min_confidence: float = 0.0) -> list[dict[str, Any]]:
        """Search maintained memory: beliefs the organism holds, not raw history."""
        return mind.memory.recall(query=query, scope=scope, kinds=list(kinds) if kinds else None,
                                  limit=limit, min_confidence=min_confidence)

    def remember(*, kind: str, claim: str, confidence: float, created_by: str,
                 supporting: Sequence[dict[str, Any]] = (),
                 opposing: Sequence[dict[str, Any]] = (),
                 tags: Sequence[str] | None = None, supersedes: str | None = None,
                 operation_id: str | None = None, mutation_id: str | None = None
                 ) -> dict[str, Any]:
        memory_id, receipt = mind.memory.remember(
            kind=kind, claim=claim, confidence=confidence, created_by=created_by,
            supporting=supporting, opposing=opposing, tags=tags, supersedes=supersedes,
            operation_id=operation_id, mutation_id=mutation_id,
        )
        return {"memory_id": memory_id, "receipt_id": receipt.receipt_id,
                "state_version": receipt.result_version, "replayed": receipt.replayed}

    def get_memory(*, memory_id: str) -> dict[str, Any]:
        """One maintained memory item, with its supporting and opposing evidence."""
        return mind.memory.get_memory(memory_id)

    def record_conclusion(*, claim: str, produced_by: str,
                          evidence: Sequence[dict[str, Any]] = (),
                          uncertainty: float | None = None,
                          alternatives: Sequence[str] | None = None,
                          operation_id: str | None = None,
                          model_identity: str | None = None,
                          snapshot_id: str | None = None,
                          mutation_id: str | None = None) -> dict[str, Any]:
        """Commit an auditable conclusion. Id may later audit it against the record."""
        cid, receipt = mind.memory.record_conclusion(
            claim=claim, produced_by=produced_by, evidence=evidence,
            uncertainty=uncertainty, alternatives=alternatives, operation_id=operation_id,
            model_identity=model_identity, snapshot_id=snapshot_id, mutation_id=mutation_id,
        )
        return {"conclusion_id": cid, "receipt_id": receipt.receipt_id,
                "state_version": receipt.result_version}

    def get_conclusion(*, conclusion_id: str) -> dict[str, Any]:
        """One recorded Ego conclusion with the evidence it rests on."""
        return mind.memory.get_conclusion(conclusion_id)

    def record_audit(*, target_kind: str, target_id: str, verdict: str,
                     actor: str = "id", focus: str | None = None,
                     findings: Sequence[str] | None = None,
                     unresolved: Sequence[str] | None = None,
                     evidence: Any = None, operation_id: str | None = None
                     ) -> dict[str, Any]:
        aid, receipt = mind.memory.record_audit(
            target_kind=target_kind, target_id=target_id, verdict=verdict, actor=actor,
            focus=focus, findings=findings, unresolved=unresolved, evidence=evidence,
            operation_id=operation_id,
        )
        return {"audit_id": aid, "receipt_id": receipt.receipt_id,
                "state_version": receipt.result_version}

    def disagreements(*, status: str = "open", limit: int = 50) -> list[dict[str, Any]]:
        """Recorded contradictions between claims, and whether they are resolved."""
        return mind.memory.get_disagreements(status=status, limit=limit)

    def operator_close_disagreement(*, disagreement_id: str, reason: str,
                                    operation_id: str | None = None
                                    ) -> dict[str, Any]:
        """Close a dispute the record is never going to settle.

        The only discretionary route. Recorded as a decision by a person
        rather than as evidence, because that is what it is: an operator
        closing a contradiction is a different fact from the contradiction
        having been resolved.
        """
        receipt = mind.memory.resolve_disagreement(
            disagreement_id=disagreement_id, resolution="closed_by_operator",
            actor="operator", detail=reason, operation_id=operation_id)
        return {"disagreement_id": disagreement_id,
                "resolution": "closed_by_operator",
                "receipt_id": receipt.receipt_id}

    def operator_retract_memory(*, memory_id: str, reason: str,
                                operation_id: str | None = None
                                ) -> dict[str, Any]:
        """Withdraw a maintained belief.

        `MemoryRepo.retract` existed and no verb reached it, so the organism
        could form a belief and never withdraw one. An operator act, which is
        consistent with memory writes already being Harness acts:
        `board_promote_to_memory` is the Harness turning discussion into
        belief, and withdrawing one is the same kind of decision.
        """
        receipt = mind.memory.retract(memory_id=memory_id, actor="operator",
                                      reason=reason, operation_id=operation_id)
        return {"memory_id": memory_id, "status": "retracted",
                "receipt_id": receipt.receipt_id}

    def open_disagreement(*, subject_kind: str, subject_id: str, claim_a: str,
                          actor_a: str, claim_b: str, actor_b: str,
                          evidence_a: Any = None, evidence_b: Any = None,
                          operation_id: str | None = None) -> dict[str, Any]:
        did, receipt = mind.memory.open_disagreement(
            subject_kind=subject_kind, subject_id=subject_id, claim_a=claim_a,
            actor_a=actor_a, claim_b=claim_b, actor_b=actor_b,
            evidence_a=evidence_a, evidence_b=evidence_b, operation_id=operation_id,
        )
        return {"disagreement_id": did, "receipt_id": receipt.receipt_id}

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    def provenance(*, operation_id: str) -> dict[str, Any]:
        """Resolve how a result came about: inputs, actors and receipts."""
        return mind.provenance(operation_id=operation_id)

    def verify_integrity(*, deep: bool = True) -> dict[str, Any]:
        """Check the event hash chain and that referenced content is present."""
        return mind.verify_integrity(deep=deep)

    def history(*, operation_id: str | None = None, correlation_id: str | None = None,
                kinds: Sequence[str] | None = None, since_seq: int = 0,
                limit: int = 100) -> list[dict[str, Any]]:
        """Raw append-only events. Evidence of what happened, not what is believed."""
        return mind.history(operation_id=operation_id, correlation_id=correlation_id,
                            kinds=list(kinds) if kinds else None,
                            since_seq=since_seq, limit=limit)

    def audit_dossier(*, conclusion_id: str | None = None,
                      operation_id: str | None = None) -> dict[str, Any]:
        """Everything Id needs to audit a conclusion, resolved from the record.

        Ego is not consulted: the dossier is built from the conclusion row, its
        evidence references, the originating operation's events and the model
        configuration recorded at the time.
        """
        dossier: dict[str, Any] = {"resolved_from": "durable record only",
                                   "ego_consulted": False}
        if not conclusion_id and not operation_id:
            # The question Id actually has: what is waiting to be audited?
            # Without this, a mind told "17 unaudited conclusions" had no way
            # to reach one, and invented identifiers instead.
            row = mind.db.conn.execute(
                "SELECT conclusion_id FROM conclusions c WHERE NOT EXISTS ("
                "SELECT 1 FROM audits a WHERE a.target_kind = 'conclusion'"
                " AND a.target_id = c.conclusion_id)"
                " ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                raise NotFound("nothing is waiting to be audited",
                               hint="every recorded conclusion has an audit")
            conclusion_id = row["conclusion_id"]
            dossier["resolved_by"] = "the oldest conclusion nobody has audited"
        if conclusion_id:
            concl = mind.memory.get_conclusion(conclusion_id)
            dossier["conclusion"] = concl
            operation_id = operation_id or concl.get("operation_id")
        if not operation_id:
            if dossier.get("conclusion"):
                # A claim recorded outside any operation is still auditable
                # against its own evidence. Refusing the dossier would make a
                # conclusion Id was told to audit unreachable.
                return {**dossier, "events": [], "hash_chain_ok": None,
                        "note": ("this conclusion was recorded outside any "
                                 "operation, so there is no operation trail; "
                                 "its own evidence is above")}
            raise NotFound("no operation to resolve", conclusion_id=conclusion_id)
        prov = mind.provenance(operation_id=operation_id)
        dossier["operation_id"] = operation_id
        dossier["hash_chain_ok"] = prov["hash_chain_ok"]
        dossier["unresolved_content"] = prov["unresolved_content"]
        dossier["events"] = [
            {"seq": e["seq"], "kind": e["kind"], "actor": e["actor_id"],
             "payload": e.get("payload")}
            for e in prov["events"]
        ][:40]
        dossier["receipts"] = prov["receipts"]
        if "conclusion" in dossier:
            refs = dossier["conclusion"].get("evidence", [])
            resolved = []
            for ref in refs:
                item = dict(ref)
                if ref.get("memory_id"):
                    try:
                        item["memory"] = mind.memory.get_memory(ref["memory_id"])
                    except MindError as exc:
                        item["memory_error"] = exc.to_dict()
                if ref.get("blob_sha256"):
                    item["content_present"] = mind.blobs.exists(ref["blob_sha256"])
                resolved.append(item)
            dossier["resolved_evidence"] = resolved
        return dossier

    def maintenance_context(*, objective: str) -> dict[str, Any]:
        """Narrow state references for a maintenance neuocyte.

        Deliberately NOT a snapshot of Id's private context: maintenance
        neuocytes get task instructions plus references, nothing more.
        """
        qs = mind.work.queue_stats()
        recent = read_events(mind.db.conn, kinds=[EventKind.WORK_FAILED,
                                                  EventKind.MEMORY_SUPERSEDED,
                                                  EventKind.AUDIT_RECORDED], limit=10)
        low_conf = mind.memory.recall(scope="active", limit=8)
        return {
            "objective": objective,
            "queue": qs,
            "state_version": mind.state_version(),
            "open_disagreements": mind.memory.get_disagreements(status="open", limit=5),
            "recent_events": [{"seq": e.seq, "kind": e.kind, "actor": e.actor_id}
                              for e in recent],
            "active_memory_sample": [
                {"memory_id": m["memory_id"], "claim": m["claim"],
                 "confidence": m["confidence"]} for m in low_conf
            ],
            "references": [f"event:{e.event_id}" for e in recent]
                          + [f"memory:{m['memory_id']}" for m in low_conf],
            "note": "maintenance neuocytes receive state references, never Ego or Id KV",
        }

    # ------------------------------------------------------------------
    # work queue
    # ------------------------------------------------------------------

    # ==================================================================
    # The external loop: what arrived with a request, and what goes back
    # ==================================================================
    def _interaction_of_turn(turn_id: str) -> str | None:
        """Which interaction this turn is answering, from the durable record.

        Walked from the turn's own triggers rather than taken as an argument.
        Ego naming an interaction would be a cross-client read, and Ego is the
        component most exposed to a confident user -- so the identifier it
        would need to do that is one it never handles.

        A turn holds at most one answer-bearing request (I81), so there is at
        most one answer here. A continuation follows `parent_turn` to the turn
        that admitted the request, because the second half of a thought is
        still answering the first half's question.
        """
        seen: set[str] = set()
        current: str | None = turn_id
        while current and current not in seen and len(seen) < 32:
            seen.add(current)
            for row in mind.db.conn.execute(
                    "SELECT payload_sha256 FROM role_triggers"
                    " WHERE turn_id = ? AND expects_answer = 1", (current,)):
                if not row["payload_sha256"]:
                    continue
                try:
                    payload = mind.blobs.get_json(row["payload_sha256"])
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(payload, dict) and payload.get("interaction_id"):
                    return str(payload["interaction_id"])
            parent = mind.db.conn.execute(
                "SELECT parent_turn FROM role_turns WHERE turn_id = ?",
                (current,)).fetchone()
            current = parent["parent_turn"] if parent else None
        return None

    def ego_read_attachment(*, input_id: str, turn_id: str,
                            max_chars: int = 8000) -> dict[str, Any]:
        """Read a file that arrived with the request this turn is answering.

        The attachment must belong to *this* turn's interaction. Ego does not
        say which interaction that is and cannot: the Harness resolves it from
        the turn, so an input id belonging to somebody else resolves to a
        refusal rather than to their file.

        Text is returned as text and bounded, because an unbounded read is a
        way to fill a context window from outside. Anything that is not text
        is described rather than decoded: handing a model a base64 wall serves
        nobody, and the digest is how the bytes are reached by something that
        can actually use them.
        """
        mine = _interaction_of_turn(turn_id)
        row = mind.db.conn.execute(
            "SELECT input_id, interaction_id, filename, sha256, bytes,"
            " media_type FROM interaction_inputs WHERE input_id = ?",
            (input_id,)).fetchone()
        # Referred to by this request, or admitted against it. An input the
        # client later used for a second question is still this one's.
        linked = mine is not None and bool(mind.db.conn.execute(
            "SELECT 1 FROM interaction_input_links"
            " WHERE interaction_id = ? AND input_id = ?",
            (mine, input_id)).fetchone())
        if row is None or not mine or not (linked or row["interaction_id"] == mine):
            # Deliberately one answer for "no such input" and "not yours": the
            # difference is only useful to somebody probing for other clients'
            # identifiers.
            raise NotFound("no attachment by that id on this request",
                           input_id=input_id)

        media = (row["media_type"] or "").lower()
        textual = (media.startswith("text/") or media in
                   ("application/json", "application/xml", "application/x-yaml")
                   or not media)
        out = {"input_id": row["input_id"], "filename": row["filename"],
               "media_type": row["media_type"], "bytes": row["bytes"],
               "sha256": row["sha256"]}
        if not textual:
            out["text"] = None
            out["note"] = (f"{row['media_type']} is not text; it is stored "
                           f"with exact-byte provenance as {row['sha256']} and "
                           "can be given to work that can read it")
            return out
        try:
            raw = mind.blobs.get(row["sha256"]).decode("utf-8")
        except UnicodeDecodeError:
            out["text"] = None
            out["note"] = ("declared as text but is not valid UTF-8; the bytes "
                           f"are stored as {row['sha256']}")
            return out
        if len(raw) > max(0, int(max_chars)):
            shown = raw[:int(max_chars)]
            out["text"] = shown
            out["truncated"] = True
            out["note"] = (f"showing the first {len(shown)} of {len(raw)} "
                           f"characters; the whole file is {row['sha256']}")
        else:
            out["text"] = raw
            out["truncated"] = False
        return out

    def ego_surface_result(*, sha256: str, turn_id: str,
                           filename: str | None = None,
                           media_type: str | None = None,
                           artifact_id: str | None = None) -> dict[str, Any]:
        """Make something fetchable by the client who asked this question.

        The counterpart to reading an attachment, and the production caller
        `surface_result` never had. Knowing a digest has never been authority
        to fetch it from outside; this is the deliberate act that makes one
        specific thing reachable, and it is recorded as such.

        The interaction comes from the turn. Ego decides *what* to surface and
        never *to whom*.
        """
        from .io_api import surface_result

        interaction_id = _interaction_of_turn(turn_id)
        if not interaction_id:
            raise InvalidInput(
                "this turn is not answering an external request, so there is "
                "nobody to surface a result to", turn_id=turn_id)
        return surface_result(sup, interaction_id=interaction_id,
                              sha256=sha256, filename=filename,
                              media_type=media_type, artifact_id=artifact_id,
                              surfaced_by="ego")

    def store_footprint() -> dict[str, Any]:
        """How much of what there is, and what may be forgotten.

        A sense. It reads and changes nothing, which is why Id may have it:
        noticing that the organism is accumulating is homeostasis, and acting
        on it is not Id's to decide.
        """
        from . import retention

        out = retention.footprint(mind.db.conn)
        cutoff = time.time() - mind.cfg.retention.working_set_seconds
        doomed = retention.prunable(mind.db.conn, older_than=cutoff)
        out["prunable_now"] = {"triggers": len(doomed["triggers"]),
                               "turns": len(doomed["turns"])}
        out["window_seconds"] = mind.cfg.retention.working_set_seconds
        out["enabled"] = mind.cfg.retention.enabled
        return out

    def store_prune(*, older_than_seconds: float | None = None
                    ) -> dict[str, Any]:
        """Forget the operational working set past its window.

        Never model-facing. What an organism is allowed to forget is policy,
        and a mind that could prune its own turn history could remove the
        record of what it did in the same motion.
        """
        from . import retention

        window = (mind.cfg.retention.working_set_seconds
                  if older_than_seconds is None else float(older_than_seconds))
        _, out = mind.writer.apply(
            lambda m: retention.prune(m, mind, older_than_seconds=window),
            actor="supervisor", bump_version=False)
        return out

    def _class_belongs_to(origin_actor: str, work_class: str) -> None:
        """A role may only admit work of its own kind.

        Enforced here as well as at `ego_request_work`, because the verb is
        the model's door and this is the Harness's. Work nobody persistent
        originated is left alone: it has no role whose kind to check against.
        """
        from .neuocyte import WORK_OF_ROLE

        expected = WORK_OF_ROLE.get(origin_actor)
        if expected is not None and work_class != expected:
            raise InvalidInput(
                f"{origin_actor} delegates {expected} work, not {work_class}",
                origin_actor=origin_actor, work_class=work_class,
                hint=("maintenance is Id's to decide on and Id's to delegate; "
                      "Ego asks for it with ego_request_id_review"))

    def admit_work(*, objective: str, work_class: str, origin_actor: str,
                   operation_id: str | None = None, priority: int = 0,
                   budget_tokens: int | None = None, wall_seconds: float | None = None,
                   maintenance_depth: int = 0, snapshot_id: str | None = None,
                   depends_on: Sequence[str] | None = None,
                   board_access: str = "read_write",
                   sandbox_allowed: bool = False,
                   specialisation: str | None = None) -> dict[str, Any]:
        _class_belongs_to(origin_actor, work_class)
        decision = sup.arbiter.admit(
            work_class=work_class, snapshot=sup.resource_snapshot(),
            requested_budget_tokens=budget_tokens, requested_wall_seconds=wall_seconds,
            maintenance_depth=maintenance_depth,
        )
        if not decision.admitted:
            receipt = mind.work.reject(objective=objective, work_class=work_class,
                                       origin_actor=origin_actor, reason=decision.reason,
                                       operation_id=operation_id)
            return {"admitted": False, "reason": decision.reason,
                    "detail": decision.detail, "receipt_id": receipt.receipt_id}
        gen = None
        try:
            gen = sup.client("inference").call("model_generation")
        except Exception:  # noqa: BLE001
            pass
        work_id, receipt = mind.work.admit(
            objective=objective, work_class=work_class, origin_actor=origin_actor,
            operation_id=operation_id, priority=priority, snapshot_id=snapshot_id,
            specialisation=specialisation,
            model_generation=gen, budget_tokens=decision.granted_budget_tokens,
            deadline=decision.granted_deadline, maintenance_depth=maintenance_depth,
            depends_on=depends_on, board_access=board_access,
            sandbox_allowed=sandbox_allowed,
        )
        return {"admitted": True, "work_id": work_id, "receipt_id": receipt.receipt_id,
                "board_access": board_access, "sandbox_allowed": sandbox_allowed,
                "granted_budget_tokens": decision.granted_budget_tokens,
                "deadline": decision.granted_deadline,
                "state_version": receipt.result_version}

    def lease_work(*, neuocyte_id: str, work_id: str | None = None,
                   work_class: str | None = None) -> dict[str, Any] | None:
        # A dispatched neuocyte asks for the item it was dispatched for. Passing
        # work_id down means it either claims that item or claims nothing --
        # it never takes an item another neuocyte is about to be dispatched for.
        return mind.work.lease(neuocyte_id=neuocyte_id, work_class=work_class,
                               work_id=work_id,
                               lease_seconds=sup.cfg.arbiter.lease_seconds)

    def _wake_owner_of_work(work_id: str, *, kind: str, summary: str) -> None:
        """Queue a trigger for the role that asked for this work, if any.

        Relevance is taken from `origin_actor` on the work row -- an explicit
        recorded relationship, not a guess. Work Ego did not originate does
        not wake Ego, which is what keeps a busy neuocyte fleet from turning
        into a wake storm.

        Maintenance work originated by Id wakes Id for the same reason. Work
        originated by the supervisor or the operator wakes nobody: no
        persistent role is waiting on it.

        Best effort on purpose. A work item is complete whether or not anyone
        was told, and failing the completion because a mailbox write failed
        would lose the result to protect a notification.
        """
        from .waking import wake_owner_of_work

        wake_owner_of_work(sup, mind, work_id, kind=kind, summary=summary)

    def complete_work(*, work_id: str, neuocyte_id: str, fencing_token: int,
                      result: Any, pinned_state_ver: int | None = None
                      ) -> dict[str, Any]:
        receipt = mind.work.complete(
            work_id=work_id, neuocyte_id=neuocyte_id, fencing_token=fencing_token,
            result=result, pinned_state_ver=pinned_state_ver,
        )
        # Scratch does not outlive the work that produced it. Anything worth
        # keeping had to be proposed and promoted, which is the only way out.
        sup.release_work_sandbox(work_id, reason="work completed")
        _wake_owner_of_work(work_id, kind="work_completed",
                            summary=f"work {work_id} you requested has completed")
        return {"receipt_id": receipt.receipt_id, "state_version": receipt.result_version,
                "replayed": receipt.replayed}

    def fail_work(*, work_id: str, neuocyte_id: str, fencing_token: int, failure: str,
                  requeue: bool = True) -> dict[str, Any]:
        receipt = mind.work.fail(work_id=work_id, neuocyte_id=neuocyte_id,
                                 fencing_token=fencing_token, failure=failure,
                                 requeue=requeue)
        sup.release_work_sandbox(work_id, reason="work failed")
        # Read the outcome that was committed, rather than deciding from the
        # caller's request. `requeue=True` -- which is the neuocyte's default
        # -- still retires the item once its attempts run out, and taking the
        # notification from the flag meant the attempt that ended the work
        # told nobody: the owner waited on a `failed` item for a message that
        # was never going to come.
        row = mind.db.conn.execute(
            "SELECT status FROM work_items WHERE work_id = ?", (work_id,)).fetchone()
        outcome = row["status"] if row is not None else None
        # A requeued failure is an attempt, not an outcome: waking the owner
        # for it would report a conclusion that has not been reached. Replay
        # of a receipt is not a second outcome either.
        if outcome in TERMINAL_WORK and not receipt.replayed:
            _wake_owner_of_work(
                work_id, kind="work_failed",
                summary=f"work {work_id} you requested failed: {failure[:200]}")
        return {"receipt_id": receipt.receipt_id, "outcome": outcome}

    def cancel_work(*, work_id: str, actor: str = "supervisor", reason: str = ""
                    ) -> dict[str, Any]:
        receipt = mind.work.cancel(work_id=work_id, actor=actor, reason=reason)
        sup.release_work_sandbox(work_id, reason=f"work cancelled: {reason}"[:200])
        _wake_owner_of_work(
            work_id, kind="work_cancelled",
            summary=f"work {work_id} you requested was cancelled: {reason[:200]}")
        return {"receipt_id": receipt.receipt_id}

    def get_work(*, work_id: str) -> dict[str, Any]:
        """The current state of one work item."""
        return mind.work.get_work(work_id)

    def queue_stats() -> dict[str, Any]:
        """Counts of productive work by status."""
        return mind.work.queue_stats()

    def kill_all_neuocytes(*, reason: str = "operator request") -> dict[str, Any]:
        """Kill every disposable neuocyte. State and unfinished work must survive."""
        killed = list(sup.neuocytes)
        for wid in killed:
            sup._kill_neuocyte(wid, reason=reason)
        expired = mind.work.expire_leases()
        return {"killed_neuocytes": killed, "requeued_work": expired,
                "state_version": mind.state_version()}

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------
    def publish_ego_snapshot(*, actor: str, model_generation: str, token_count: int,
                             tokens: Sequence[int], text: str | None, kv_mode: str,
                             backend_handle: str | None,
                             operation_id: str | None = None) -> dict[str, Any]:
        snapshot_id, version, receipt = mind.work.publish_snapshot(
            actor=actor, model_generation=model_generation, token_count=token_count,
            tokens=tokens, text=text, kv_mode=kv_mode, backend_handle=backend_handle,
            operation_id=operation_id,
        )
        sup._last_snapshot_publish = time.time()
        return {"snapshot_id": snapshot_id, "version": version, "kv_mode": kv_mode,
                "receipt_id": receipt.receipt_id, "token_count": token_count,
                "model_generation": model_generation}

    def acquire_snapshot(*, holder: str, snapshot_id: str | None = None
                         ) -> dict[str, Any]:
        """Take a reference to a published snapshot.

        With no id, the newest published snapshot is used -- a replacement
        neuocyte always starts from the newest, while running neuocytes keep the
        one they were pinned to.
        """
        if snapshot_id:
            snap = mind.work.get_snapshot(snapshot_id)
        else:
            snap = mind.work.latest_snapshot(actor="ego")
            if snap is None:
                # Work can be admitted before Ego has ever published. Publishing
                # is the harness's job, not the neuocyte's, so do it here rather
                # than failing work that is otherwise perfectly runnable.
                published = ensure_snapshot(max_age_seconds=0.0)
                snap = mind.work.get_snapshot(published["snapshot_id"])
        ref_id, receipt = mind.work.acquire_snapshot_ref(
            snapshot_id=snap["snapshot_id"], holder=holder
        )
        out = dict(snap)
        out["ref_id"] = ref_id
        out["receipt_id"] = receipt.receipt_id
        return out

    def release_snapshot_ref(*, ref_id: str, actor: str) -> dict[str, Any]:
        receipt = mind.work.release_snapshot_ref(ref_id=ref_id, actor=actor)
        return {"receipt_id": receipt.receipt_id}

    def snapshot_tokens(*, snapshot_id: str) -> list[int]:
        return mind.work.snapshot_tokens(snapshot_id)

    def list_snapshots(*, actor: str = "ego", limit: int = 20) -> list[dict[str, Any]]:
        return [dict(r) for r in mind.db.conn.execute(
            "SELECT * FROM snapshots WHERE actor = ? ORDER BY version DESC LIMIT ?",
            (actor, limit),
        )]

    def ensure_snapshot(*, operation_id: str | None = None,
                        max_age_seconds: float = 60.0) -> dict[str, Any]:
        """Publish a fresh Ego snapshot if the newest one is stale."""
        latest = mind.work.latest_snapshot(actor="ego")
        if latest and (time.time() - latest["created_at"]) < max_age_seconds:
            return {"snapshot_id": latest["snapshot_id"], "version": latest["version"],
                    "reused": True}
        result = sup.client("ego").call("publish_snapshot", operation_id=operation_id)
        result["reused"] = False
        return result

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def open_operation(*, kind: str, actor: str, request: Any,
                       idempotency_key: str | None = None) -> dict[str, Any]:
        op_id, receipt, replayed = mind.memory.open_operation(
            kind=kind, actor=actor, request=request, idempotency_key=idempotency_key
        )
        return {"operation_id": op_id, "receipt_id": receipt.receipt_id,
                "replayed": replayed, "state_version": receipt.result_version}

    def get_operation(*, operation_id: str) -> dict[str, Any]:
        return mind.memory.get_operation(operation_id)

    def update_operation(*, operation_id: str, status: str, actor: str,
                         result: Any = None, limitations: Sequence[str] | None = None,
                         work_id: str | None = None) -> dict[str, Any]:
        receipt = mind.memory.update_operation(
            operation_id=operation_id, status=status, actor=actor, result=result,
            limitations=limitations, work_id=work_id,
        )
        return {"receipt_id": receipt.receipt_id, "state_version": receipt.result_version}

    # ------------------------------------------------------------------
    # cognitive verbs
    # ------------------------------------------------------------------
    def _run_operation(kind: str, actor: str, request: dict[str, Any],
                       idempotency_key: str | None, fn) -> dict[str, Any]:
        """Open an operation, run it, commit the outcome, return a receipt.

        A replayed idempotency key returns the stored result rather than
        re-running the cognition.
        """
        op_id, receipt, replayed = mind.memory.open_operation(
            kind=kind, actor=actor, request=request, idempotency_key=idempotency_key
        )
        if replayed:
            op = mind.memory.get_operation(op_id)
            return {
                "schema_version": SCHEMA_VERSION, "operation_id": op_id,
                "status": op["status"], "receipt_id": receipt.receipt_id,
                "state_version": op["state_version"], "result": op.get("result"),
                "limitations": op.get("limitations") or [],
                "replayed": True,
            }
        limitations: list[str] = []
        try:
            result = fn(op_id, limitations)
            status = "completed"
        except MindError as exc:
            result = {"error": exc.to_dict()}
            status = "failed"
            limitations.append(f"{exc.code}: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            result = {"error": {"code": "internal_error",
                                "message": f"{type(exc).__name__}: {exc}"}}
            status = "failed"
            limitations.append(f"internal_error: {exc}")
        commit = mind.memory.update_operation(
            operation_id=op_id, status=status, actor=actor, result=result,
            limitations=limitations,
        )
        return {
            "schema_version": SCHEMA_VERSION, "operation_id": op_id, "status": status,
            "receipt_id": commit.receipt_id, "state_version": commit.result_version,
            "result": result, "limitations": limitations, "replayed": False,
        }

    def _await_turn(trigger_id: str, *, timeout: float) -> dict[str, Any]:
        """Wait for the answer to **this request**.

        It used to wait for the turn that consumed the trigger and return that
        turn's result, which made an answer a property of a turn rather than
        of the question asked. Two requests bundled into one turn received the
        same reply, and a thought continued across turns returned only its
        first turn's text -- the rest was produced and then unreachable.

        The mailbox now writes an answer against the request itself, following
        the continuation chain, so this waits for exactly that. Waiting is
        still only a convenience: the request is durable and ordered the
        moment it is queued, and a caller that gives up changes nothing except
        its own patience.
        """
        deadline = time.time() + max(0.0, timeout)
        while True:
            row = mind.db.conn.execute(
                "SELECT status, turn_id, answer_status, answer_sha256,"
                " answered_by_turn FROM role_triggers WHERE trigger_id = ?",
                (trigger_id,)).fetchone()
            if row is None:
                raise NotFound("trigger disappeared", trigger_id=trigger_id)
            if row["status"] == "expired":
                return {"status": "expired", "turn_id": None,
                        "note": "undeliverable after repeated failures"}

            if row["answer_status"] in ("answered", "incomplete"):
                answer, record = "", {}
                if row["answer_sha256"]:
                    try:
                        record = mind.blobs.get_json(row["answer_sha256"]) or {}
                        answer = record.get("answer", "")
                    except Exception:  # noqa: BLE001
                        answer, record = "", {}
                turn = mind.db.conn.execute(
                    "SELECT * FROM role_turns WHERE turn_id = ?",
                    (row["answered_by_turn"],)).fetchone()
                result = None
                if turn is not None and turn["result_sha256"]:
                    try:
                        result = mind.blobs.get_json(turn["result_sha256"])
                    except Exception:  # noqa: BLE001
                        result = None
                if isinstance(result, dict):
                    # The answer of record wins over whatever the turn result
                    # happened to carry.
                    result = {**result, "answer": answer}
                else:
                    result = {"answer": answer}
                # So does its conclusion: recorded with the whole answer, not
                # by the turn that happened to finish it.
                result["conclusion_id"] = record.get("conclusion_id")
                result["conclusion_ids"] = record.get("conclusion_ids") or []
                # Terminal either way, and only one of them is finished.
                # "completed" is reserved for a thought that concluded; one
                # that was stopped -- by the continuation limit, a deadline, a
                # failure -- is "incomplete", carrying everything it said and
                # why it stopped, rather than posing as the reply.
                finished = row["answer_status"] == "answered"
                return {"status": "completed" if finished else "incomplete",
                        "complete": finished,
                        "ended_because": record.get("ended_because"),
                        "withheld": record.get("withheld") or [],
                        "turn_id": row["answered_by_turn"],
                        "stop_reason": turn["stop_reason"] if turn else None,
                        "result": result,
                        "environment_sha256": (turn["environment_sha256"]
                                               if turn else None),
                        "profile_ref": turn["profile_ref"] if turn else None}

            if row["answer_status"] == "unanswerable":
                record = {}
                if row["answer_sha256"]:
                    try:
                        record = mind.blobs.get_json(row["answer_sha256"]) or {}
                    except Exception:  # noqa: BLE001
                        record = {}
                return {"status": "unanswerable",
                        "ended_because": record.get("ended_because"),
                        "withheld": record.get("withheld") or [],
                        "turn_id": row["answered_by_turn"],
                        "result": {"answer": ""},
                        "note": ("the thought ended without producing an "
                                 "answer to this request")}

            if time.time() >= deadline:
                return {"status": row["status"], "turn_id": row["turn_id"],
                        "note": ("still queued; Ego will answer it at a turn "
                                 "boundary and the answer is retrievable by "
                                 "trigger id")}
            time.sleep(0.1)

    def role_answer(*, trigger_id: str, wait_seconds: float = 0.0
                    ) -> dict[str, Any]:
        """The answer of record for one queued request.

        `_await_turn` says an unanswered request's answer stays retrievable by
        trigger id. Nothing exposed it, so a caller whose wait elapsed had the
        answer written durably against its request and no way to read it.

        A conversational surface needs exactly this: ask, let go of the
        connection, and come back. A thought that takes continuation turns
        outlives any single HTTP request, and holding a socket open for the
        duration is not the same thing as being able to collect an answer.

        Read-only and scoped to one request. It reports on the question the
        caller asked and carries no authority over the role that answers it.
        """
        settled = _await_turn(trigger_id,
                              timeout=max(0.0, float(wait_seconds)))
        result = settled.get("result") or {}
        out = {"trigger_id": trigger_id, "status": settled["status"],
               "answer": result.get("answer", ""),
               "is_simulated": bool(result.get("is_simulated"))}
        if settled["status"] in ("incomplete", "unanswerable"):
            out["ended_because"] = settled.get("ended_because")
        if settled.get("withheld"):
            out["withheld"] = settled["withheld"]
        if settled.get("note"):
            out["note"] = settled["note"]
        return out

    def _label_simulation(result: dict[str, Any] | None,
                          limitations: list[str]) -> None:
        """Say plainly when text came from a stub rather than a model.

        Restored after the turn model briefly lost it. A caller must never
        have to guess whether a cognitive result is real inference, and the
        organism claiming otherwise by omission is exactly the kind of
        overclaiming every other surface here is built to avoid.
        """
        if isinstance(result, dict) and result.get("is_simulated"):
            limitations.append(
                "SIMULATED BACKEND: this text was produced by a deterministic "
                "stub, not by model inference")

    def ego_converse(*, message: str, conversation_id: str | None = None,
                     idempotency_key: str | None = None,
                     max_tokens: int | None = None,
                     temperature: float | None = None,
                     wait: bool = True, wait_seconds: float | None = None,
                     # Supplied by the io worker, never by a model: they are
                     # how the Harness scopes Ego's attachment and surfacing
                     # effectors to this request and no other.
                     interaction_id: str | None = None,
                     attachments: list | None = None,
                     ) -> dict[str, Any]:
        """Give Ego something to think about, and optionally wait for it.

        This used to call into the Ego process synchronously, which meant two
        callers ran two cognitive turns concurrently against one inference
        session. Input is now queued in the Harness-owned mailbox and consumed
        at a turn boundary.

        There is deliberately no fast path for an idle Ego. One truthful
        ingestion path means conversational ordering is the queue order, not a
        race between whoever called while Ego happened to be free.
        """
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            queued = sup.methods()["role_enqueue_trigger"](
                role="ego", kind="user_input", source="operator",
                expects_answer=True,
                summary=message.strip()[:400],
                payload={"message": message,
                         "conversation_id": conversation_id,
                         # Who is waiting, and what they sent. Ego never
                         # supplies either: both are how the Harness scopes
                         # `ego_read_attachment` and `ego_surface_result` to
                         # this request and no other.
                         "interaction_id": interaction_id,
                         "attachments": list(attachments or [])},
                correlation_id=conversation_id, operation_id=op_id,
                # Recorded against the interaction in this same commit, so
                # the answer can be delivered from the record rather than by
                # whoever happens to be waiting.
                answers_interaction=interaction_id,
                # The operation, not the conversation. A conversation is many
                # interactions, and work delegated while answering this one
                # comes back tagged with *this* operation -- tagging the
                # request with the conversation instead would make the
                # returning result a stranger to the turn that asked for it.
                # Continuity across a conversation is Ego's persistent
                # context, which is what `conversation_id` still correlates.
                lineage=op_id)
            out: dict[str, Any] = {"trigger_id": queued["trigger_id"],
                                   "status": "queued",
                                   "conversation_id": conversation_id}
            if not wait:
                limitations.append(
                    "queued only; the answer is not in this response")
                return out
            settled = _await_turn(
                queued["trigger_id"],
                timeout=(sup.cfg.scheduler.submit_wait_seconds
                         if wait_seconds is None else wait_seconds))
            out.update(settled)
            result = settled.get("result") or {}
            _label_simulation(result, limitations)
            out["is_simulated"] = bool(result.get("is_simulated"))
            out["model_generation"] = result.get("model_generation")
            out["tool_requests"] = result.get("tool_requests", [])
            out["answer"] = result.get("answer", "")
            out["conclusion_id"] = result.get("conclusion_id")
            out["conclusion_ids"] = result.get("conclusion_ids") or []
            out["tool_calls"] = result.get("tool_calls", [])
            if settled["status"] == "incomplete":
                # Terminal, with everything said so far, and plainly not
                # a finished answer.
                limitations.append(
                    "the answer is incomplete: it stopped before "
                    f"finishing ({settled.get('ended_because')}); "
                    "everything said so far is included")
            elif settled["status"] != "completed":
                limitations.append(
                    "Ego had not reached this input before the wait elapsed; "
                    "it remains queued and will be processed")
            elif settled.get("stop_reason") not in ("model_stop", None):
                limitations.append(
                    f"the turn ended with {settled['stop_reason']!r}; the "
                    "Harness may have scheduled a continuation")
            return out

        return _run_operation("ego_converse", "ego",
                              {"message": message, "conversation_id": conversation_id},
                              idempotency_key, run)

    def ego_investigate(*, question: str, constraints: str = "",
                        budget_tokens: int | None = None,
                        idempotency_key: str | None = None,
                        interaction_id: str | None = None,
                        attachments: Sequence[dict[str, Any]] | None = None,
                        wait: bool = True, wait_seconds: float | None = None
                        ) -> dict[str, Any]:
        """Ask Ego to investigate something.

        Through the mailbox, like every other input. This used to call into
        the Ego process directly, which meant it ran a generation against the
        same inference session a queued turn might already be using.

        Ego is told what to investigate and reaches for its own senses and
        effectors from there -- requesting work, reading the board, recording
        a conclusion. That is better than handing it a pre-assembled context:
        the environment already tells it what it can do, and what it actually
        used is then on the record as tool calls.
        """
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            framing = f"investigate: {question.strip()}"
            if constraints.strip():
                framing += f"\nconstraints: {constraints.strip()}"
            queued = sup.methods()["role_enqueue_trigger"](
                role="ego", kind="user_input", source="operator",
                expects_answer=True,
                summary=framing[:400],
                # Whole, not clipped: the door already refuses more than
                # `MAX_INPUT_CHARS`, and what gets past it is what was asked.
                # Rendering bounds the body and says where the rest is
                # (`mailbox.trigger_body`); storage silently dropping the end
                # of a request cost investigations their trailing
                # instructions.
                payload={"question": question,
                         "constraints": constraints,
                         "budget_tokens": budget_tokens,
                         "intent": "investigate",
                         # The same context a conversation carries. Without
                         # these an investigation could not resolve its own
                         # request's attachments or return a file through
                         # `ego_surface_result`.
                         "interaction_id": interaction_id,
                         "attachments": list(attachments or [])},
                operation_id=op_id, lineage=op_id,
                # In the same commit as the trigger; see `ego_converse`.
                answers_interaction=interaction_id)
            out: dict[str, Any] = {"trigger_id": queued["trigger_id"],
                                   "status": "queued", "question": question}
            if not wait:
                limitations.append("queued only; the answer is not in this "
                                   "response")
                return out
            settled = _await_turn(
                queued["trigger_id"],
                timeout=(sup.cfg.scheduler.submit_wait_seconds
                         if wait_seconds is None else wait_seconds))
            out.update(settled)
            result = settled.get("result") or {}
            _label_simulation(result, limitations)
            out["is_simulated"] = bool(result.get("is_simulated"))
            out["claim"] = result.get("answer", "")
            out["conclusion_id"] = result.get("conclusion_id")
            out["conclusion_ids"] = result.get("conclusion_ids") or []
            if settled["status"] == "incomplete":
                # Terminal, with everything said so far, and plainly not
                # a finished answer.
                limitations.append(
                    "the answer is incomplete: it stopped before "
                    f"finishing ({settled.get('ended_because')}); "
                    "everything said so far is included")
            elif settled["status"] != "completed":
                limitations.append(
                    "Ego had not reached this before the wait elapsed; it "
                    "remains queued and will be processed")
            return out

        return _run_operation("ego_investigate", "ego",
                              {"question": question, "constraints": constraints},
                              idempotency_key, run)

    def ego_recall(*, query: str = "", scope: str = "active", limit: int = 10
                   ) -> dict[str, Any]:
        items = mind.memory.recall(query=query or None, scope=scope, limit=limit)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "state_version": mind.state_version(),
            "result": {"memories": items, "count": len(items)},
            "limitations": ["recall searches MAINTAINED memory only; raw history is "
                            "evidence and is reached through id_audit"],
        }

    def ego_status(*, operation_id: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
                               "state_version": mind.state_version()}
        try:
            out["ego"] = sup.client("ego").call("status")
        except Exception as exc:  # noqa: BLE001
            out["ego"] = {"unreachable": repr(exc)}
        if operation_id:
            op = mind.memory.get_operation(operation_id)
            out["operation"] = op
            if op.get("work_id"):
                out["work"] = mind.work.get_work(op["work_id"])
        out["queue"] = mind.work.queue_stats()
        return out

    def id_introspect(*, question: str, scope: str = "all",
                      idempotency_key: str | None = None,
                      wait: bool = True, wait_seconds: float | None = None
                      ) -> dict[str, Any]:
        """Ask Id to look at the organism and say what it sees.

        Queued rather than called. Id previously answered this on whatever
        thread the RPC arrived on, against the same session its own turns use.

        The measured state is no longer pre-assembled into the prompt: Id has
        `system_pulse`, `verify_integrity`, `disagreements` and the rest in its
        environment, and reaching for them leaves a record of what it actually
        looked at rather than what it was handed.
        """
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            queued = sup.methods()["role_enqueue_trigger"](
                role="id", kind="operator_message", source="operator",
                expects_answer=True,
                summary=f"introspect ({scope}): {question.strip()}"[:400],
                payload={"question": question[:8000], "scope": scope,
                         "intent": "introspect"},
                operation_id=op_id, lineage=op_id)
            out: dict[str, Any] = {"trigger_id": queued["trigger_id"],
                                   "status": "queued", "question": question}
            if not wait:
                limitations.append("queued only")
                return out
            settled = _await_turn(
                queued["trigger_id"],
                timeout=(sup.cfg.scheduler.submit_wait_seconds
                         if wait_seconds is None else wait_seconds))
            out.update(settled)
            result = settled.get("result") or {}
            _label_simulation(result, limitations)
            out["is_simulated"] = bool(result.get("is_simulated"))
            out["interpretation"] = result.get("text", "")
            out["measurement_source"] = (
                "Id's own senses, invoked during the turn and recorded as "
                "tool calls")
            out["interpretation_source"] = "model inference over what it read"
            if settled["status"] == "incomplete":
                # Terminal, with everything said so far, and plainly not
                # a finished answer.
                limitations.append(
                    "the answer is incomplete: it stopped before "
                    f"finishing ({settled.get('ended_because')}); "
                    "everything said so far is included")
            elif settled["status"] != "completed":
                limitations.append("Id had not reached this before the wait "
                                   "elapsed; it remains queued")
            return out

        return _run_operation("id_introspect", "id",
                              {"question": question, "scope": scope},
                              idempotency_key, run)

    def id_health(*, scope: str = "all") -> dict[str, Any]:
        """Whether Id itself is running, and how recently it reported."""
        base = health()
        base["capabilities"] = capabilities()
        try:
            base["id"] = sup.client("id").call("health")
        except Exception as exc:  # noqa: BLE001
            base["id"] = {"unreachable": repr(exc)}
        base["integrity"] = mind.verify_integrity(deep=False)
        try:
            base["context"] = sup.homeostasis.assess()
        except Exception as exc:  # noqa: BLE001
            base["context"] = {"unavailable": repr(exc)}
        base["board"] = mind.board.stats()
        try:
            from .security import audit_paths

            base["filesystem"] = audit_paths(
                [sup.cfg.state_dir, sup.cfg.blob_dir, sup.cfg.sandbox_dir,
                 sup.cfg.artifact_dir])
        except Exception as exc:  # noqa: BLE001
            base["filesystem"] = {"error": repr(exc)}
        try:
            base["sandbox"] = (sup.sandboxes.capabilities() if sup.sandboxes
                               else {"sandbox_available": False})
        except Exception as exc:  # noqa: BLE001
            base["sandbox"] = {"error": repr(exc)}
        return base

    def id_audit(*, conclusion_id: str | None = None,
                 operation_id: str | None = None, focus: str = "",
                 idempotency_key: str | None = None,
                 wait: bool = True, wait_seconds: float | None = None
                 ) -> dict[str, Any]:
        """Ask Id to audit a conclusion against the record.

        Queued like any other input, so it cannot run a generation against the
        session a claimed turn may already hold.

        The split is sharper than before. The **evidence review is resolved by
        the Harness** -- the dossier, the hash chain, whatever content could
        not be resolved -- so it is measured fact rather than something the
        model reported about itself. Only the verdict and the findings come
        from Id's turn. Id still reaches for `audit_dossier` during that turn
        through its own senses, and what it consulted is recorded as tool
        calls.

        Ego is never consulted. Not once, anywhere in this path.
        """
        if not conclusion_id and not operation_id:
            raise NotFound("audit needs a conclusion_id or an operation_id")
        target = conclusion_id or operation_id

        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            # Measured first. A target the record cannot resolve is refused
            # here, before anyone is woken: queueing first cost Id a turn on an
            # audit whose verdict could never be committed.
            dossier = sup.methods()["audit_dossier"](
                conclusion_id=conclusion_id, operation_id=operation_id)
            framing = (f"audit conclusion {target}"
                       + (f" (focus: {focus})" if focus.strip() else "")
                       + ". " + AUDIT_FRAMING)
            queued = sup.methods()["role_enqueue_trigger"](
                role="id", kind="operator_message", source="operator",
                expects_answer=True,
                source_ref=target, summary=framing[:400],
                payload={"conclusion_id": conclusion_id,
                         "operation_id_target": operation_id,
                         "focus": focus[:2000], "intent": "audit",
                         # Who records the verdict. A caller that waits does;
                         # one that does not leaves it to the Harness when Id
                         # answers -- otherwise the verdict is produced and
                         # dropped, which is what every unwaited audit did.
                         "commit": "waiter" if wait else "on_completion"},
                operation_id=op_id, lineage=op_id)

            # Measured, by the Harness, regardless of what Id says -- above.
            reviewed = {
                "ego_consulted": False,
                "resolved_from": "durable record only",
                "hash_chain_ok": dossier.get("hash_chain_ok"),
                "events": dossier.get("events", []),
                "operation_id": dossier.get("operation_id"),
                "conclusion": dossier.get("conclusion"),
                "unresolved_content": dossier.get("unresolved_content", []),
            }
            out: dict[str, Any] = {
                "trigger_id": queued["trigger_id"], "status": "queued",
                "conclusion_id": conclusion_id,
                "target_kind": "conclusion" if conclusion_id else "operation",
                "target_id": target,
                "asked_ego_to_defend_itself": False,
                "evidence_reviewed": reviewed,
                "verdict": "inconclusive", "findings": [], "unresolved": "",
            }
            if not wait:
                limitations.append("queued; the verdict is recorded when Id "
                                   "answers")
                return out

            settled = _await_turn(
                queued["trigger_id"],
                timeout=(sup.cfg.scheduler.submit_wait_seconds
                         if wait_seconds is None else wait_seconds))
            out.update(settled)
            out["evidence_reviewed"] = reviewed
            out["asked_ego_to_defend_itself"] = False
            result = settled.get("result") or {}
            _label_simulation(result, limitations)
            out["is_simulated"] = bool(result.get("is_simulated"))

            # Not yet answered: the verdict is owed to this waiter, and a
            # waiter that stops waiting records nothing. Said plainly.
            if settled["status"] not in ("completed", "incomplete"):
                limitations.append("Id had not reached this before the wait "
                                   "elapsed; it remains queued, and this call "
                                   "will not record its verdict")
                return out
            if settled["status"] == "incomplete":
                limitations.append(
                    "the answer is incomplete: it stopped before finishing "
                    f"({settled.get('ended_because')}); everything said so far "
                    "is included")
            # The structured verdict survived the move onto the turn model. It
            # used to be parsed inside the role; the turn is generic, so it is
            # parsed where it is committed. Losing it because the transport
            # changed would be a capability quietly disappearing.
            out.update(commit_audit(
                mind, dossier, conclusion_id=conclusion_id,
                operation_id=operation_id, focus=focus,
                text=result.get("text") or result.get("answer") or "",
                op_id=op_id))
            if not out.pop("verdict_stated"):
                limitations.append(
                    "Id did not state a verdict in the expected form; "
                    "'inconclusive' here means unparsed, not judged")
            if reviewed.get("unresolved_content"):
                limitations.append("some referenced content could not be "
                                   "resolved; the audit is incomplete")
            return out

        return _run_operation("id_audit", "id",
                              {"conclusion_id": conclusion_id,
                               "operation_id": operation_id, "focus": focus},
                              idempotency_key, run)

    def harness_request_audit(*, conclusion_id: str) -> dict[str, Any]:
        """Ego put a claim into the auditable record, so Id is woken to audit it.

        Not the operator asking, and it does not say it is: the trigger is a
        `conclusion_recorded` event from the Harness. Id reads the dossier and
        states a verdict like any other audit, and the verdict is committed
        when Id answers -- nobody is waiting on it, which is exactly the case
        that used to lose it. Only intentional conclusions reach here, because
        answering no longer records one (I116), so this wakes Id for things
        worth auditing rather than for every reply.
        """
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            sup.methods()["audit_dossier"](conclusion_id=conclusion_id)
            queued = sup.methods()["role_enqueue_trigger"](
                role="id", kind="conclusion_recorded", source="harness",
                expects_answer=True, source_ref=conclusion_id,
                summary=(f"Ego recorded conclusion {conclusion_id}. Audit it. "
                         + AUDIT_FRAMING)[:400],
                payload={"conclusion_id": conclusion_id, "focus": "",
                         "intent": "audit", "commit": "on_completion"},
                operation_id=op_id, lineage=op_id)
            limitations.append("queued; the verdict is recorded when Id answers")
            return {"trigger_id": queued["trigger_id"], "status": "queued",
                    "conclusion_id": conclusion_id}

        return _run_operation("id_audit", "harness",
                              {"conclusion_id": conclusion_id,
                               "requested_by": "harness"}, None, run)

    def harness_commit_audit(*, trigger_id: str) -> dict[str, Any] | None:
        """Commit the verdict of an audit nobody waited for, now that Id answered.

        Called by the Harness when a turn closes, for each request it
        answered. Anything but an unwaited audit is left alone, so a waiter's
        audit is still committed exactly once, by the waiter.
        """
        row = mind.db.conn.execute(
            "SELECT payload_sha256, operation_id, answer_status, answer_sha256"
            " FROM role_triggers WHERE trigger_id = ?", (trigger_id,)).fetchone()
        if row is None or not row["payload_sha256"]:
            return None
        payload = mind.blobs.get_json(row["payload_sha256"]) or {}
        if payload.get("intent") != "audit" or payload.get("commit") != "on_completion":
            return None
        text = ""
        if row["answer_sha256"]:
            text = (mind.blobs.get_json(row["answer_sha256"]) or {}).get("answer", "")
        conclusion_id = payload.get("conclusion_id")
        target_op = payload.get("operation_id_target")
        dossier = sup.methods()["audit_dossier"](
            conclusion_id=conclusion_id, operation_id=target_op)
        return commit_audit(mind, dossier, conclusion_id=conclusion_id,
                            operation_id=target_op,
                            focus=payload.get("focus") or "", text=text,
                            op_id=row["operation_id"])

    def id_disagreements(*, scope: str = "open", limit: int = 20) -> dict[str, Any]:
        items = mind.memory.get_disagreements(status=scope, limit=limit)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "state_version": mind.state_version(),
            "result": {"disagreements": items, "count": len(items)},
            "limitations": ["competing claims with their evidence; this is not a "
                            "majority-truth score and no side is marked correct"],
        }

    def id_maintenance(*, objective: str, scope: str = "",
                       budget_tokens: int | None = None, maintenance_depth: int = 0,
                       idempotency_key: str | None = None) -> dict[str, Any]:
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            out = admit_work(objective=objective, work_class="maintenance",
                             origin_actor="id", operation_id=op_id,
                             budget_tokens=budget_tokens,
                             maintenance_depth=maintenance_depth)
            if not out["admitted"]:
                limitations.append(f"refused by the arbiter: {out['reason']}")
            return out

        return _run_operation("id_maintenance", "id",
                              {"objective": objective, "scope": scope},
                              idempotency_key, run)

    # ------------------------------------------------------------------
    # cancellation
    # ------------------------------------------------------------------
    def cancel_operation(*, operation_id: str | None = None,
                         idempotency_key: str | None = None,
                         reason: str = "client cancelled",
                         actor: str = "client") -> dict[str, Any]:
        """Stop an operation and everything downstream of it.

        Cancellation is cooperative first and forceful second: the in-flight
        generation is asked to stop between tokens, queued work is cancelled so
        no neuocyte picks it up, and only a neuocyte already running is killed.

        Idempotent. Cancelling a finished operation is not an error -- a client
        that cancels just as the work lands should get a truthful "already
        completed", not a failure.
        """
        if not operation_id and not idempotency_key:
            raise InvalidInput("cancel needs an operation_id or an idempotency_key")
        if not operation_id:
            row = mind.db.conn.execute(
                "SELECT operation_id FROM operations WHERE idempotency_key = ?",
                (idempotency_key,)).fetchone()
            if row is None:
                # Nothing was ever registered under that key. Saying so beats
                # inventing a failure for a client that gave up early.
                return {"operation_id": None, "cancelled": False,
                        "already_terminal": False,
                        "detail": "no operation found for that idempotency key"}
            operation_id = row["operation_id"]

        op = mind.memory.get_operation(operation_id)
        terminal = op["status"] in ("completed", "failed", "cancelled", "interrupted")

        cancelled_work: list[str] = []
        killed: list[str] = []
        generations: list[dict[str, Any]] = []

        if not terminal:
            for row in mind.db.conn.execute(
                    "SELECT work_id, status, lease_owner FROM work_items"
                    " WHERE operation_id = ? AND status NOT IN ('done','failed','cancelled')",
                    (operation_id,)):
                mind.work.cancel(work_id=row["work_id"], actor=actor, reason=reason)
                cancelled_work.append(row["work_id"])
                owner = row["lease_owner"]
                if owner and owner in sup.neuocytes:
                    sup._kill_neuocyte(owner, reason=f"operation cancelled: {reason}")
                    killed.append(owner)

            # Only the role that OWNS this operation. Cancelling both would
            # stop an unrelated in-flight audit whenever a conversation was
            # cancelled -- the operation row records its actor, so there is no
            # need to guess.
            owning = op["actor"] if op["actor"] in ("ego", "id") else None
            for role in ([owning] if owning else []):
                agent = next((a for a in mind.work.live_agents(role=role)), None)
                handle = (agent or {}).get("session_handle")
                if not handle:
                    continue
                try:
                    out = sup.client("inference").call("cancel_generation",
                                                       session_id=handle)
                    if out.get("cancel_requested"):
                        generations.append({"role": role, "session_id": handle})
                except Exception:  # noqa: BLE001
                    pass

        def body(m: Mutation) -> None:
            if not terminal:
                m.sql("UPDATE operations SET status = 'cancelled', updated_at = ?"
                      " WHERE operation_id = ?", (time.time(), operation_id))
            m.emit(EventKind.OPERATION_CANCELLED, {
                "operation_id": operation_id, "actor": actor, "reason": reason,
                "already_terminal": terminal, "prior_status": op["status"],
                "cancelled_work": cancelled_work, "killed_neuocytes": killed,
                "generations_stopped": generations,
            })
            for g in generations:
                m.emit(EventKind.GENERATION_CANCELLED, g)

        receipt, _ = mind.writer.apply(
            body, actor=actor, operation_id=operation_id, bump_version=not terminal,
            mutation_id=f"cancel:{operation_id}")

        return {
            "operation_id": operation_id,
            "cancelled": not terminal,
            "already_terminal": terminal,
            "prior_status": op["status"],
            "cancelled_work": cancelled_work,
            "killed_neuocytes": killed,
            "generations_stopped": generations,
            "receipt_id": receipt.receipt_id,
            "detail": ("operation was already terminal; nothing to stop"
                       if terminal else
                       "queued work cancelled, running neuocytes killed, "
                       "in-flight generation asked to stop between tokens"),
        }

    def cancel_generation(*, role: str, reason: str = "cancelled") -> dict[str, Any]:
        """Stop a role's current generation without touching its operation."""
        agent = next((a for a in mind.work.live_agents(role=role)), None)
        handle = (agent or {}).get("session_handle")
        if not handle:
            raise NotFound("role has no live inference session", role=role)
        out = sup.client("inference").call("cancel_generation", session_id=handle)
        mind.writer.record_rejection(
            actor="supervisor", reason=reason, kind=EventKind.GENERATION_CANCELLED,
            payload={"role": role, "session_id": handle})
        return out

    # ------------------------------------------------------------------
    def side_channel(*, to_role: str, kind: str, payload: dict[str, Any] | None = None,
                     from_role: str = "supervisor", durable: bool = True
                     ) -> dict[str, Any]:
        """Deliver a message to a persistent role.

        Delivery may be transient; influence may not be unaudited. This used
        to push into an in-memory list on the role process, bounded at 64 and
        dropping the oldest -- and Id's `status()` drained that list into its
        own cognition. A receipt-free message was therefore shaping persistent
        cognition with no author, no body and no record.

        A message that can wake a role or enter its cognitive input is now
        queued in the durable mailbox, with its author, its exact body and its
        consumption relationship on the record. The transient signal is still
        delivered for liveness-style nudges, and `durable=False` keeps that
        behaviour for a caller that genuinely wants a hint rather than an
        input -- but it then cannot enter a trigger bundle.
        """
        from . import mailbox

        if to_role not in ("ego", "id"):
            raise InvalidInput("side channel targets ego or id", to_role=to_role)
        body = payload or {}
        queued = None
        if durable:
            summary = str(body.get("message") or body.get("summary")
                          or f"{kind} from {from_role}")
            queued = sup.methods()["role_enqueue_trigger"](
                role=to_role, kind="role_message", source=from_role or "supervisor",
                summary=summary[:mailbox.MAX_SUMMARY],
                payload={"kind": kind, "from_role": from_role, "body": body},
                # One role speaking to the other addresses the role itself, not
                # any one of its interactions.
                ambient=True)
        # Peer traffic reaches the live room from here, which is the one
        # place it all passes through, so the view is assembled from what was
        # actually carried rather than from each caller remembering to report
        # itself. Authorship is `from_role`, which the Harness sets.
        #
        # Not the Operator: a room entry is an utterance, and the Operator
        # utters once even though the Harness delivers to both minds. That
        # entry is posted by `operator_backchannel`, where the one act is.
        if from_role in ("ego", "id"):
            sup.room.post(author=from_role,
                          text=str(body.get("message") or ""), kind=kind)

        transient: dict[str, Any] = {}
        try:
            transient = sup.client(to_role).call(
                "signal", kind=kind, payload=body, from_role=from_role)
        except Exception as exc:  # noqa: BLE001
            # The durable trigger is what matters; a role that is restarting
            # will still see the message at its next turn.
            transient = {"accepted": False, "reason": str(exc)[:200]}
        return {**transient, "durable": bool(queued),
                "trigger_id": (queued or {}).get("trigger_id"),
                "note": ("queued in the role mailbox and attributable; it "
                         "becomes cognitive input at a turn boundary"
                         if queued else
                         "transient only; this cannot enter a trigger bundle")}

    def shutdown() -> dict[str, Any]:
        sup._stop.set()
        return {"stopping": True}

    return {
        # health / status
        "health": health, "capabilities": capabilities, "status": status,
        "debug_threads": debug_threads,
        # agents
        "register_agent": register_agent, "heartbeat": heartbeat,
        "role_environment": role_environment,
        "retire_agent": retire_agent,
        # memory
        "recall": recall, "remember": remember, "get_memory": get_memory,
        "record_conclusion": record_conclusion, "get_conclusion": get_conclusion,
        "record_audit": record_audit, "disagreements": disagreements,
        "open_disagreement": open_disagreement,
        "operator_close_disagreement": operator_close_disagreement,
        "operator_retract_memory": operator_retract_memory,
        # provenance
        "provenance": provenance, "verify_integrity": verify_integrity,
        "history": history, "audit_dossier": audit_dossier,
        "maintenance_context": maintenance_context,
        # work
        "store_footprint": store_footprint, "store_prune": store_prune,
        "ego_read_attachment": ego_read_attachment,
        "ego_surface_result": ego_surface_result,
        "admit_work": admit_work, "lease_work": lease_work,
        "complete_work": complete_work, "fail_work": fail_work,
        "cancel_work": cancel_work, "get_work": get_work, "queue_stats": queue_stats,
        "kill_all_neuocytes": kill_all_neuocytes,
        # snapshots
        "publish_ego_snapshot": publish_ego_snapshot,
        "acquire_snapshot": acquire_snapshot,
        "release_snapshot_ref": release_snapshot_ref,
        "snapshot_tokens": snapshot_tokens, "list_snapshots": list_snapshots,
        "ensure_snapshot": ensure_snapshot,
        # operations
        "open_operation": open_operation, "get_operation": get_operation,
        "update_operation": update_operation,
        # cognitive verbs
        "ego_converse": ego_converse, "ego_investigate": ego_investigate,
        "ego_recall": ego_recall, "ego_status": ego_status,
        "role_answer": role_answer,
        "id_introspect": id_introspect, "id_health": id_health, "id_audit": id_audit,
        "harness_request_audit": harness_request_audit,
        "harness_commit_audit": harness_commit_audit,
        "id_disagreements": id_disagreements, "id_maintenance": id_maintenance,
        # misc
        "cancel_operation": cancel_operation,
        "cancel_generation": cancel_generation,
        "side_channel": side_channel, "shutdown": shutdown,
    }
