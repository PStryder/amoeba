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

import time
from typing import TYPE_CHECKING, Any, Sequence

from .arbiter import ResourceSnapshot
from .errors import (
    BackendUnavailable, InvalidInput, MindError, NotFound, ResourceExhausted,
)
from .ids import new_id
from .store.events import EventKind, read_events
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

SCHEMA_VERSION = "1.0.0"


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
                       model_generation: str | None = None, work_id: str | None = None
                       ) -> dict[str, Any]:
        incarnation, receipt = mind.work.register_agent(
            agent_id=agent_id, role=role, pid=pid, session_handle=session_handle,
            snapshot_id=snapshot_id, model_generation=model_generation, work_id=work_id,
        )
        return {"agent_id": agent_id, "incarnation": incarnation,
                "receipt_id": receipt.receipt_id}

    def heartbeat(*, agent_id: str) -> dict[str, Any]:
        mind.work.heartbeat(agent_id)
        return {"ok": True, "at": time.time()}

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
        return mind.memory.get_memory(memory_id)

    def record_conclusion(*, claim: str, produced_by: str,
                          evidence: Sequence[dict[str, Any]] = (),
                          uncertainty: float | None = None,
                          alternatives: Sequence[str] | None = None,
                          operation_id: str | None = None,
                          model_identity: str | None = None,
                          snapshot_id: str | None = None,
                          mutation_id: str | None = None) -> dict[str, Any]:
        cid, receipt = mind.memory.record_conclusion(
            claim=claim, produced_by=produced_by, evidence=evidence,
            uncertainty=uncertainty, alternatives=alternatives, operation_id=operation_id,
            model_identity=model_identity, snapshot_id=snapshot_id, mutation_id=mutation_id,
        )
        return {"conclusion_id": cid, "receipt_id": receipt.receipt_id,
                "state_version": receipt.result_version}

    def get_conclusion(*, conclusion_id: str) -> dict[str, Any]:
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
        return mind.memory.get_disagreements(status=status, limit=limit)

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
        return mind.provenance(operation_id=operation_id)

    def verify_integrity(*, deep: bool = True) -> dict[str, Any]:
        return mind.verify_integrity(deep=deep)

    def history(*, operation_id: str | None = None, correlation_id: str | None = None,
                kinds: Sequence[str] | None = None, since_seq: int = 0,
                limit: int = 100) -> list[dict[str, Any]]:
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
        if conclusion_id:
            concl = mind.memory.get_conclusion(conclusion_id)
            dossier["conclusion"] = concl
            operation_id = operation_id or concl.get("operation_id")
        if not operation_id:
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
    def admit_work(*, objective: str, work_class: str, origin_actor: str,
                   operation_id: str | None = None, priority: int = 0,
                   budget_tokens: int | None = None, wall_seconds: float | None = None,
                   maintenance_depth: int = 0, snapshot_id: str | None = None,
                   depends_on: Sequence[str] | None = None,
                   board_access: str = "read_write",
                   sandbox_allowed: bool = False) -> dict[str, Any]:
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

    def complete_work(*, work_id: str, neuocyte_id: str, fencing_token: int,
                      result: Any, pinned_state_ver: int | None = None
                      ) -> dict[str, Any]:
        receipt = mind.work.complete(
            work_id=work_id, neuocyte_id=neuocyte_id, fencing_token=fencing_token,
            result=result, pinned_state_ver=pinned_state_ver,
        )
        return {"receipt_id": receipt.receipt_id, "state_version": receipt.result_version,
                "replayed": receipt.replayed}

    def fail_work(*, work_id: str, neuocyte_id: str, fencing_token: int, failure: str,
                  requeue: bool = True) -> dict[str, Any]:
        receipt = mind.work.fail(work_id=work_id, neuocyte_id=neuocyte_id,
                                 fencing_token=fencing_token, failure=failure,
                                 requeue=requeue)
        return {"receipt_id": receipt.receipt_id}

    def cancel_work(*, work_id: str, actor: str = "supervisor", reason: str = ""
                    ) -> dict[str, Any]:
        receipt = mind.work.cancel(work_id=work_id, actor=actor, reason=reason)
        return {"receipt_id": receipt.receipt_id}

    def get_work(*, work_id: str) -> dict[str, Any]:
        return mind.work.get_work(work_id)

    def queue_stats() -> dict[str, Any]:
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

    def ego_converse(*, message: str, conversation_id: str | None = None,
                     idempotency_key: str | None = None, max_tokens: int = 384,
                     temperature: float = 0.0) -> dict[str, Any]:
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            out = sup.client("ego").call(
                "converse", message=message, conversation_id=conversation_id,
                operation_id=op_id, max_tokens=max_tokens, temperature=temperature,
            )
            if out.get("is_simulated"):
                limitations.append(
                    "SIMULATED BACKEND: this text was produced by a deterministic "
                    "stub, not by model inference"
                )
            evidence = [{"memory_id": m} for m in out.get("cited_memory_ids", [])]
            evidence.append({"note": f"operation {op_id} inference events"})
            cid, _ = mind.memory.record_conclusion(
                claim=out["answer"][:2000] or "(empty answer)",
                produced_by="ego", evidence=evidence,
                uncertainty=None, operation_id=op_id,
                model_identity=out.get("model_generation"),
            )
            out["conclusion_id"] = cid
            if out.get("tool_requests"):
                limitations.append(
                    "model requested tools; they were parsed and recorded but "
                    "ego_converse does not execute tools"
                )
            return out

        return _run_operation("ego_converse", "ego",
                              {"message": message, "conversation_id": conversation_id},
                              idempotency_key, run)

    def ego_investigate(*, question: str, constraints: str = "",
                        budget_tokens: int | None = None,
                        idempotency_key: str | None = None) -> dict[str, Any]:
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            out = sup.client("ego").call("investigate", question=question,
                                         constraints=constraints, operation_id=op_id)
            if out.get("is_simulated"):
                limitations.append("SIMULATED BACKEND: not model inference")
            snap = ensure_snapshot(operation_id=op_id)
            admitted = admit_work(
                objective=f"{question} :: {out.get('plan') or 'investigate'}",
                work_class="user", origin_actor="ego", operation_id=op_id,
                budget_tokens=budget_tokens, snapshot_id=snap["snapshot_id"],
            )
            if not admitted["admitted"]:
                limitations.append(f"work not admitted: {admitted['reason']}")
            out["snapshot_id"] = snap["snapshot_id"]
            out["work"] = admitted
            out["accepted_scope"] = {
                "question": question, "constraints": constraints,
                "budget_tokens": admitted.get("granted_budget_tokens"),
                "deadline": admitted.get("deadline"),
            }
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
                      idempotency_key: str | None = None) -> dict[str, Any]:
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            out = sup.client("id").call("introspect", question=question, scope=scope,
                                        operation_id=op_id)
            if out.get("is_simulated"):
                limitations.append("SIMULATED BACKEND: not model inference")
            limitations.append(
                "the 'measured' block is read from durable state; the "
                "'interpretation' block is model output and is not evidence"
            )
            return out

        return _run_operation("id_introspect", "id", {"question": question, "scope": scope},
                              idempotency_key, run)

    def id_health(*, scope: str = "all") -> dict[str, Any]:
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
            base["sandbox"] = (sup.sandboxes.capabilities() if sup.sandboxes
                               else {"sandbox_available": False})
        except Exception as exc:  # noqa: BLE001
            base["sandbox"] = {"error": repr(exc)}
        return base

    def id_audit(*, conclusion_id: str | None = None, operation_id: str | None = None,
                 focus: str = "", idempotency_key: str | None = None) -> dict[str, Any]:
        def run(op_id: str, limitations: list[str]) -> dict[str, Any]:
            out = sup.client("id").call("audit", conclusion_id=conclusion_id,
                                        operation_id_target=operation_id, focus=focus,
                                        operation_id=op_id)
            if out.get("is_simulated"):
                limitations.append("SIMULATED BACKEND: not model inference")
            audit_id, _ = mind.memory.record_audit(
                target_kind=out["target_kind"], target_id=out["target_id"],
                verdict=out["verdict"], focus=focus or None,
                findings=out.get("findings"), unresolved=out.get("unresolved"),
                evidence={"hash_chain_ok": out["evidence_reviewed"].get("hash_chain_ok"),
                          "event_count": len(out["evidence_reviewed"].get("events", [])),
                          "unresolved_content":
                              out["evidence_reviewed"].get("unresolved_content", [])},
                operation_id=op_id,
            )
            out["audit_id"] = audit_id
            if out["evidence_reviewed"].get("unresolved_content"):
                limitations.append("some referenced content could not be resolved; "
                                   "the audit is incomplete")
            # A contested audit of an Ego conclusion is a real disagreement, and
            # it is recorded as one rather than quietly overwriting the claim.
            if conclusion_id and out["verdict"] in ("contested", "unsupported"):
                concl = mind.memory.get_conclusion(conclusion_id)
                did, _ = mind.memory.open_disagreement(
                    subject_kind="conclusion", subject_id=conclusion_id,
                    claim_a=concl["claim"], actor_a=concl["produced_by"],
                    claim_b="; ".join(out.get("findings") or ["contested"]),
                    actor_b="id",
                    evidence_a={"conclusion_evidence": concl.get("evidence", [])},
                    evidence_b={"audit_id": audit_id},
                    operation_id=op_id,
                )
                out["disagreement_id"] = did
            return out

        return _run_operation("id_audit", "id",
                              {"conclusion_id": conclusion_id, "operation_id": operation_id,
                               "focus": focus}, idempotency_key, run)

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
                     from_role: str = "supervisor") -> dict[str, Any]:
        if to_role not in ("ego", "id"):
            raise InvalidInput("side channel targets ego or id", to_role=to_role)
        return sup.client(to_role).call("signal", kind=kind, payload=payload or {},
                                        from_role=from_role)

    def shutdown() -> dict[str, Any]:
        sup._stop.set()
        return {"stopping": True}

    return {
        # health / status
        "health": health, "capabilities": capabilities, "status": status,
        "debug_threads": debug_threads,
        # agents
        "register_agent": register_agent, "heartbeat": heartbeat,
        "retire_agent": retire_agent,
        # memory
        "recall": recall, "remember": remember, "get_memory": get_memory,
        "record_conclusion": record_conclusion, "get_conclusion": get_conclusion,
        "record_audit": record_audit, "disagreements": disagreements,
        "open_disagreement": open_disagreement,
        # provenance
        "provenance": provenance, "verify_integrity": verify_integrity,
        "history": history, "audit_dossier": audit_dossier,
        "maintenance_context": maintenance_context,
        # work
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
        "id_introspect": id_introspect, "id_health": id_health, "id_audit": id_audit,
        "id_disagreements": id_disagreements, "id_maintenance": id_maintenance,
        # misc
        "cancel_operation": cancel_operation,
        "cancel_generation": cancel_generation,
        "side_channel": side_channel, "shutdown": shutdown,
    }
