"""Ego and Id: two long-lived processes with separate private contexts.

Ego does outward cognition -- conversation, investigation, synthesis. Id does
inward homeostasis -- introspection, audit, contradiction detection,
maintenance proposals.

They are distinct OS processes with distinct inference sessions and distinct
restart identities. Neither is the scheduler and neither is an administrator:
both must go through the supervisor's single writer for any consequential
change, and both receive a receipt in return.

Id's private context is never published as a snapshot. Only Ego's is.

Signals between the two are relayed by the supervisor rather than sent
peer-to-peer: each role exposes a ``signal`` method, and the supervisor's
``side_channel`` verb delivers to it. A direct peer connection was written and
removed during review, because it was never called and its presence implied a
path that did not exist.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .config import Config, load_config
from .errors import BackendUnavailable, NotFound
from .ids import new_id
from .logging_setup import get_logger, setup_logging
from .rpc import RpcClient, RpcServer, read_or_create_token
from .tools import parse_tool_calls, strip_tool_calls

EGO_SYSTEM = """You are Ego, the outward-facing half of a persistent amoeba.
You converse, investigate and synthesise. You have a durable memory of maintained
beliefs, and an append-only history that is evidence rather than memory.
Be concrete and brief. State uncertainty plainly rather than hedging everywhere.
When you assert something substantive, you are producing a conclusion that Id may
later audit against the recorded evidence, so do not claim support you do not have."""

ID_SYSTEM = """You are Id, the inward half of a persistent amoeba.
You observe outcomes, resource pressure, unfinished obligations and contradictions.
You audit Ego's conclusions against recorded evidence, not against Ego's own defence
of them. You may propose maintenance work; you may not perform administration.
Be terse and specific. Separate what you measured from what you inferred."""


class RoleProcess:
    """Shared machinery: inference session, supervisor link, RPC server, heartbeat."""

    role = "role"
    system_prompt = ""

    def __init__(self, cfg: Config, *, port: int) -> None:
        self.cfg = cfg
        self.port = port
        self.log = get_logger(self.role)
        self.token = read_or_create_token(cfg.token_path)
        self.sup = RpcClient(cfg.supervisor_host, cfg.supervisor_port, self.token,
                             name=f"{self.role}->supervisor")
        self.inf = RpcClient(cfg.supervisor_host, cfg.inference_port, self.token,
                             name=f"{self.role}->inference")
        self.session_id: str | None = None
        self.model_generation = ""
        self.capabilities: dict[str, Any] = {}
        self.incarnation = 0
        self.started_at = time.time()
        self.turns = 0
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._signals: list[dict[str, Any]] = []
        self.role_cfg = cfg.ego if self.role == "ego" else cfg.id

    # ------------------------------------------------------------------
    def connect(self) -> None:
        self.sup.connect(retries=60, delay=0.5)
        self.inf.connect(retries=60, delay=0.5)
        self.capabilities = self.inf.call("capabilities")
        self.model_generation = self.capabilities.get("model_generation", "")
        sess = self.inf.call("open_session", role=self.role)
        self.session_id = sess["session_id"]
        reg = self.sup.call("register_agent", agent_id=self.role, role=self.role,
                            pid=os.getpid(), session_handle=self.session_id,
                            model_generation=self.model_generation)
        self.incarnation = reg["incarnation"]
        self.log.info("%s incarnation %s on session %s (backend=%s simulated=%s)",
                      self.role, self.incarnation, self.session_id,
                      self.capabilities.get("backend_kind"),
                      self.capabilities.get("is_simulated"))
        self._prime_context()

    def _system_text(self) -> str:
        extra = self.role_cfg.system_prompt
        return self.system_prompt + (f"\n{extra}" if extra else "")

    def _prime_context(self) -> None:
        """Seed the private context with the role's system prompt only.

        The role's broader working context accumulates from here; it is not a
        curated shared-memory summary.
        """
        rendered = self.inf.call(
            "apply_chat_template",
            messages=[{"role": "system", "content": self._system_text()}],
            add_assistant=False,
        )
        self.inf.call("ingest_text", session_id=self.session_id, text=rendered,
                      parse_special=True)

    # ------------------------------------------------------------------
    def _infer(self, user_text: str, *, max_tokens: int = 384,
               temperature: float = 0.0, seed: int = 1234) -> dict[str, Any]:
        """Append one user turn to the private context and generate a reply."""
        rendered = self.inf.call(
            "apply_chat_template",
            messages=[{"role": "user", "content": user_text}],
            add_assistant=True,
        )
        self.inf.call("ingest_text", session_id=self.session_id, text=rendered,
                      parse_special=True)
        out = self.inf.call("generate", session_id=self.session_id,
                            max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.turns += 1
        return out

    def heartbeat_loop(self) -> None:
        while not self._stop.wait(5.0):
            try:
                self.sup.call("heartbeat", agent_id=self.role)
            except Exception:  # noqa: BLE001
                self.log.debug("heartbeat failed", exc_info=True)

    # ------------------------------------------------------------------
    def base_methods(self) -> dict[str, Any]:
        return {
            "health": self.health,
            "signal": self.receive_signal,
            "context_stats": self.context_stats,
            "shutdown": self.shutdown,
        }

    def health(self) -> dict[str, Any]:
        """Answerable even when inference is broken."""
        inf_ok, inf_detail = True, "ok"
        try:
            self.inf.call("health")
        except Exception as exc:  # noqa: BLE001
            inf_ok, inf_detail = False, repr(exc)
        return {
            "role": self.role,
            "status": "alive",
            "pid": os.getpid(),
            "incarnation": self.incarnation,
            "uptime_seconds": time.time() - self.started_at,
            "session_id": self.session_id,
            "model_generation": self.model_generation,
            "turns": self.turns,
            "inference_reachable": inf_ok,
            "inference_detail": inf_detail,
            "pending_signals": len(self._signals),
            "is_simulated_backend": bool(self.capabilities.get("is_simulated")),
        }

    def context_stats(self) -> dict[str, Any]:
        try:
            info = self.inf.call("session_tokens", session_id=self.session_id, limit=0)
            return {"n_past": info["n_past"], "prefix_len": info["prefix_len"],
                    "max_context_tokens": self.role_cfg.max_context_tokens}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def receive_signal(self, *, kind: str, payload: dict[str, Any] | None = None,
                       from_role: str = "", correlation_id: str | None = None
                       ) -> dict[str, Any]:
        """Accept a transient side-channel signal.

        Bounded queue with explicit backpressure: an overflowing channel drops
        the oldest signal and says so rather than growing without limit.
        """
        with self._lock:
            dropped = 0
            while len(self._signals) >= 64:
                self._signals.pop(0)
                dropped += 1
            self._signals.append({
                "kind": kind, "payload": payload or {}, "from_role": from_role,
                "correlation_id": correlation_id, "received_at": time.time(),
            })
        return {"accepted": True, "queued": len(self._signals), "dropped_oldest": dropped,
                "note": "transient signal; no state changed and no receipt issued"}

    def drain_signals(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._signals = self._signals, []
            return out

    def shutdown(self) -> dict[str, Any]:
        self._stop.set()
        return {"stopping": True}

    # ------------------------------------------------------------------
    def serve(self) -> int:
        self.connect()
        server = RpcServer(self.cfg.supervisor_host, self.port, token=self.token,
                           service_name=self.role,
                           on_error=lambda m, e: self.log.exception("rpc %s: %s", m, e))
        server.register_all(self.base_methods())
        server.register_all(self.methods())
        server.serve_in_thread()
        threading.Thread(target=self.heartbeat_loop, daemon=True,
                         name=f"{self.role}-heartbeat").start()
        self.log.info("%s listening on %s:%s", self.role, self.cfg.supervisor_host, self.port)
        (self.cfg.state_dir / f"{self.role}.ready").write_text(
            json.dumps({"pid": os.getpid(), "port": self.port,
                        "incarnation": self.incarnation}), encoding="utf-8")

        def _sig(_s: int, _f: Any) -> None:
            self._stop.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)
            except (ValueError, OSError):
                pass
        try:
            while not self._stop.is_set():
                self._stop.wait(0.5)
        finally:
            self.log.info("%s stopping", self.role)
            server.shutdown()
            try:
                if self.session_id:
                    self.inf.call("close_session", session_id=self.session_id)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.sup.call("retire_agent", agent_id=self.role, reason="clean shutdown")
            except Exception:  # noqa: BLE001
                pass
            for p in (self.cfg.state_dir / f"{self.role}.ready",):
                try:
                    p.unlink()
                except OSError:
                    pass
        return 0

    def methods(self) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ego
# ---------------------------------------------------------------------------
class EgoProcess(RoleProcess):
    role = "ego"
    system_prompt = EGO_SYSTEM

    def methods(self) -> dict[str, Any]:
        return {
            "converse": self.converse,
            "investigate": self.investigate,
            "publish_snapshot": self.publish_snapshot,
            "status": self.status,
        }

    # -- cognition ------------------------------------------------------
    def converse(self, *, message: str, conversation_id: str | None = None,
                 operation_id: str | None = None, max_tokens: int = 384,
                 temperature: float = 0.0, recall_limit: int = 5) -> dict[str, Any]:
        """One conversational turn, grounded in maintained memory.

        Recall searches *memory*, not raw history: contradictory raw events do
        not silently become active beliefs.
        """
        recalled = self.sup.call("recall", query=message, limit=recall_limit)
        memo = "\n".join(
            f"- [{m['memory_id']}] ({m['kind']}, confidence {m['confidence']:.2f}) {m['claim']}"
            for m in recalled
        ) or "- (no maintained memory matched this)"
        prompt = (
            f"Maintained memory that may be relevant:\n{memo}\n\n"
            f"User message:\n{message}\n\n"
            "Answer briefly. If you rely on a memory above, cite its id in square brackets."
        )
        out = self._infer(prompt, max_tokens=max_tokens, temperature=temperature)
        text = strip_tool_calls(out["text"]).strip()
        tool_requests = [
            {"name": r.name, "arguments": r.arguments} for r in parse_tool_calls(out["text"])
        ]
        cited = [m["memory_id"] for m in recalled if m["memory_id"] in out["text"]]
        return {
            "answer": text,
            "raw_text": out["text"],
            "recalled": [{"memory_id": m["memory_id"], "claim": m["claim"],
                          "confidence": m["confidence"]} for m in recalled],
            "cited_memory_ids": cited,
            "tool_requests": tool_requests,
            "finish_reason": out["finish_reason"],
            "completion_tokens": out["completion_tokens"],
            "time_to_first_token": out["time_to_first_token"],
            "model_generation": out.get("model_generation"),
            "is_simulated": out.get("is_simulated", False),
            "conversation_id": conversation_id,
            "operation_id": operation_id,
            "session_n_past": self.context_stats().get("n_past"),
        }

    def investigate(self, *, question: str, constraints: str = "",
                    operation_id: str | None = None, max_tokens: int = 256
                    ) -> dict[str, Any]:
        """Turn a question into a bounded plan and a claim worth committing.

        The actual fan-out work happens in disposable neuocytes forked from a
        published Ego snapshot; this produces the framing and the scope.
        """
        prompt = (
            f"Question to investigate:\n{question}\n"
            f"Constraints: {constraints or 'none stated'}\n\n"
            "Reply with exactly two lines:\n"
            "PLAN: <one sentence describing what to check>\n"
            "CLAIM: <your current best answer, with explicit uncertainty>"
        )
        out = self._infer(prompt, max_tokens=max_tokens)
        text = strip_tool_calls(out["text"])
        plan, claim = "", text.strip()
        for line in text.splitlines():
            if line.upper().startswith("PLAN:"):
                plan = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CLAIM:"):
                claim = line.split(":", 1)[1].strip()
        return {
            "question": question, "plan": plan or text.strip()[:280],
            "claim": claim, "raw_text": out["text"],
            "completion_tokens": out["completion_tokens"],
            "model_generation": out.get("model_generation"),
            "is_simulated": out.get("is_simulated", False),
            "operation_id": operation_id,
        }

    # -- snapshot publication -------------------------------------------
    def publish_snapshot(self, *, operation_id: str | None = None) -> dict[str, Any]:
        """Publish an immutable prefix of Ego's ACTUAL inference context.

        This is a snapshot of a valid prefix of the live context, not a curated
        summary of it. Ego keeps running on the same session afterwards: the
        published prefix is frozen only in the sense that nobody rewrites those
        positions, while Ego continues appending past them.
        """
        info = self.inf.call("session_tokens", session_id=self.session_id)
        tokens = info["tokens"]
        if not tokens:
            raise BackendUnavailable("ego context is empty; nothing to publish")
        text = self.inf.call("detokenize", tokens=tokens, special=True)
        result = self.sup.call(
            "publish_ego_snapshot",
            actor="ego",
            model_generation=self.model_generation,
            token_count=len(tokens),
            tokens=tokens,
            text=text,
            kv_mode=self.capabilities.get("kv_mode", "recomputed"),
            backend_handle=self.session_id,
            operation_id=operation_id,
        )
        self.log.info("published snapshot %s v%s (%d tokens, kv_mode=%s)",
                      result["snapshot_id"], result["version"], len(tokens),
                      result["kv_mode"])
        return result

    def status(self) -> dict[str, Any]:
        base = self.health()
        base.update({
            "context": self.context_stats(),
            "pending_signals": self.drain_signals(),
        })
        return base


# ---------------------------------------------------------------------------
# Id
# ---------------------------------------------------------------------------
class IdProcess(RoleProcess):
    role = "id"
    system_prompt = ID_SYSTEM

    def methods(self) -> dict[str, Any]:
        return {
            "introspect": self.introspect,
            "audit": self.audit,
            "health_report": self.health_report,
            "propose_maintenance": self.propose_maintenance,
        }

    def introspect(self, *, question: str, scope: str = "all",
                   operation_id: str | None = None) -> dict[str, Any]:
        """Observed account of the mind's own operation.

        The measured part comes from durable state; only the interpretation
        comes from the model, and the two are returned separately.
        """
        measured = self.sup.call("status")
        prompt = (
            f"Question: {question}\n\nMeasured state of this mind:\n"
            f"{json.dumps(measured, indent=2, default=str)[:3000]}\n\n"
            "In at most four sentences, say what this state implies. "
            "Mark anything you are inferring rather than reading."
        )
        out = self._infer(prompt, max_tokens=256)
        return {
            "question": question,
            "measured": measured,
            "interpretation": strip_tool_calls(out["text"]).strip(),
            "measurement_source": "durable state (events, work queue, memory tables)",
            "interpretation_source": "model inference over the measured state",
            "is_simulated": out.get("is_simulated", False),
            "operation_id": operation_id,
        }

    def audit(self, *, conclusion_id: str | None = None, operation_id_target: str | None = None,
              focus: str = "", operation_id: str | None = None) -> dict[str, Any]:
        """Audit an Ego conclusion against recorded evidence.

        Id resolves the conclusion to its original inputs, model configuration
        and evidence through the event log. It never asks Ego to defend itself:
        Ego is not consulted anywhere in this path.
        """
        if not conclusion_id and not operation_id_target:
            raise NotFound("audit needs a conclusion_id or an operation_id")
        dossier = self.sup.call("audit_dossier", conclusion_id=conclusion_id,
                                operation_id=operation_id_target)
        prompt = (
            "Audit the following conclusion using ONLY the recorded evidence below. "
            "Do not assume anything the record does not show.\n\n"
            f"Focus: {focus or 'general support for the claim'}\n\n"
            f"{json.dumps(dossier, indent=2, default=str)[:4000]}\n\n"
            "Reply with exactly three lines:\n"
            "VERDICT: supported | contested | unsupported | inconclusive\n"
            "FINDING: <one sentence>\n"
            "UNRESOLVED: <one sentence naming what the record does not establish>"
        )
        out = self._infer(prompt, max_tokens=256)
        text = strip_tool_calls(out["text"])
        verdict, finding, unresolved = "inconclusive", "", ""
        for line in text.splitlines():
            upper = line.upper()
            if upper.startswith("VERDICT:"):
                candidate = line.split(":", 1)[1].strip().lower().split()[0] if ":" in line else ""
                if candidate in ("supported", "contested", "unsupported", "inconclusive"):
                    verdict = candidate
            elif upper.startswith("FINDING:"):
                finding = line.split(":", 1)[1].strip()
            elif upper.startswith("UNRESOLVED:"):
                unresolved = line.split(":", 1)[1].strip()
        return {
            "target_kind": "conclusion" if conclusion_id else "operation",
            "target_id": conclusion_id or operation_id_target,
            "verdict": verdict,
            "findings": [finding] if finding else [],
            "unresolved": [unresolved] if unresolved else [],
            "evidence_reviewed": dossier,
            "raw_text": out["text"],
            "asked_ego_to_defend_itself": False,
            "is_simulated": out.get("is_simulated", False),
            "operation_id": operation_id,
        }

    def health_report(self, *, scope: str = "all") -> dict[str, Any]:
        measured = self.sup.call("health")
        measured["id_process"] = self.health()
        return measured

    def propose_maintenance(self, *, objective: str, scope: str = "",
                            budget_tokens: int | None = None,
                            maintenance_depth: int = 0,
                            operation_id: str | None = None) -> dict[str, Any]:
        """Propose maintenance work. The arbiter decides; Id cannot self-admit."""
        return self.sup.call(
            "admit_work", objective=objective, work_class="maintenance",
            origin_actor="id", operation_id=operation_id,
            budget_tokens=budget_tokens, maintenance_depth=maintenance_depth,
        )


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amoeba.roles")
    ap.add_argument("role", choices=["ego", "id"])
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, args.role)
    if args.role == "ego":
        return EgoProcess(cfg, port=cfg.ego_port).serve()
    return IdProcess(cfg, port=cfg.id_port).serve()


if __name__ == "__main__":
    raise SystemExit(main())
