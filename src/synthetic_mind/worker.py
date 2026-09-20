"""Disposable, Ego-derived workers.

Lifecycle, in the order the design requires:

1. The supervisor publishes (or reuses) a consistent Ego snapshot taken at an
   inference boundary.
2. The worker takes a reference on that snapshot so the published storage
   cannot be reclaimed while it is reading from it.
3. It instantiates a session from the snapshot -- by forking the physically
   shared prefix when the backend supports it, otherwise by recomputing the
   exact recorded token prefix.
4. It appends its own private instruction tail and lets that continuation
   evolve. The tail is private: the source Ego session and sibling workers
   never see it.
5. It stays pinned to that snapshot and model generation for its whole life.
   It never reads a newer snapshot; a replacement worker is forked from the
   newer one instead.
6. It commits findings as evidence/proposals through a durable receipt, with
   its fencing token, and retires.

Killing a worker is always safe: its lease expires, the fencing token advances,
and its late result can no longer commit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any, Sequence

from .config import Config, load_config
from .errors import CapabilityUnsupported, Fenced, MindError
from .ids import new_id
from .logging_setup import get_logger, setup_logging
from .rpc import RpcClient, read_or_create_token
from .tools import parse_tool_calls, strip_tool_calls

WORKER_INSTRUCTION = """You are a bounded worker forked from the mind's Ego context.
You inherited the context above. Do one narrow task and stop.

Task: {objective}

Reply with exactly three lines:
FINDING: <one sentence, the thing you actually determined>
CONFIDENCE: <a number between 0 and 1>
EVIDENCE: <what in the context above supports it, or "none in context">"""

MAINTENANCE_INSTRUCTION = """You are a bounded maintenance worker for a synthetic mind.
You were NOT given Ego's private context: you get a narrow task and references to
durable state. Do the task and stop.

Task: {objective}

Relevant durable state:
{state}

Reply with exactly three lines:
FINDING: <one sentence>
CONFIDENCE: <a number between 0 and 1>
EVIDENCE: <which state references support it, or "none">"""


class Worker:
    def __init__(self, cfg: Config, *, worker_id: str) -> None:
        self.cfg = cfg
        self.worker_id = worker_id
        self.log = get_logger("worker")
        self.token = read_or_create_token(cfg.token_path)
        self.sup = RpcClient(cfg.supervisor_host, cfg.supervisor_port, self.token,
                             name=f"{worker_id}->supervisor")
        self.inf = RpcClient(cfg.supervisor_host, cfg.inference_port, self.token,
                             name=f"{worker_id}->inference")
        self.session_id: str | None = None
        self.ref_id: str | None = None
        self.snapshot_id: str | None = None
        self.model_generation = ""
        self.pinned_state_version: int | None = None

    # ------------------------------------------------------------------
    def run(self, *, work_id: str | None = None) -> int:
        self.sup.connect(retries=20, delay=0.25)
        self.inf.connect(retries=20, delay=0.25)
        caps = self.inf.call("capabilities")
        self.model_generation = caps.get("model_generation", "")

        item = self.sup.call("lease_work", worker_id=self.worker_id, work_id=work_id)
        if not item:
            self.log.info("%s: nothing to lease", self.worker_id)
            return 0
        work_id = item["work_id"]
        fencing_token = item["fencing_token"]
        self.pinned_state_version = item.get("pinned_state_ver")
        deadline = item.get("deadline") or (time.time() + self.cfg.arbiter.worker_wall_seconds)
        budget = item.get("budget_tokens") or self.cfg.arbiter.worker_token_budget

        self.sup.call("register_agent", agent_id=self.worker_id, role="worker",
                      pid=os.getpid(), work_id=work_id,
                      model_generation=self.model_generation)
        try:
            result = self._execute(item, caps=caps, budget=budget, deadline=deadline)
            self.sup.call("complete_work", work_id=work_id, worker_id=self.worker_id,
                          fencing_token=fencing_token, result=result,
                          pinned_state_ver=self.pinned_state_version)
            self.log.info("%s completed %s", self.worker_id, work_id)
            return 0
        except Fenced as exc:
            self.log.warning("%s fenced on %s: %s", self.worker_id, work_id, exc.message)
            return 0
        except MindError as exc:
            self._fail(work_id, fencing_token, exc.message)
            return 1
        except Exception as exc:  # noqa: BLE001
            self._fail(work_id, fencing_token,
                       f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")
            return 1
        finally:
            self._retire()

    def _fail(self, work_id: str, fencing_token: int, message: str) -> None:
        try:
            self.sup.call("fail_work", work_id=work_id, worker_id=self.worker_id,
                          fencing_token=fencing_token, failure=message[:2000])
        except Exception:  # noqa: BLE001
            self.log.exception("could not report failure for %s", work_id)

    # ------------------------------------------------------------------
    def _execute(self, item: dict[str, Any], *, caps: dict[str, Any],
                 budget: int, deadline: float) -> dict[str, Any]:
        objective = item["objective"]
        work_class = item["work_class"]
        if work_class == "maintenance":
            return self._execute_maintenance(item, budget=budget, deadline=deadline)
        return self._execute_ego_derived(item, caps=caps, budget=budget, deadline=deadline)

    # -- user-directed work: forked from an Ego snapshot -----------------
    def _execute_ego_derived(self, item: dict[str, Any], *, caps: dict[str, Any],
                             budget: int, deadline: float) -> dict[str, Any]:
        snapshot = self.sup.call("acquire_snapshot", holder=self.worker_id,
                                 snapshot_id=item.get("snapshot_id"))
        self.snapshot_id = snapshot["snapshot_id"]
        self.ref_id = snapshot["ref_id"]

        if snapshot["model_generation"] != self.model_generation:
            # Cached tensors from another weight/tokenizer/positional
            # configuration are never reinterpreted.
            self.sup.call("release_snapshot_ref", ref_id=self.ref_id,
                          actor=self.worker_id)
            self.ref_id = None
            raise CapabilityUnsupported(
                "snapshot belongs to a different model generation; refusing reuse",
                snapshot_generation=snapshot["model_generation"],
                current_generation=self.model_generation,
            )

        instantiation = self._instantiate_from_snapshot(snapshot, caps=caps)
        prompt = WORKER_INSTRUCTION.format(objective=item["objective"])
        rendered = self.inf.call(
            "apply_chat_template",
            messages=[{"role": "user", "content": prompt}], add_assistant=True,
        )
        self.inf.call("ingest_text", session_id=self.session_id, text=rendered,
                      parse_special=True)
        out = self.inf.call("generate", session_id=self.session_id,
                            max_tokens=min(budget, 256), temperature=0.0,
                            deadline=deadline)
        parsed = _parse_finding(out["text"])
        return {
            "kind": "ego_derived_finding",
            "objective": item["objective"],
            "snapshot_id": self.snapshot_id,
            "snapshot_version": snapshot["version"],
            "model_generation": self.model_generation,
            "pinned_state_version": self.pinned_state_version,
            "instantiation": instantiation,
            "finding": parsed["finding"],
            "confidence": parsed["confidence"],
            "evidence_note": parsed["evidence"],
            "raw_text": out["text"],
            "tool_requests": [{"name": r.name, "arguments": r.arguments}
                              for r in parse_tool_calls(out["text"])],
            "completion_tokens": out["completion_tokens"],
            "is_simulated": out.get("is_simulated", False),
            "worker_id": self.worker_id,
        }

    def _instantiate_from_snapshot(self, snapshot: dict[str, Any],
                                   *, caps: dict[str, Any]) -> dict[str, Any]:
        """Fork the published prefix, or recompute it exactly if forking is not
        available. The two are reported distinctly, never conflated."""
        handle = snapshot.get("backend_handle")
        can_fork = (
            caps.get("kv_mode") == "shared_prefix"
            and handle
            and snapshot.get("status") == "published"
        )
        if can_fork:
            try:
                sess = self.inf.call(
                    "fork_prefix", src_session_id=handle,
                    prefix_len=snapshot["token_count"], role="worker",
                    snapshot_id=snapshot["snapshot_id"],
                )
                self.session_id = sess["session_id"]
                return {"method": "forked_shared_prefix", "kv_mode": sess["kv_mode"],
                        "prefix_len": sess["prefix_len"],
                        "note": "KV cells physically shared with the source sequence"}
            except Exception as exc:  # noqa: BLE001
                self.log.warning("fork failed (%s); falling back to recomputation", exc)

        sess = self.inf.call("open_session", role="worker")
        self.session_id = sess["session_id"]
        tokens = self.sup.call("snapshot_tokens", snapshot_id=snapshot["snapshot_id"])
        t0 = time.perf_counter()
        self.inf.call("restore_prefix", session_id=self.session_id, tokens=tokens,
                      snapshot_id=snapshot["snapshot_id"])
        return {
            "method": "recomputed_exact_prefix",
            "kv_mode": "recomputed",
            "prefix_len": len(tokens),
            "recompute_seconds": time.perf_counter() - t0,
            "note": ("the exact recorded token prefix was re-evaluated; this reproduces "
                     "the context, unlike summarising it, which would be different "
                     "behaviour"),
        }

    # -- maintenance work: narrow task, no Ego snapshot -------------------
    def _execute_maintenance(self, item: dict[str, Any], *, budget: int,
                             deadline: float) -> dict[str, Any]:
        """Id maintenance workers get references to state, never a snapshot of
        Id's private context."""
        state = self.sup.call("maintenance_context", objective=item["objective"])
        sess = self.inf.call("open_session", role="worker")
        self.session_id = sess["session_id"]
        prompt = MAINTENANCE_INSTRUCTION.format(
            objective=item["objective"],
            state=json.dumps(state, indent=2, default=str)[:2500],
        )
        rendered = self.inf.call(
            "apply_chat_template",
            messages=[{"role": "user", "content": prompt}], add_assistant=True,
        )
        self.inf.call("ingest_text", session_id=self.session_id, text=rendered,
                      parse_special=True)
        out = self.inf.call("generate", session_id=self.session_id,
                            max_tokens=min(budget, 256), temperature=0.0,
                            deadline=deadline)
        parsed = _parse_finding(out["text"])
        return {
            "kind": "maintenance_finding",
            "objective": item["objective"],
            "snapshot_id": None,
            "received_ego_snapshot": False,
            "state_references": state.get("references", []),
            "model_generation": self.model_generation,
            "pinned_state_version": self.pinned_state_version,
            "finding": parsed["finding"],
            "confidence": parsed["confidence"],
            "evidence_note": parsed["evidence"],
            "raw_text": out["text"],
            "completion_tokens": out["completion_tokens"],
            "is_simulated": out.get("is_simulated", False),
            "worker_id": self.worker_id,
        }

    # ------------------------------------------------------------------
    def _retire(self) -> None:
        """Retirement releases inference and snapshot resources but never
        destroys authoritative work state."""
        try:
            if self.session_id:
                self.inf.call("close_session", session_id=self.session_id,
                              keep_prefix=bool(self.snapshot_id))
        except Exception:  # noqa: BLE001
            self.log.debug("session close failed", exc_info=True)
        try:
            if self.ref_id:
                self.sup.call("release_snapshot_ref", ref_id=self.ref_id,
                              actor=self.worker_id)
        except Exception:  # noqa: BLE001
            self.log.debug("snapshot ref release failed", exc_info=True)
        try:
            self.sup.call("retire_agent", agent_id=self.worker_id, reason="task complete")
        except Exception:  # noqa: BLE001
            pass
        self.sup.close()
        self.inf.close()


def _parse_finding(text: str) -> dict[str, Any]:
    clean = strip_tool_calls(text)
    finding, confidence, evidence = "", 0.5, ""
    for line in clean.splitlines():
        upper = line.upper()
        if upper.startswith("FINDING:"):
            finding = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            raw = line.split(":", 1)[1].strip().split()[0] if ":" in line else ""
            try:
                confidence = max(0.0, min(1.0, float(raw.rstrip(".,"))))
            except ValueError:
                confidence = 0.5
        elif upper.startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()
    if not finding:
        finding = clean.strip()[:400] or "(worker produced no parsable finding)"
    return {"finding": finding, "confidence": confidence, "evidence": evidence}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synthetic_mind.worker")
    ap.add_argument("--config", default=os.environ.get("SYNTHETIC_MIND_CONFIG"))
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--work-id", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, "worker")
    worker_id = args.worker_id or new_id("wk")
    return Worker(cfg, worker_id=worker_id).run(work_id=args.work_id)


if __name__ == "__main__":
    raise SystemExit(main())
