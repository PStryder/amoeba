"""The single GPU-owner process.

One process holds one set of weights. Ego, Id and every neuocyte are *sessions*
inside it, addressed by session id. Separating persistent agent identity (which
lives in the durable state) from ownership of the inference buffers (which
lives here) is what lets an agent be restarted without reloading the model, and
lets this process be restarted without destroying the mind.

When this process dies, every KV handle it hosted is gone. The snapshots
survive, because each one records the exact token prefix that produced it, and
can be rebuilt by recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import queue
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .arbiter import Arbiter
from .config import Config, load_config
from .errors import BackendUnavailable, CapabilityUnsupported, MindError
from .logging_setup import get_logger, setup_logging
from .rpc import RpcServer, read_or_create_token

SERVICE_NAME = "inference"


class InferenceService:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = get_logger("inference")
        self.backend: Any = None
        self.arbiter = Arbiter(cfg.arbiter)
        self.started_at = time.time()
        self.incarnation_file = cfg.state_dir / "inference.incarnation"
        self.incarnation = self._bump_incarnation()
        self._stats = {"generate_calls": 0, "tokens_generated": 0, "errors": 0,
                       "batched_calls": 0, "forks": 0, "recomputes": 0}
        self._stop = threading.Event()
        # One dispatcher thread owns the backend; callers wait on their own
        # event. FIFO, so a batch is a prefix of the queue and there is no
        # ordering anybody can influence.
        self._pending: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._batch_lock = threading.Lock()
        self._dispatcher: threading.Thread | None = None

    def _bump_incarnation(self) -> int:
        n = 0
        if self.incarnation_file.exists():
            try:
                n = int(self.incarnation_file.read_text().strip() or 0)
            except ValueError:
                n = 0
        n += 1
        self.incarnation_file.write_text(str(n), encoding="utf-8")
        return n

    # ------------------------------------------------------------------
    def start_backend(self) -> dict[str, Any]:
        kind = self.cfg.backend.kind
        if kind == "deterministic":
            from .backends.deterministic import DeterministicBackend

            self.backend = DeterministicBackend(
                n_seq_max=self.cfg.backend.n_seq_max, n_ctx=self.cfg.backend.n_ctx
            )
            report = self.backend.load()
            self.log.warning("SIMULATED BACKEND ACTIVE: output is not model inference")
        elif kind == "llama_cpp":
            from .backends.llama_engine import LlamaEngine

            bc = self.cfg.backend
            runtime_dir = Path(bc.lib_path).parent if bc.lib_path else self.cfg.runtime_dir
            self.backend = LlamaEngine(
                runtime_dir=runtime_dir,
                model_path=bc.model_path,
                n_ctx=bc.n_ctx, n_seq_max=bc.n_seq_max,
                n_batch=bc.n_batch, n_ubatch=bc.n_ubatch,
                n_gpu_layers=bc.n_gpu_layers, n_threads=bc.n_threads,
                kv_unified=True, flash_attn=bc.flash_attn,
                type_k=bc.type_k, type_v=bc.type_v,
                log_sink=lambda lvl, txt: self.log.debug("llama: %s", txt.strip()),
            )
            report = self.backend.load()
            self.log.info("model loaded: %s (%s)", report.get("model_desc"),
                          report.get("model_generation"))
        else:
            raise BackendUnavailable("unknown backend kind", kind=kind)
        return report

    # ------------------------------------------------------------------
    # RPC surface
    # ------------------------------------------------------------------
    def methods(self) -> dict[str, Any]:
        b = self.backend
        return {
            "health": self.health,
            "capabilities": lambda: b.capabilities(),
            "load_report": lambda: getattr(b, "load_report", {}) or {},
            "model_generation": lambda: b.model_generation,
            "open_session": self.open_session,
            "close_session": self.close_session,
            "reset_session": lambda session_id: b.reset_session(session_id),
            "cancel_generation": self.cancel_generation,
            "active_sessions": lambda: b.active_sessions(),
            "tokenize": lambda text, add_special=False, parse_special=True: b.tokenize(
                text, add_special=add_special, parse_special=parse_special),
            "detokenize": lambda tokens, special=False: b.detokenize(tokens, special=special),
            "apply_chat_template": lambda messages, add_assistant=True: b.apply_chat_template(
                messages, add_assistant=add_assistant),
            "ingest": self.ingest,
            "ingest_text": self.ingest_text,
            "generate": self.generate,
            "chat": self.chat,
            "generate_batched": self.generate_batched,
            "fork_prefix": self.fork_prefix,
            "restore_prefix": self.restore_prefix,
            "top_logits": lambda session_id, k=5: b.top_logits(session_id, k),
            "seq_pos_max": lambda session_id: b.seq_pos_max(session_id),
            "state_seq_size": lambda session_id: b.state_seq_size(session_id),
            "vram_free": lambda: b.vram_free(),
            "session_tokens": self.session_tokens,
            "context_report": self.context_report,
            "stats": lambda: dict(self._stats),
            "shutdown": self.shutdown,
            # Only present when the backend offers it, which is only the
            # deterministic one. With a real model loaded this key does not
            # exist and the verb is simply unknown -- there is no way to make
            # an actual model return a canned string.
            **({"script_responses":
                lambda responses: {"queued": b.script_responses(responses)}}
               if hasattr(b, "script_responses") else {}),
        }

    def cancel_generation(self, *, session_id: str) -> dict[str, Any]:
        """Stop an in-flight generation on one session.

        Runs on an RPC handler thread while the engine lock is held by the
        generation being cancelled, which is exactly why the backend's
        request_cancel does not take that lock.
        """
        ok = bool(self.backend.request_cancel(session_id))
        if ok:
            self._stats["cancellations"] = self._stats.get("cancellations", 0) + 1
        return {"session_id": session_id, "cancel_requested": ok,
                "granularity": "one decode step (~6ms); prefill is not interruptible"}

    def context_report(self) -> dict[str, Any]:
        """Measured KV occupancy. Homeostasis never estimates this."""
        report = getattr(self.backend, "load_report", {}) or {}
        capacity = int(report.get("n_ctx_total") or self.cfg.backend.n_ctx)
        sessions = []
        used = 0
        try:
            for s in self.backend.active_sessions():
                n = int(s.get("n_past", 0))
                # A forked prefix is physically shared, so charging it to
                # every owner would overstate the pool. A RECOMPUTED prefix is
                # not shared -- it occupies its own cells -- so keying off
                # snapshot_id would under-report occupancy for exactly the
                # sessions restored after an inference restart.
                private = n - int(s.get("prefix_len", 0) or 0)
                charged = private if s.get("shares_prefix") else n
                used += charged
                # The session's own ceiling, not the pool's. Reporting the
                # pool here made `role_context_high` measure every role
                # against 49152, so the proactive threshold sat far above the
                # hard refusal and rejuvenation could only happen by
                # collision. A session with no policy falls back to the pool,
                # which is visible rather than silent.
                sessions.append({
                    **s,
                    "private_tokens": private,
                    "charged_tokens": charged,
                    "budget_tokens": int(s.get("context_budget_tokens") or capacity),
                    "budget_basis": s.get("budget_basis", "total"),
                    "budgeted_tokens": int(
                        s.get("budgeted_tokens",
                              private if s.get("budget_basis") == "private_growth"
                              else n)),
                    "pool_capacity": capacity,
                })
        except Exception as exc:  # noqa: BLE001
            return {"pool_tokens_used": 0, "pool_capacity": capacity,
                    "sessions": [], "detail": f"backend error: {exc!r}"}
        return {
            "pool_tokens_used": used,
            "pool_capacity": capacity,
            "occupancy": used / max(capacity, 1),
            "sessions": sessions,
            "detail": ("physically shared prefixes counted once; a fork's private "
                       "tail is charged to the fork; recomputed prefixes are "
                       "charged in full because they are not shared"),
        }

    def health(self) -> dict[str, Any]:
        """Must stay answerable even while inference is failing or saturated."""
        try:
            caps = self.backend.capabilities() if self.backend else {}
        except Exception as exc:  # noqa: BLE001
            caps = {"error": repr(exc)}
        sessions = []
        try:
            sessions = self.backend.active_sessions() if self.backend else []
        except Exception:  # noqa: BLE001
            pass
        return {
            "service": SERVICE_NAME,
            "status": "alive" if self.backend is not None else "no_backend",
            "pid": os.getpid(),
            "incarnation": self.incarnation,
            "uptime_seconds": time.time() - self.started_at,
            "capabilities": caps,
            "active_sessions": len(sessions),
            "max_sessions": caps.get("max_sessions", 0),
            "vram_free_bytes": self._safe_vram(),
            "stats": dict(self._stats),
        }

    def _safe_vram(self) -> int:
        try:
            return int(self.backend.vram_free()) if self.backend else 0
        except Exception:  # noqa: BLE001
            return 0

    def open_session(self, *, role: str, session_id: str | None = None,
                     context_budget_tokens: int | None = None,
                     budget_basis: str = "total") -> dict[str, Any]:
        sess = self.backend.open_session(
            role=role, session_id=session_id,
            context_budget_tokens=context_budget_tokens,
            budget_basis=budget_basis)
        return {"session_id": sess.session_id, "role": sess.role, "seq_id": sess.seq_id,
                "context_budget_tokens": sess.context_budget_tokens,
                "budget_basis": sess.budget_basis,
                "model_generation": self.backend.model_generation}

    def close_session(self, *, session_id: str, keep_prefix: bool = False) -> dict[str, Any]:
        self.backend.close_session(session_id, keep_prefix=keep_prefix)
        return {"closed": session_id}

    def session_tokens(self, *, session_id: str, limit: int | None = None) -> dict[str, Any]:
        sess = self.backend.get_session(session_id)
        toks = sess.tokens if limit is None else sess.tokens[:limit]
        return {"session_id": session_id, "n_past": sess.n_past,
                "prefix_len": sess.prefix_len, "tokens": list(toks)}

    def ingest(self, *, session_id: str, tokens: Sequence[int],
               compute_logits: bool = True) -> dict[str, Any]:
        n = self.backend.ingest(session_id, tokens, compute_logits=compute_logits)
        return {"session_id": session_id, "n_past": n}

    def ingest_text(self, *, session_id: str, text: str,
                    parse_special: bool = True) -> dict[str, Any]:
        toks = self.backend.tokenize(text, add_special=False, parse_special=parse_special)
        n = self.backend.ingest(session_id, toks)
        return {"session_id": session_id, "n_past": n, "tokens_added": len(toks)}

    def generate(self, *, session_id: str, max_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 0.95, top_k: int = 40, seed: int = 1234,
                 stop_strings: Sequence[str] = (), deadline: float | None = None
                 ) -> dict[str, Any]:
        """One generation, possibly decoded alongside others.

        The caller's thread waits here exactly as it did when this called the
        backend directly. What changed is that a single dispatcher owns the
        backend, so several waiting requests can share one decode step instead
        of queueing behind each other's locks.
        """
        sess = self.backend.get_session(session_id)
        # Whichever measure this session's allowance is written against. A
        # role is judged on everything it holds; a worker that inherited a
        # prefix is judged on what it has added to it, because the prefix was
        # not its doing and the allowance was written for its own growth.
        clamp = self.arbiter.clamp_inference(
            prompt_tokens=sess.budgeted_tokens, max_tokens=max_tokens,
            deadline=deadline,
            budget_tokens=sess.context_budget_tokens,
            budget_basis=sess.budget_basis,
        )
        req: dict[str, Any] = {
            "session_id": session_id, "clamp": clamp, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "seed": seed,
            "stop_strings": list(stop_strings), "done": threading.Event(),
            "queued": time.perf_counter(), "result": None, "error": None,
        }
        req["started"] = req["queued"]
        self._ensure_dispatcher()
        self._pending.put(req)
        req["done"].wait()
        if req["error"] is not None:
            raise req["error"]
        return req["result"]


    # ==================================================================
    # Grouping concurrent generations
    # ==================================================================
    def _ensure_dispatcher(self) -> None:
        """Start the one thread that talks to the backend, on first use."""
        if getattr(self, "_dispatcher", None) is not None:
            return
        with self._batch_lock:
            if getattr(self, "_dispatcher", None) is not None:
                return
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop, name="inference-batcher",
                daemon=True)
            self._dispatcher.start()

    def _dispatch_loop(self) -> None:
        """Take the first waiting request, then everything else already there.

        Never waits for company. The block below is on the *first* request
        only; once it has one, it drains whatever else is queued and goes. So
        a batch forms exactly when there was contention and never manufactures
        any.
        """
        while not self._stop.is_set():
            try:
                first = self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            group = [first]
            limit = max(1, int(self.cfg.batching.max_batch))
            while len(group) < limit:
                try:
                    group.append(self._pending.get_nowait())
                except queue.Empty:
                    break
            self._run_group(group)

    def _run_group(self, group: list[dict[str, Any]]) -> None:
        if len(group) > 1 and self.cfg.batching.enabled:
            try:
                self._run_batched(group)
                return
            except CapabilityUnsupported:
                # A backend that cannot batch is not an error; it is a backend
                # that cannot batch. Everything still runs, one at a time.
                pass
            except MindError:
                # One caller's failure must not become eight. Re-run the group
                # serially so the error lands on the request that caused it.
                self.log.debug("batched generate failed; running the group "
                               "one at a time", exc_info=True)
        for req in group:
            self._run_one(req)

    def _run_batched(self, group: list[dict[str, Any]]) -> None:
        requests = [{"session_id": r["session_id"],
                     "max_tokens": r["clamp"]["max_tokens"],
                     "stop_strings": list(r["stop_strings"])}
                    for r in group]
        head = group[0]
        results = self.backend.generate_batched(
            requests, temperature=head["temperature"], seed=head["seed"],
            deadline=head["clamp"]["deadline"])
        self._stats["batched_calls"] += 1
        for req in group:
            res = results.get(req["session_id"])
            if res is None:
                self._run_one(req)
                continue
            req["result"] = self._finish(req, res, batch_size=len(group))
            req["done"].set()

    def _run_one(self, req: dict[str, Any]) -> None:
        try:
            res = self.backend.generate(
                req["session_id"], max_tokens=req["clamp"]["max_tokens"],
                temperature=req["temperature"], top_p=req["top_p"],
                top_k=req["top_k"], seed=req["seed"],
                stop_strings=list(req["stop_strings"]),
                deadline=req["clamp"]["deadline"])
        except MindError as exc:
            self._stats["errors"] += 1
            req["error"] = exc
            req["done"].set()
            return
        req["result"] = self._finish(req, res, batch_size=1)
        req["done"].set()

    def _finish(self, req: dict[str, Any], res: Any, *, batch_size: int
                ) -> dict[str, Any]:
        self._stats["generate_calls"] += 1
        self._stats["tokens_generated"] += res.completion_tokens
        out = res.to_dict()
        # Measured, not assumed: how long this request sat behind others is
        # exactly what a caller needs to tell contention from slow inference.
        out["queue_delay_seconds"] = req["started"] - req["queued"]
        out["service_seconds"] = time.perf_counter() - req["started"]
        out["budget_clamped"] = req["clamp"]["clamped"]
        out["batch_size"] = batch_size
        out["model_generation"] = self.backend.model_generation
        out["is_simulated"] = bool(getattr(self.backend, "is_simulated", False))
        return out

    def chat(self, *, session_id: str, messages: Sequence[dict[str, str]],
             max_tokens: int = 256, temperature: float = 0.0, seed: int = 1234,
             stop_strings: Sequence[str] = (), reset: bool = False,
             add_assistant: bool = True) -> dict[str, Any]:
        """Render a chat turn with the model's own template, ingest and generate."""
        if reset:
            self.backend.reset_session(session_id)
        rendered = self.backend.apply_chat_template(messages, add_assistant=add_assistant)
        toks = self.backend.tokenize(rendered, add_special=False, parse_special=True)
        self.backend.ingest(session_id, toks)
        out = self.generate(session_id=session_id, max_tokens=max_tokens,
                            temperature=temperature, seed=seed, stop_strings=stop_strings)
        out["rendered_prompt_tokens"] = len(toks)
        return out

    def generate_batched(self, *, requests: Sequence[dict[str, Any]],
                         max_tokens: int = 128, temperature: float = 0.0,
                         seed: int = 1234) -> dict[str, Any]:
        t0 = time.perf_counter()
        res = self.backend.generate_batched(
            requests, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        self._stats["batched_calls"] += 1
        total = sum(r.completion_tokens for r in res.values())
        self._stats["tokens_generated"] += total
        return {
            "results": {k: v.to_dict() for k, v in res.items()},
            "wall_seconds": time.perf_counter() - t0,
            "total_completion_tokens": total,
            # An honest label: one fused kernel over several sequences is
            # continuous batching, not independent overlapping execution.
            "execution_mode": "continuous_batching",
            "physical_overlap_verified": False,
        }

    def fork_prefix(self, *, src_session_id: str, prefix_len: int, role: str,
                    session_id: str | None = None, snapshot_id: str | None = None,
                    context_budget_tokens: int | None = None,
                    budget_basis: str = "private_growth") -> dict[str, Any]:
        sess = self.backend.fork_prefix(
            src_session_id=src_session_id, prefix_len=prefix_len, role=role,
            session_id=session_id, snapshot_id=snapshot_id,
            context_budget_tokens=context_budget_tokens,
            budget_basis=budget_basis,
        )
        self._stats["forks"] += 1
        return {"session_id": sess.session_id, "seq_id": sess.seq_id,
                "prefix_len": sess.prefix_len, "snapshot_id": sess.snapshot_id,
                "context_budget_tokens": sess.context_budget_tokens,
                "budget_basis": sess.budget_basis,
                "kv_mode": self.backend.capabilities().get("kv_mode")}

    def restore_prefix(self, *, session_id: str, tokens: Sequence[int],
                       snapshot_id: str | None = None) -> dict[str, Any]:
        n = self.backend.restore_prefix(session_id=session_id, tokens=tokens,
                                        snapshot_id=snapshot_id)
        self._stats["recomputes"] += 1
        return {"session_id": session_id, "n_past": n, "kv_mode": "recomputed"}

    def shutdown(self) -> dict[str, Any]:
        self._stop.set()
        return {"stopping": True}

    # ------------------------------------------------------------------
    def serve(self) -> int:
        token = read_or_create_token(self.cfg.token_path)
        report = self.start_backend()
        server = RpcServer(
            self.cfg.supervisor_host, self.cfg.inference_port,
            token=token, service_name=SERVICE_NAME,
            on_error=lambda m, e: self.log.exception("rpc handler %s failed: %s", m, e),
        )
        server.register_all(self.methods())
        server.serve_in_thread()
        self.log.info("inference service listening on %s:%s (incarnation %s)",
                      self.cfg.supervisor_host, self.cfg.inference_port, self.incarnation)
        (self.cfg.state_dir / "inference.ready").write_text(
            json.dumps({"pid": os.getpid(), "port": self.cfg.inference_port,
                        "incarnation": self.incarnation,
                        "model_generation": report.get("model_generation", ""),
                        "started_at": self.started_at}),
            encoding="utf-8",
        )

        def _sig(_signum: int, _frame: Any) -> None:
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
            self.log.info("inference service stopping")
            server.shutdown()
            try:
                if self.backend:
                    self.backend.close()
            except Exception:  # noqa: BLE001
                self.log.exception("backend close failed")
            try:
                (self.cfg.state_dir / "inference.ready").unlink()
            except OSError:
                pass
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amoeba.inference_service")
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, "inference")
    return InferenceService(cfg).serve()


if __name__ == "__main__":
    raise SystemExit(main())
