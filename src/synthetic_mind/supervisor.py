"""The fixed harness: lifecycle, admission control, scheduling and the writer.

The supervisor is the only writable holder of durable state, the only spawner
of processes, and the only enforcer of resource limits. Ego and Id may request
and propose; they cannot rewrite the arbiter or bypass a cap.

It also proxies the cognitive verbs so that a client disconnect cannot end an
operation: the MCP facade is a transient stdio process that talks to this
long-lived one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .arbiter import Arbiter, ResourceSnapshot
from .config import Config, load_config
from .errors import (
    BackendUnavailable, InvalidInput, MindError, NotFound, ResourceExhausted,
)
from .ids import new_id
from .logging_setup import get_logger, setup_logging
from .homeostasis import ContextHomeostasis, HomeostasisConfig
from .mind import Mind
from .rpc import RpcClient, RpcServer, read_or_create_token, wait_for_port
from .sandbox import SandboxManager
from .store.events import read_events
from .store.writer import Mutation

SERVICE_NAME = "supervisor"
CHILDREN = ("inference", "ego", "id")
CHILD_GRACE_SECONDS = 8.0
"""How long a child may be unreachable before it is treated as dead."""


class SingleInstanceLock:
    """Exactly one supervisor may own a state directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            stale = self._is_stale()
            if not stale:
                raise ResourceExhausted(
                    "another supervisor already owns this state directory",
                    lock=str(self.path), holder=self._holder(),
                )
            os.unlink(self.path)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())
        self.fh = fd

    def _holder(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _is_stale(self) -> bool:
        info = self._holder()
        pid = info.get("pid")
        if not pid:
            return True
        return not _pid_alive(int(pid))

    def release(self) -> None:
        if self.fh is not None:
            try:
                os.close(self.fh)
            except OSError:
                pass
            self.fh = None
        try:
            self.path.unlink()
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            return str(pid) in out.stdout
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Supervisor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log = get_logger("supervisor")
        self.lock = SingleInstanceLock(cfg.lock_path)
        self.token = read_or_create_token(cfg.token_path)
        self.mind: Mind | None = None
        self.arbiter = Arbiter(cfg.arbiter)
        self.started_at = time.time()
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.workers: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, RpcClient] = {}
        self._stop = threading.Event()
        self._sched_lock = threading.RLock()
        self._server: RpcServer | None = None
        self._last_snapshot_publish = 0.0
        self._unreachable_since: dict[str, float] = {}
        self.sandboxes: SandboxManager | None = None
        self.homeostasis = ContextHomeostasis(
            HomeostasisConfig(**{k: getattr(cfg.homeostasis, k)
                                 for k in HomeostasisConfig.__slots__}),
            mind=None, inference=lambda: self.client("inference"))

    # ------------------------------------------------------------------
    # child processes
    # ------------------------------------------------------------------
    def _spawn(self, name: str) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        if self.cfg.source_path:
            env["SYNTHETIC_MIND_CONFIG"] = str(self.cfg.source_path)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
        )
        if name == "inference":
            cmd = [sys.executable, "-m", "synthetic_mind.inference_service"]
        elif name in ("ego", "id"):
            cmd = [sys.executable, "-m", "synthetic_mind.roles", name]
        else:
            raise InvalidInput("unknown child", name=name)
        if self.cfg.source_path:
            cmd += ["--config", str(self.cfg.source_path)]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            cmd, env=env, creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        self.procs[name] = proc
        self.log.info("spawned %s pid=%s", name, proc.pid)
        return proc

    def _terminate(self, name: str, *, timeout: float = 10.0) -> None:
        proc = self.procs.pop(name, None)
        if proc is None or proc.poll() is not None:
            return
        # Cooperative first: the child returns its resources and deregisters.
        try:
            client = self.clients.pop(name, None)
            if client is not None:
                try:
                    client.call("shutdown")
                except Exception:  # noqa: BLE001
                    pass
                client.close()
        except Exception:  # noqa: BLE001
            pass
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            self.log.warning("%s did not stop cooperatively; terminating tree", name)
            _kill_tree(proc.pid)

    def client(self, name: str) -> RpcClient:
        port = {"inference": self.cfg.inference_port, "ego": self.cfg.ego_port,
                "id": self.cfg.id_port}[name]
        cl = self.clients.get(name)
        if cl is None or not cl.connected:
            cl = RpcClient(self.cfg.supervisor_host, port, self.token, name=f"sup->{name}")
            cl.connect(retries=40, delay=0.5)
            self.clients[name] = cl
        return cl

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.lock.acquire()
        self.mind = Mind(self.cfg)
        self.homeostasis.mind = self.mind
        if self.cfg.sandbox.enabled:
            self.sandboxes = SandboxManager(self.cfg.sandbox_dir)
            ok, detail = self.sandboxes.available()
            if not ok:
                self.log.warning("sandboxing unavailable: %s", detail)
                self.sandboxes = None
            else:
                self.log.info("sandbox isolation: windows_appcontainer")
        recovery = self.mind.recover()
        self.log.info("recovery: %s", recovery["summary"])

        self._server = RpcServer(
            self.cfg.supervisor_host, self.cfg.supervisor_port, token=self.token,
            service_name=SERVICE_NAME,
            on_error=lambda m, e: self.log.exception("rpc %s: %s", m, e),
        )
        self._server.register_all(self.methods())
        self._server.serve_in_thread()
        self.log.info("supervisor listening on %s:%s",
                      self.cfg.supervisor_host, self.cfg.supervisor_port)

        self._spawn("inference")
        if not wait_for_port(self.cfg.supervisor_host, self.cfg.inference_port, timeout=300):
            raise BackendUnavailable("inference service did not start")
        for role in ("ego", "id"):
            self._spawn(role)
        for role, port in (("ego", self.cfg.ego_port), ("id", self.cfg.id_port)):
            if not wait_for_port(self.cfg.supervisor_host, port, timeout=180):
                self.log.error("%s did not start", role)

        self.cfg.ready_path.write_text(json.dumps({
            "pid": os.getpid(), "port": self.cfg.supervisor_port,
            "state_dir": str(self.cfg.state_dir), "started_at": self.started_at,
            "run_id": self.mind.run_id,
        }), encoding="utf-8")

    def stop(self) -> None:
        self._stop.set()
        for name in reversed(CHILDREN):
            try:
                self._terminate(name)
            except Exception:  # noqa: BLE001
                self.log.exception("failed stopping %s", name)
        for wid in list(self.workers):
            self._kill_worker(wid, reason="supervisor shutdown")
        if self.sandboxes is not None:
            killed = self.sandboxes.destroy_all()
            if killed:
                self.log.info("destroyed %d sandboxes on shutdown", len(killed))
        if self._server is not None:
            self._server.shutdown()
        if self.mind is not None:
            self.mind.close()
        try:
            self.cfg.ready_path.unlink()
        except OSError:
            pass
        self.lock.release()
        self.log.info("supervisor stopped")

    # ------------------------------------------------------------------
    # scheduler
    # ------------------------------------------------------------------
    def resource_snapshot(self) -> ResourceSnapshot:
        assert self.mind is not None
        qs = self.mind.work.queue_stats()
        active = [w for w in self.workers.values() if w["proc"].poll() is None]
        inf_sessions, max_sessions, vram = 0, 0, 0
        try:
            health = self.client("inference").call("health")
            inf_sessions = health.get("active_sessions", 0)
            max_sessions = health.get("max_sessions", 0)
            vram = health.get("vram_free_bytes", 0)
        except Exception:  # noqa: BLE001
            pass
        return ResourceSnapshot(
            active_workers=len(active),
            active_user_workers=sum(1 for w in active if w["work_class"] == "user"),
            active_maintenance_workers=sum(1 for w in active
                                           if w["work_class"] == "maintenance"),
            queued_user=qs["by_class"].get("user:queued", 0),
            queued_maintenance=qs["by_class"].get("maintenance:queued", 0),
            outstanding_work=(qs["by_status"].get("queued", 0)
                              + qs["by_status"].get("leased", 0)),
            oldest_queued_age=qs["oldest_queued_age"],
            recent_maintenance_count=self.mind.work.count_recent_maintenance(),
            inference_sessions=inf_sessions,
            max_inference_sessions=max_sessions,
            vram_free_bytes=vram,
        )

    def scheduler_loop(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self._scheduler_tick()
            except Exception:  # noqa: BLE001
                self.log.exception("scheduler tick failed")

    def _scheduler_tick(self) -> None:
        assert self.mind is not None
        with self._sched_lock:
            self._reap_workers()
            expired = self.mind.work.expire_leases()
            if expired:
                self.log.info("expired leases requeued: %s", expired)
            snapshot = self.resource_snapshot()
            work_class = self.arbiter.next_class_to_serve(snapshot)
            if work_class is not None:
                self._dispatch_worker(work_class)
            self._reclaim_snapshots()
            self._homeostasis_tick()

    def _homeostasis_tick(self) -> None:
        """Context pressure is checked on a slow cadence, not every second."""
        now = time.time()
        if now - getattr(self, "_last_homeo", 0.0) < 15.0:
            return
        self._last_homeo = now
        try:
            out = self.homeostasis.tick()
            if out and out.get("performed"):
                self.log.warning("auto-rejuvenated %s: %s", out["role"], out["reason"])
        except Exception:  # noqa: BLE001
            self.log.exception("homeostasis tick failed")

    def _reap_workers(self) -> None:
        for wid, info in list(self.workers.items()):
            proc = info["proc"]
            if proc.poll() is None:
                if time.time() > info["hard_deadline"]:
                    self._kill_worker(wid, reason="exceeded wall-clock budget")
                continue
            self.workers.pop(wid, None)
            if proc.returncode not in (0, None):
                self.log.warning("worker %s exited rc=%s", wid, proc.returncode)

    def _dispatch_worker(self, work_class: str) -> None:
        assert self.mind is not None
        row = self.mind.db.conn.execute(
            "SELECT work_id FROM work_items WHERE status = 'queued' AND work_class = ?"
            " ORDER BY priority DESC, created_at ASC LIMIT 1", (work_class,),
        ).fetchone()
        if row is None:
            return
        worker_id = new_id("wk")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
        )
        cmd = [sys.executable, "-m", "synthetic_mind.worker", "--worker-id", worker_id,
               "--work-id", row["work_id"]]
        if self.cfg.source_path:
            cmd += ["--config", str(self.cfg.source_path)]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, env=env, creationflags=flags,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=str(Path(__file__).resolve().parents[2]))
        self.workers[worker_id] = {
            "proc": proc, "work_class": work_class, "work_id": row["work_id"],
            "started": time.time(),
            "hard_deadline": time.time() + self.cfg.arbiter.worker_wall_seconds + 30,
        }
        self.arbiter.note_served(work_class)
        self.log.info("dispatched %s worker %s for %s", work_class, worker_id, row["work_id"])

    def _kill_worker(self, worker_id: str, *, reason: str) -> None:
        info = self.workers.pop(worker_id, None)
        if info is None:
            return
        _kill_tree(info["proc"].pid)
        assert self.mind is not None
        try:
            self.mind.work.retire_agent(agent_id=worker_id, reason=reason, crashed=True)
        except MindError:
            self.log.debug("retire of %s failed", worker_id, exc_info=True)
        self.log.warning("killed worker %s: %s", worker_id, reason)

    def _reclaim_snapshots(self) -> None:
        """Release unreferenced old snapshots; never the newest, never one with
        a live reference."""
        assert self.mind is not None
        for sid in self.mind.work.reclaimable_snapshots(actor="ego", keep_latest=1):
            try:
                self.mind.work.mark_snapshot_released(
                    snapshot_id=sid, actor="supervisor", reason="unreferenced and superseded"
                )
                snap = self.mind.work.get_snapshot(sid)
                handle = snap.get("backend_handle")
                if handle:
                    try:
                        self.client("inference").call("close_session", session_id=handle)
                    except Exception:  # noqa: BLE001
                        pass
                self.log.info("reclaimed snapshot %s", sid)
            except ResourceExhausted:
                continue
            except MindError:
                self.log.debug("snapshot reclaim failed for %s", sid, exc_info=True)

    # ------------------------------------------------------------------
    def serve(self) -> int:
        self.start()
        threading.Thread(target=self.scheduler_loop, daemon=True,
                         name="scheduler").start()
        import signal

        def _sig(_s: int, _f: Any) -> None:
            self._stop.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)
            except (ValueError, OSError):
                pass
        try:
            next_check = time.time() + CHILD_GRACE_SECONDS
            while not self._stop.is_set():
                self._stop.wait(0.5)
                if time.time() >= next_check:
                    next_check = time.time() + 2.0
                    self._supervise_children()
        finally:
            self.stop()
        return 0

    def _supervise_children(self) -> None:
        """Restart a child that has exited OR become unreachable.

        Process exit alone is not a reliable liveness signal here: on Windows
        the venv ``python.exe`` is a trampoline, so ``Popen.pid`` is not the pid
        of the interpreter that actually serves RPC. Killing the real process
        can leave the trampoline behind, and ``poll()`` keeps returning None
        while the child is gone. Reachability is the ground truth, with a grace
        period so a slow health call is not mistaken for a death.
        """
        for name in CHILDREN:
            proc = self.procs.get(name)
            if proc is None:
                continue
            exited = proc.poll() is not None
            reason = ""
            if exited:
                reason = f"process exited rc={proc.returncode}"
            else:
                if self._child_reachable(name):
                    self._unreachable_since.pop(name, None)
                    continue
                since = self._unreachable_since.setdefault(name, time.time())
                if time.time() - since < CHILD_GRACE_SECONDS:
                    continue
                reason = (f"unreachable for {time.time() - since:.0f}s "
                          f"while its process object still looked alive")
            self._restart_child(name, reason)

    def _child_reachable(self, name: str) -> bool:
        try:
            self.client(name).call("health")
            return True
        except Exception:  # noqa: BLE001
            self.clients.pop(name, None)
            return False

    def _restart_child(self, name: str, reason: str) -> None:
        assert self.mind is not None
        self.log.error("%s: %s; restarting", name, reason)
        self._unreachable_since.pop(name, None)
        proc = self.procs.get(name)
        if proc is not None and proc.poll() is None:
            # The trampoline (or a hung child) is still around; take the tree.
            _kill_tree(proc.pid)
        try:
            self.mind.work.retire_agent(agent_id=name, reason=reason, crashed=True)
        except MindError:
            pass
        if name == "inference":
            # Every KV handle the old process hosted is gone. Snapshot rows
            # survive because they record exact tokens.
            n = self.mind.work.invalidate_backend_handles(
                reason="inference service restarted"
            )
            self.log.warning("invalidated %d backend handles", n)
        self.clients.pop(name, None)
        self.procs.pop(name, None)
        self._spawn(name)

    # ------------------------------------------------------------------
    def methods(self) -> dict[str, Any]:
        from . import harness_api, supervisor_api

        methods = supervisor_api.build(self)
        methods.update(harness_api.build(self))
        return methods


def _kill_tree(pid: int) -> None:
    """Terminate a process and its descendants.

    Only processes this supervisor owns are ever targeted.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=30, check=False)
    else:
        import signal as _signal

        try:
            os.killpg(os.getpgid(pid), _signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, _signal.SIGKILL)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synthetic_mind.supervisor")
    ap.add_argument("--config", default=os.environ.get("SYNTHETIC_MIND_CONFIG"))
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, "supervisor")
    return Supervisor(cfg).serve()


if __name__ == "__main__":
    raise SystemExit(main())
