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
from .filespace import Filespace
from .errors import (
    BackendUnavailable, InvalidInput, MindError, NotFound, ResourceExhausted,
)
from .ids import new_id
from .logging_setup import get_logger, setup_logging
from .homeostasis import ContextHomeostasis, HomeostasisConfig
from .mind import Mind
from .pulse import PulseCollector
from .rpc import RpcClient, RpcServer, read_or_create_token, wait_for_port
from .sandbox import SandboxManager
from .security import audit_paths, harden_state_tree
from .store.events import read_events
from .store.writer import Mutation

SERVICE_NAME = "supervisor"
CHILDREN = ("inference", "ego", "id")
CHILD_GRACE_SECONDS = 8.0
PROBE_TIMEOUT_SECONDS = 2.0
"""A liveness probe must fail fast; see Supervisor.client."""
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
        self.neuocytes: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, RpcClient] = {}
        self._probes: dict[str, RpcClient] = {}
        self._supervision_passes = 0
        self._supervision_last = 0.0
        self._stop = threading.Event()
        self._sched_lock = threading.RLock()
        self._server: RpcServer | None = None
        self._last_snapshot_publish = 0.0
        self._unreachable_since: dict[str, float] = {}
        self.sandboxes: SandboxManager | None = None
        self.filespace: Filespace | None = None
        self._method_cache: dict[str, Any] | None = None
        # One sandbox per work item, owned by the Harness. A neuocyte never
        # names a sandbox and never creates or destroys one: it is resolved
        # from the work id, so there is no identifier for a model to forge.
        self._work_sandboxes: dict[str, str] = {}
        self._work_sandbox_lock = threading.RLock()
        self.pulse = PulseCollector(self)
        # What a *running* role actually primed its context with, as
        # reported at registration. Editing configuration changes what the
        # next incarnation would run, not this. In memory on purpose: it
        # describes live processes, and the roles are this supervisor's
        # own children, so it cannot outlive what it describes.
        self.role_prompt_digest: dict[str, str] = {}
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
            env["AMOEBA_CONFIG"] = str(self.cfg.source_path)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
        )
        if name == "inference":
            cmd = [sys.executable, "-m", "amoeba.inference_service"]
        elif name in ("ego", "id"):
            cmd = [sys.executable, "-m", "amoeba.roles", name]
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
        # Actively obtain a connection rather than relying on a cached one --
        # health checks use the probe pool, so the work pool is often empty,
        # and skipping the shutdown here means every child has to be force
        # killed after a timeout. That made teardown slow enough for
        # consecutive stacks to overlap.
        probe = self._probes.pop(name, None)
        if probe is not None:
            try:
                probe.close()
            except Exception:  # noqa: BLE001
                pass
        client = self.clients.pop(name, None)
        try:
            if client is None or not client.connected:
                client = self.client(name, probe=True)
                self._probes.pop(name, None)
            client.call("shutdown")
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        if proc.poll() is None:
            self.log.warning("%s did not stop cooperatively; terminating tree", name)
            _kill_tree(proc.pid)

    def client(self, name: str, *, probe: bool = False) -> RpcClient:
        """Connection to a child.

        ``probe=True`` is for liveness checks and must fail fast. The patient
        path retries for ~20s, which is right when a caller genuinely needs a
        restarting child -- but catastrophic for polling: health() touches all
        three children, so one dead child would make every health call block
        20s, including the supervision loop trying to restart it. A status
        endpoint that takes 20s to say "it is down" is not responsive.
        """
        port = {"inference": self.cfg.inference_port, "ego": self.cfg.ego_port,
                "id": self.cfg.id_port}[name]
        # Probe and work connections are kept in SEPARATE pools. Sharing them
        # would let a 2s probe socket be reused for a call that legitimately
        # takes a minute, and the resulting timeout would look like the child
        # had died.
        pool = self._probes if probe else self.clients
        cl = pool.get(name)
        if cl is None or not cl.connected:
            cl = RpcClient(
                self.cfg.supervisor_host, port, self.token,
                timeout=PROBE_TIMEOUT_SECONDS if probe else 600.0,
                name=f"sup->{name}{'/probe' if probe else ''}")
            cl.connect(retries=1, delay=0.0) if probe else cl.connect(retries=40, delay=0.5)
            pool[name] = cl
        return cl

    # ------------------------------------------------------------------
    # startup
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.lock.acquire()
        # Before opening the database: these directories inherit their parent's
        # DACL, and a permissive parent leaves the event log, the blobs and the
        # sandbox runtime writable by anyone on the machine.
        hardening = harden_state_tree(self.cfg)
        if hardening["audit"]["exposed_count"]:
            self.log.error("state tree exposed to %s after hardening",
                           hardening["audit"]["exposed_paths"])
        self.mind = Mind(self.cfg)
        self.homeostasis.mind = self.mind
        # Configured host roots are Peter's directories, not Amoeba's state,
        # so they are deliberately NOT hardened: locking down a directory the
        # user works in would be a surprising side effect of pointing Amoeba
        # at it.
        self.filespace = Filespace(self.cfg.filespace)
        if self.filespace.available():
            self.log.info("filespace roots: %s",
                          [r["name"] for r in self.filespace.roots()])
        else:
            self.log.info("no filespace roots configured; Amoeba has no host "
                          "filesystem access")
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
        self._register_scopes()
        self._server.serve_in_thread()
        self.log.info("supervisor listening on %s:%s",
                      self.cfg.supervisor_host, self.cfg.supervisor_port)

        # The inference service is a hard dependency: the roles cannot register
        # without it, so this one wait is worth blocking on.
        self._spawn("inference")
        if not wait_for_port(self.cfg.supervisor_host, self.cfg.inference_port, timeout=300):
            raise BackendUnavailable("inference service did not start")
        for role in ("ego", "id"):
            self._spawn(role)
        # Deliberately NOT blocking on role readiness here. Startup used to
        # wait up to 180s per role, and supervision only began afterwards -- so
        # a role that died inside that window was not restarted for three
        # minutes, during which the supervisor was blind. Supervision starts
        # first now and brings up whatever is not yet answering.

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
        for wid in list(self.neuocytes):
            self._kill_neuocyte(wid, reason="supervisor shutdown")
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
        active = [w for w in self.neuocytes.values() if w["proc"].poll() is None]
        inf_sessions, max_sessions, vram = 0, 0, 0
        try:
            health = self.client("inference").call("health")
            inf_sessions = health.get("active_sessions", 0)
            max_sessions = health.get("max_sessions", 0)
            vram = health.get("vram_free_bytes", 0)
        except Exception:  # noqa: BLE001
            pass
        return ResourceSnapshot(
            active_neuocytes=len(active),
            active_user_neuocytes=sum(1 for w in active if w["work_class"] == "user"),
            active_maintenance_neuocytes=sum(1 for w in active
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
            self._reap_neuocytes()
            expired = self.mind.work.expire_leases()
            if expired:
                self.log.info("expired leases requeued: %s", expired)
            snapshot = self.resource_snapshot()
            work_class = self.arbiter.next_class_to_serve(snapshot)
            if work_class is not None:
                self._dispatch_neuocyte(work_class)
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

    def _reap_neuocytes(self) -> None:
        for wid, info in list(self.neuocytes.items()):
            proc = info["proc"]
            if proc.poll() is None:
                if time.time() > info["hard_deadline"]:
                    self._kill_neuocyte(wid, reason="exceeded wall-clock budget")
                continue
            self.neuocytes.pop(wid, None)
            if proc.returncode not in (0, None):
                self.log.warning("neuocyte %s exited rc=%s", wid, proc.returncode)

    def _dispatch_neuocyte(self, work_class: str) -> None:
        assert self.mind is not None
        row = self.mind.db.conn.execute(
            "SELECT work_id FROM work_items WHERE status = 'queued' AND work_class = ?"
            " ORDER BY priority DESC, created_at ASC LIMIT 1", (work_class,),
        ).fetchone()
        if row is None:
            return
        neuocyte_id = new_id("nc")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
        )
        cmd = [sys.executable, "-m", "amoeba.neuocyte", "--neuocyte-id", neuocyte_id,
               "--work-id", row["work_id"]]
        if self.cfg.source_path:
            cmd += ["--config", str(self.cfg.source_path)]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, env=env, creationflags=flags,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=str(Path(__file__).resolve().parents[2]))
        self.neuocytes[neuocyte_id] = {
            "proc": proc, "work_class": work_class, "work_id": row["work_id"],
            "started": time.time(),
            "hard_deadline": time.time() + self.cfg.arbiter.neuocyte_wall_seconds + 30,
        }
        self.arbiter.note_served(work_class)
        self.log.info("dispatched %s neuocyte %s for %s", work_class, neuocyte_id, row["work_id"])

    def _kill_neuocyte(self, neuocyte_id: str, *, reason: str) -> None:
        info = self.neuocytes.pop(neuocyte_id, None)
        if info is None:
            return
        _kill_tree(info["proc"].pid)
        assert self.mind is not None
        try:
            self.mind.work.retire_agent(agent_id=neuocyte_id, reason=reason, crashed=True)
        except MindError:
            self.log.debug("retire of %s failed", neuocyte_id, exc_info=True)
        self.log.warning("killed neuocyte %s: %s", neuocyte_id, reason)

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
        threading.Thread(target=self.supervision_loop, daemon=True,
                         name="supervision").start()
        self._await_roles()
        try:
            while not self._stop.is_set():
                self._stop.wait(0.5)
        finally:
            self.stop()
        return 0

    def _await_roles(self, timeout: float = 60.0) -> None:
        """Note when the roles come up. Advisory only.

        Supervision is already running by the time this is called, so a role
        that is slow or dead gets restarted regardless of what this observes.
        It exists to put readiness in the log, not to gate anything.
        """
        deadline = time.time() + timeout
        pending = {"ego": self.cfg.ego_port, "id": self.cfg.id_port}
        while pending and time.time() < deadline and not self._stop.is_set():
            for role, port in list(pending.items()):
                if wait_for_port(self.cfg.supervisor_host, port, timeout=0.5):
                    self.log.info("%s is listening", role)
                    pending.pop(role)
            if pending:
                self._stop.wait(0.5)
        for role in pending:
            self.log.warning("%s not listening yet; supervision will restart it "
                             "if it stays down", role)

    def supervision_loop(self) -> None:
        """Watch the children on a dedicated thread.

        This used to run inline in the main loop. A single probe that blocked
        longer than expected then stalled supervision entirely, and a child
        that died during that window was never restarted -- the failure mode
        looked like "the supervisor ignored it", which is much worse than a
        slow restart. On its own thread, a stall costs one pass rather than the
        whole mechanism, and the pass counter makes a stall visible.
        """
        # A short settle before the first pass: children need a moment to bind
        # their ports, and restarting one that is merely still starting would
        # be worse than waiting.
        self._stop.wait(3.0)
        passes = 0
        while not self._stop.is_set():
            started = time.time()
            try:
                self._supervise_children()
            except Exception:  # noqa: BLE001
                self.log.exception("supervision pass failed")
            passes += 1
            self._supervision_passes = passes
            self._supervision_last = time.time()
            elapsed = time.time() - started
            if elapsed > CHILD_GRACE_SECONDS:
                self.log.warning("supervision pass took %.1fs", elapsed)
            self._stop.wait(max(0.5, 2.0 - elapsed))

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
        """Is this child answering right now?

        Runs on a throwaway thread with a hard join deadline. A probe socket
        can wedge in ways a socket timeout does not cover (a peer that accepts
        and never replies, for instance), and a supervision pass must not be
        hostage to that.
        """
        result: list[bool] = []

        def probe() -> None:
            try:
                self.client(name, probe=True).call("health")
                result.append(True)
            except Exception:  # noqa: BLE001
                result.append(False)

        t = threading.Thread(target=probe, daemon=True, name=f"probe-{name}")
        t.start()
        t.join(timeout=PROBE_TIMEOUT_SECONDS * 2)
        if not result:
            # Still stuck: treat as unreachable and drop the connection so the
            # next pass starts a fresh one.
            self._probes.pop(name, None)
            self.log.warning("%s health probe did not return within %.1fs",
                             name, PROBE_TIMEOUT_SECONDS * 2)
            return False
        if not result[0]:
            self._probes.pop(name, None)
        return result[0]

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
        self._probes.pop(name, None)
        self.procs.pop(name, None)
        self._spawn(name)

    # ------------------------------------------------------------------
    def _system_pulse(self, *, max_age_seconds: float = 1.0) -> dict[str, Any]:
        """Bounded live telemetry: what the organism is doing right now.

        Facts only. Deciding whether any of it is a problem is Id's job, and
        putting that conclusion here would move cognition into the Harness.
        """
        return self.pulse.capture(max_age_seconds=float(max_age_seconds))

    # ------------------------------------------------------------------
    # Capability scopes
    # ------------------------------------------------------------------
    def _register_scopes(self) -> None:
        """Give each kind of caller its own method table.

        Absence, not a guard. A verb outside a scope's table does not exist for
        that caller: it cannot be listed, cannot be dispatched, and there is no
        shared implementation containing an `if caller != id` to get wrong. The
        scope comes from the presented secret, so it cannot be claimed.

        Membership is derived from what each caller actually calls, not from
        what seems reasonable -- an unused verb in a scope is capability nobody
        asked for.
        """
        from .scopes import scope_tables

        methods = self.methods()
        for scope, names in scope_tables().items():
            token = read_or_create_token(self.cfg.scope_token_path(scope))
            missing = [n for n in names if n not in methods]
            if missing:
                raise InvalidInput("scope names unknown methods",
                                   scope=scope, missing=missing)
            self._server.register_scope(
                scope, {n: methods[n] for n in names}, token=token)
            self.log.info("scope %s: %d verbs", scope, len(names))

    # ------------------------------------------------------------------
    def methods(self) -> dict[str, Any]:
        # Cached: the handlers are closures, and the tool registry looks them
        # up per call. Rebuilding would make every lookup a fresh closure set.
        if self._method_cache is None:
            from . import harness_api, id_api, supervisor_api

            methods = supervisor_api.build(self)
            methods.update(harness_api.build(self))
            methods.update(id_api.build(self))
            methods["system_pulse"] = self._system_pulse
            self._method_cache = methods
        return self._method_cache

    # ------------------------------------------------------------------
    # Per-work sandboxes, owned by the Harness
    # ------------------------------------------------------------------
    def sandbox_for_work(self, work_id: str, *, owner: str) -> str:
        """The sandbox for one work item, created on first use.

        Resolved from ``work_id`` rather than accepted as an argument. That is
        what stops one neuocyte reaching another's scratch: there is no
        parameter in which to name a sandbox, so a model cannot ask for one it
        does not own.
        """
        with self._work_sandbox_lock:
            existing = self._work_sandboxes.get(work_id)
            if existing is not None:
                try:
                    self.sandboxes.get(existing)  # type: ignore[union-attr]
                    return existing
                except Exception:  # noqa: BLE001
                    self._work_sandboxes.pop(work_id, None)
            res = self.methods()["sandbox_create"](owner=owner, work_id=work_id)
            self._work_sandboxes[work_id] = res["sandbox_id"]
            self.log.info("sandbox %s opened for work %s", res["sandbox_id"], work_id)
            return res["sandbox_id"]

    def release_work_sandbox(self, work_id: str, *, reason: str) -> str | None:
        """Destroy a work item's sandbox when the work item finishes.

        Scratch does not outlive the work that produced it. Anything worth
        keeping had to be proposed and promoted, which is the only path out.
        """
        with self._work_sandbox_lock:
            sandbox_id = self._work_sandboxes.pop(work_id, None)
        if sandbox_id is None:
            return None
        try:
            self.methods()["sandbox_destroy"](sandbox_id=sandbox_id,
                                              actor="supervisor", reason=reason)
        except Exception:  # noqa: BLE001
            # Loud on purpose. Scratch that survives its work item is live
            # state nobody owns, and this failing quietly is how it would
            # accumulate unnoticed -- which is exactly what happened when this
            # call was made with an argument the handler did not accept.
            self.log.warning("could not destroy sandbox %s for work %s",
                             sandbox_id, work_id, exc_info=True)
            with self._work_sandbox_lock:
                self._work_sandboxes[work_id] = sandbox_id
            raise
        return sandbox_id


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
    ap = argparse.ArgumentParser(prog="amoeba.supervisor")
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, "supervisor")
    return Supervisor(cfg).serve()


if __name__ == "__main__":
    raise SystemExit(main())
