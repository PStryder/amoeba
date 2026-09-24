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
from . import conditions, heartbeat, mailbox
from .filespace import Filespace
from .errors import (
    BackendUnavailable, InvalidInput, MindError, NotFound, ResourceExhausted,
)
from .ids import new_id
from .logging_setup import get_logger, setup_logging
from .homeostasis import ContextHomeostasis, HomeostasisConfig
from .http_api import ApiServer
from .mind import Mind
from .pulse import PulseCollector
from .rpc import RpcClient, RpcServer, read_or_create_token, wait_for_port
from .room import Room
from .sandbox import SandboxManager
from .security import audit_paths, harden_state_tree
from .store.events import EventKind, read_events
from .store.writer import Mutation

SERVICE_NAME = "supervisor"
CHILDREN = ("inference", "ego", "id")
STALE_TURN_GRACE_SECONDS = 120.0
"""Extra time beyond a role's own per-turn deadline before the Harness
treats an open turn as abandoned. A turn that reaches this has already
ignored the bound it enforces on itself."""

CHILD_GRACE_SECONDS = 8.0
PROBE_TIMEOUT_SECONDS = 2.0
"""A liveness probe must fail fast; see Supervisor.client."""
"""How long a child may be unreachable before it is treated as dead."""


class SingleInstanceLock:
    """Exactly one supervisor may own a state directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh: Any = None

    ATTEMPTS = 5

    def acquire(self) -> None:
        """Take the lock, clearing a dead holder's if that is what is there.

        Clearing a stale lock races two ways, and a restart hits both. The
        previous supervisor can release its own lock between our seeing it and
        our unlinking it, so the unlink finds nothing; and another supervisor
        can create a fresh lock between our unlink and our open, so the open
        finds one. The first killed a restart with FileNotFoundError. So every
        attempt goes back to the top: the exclusive create is the only step
        that decides ownership, and whoever holds the lock is re-judged each
        time rather than assumed from a check that has since gone stale.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(self.ATTEMPTS):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except FileExistsError:
                if not self._is_stale():
                    raise ResourceExhausted(
                        "another supervisor already owns this state directory",
                        lock=str(self.path), holder=self._holder(),
                    ) from None
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass    # its holder released it first; the next create decides
                continue
            os.write(fd, json.dumps({"pid": os.getpid(),
                                     "started": time.time()}).encode())
            self.fh = fd
            return
        raise ResourceExhausted(
            "could not take the state directory lock; it kept changing hands",
            lock=str(self.path), holder=self._holder(),
        )

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
        # The live backchannel, for watching rather than for the record. It is
        # capped and it dies with this process; what a message actually did
        # lives in its trigger and in the turn that consumed it.
        self.room = Room()
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
        self.api: ApiServer | None = None
        # What a *running* role actually primed its context with, as
        # reported at registration. Editing configuration changes what the
        # next incarnation would run, not this. In memory on purpose: it
        # describes live processes, and the roles are this supervisor's
        # own children, so it cannot outlive what it describes.
        self.role_prompt_digest: dict[str, str] = {}
        # Scheduler state for the persistent roles. Deterministic substrate:
        # it decides when a role gets another bounded turn, never what the
        # role should conclude.
        self._role_activity: dict[str, str] = {"ego": "idle", "id": "idle"}
        self._next_heartbeat: dict[str, float] = {}
        self._heartbeat_interval: dict[str, float] = {}
        # When the current run of pressure deferrals began, per role. Cleared
        # the moment pressure drops, so the ceiling measures one continuous
        # period of strain rather than a lifetime total.
        self._heartbeat_deferred_since: dict[str, float] = {}
        # Streaks already on the record, and the repairs spent on them, so a
        # role that cannot think is reported once and restarted a bounded
        # number of times rather than in a loop.
        self._not_thinking: dict[str, float | None] = {}
        # When each condition last woke a role, so a bad hour costs a handful
        # of turns rather than a wake storm.
        self._condition_woke: dict[tuple[str, str], float] = {}
        self._role_repairs: dict[str, list[float]] = {}
        # NB: guarded by the existing `_sched_lock` above, which is also held
        # across the whole scheduler tick. Sharing it keeps the ordering
        # obvious; it also means a trigger enqueue can wait behind a neuocyte
        # spawn, which is why the stale-turn sweep runs outside that lock.
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
        self.homeostasis.governed_prompt = self._governed_prompt
        self.homeostasis.hand_over = self.hand_over_session
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
        self._ingest_prompt_library()
        self._validate_output_ceilings()
        self._recover_role_turns()

        self._server = RpcServer(
            self.cfg.supervisor_host, self.cfg.supervisor_port, token=self.token,
            service_name=SERVICE_NAME,
            on_error=lambda m, e: self.log.exception("rpc %s: %s", m, e),
        )
        self._server.register_all(self.methods())
        self._register_scopes()
        self._server.serve_in_thread()
        if self.cfg.api_enabled:
            self._start_api()
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

        # Queued after the roles are spawned, so the first thing Id does is
        # think about an organism that exists rather than one still starting.
        self._startup_triggers()

        self.cfg.ready_path.write_text(json.dumps({
            "pid": os.getpid(), "port": self.cfg.supervisor_port,
            "state_dir": str(self.cfg.state_dir), "started_at": self.started_at,
            "run_id": self.mind.run_id,
        }), encoding="utf-8")

    def _ingest_prompt_library(self) -> None:
        """Compare the shipped prompt files against the library, once, at boot.

        A fresh state directory gets its baseline here -- roots have no other
        origin. An edited file becomes a *candidate*, so restarting with a
        changed prompt never silently changes how the organism thinks; the
        selected version keeps running until somebody approves the new one.
        """
        from .promptlib import bootstrap as prompt_bootstrap
        from .promptlib.store import PromptStore

        store = PromptStore(self.mind)
        try:
            _, out = self.mind.writer.apply(
                lambda m: prompt_bootstrap.ingest(m, store), actor="bootstrap")
        except Exception:
            # A malformed prompt file must not take the organism down, but it
            # must not pass unnoticed either: whatever is already selected
            # keeps running and the failure is on the record.
            self.log.exception("prompt library bootstrap failed; running on "
                               "the previously selected profiles")
            return
        counts = out["counts"]
        self.log.info("prompt library: %s baseline, %s matched, %s present, "
                      "%s new candidate(s)", counts["baseline"],
                      counts["matched"], counts["present"], counts["delta"])
        if counts["delta"]:
            self.log.warning(
                "%s prompt file(s) differ from the selected versions and are "
                "waiting for approval: %s", counts["delta"],
                [r["namespace"] for r in out["ingested"]
                 if r["outcome"] == "delta"])

    def _validate_output_ceilings(self) -> None:
        """Every selected profile's output ceiling must fit the platform cap.

        Checked once, before any mind is born, and fatal when it fails. A
        profile stating 3072 under a cap of 512 is a contradiction, and the
        only two ways to resolve it silently -- clamp the profile, or ignore
        the cap -- each make one of the two numbers a lie. So the supervisor
        refuses to start and says which profile and which setting disagree.

        A selected profile that states no ceiling at all is not fatal: an
        approved root can predate the setting, and the binding supplies the
        shipped value for its namespace and records that it did. It is said
        aloud here, with what to approve, because a ceiling nobody chose is
        a thing an operator should know about.
        """
        from .promptlib.model import FALLBACK_OUTPUT_CEILINGS
        from .promptlib.resolver import Resolver
        from .promptlib.store import PromptStore

        cap = int(self.cfg.arbiter.max_completion_tokens)
        store = PromptStore(self.mind)
        resolver = Resolver(store)
        contradictions: list[str] = []
        silent: list[str] = []
        for namespace, shipped in sorted(FALLBACK_OUTPUT_CEILINGS.items()):
            if shipped > cap:
                contradictions.append(
                    f"the shipped ceiling for {namespace} is {shipped}")
        for namespace in store.namespaces():
            if store.selected(namespace) is None:
                continue
            resolved = resolver.resolve_selected(namespace)
            stated = resolved.model_vars.get("max_output_tokens")
            if stated is None:
                silent.append(str(resolved.ref))
            elif int(stated) > cap:
                contradictions.append(
                    f"{resolved.ref} states max_output_tokens {stated}")
        if contradictions:
            raise RuntimeError(
                f"output ceilings exceed the platform cap of {cap} "
                "([arbiter] max_completion_tokens): "
                + "; ".join(contradictions)
                + ". Lower the profile or raise the cap; the supervisor will "
                "not start with the two in contradiction.")
        if silent:
            self.log.warning(
                "selected profile(s) state no max_output_tokens: %s. Each is "
                "bound with the shipped ceiling for its namespace, recorded in "
                "the binding's harness_constraints; approve the candidate that "
                "states it to make the ceiling governed.", ", ".join(silent))

    # ------------------------------------------------------------------
    # Turn scheduling: deterministic substrate, never a cognitive component
    # ------------------------------------------------------------------
    def role_activity(self, role: str) -> str:
        """What a persistent role is doing, for the operator surface.

        Derived from durable state where it can be -- an open turn row is the
        ground truth for "processing" -- so a stale in-memory flag cannot
        claim a role is idle while a turn is running.
        """
        from . import mailbox

        if self.mind is None:
            return "unknown"
        try:
            if mailbox.open_turn(self.mind.db.conn, role) is not None:
                return "processing"
            if mailbox.pending_count(self.mind.db.conn, role):
                return "queued"
        except Exception:  # noqa: BLE001
            return "unknown"
        with self._sched_lock:
            if self._role_activity.get(role) == "recovering":
                return "recovering"
        if role == "id" and self.cfg.scheduler.id_heartbeat_seconds > 0:
            return "heartbeat_wait"
        return "idle"

    def next_heartbeat(self, role: str) -> float | None:
        """When this role is next due a heartbeat turn, if it gets one."""
        with self._sched_lock:
            return self._next_heartbeat.get(role)

    def note_trigger(self, role: str) -> None:
        """A trigger arrived: reset this role's heartbeat backoff.

        The organism stopped being quiet, so the next quiet period starts from
        the configured interval rather than from however far the backoff had
        climbed.
        """
        with self._sched_lock:
            self._heartbeat_interval.pop(role, None)

    def note_turn_finished(self, stop_reason: str | None, *, turn_id: str) -> None:
        """Record that a turn closed, for the heartbeat cadence."""
        with self._sched_lock:
            self._role_activity["id"] = "idle"


    def _record_heartbeat_deferral(self, role: str, held: dict) -> None:
        """Note that a review was held back, once per run of pressure.

        Recorded rather than merely logged: an operator asking why Id went
        quiet deserves an answer in the same record as everything else, and a
        silence with no entry beside it is indistinguishable from a scheduler
        that stopped working.
        """
        if self.mind is None:
            return
        try:
            self.mind.writer.apply(
                lambda m: m.emit(EventKind.HEARTBEAT_DEFERRED, {
                    "role": role, **held,
                    "note": ("the discretionary heartbeat only; event-driven "
                             "turns are never held back, and this one runs "
                             "regardless once the ceiling is reached")}),
                actor="supervisor", bump_version=False)
        except Exception:  # noqa: BLE001
            self.log.debug("could not record the heartbeat deferral",
                           exc_info=True)

    def _heartbeat_deferred_for_pressure(self, role: str, now: float) -> dict | None:
        """Should this discretionary heartbeat wait for the pool to settle?

        Returns a reason to defer, or None to proceed. Only the heartbeat ever
        asks: an event-driven turn is something that happened, and holding one
        back would make the organism unable to think about its own world
        because it was busy.

        Unknown pressure proceeds. An absent measurement is not evidence of
        pressure, and refusing cognition on it would stop the organism
        thinking because a monitor was down.
        """
        from .homeostasis import PRESSURE_LEVELS

        want = (self.cfg.scheduler.heartbeat_defer_at_pressure or "never").lower()
        if want == "never" or want not in PRESSURE_LEVELS:
            return None
        level, _age = self.homeostasis.last_pressure()
        if level is None or level not in PRESSURE_LEVELS:
            return None
        if PRESSURE_LEVELS.index(level) < PRESSURE_LEVELS.index(want):
            self._heartbeat_deferred_since.pop(role, None)
            return None

        since = self._heartbeat_deferred_since.setdefault(role, now)
        waited = now - since
        ceiling = max(0.0, self.cfg.scheduler.heartbeat_max_deferral_seconds)
        if waited >= ceiling:
            # Long enough. Id's heartbeat is the homeostatic review, and a
            # review that never happens is worse than a turn that costs a
            # prefill.
            self._heartbeat_deferred_since.pop(role, None)
            return None
        return {"pressure": level, "threshold": want,
                "deferred_seconds": round(waited, 1), "ceiling": ceiling}

    def _schedule_heartbeats(self) -> None:
        """Give Id a turn when the organism has been quiet.

        Id has continuing responsibility for internal state even when nobody
        sends it anything, so it is logically always on. It is not a token
        furnace: a heartbeat that finds nothing backs the interval off toward
        a ceiling, so a quiet organism gets quieter instead of paying full
        price to keep discovering that nothing happened. Any real trigger
        resets it.

        Ego has no heartbeat. A persistent identity that talks to itself
        because its process exists is not the same as one that responds.
        """
        from . import mailbox

        sched = self.cfg.scheduler
        if sched.id_heartbeat_seconds <= 0 or self.mind is None:
            return
        role = "id"
        now = time.time()
        with self._sched_lock:
            due = self._next_heartbeat.get(role)
            if due is None:
                self._next_heartbeat[role] = now + sched.id_heartbeat_seconds
                return
            if now < due:
                return
            interval = self._heartbeat_interval.get(
                role, sched.id_heartbeat_seconds)

        # Do not stack a heartbeat on top of work that is already waiting:
        # Id will see everything queued at its next boundary anyway, and a
        # heartbeat it did not need is noise in its own transcript.
        try:
            if (mailbox.pending_count(self.mind.db.conn, role)
                    or mailbox.open_turn(self.mind.db.conn, role) is not None):
                with self._sched_lock:
                    self._next_heartbeat[role] = now + interval
                return
            held = self._heartbeat_deferred_for_pressure(role, now)
            if held is not None:
                # Re-examined sooner than a full interval, so the review
                # resumes shortly after the pool settles rather than waiting
                # out a cycle it did not use.
                with self._sched_lock:
                    self._next_heartbeat[role] = now + min(interval, 60.0)
                if held["deferred_seconds"] <= 0.0:
                    self.log.info(
                        "holding %s's heartbeat while pressure is %s", role,
                        held["pressure"])
                    self._record_heartbeat_deferral(role, held)
                return

            overdue = self._heartbeat_deferred_since.pop(role, None)
            deferred = round(now - overdue, 1) if overdue else 0.0
            # What actually changed, measured from an event watermark, so the
            # review does not begin by spending five tool calls to discover a
            # row of zeros. Measurement only: what it means is Id's to say.
            digest = heartbeat.measure(
                self.mind.db.conn, role,
                since=heartbeat.watermark_of(self.mind.db.conn, self.mind.blobs, role))
            body = heartbeat.render(digest, interval_seconds=interval,
                                    deferred_seconds=deferred)
            payload = {"reason": "periodic_homeostatic_review",
                       "interval_seconds": interval,
                       # Said plainly, because a review that ran late under
                       # strain is a different fact from one that ran on
                       # time, and Id is the component that should know.
                       "deferred_for_pressure_seconds": deferred,
                       "message": body,
                       "measured_to_seq": digest["seq"],
                       "events_since": digest["events"]}
            if heartbeat.quiet(digest):
                # Nothing happened and nothing is owed. A ceiling, not an
                # instruction: Id may still say whatever it likes, and if it
                # needs more room the turn continues as any other does.
                payload["output_ceiling"] = int(sched.heartbeat_quiet_ceiling_tokens)
            self.methods()["role_enqueue_trigger"](
                role=role, kind="heartbeat", source="scheduler",
                summary="periodic homeostatic review",
                payload=payload,
                # About the organism, not about anybody's question.
                ambient=True)
        except Exception:  # noqa: BLE001
            self.log.debug("heartbeat scheduling failed", exc_info=True)
            return
        with self._sched_lock:
            # Back off *after* enqueueing, so a quiet organism's next review is
            # further away; note_trigger resets this the moment anything real
            # arrives.
            grown = min(interval * sched.id_heartbeat_backoff,
                        sched.id_heartbeat_max_seconds)
            self._heartbeat_interval[role] = grown
            self._next_heartbeat[role] = time.time() + grown

    def _startup_triggers(self) -> None:
        """Id forms an initial view of the organism it woke up in.

        Ego does not get one by default: it wakes because something relevant
        happened, and starting a process is not that.
        """
        sched = self.cfg.scheduler
        for role, wanted, summary in (
            ("id", sched.id_startup_turn,
             "the organism has started; form an initial view of its internal state"),
            ("ego", sched.ego_startup_turn,
             "the organism has started"),
        ):
            if not wanted:
                continue
            try:
                self.methods()["role_enqueue_trigger"](
                    role=role, kind="startup", source="supervisor",
                    summary=summary,
                    payload={"run_id": self.mind.run_id if self.mind else None},
                    ambient=True)
            except Exception:  # noqa: BLE001
                self.log.warning("could not queue %s startup turn", role,
                                 exc_info=True)

    def _recover_role_turns(self, role: str | None = None,
                            reason: str = "process did not survive the turn"
                            ) -> None:
        """Re-open turns left running by a process that did not survive.

        Called at startup for every role, and for a single role whenever
        supervision restarts it. The second case is the one that matters
        operationally: one open turn per role is a database constraint, so a
        turn its owner never closed blocks every future turn for that role.
        The role keeps heartbeating and reporting healthy while its mailbox
        fills and nothing runs.
        """
        from . import mailbox

        if self.mind is None:
            return
        try:
            _, out = self.mind.writer.apply(
                lambda m: mailbox.recover(m, self.mind, role=role,
                                          reason=reason),
                actor="supervisor", bump_version=False)
        except Exception:  # noqa: BLE001
            self.log.exception("role turn recovery failed")
            return
        recovered = out.get("recovered_turns") or []
        if recovered:
            self.log.warning(
                "recovered %d interrupted %s turn(s); %d trigger(s) requeued",
                len(recovered), role or "role",
                sum(len(r["requeued"]) for r in recovered))

    def _governed_prompt(self, role: str) -> str | None:
        """The prompt this role's incarnation was bound to, for a rebuild.

        Resolved from the library by the binding's own profile ref and held to
        the digest the binding froze, so a rebuilt session is primed with
        exactly what the incarnation was born with -- not with whatever the
        library selects now. `None` when that cannot be shown, and the
        rebuild then keeps the message the session was primed with.
        """
        from .ids import sha256_hex

        row = self.mind.db.conn.execute(
            "SELECT profile_ref, prompt_sha256 FROM incarnation_profiles"
            " WHERE actor_id = ? AND actor_kind = ?"
            " ORDER BY created_at DESC LIMIT 1", (role, role)).fetchone()
        if row is None or not row["profile_ref"]:
            return None
        try:
            text = self.methods()["prompt_resolve"](
                profile_ref=row["profile_ref"], include_text=True)["prompt_text"]
        except Exception:  # noqa: BLE001
            self.log.warning("could not resolve %s for a rebuild",
                             row["profile_ref"], exc_info=True)
            return None
        if sha256_hex(text.encode("utf-8")) != row["prompt_sha256"]:
            return None
        return text

    def hand_over_session(self, role: str, session_id: str | None, *,
                          reason: str = "") -> bool:
        """Tell a role which inference session it now has.

        Called after anything replaces a role's session. Identity survives --
        incarnation, profile binding and mailbox are untouched, and a
        replacement session is not a new mind -- but the role holds the handle
        in memory, so nobody else can update it for it.
        """
        if not session_id or role not in ("ego", "id"):
            return False
        try:
            self.client(role).call("refresh_session", session_id=session_id,
                                   reason=reason[:200])
            return True
        except Exception:  # noqa: BLE001
            # Supervision will restart it if it is genuinely unreachable; a
            # role that missed the handover fails its next turn and recovers.
            self.log.warning("could not hand %s its new session %s", role,
                             session_id, exc_info=True)
            return False

    def _expire_stale_turns(self) -> None:
        """Backstop for a role that is alive but stuck.

        A crash is recoverable because the process is visibly gone. A hang is
        not: nothing dies, nothing is restarted, and the open turn wedges the
        role just as thoroughly.
        """
        from . import mailbox

        if self.mind is None:
            return
        grace = self.cfg.scheduler.turn_wall_seconds + STALE_TURN_GRACE_SECONDS
        try:
            _, out = self.mind.writer.apply(
                lambda m: mailbox.expire_stale_turns(m, self.mind,
                                                     max_seconds=grace),
                actor="supervisor", bump_version=False)
        except Exception:  # noqa: BLE001
            self.log.exception("stale turn sweep failed")
            return
        for entry in out.get("expired_turns") or []:
            self.log.warning("expired stale turn %s; %d trigger(s) requeued",
                             entry["turn_id"], len(entry["requeued"]))

    def stop(self) -> None:
        self._stop.set()
        for name in reversed(CHILDREN):
            try:
                self._terminate(name)
            except Exception:  # noqa: BLE001
                self.log.exception("failed stopping %s", name)
        if self.api is not None:
            try:
                self.api.stop()
            except Exception:  # noqa: BLE001
                pass
            self.api = None
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
        kv_used, kv_capacity = 0, 0
        try:
            health = self.client("inference").call("health")
            inf_sessions = health.get("active_sessions", 0)
            max_sessions = health.get("max_sessions", 0)
            vram = health.get("vram_free_bytes", 0)
        except Exception:  # noqa: BLE001
            pass
        try:
            # Measured, not inferred from the health summary: this is the
            # figure admission refuses on, and it counts a shared prefix once
            # and a recomputed one in full.
            report = self.client("inference").call("context_report")
            kv_used = int(report.get("pool_tokens_used") or 0)
            kv_capacity = int(report.get("pool_capacity") or 0)
        except Exception:  # noqa: BLE001
            # Left at zero, which `kv_admission` reads as "not measurable"
            # and declines to guess about.
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
            kv_tokens_used=kv_used,
            kv_pool_capacity=kv_capacity,
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
        # Outside the dispatch critical section on purpose: this takes the
        # writer lock, and holding the scheduler lock across it would make
        # every trigger enqueue wait behind a neuocyte spawn.
        self._expire_stale_turns()
        self._retention_tick()

    def _retention_tick(self) -> None:
        """Housekeeping, on a cadence measured in hours.

        Not a cognitive decision and not an urgent one. The working set grows
        slowly, so a sweep that finds nothing should be rare rather than
        merely cheap.
        """
        policy = self.cfg.retention
        if not policy.enabled or self.mind is None:
            return
        now = time.time()
        if now - getattr(self, "_last_prune", 0.0) < policy.sweep_seconds:
            return
        self._last_prune = now
        try:
            out = self.methods()["store_prune"]()
        except Exception:  # noqa: BLE001
            self.log.exception("retention sweep failed")
            return
        if out.get("triggers_pruned") or out.get("turns_pruned"):
            self.log.info("pruned %d trigger(s) and %d turn(s) older than %.0fs",
                          out["triggers_pruned"], out["turns_pruned"],
                          policy.working_set_seconds)

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
                # Rejuvenation replaces the inference session and closes the
                # old one. Without telling the role, it keeps a dead handle
                # and every subsequent turn fails against a session that is
                # gone -- alive, heartbeating, and unable to think.
                self.hand_over_session(out["role"], out.get("new_session_id"),
                                       reason=out.get("reason", ""))
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
            try:
                # Answering is not thinking: a role can pass every probe
                # while failing every turn.
                self._supervise_thinking()
            except Exception:  # noqa: BLE001
                self.log.exception("thinking check failed")
            try:
                # An answer that exists belongs to the client that asked,
                # whatever became of the thread that was waiting for it.
                self.methods()["io_reconcile"]()
            except Exception:  # noqa: BLE001
                self.log.exception("interaction reconcile failed")
            try:
                self._wake_on_conditions()
            except Exception:  # noqa: BLE001
                self.log.exception("condition wake failed")
            try:
                self._schedule_heartbeats()
            except Exception:  # noqa: BLE001
                self.log.exception("heartbeat scheduling failed")
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

    def _supervise_thinking(self) -> None:
        """Restart a role that answers but cannot complete a turn.

        Reachability is not health. Live, Id held a session the Harness had
        closed -- after a rejuvenation Id itself requested -- and failed every
        turn for thirty-seven hours, reporting itself alive throughout,
        because nothing asked whether its turns were working. The record knew
        all along: seventy-five failures and not one successful turn.

        Bounded: the streak is recorded once, the repair is rate limited, and
        a role that keeps failing after its repairs are spent is left alone
        and stays visibly unwell rather than being restarted in a loop.
        """
        if self.mind is None:
            return
        threshold = int(self.cfg.scheduler.role_failure_threshold_turns)
        if threshold <= 0:
            return
        for role in ("ego", "id"):
            try:
                streak = mailbox.failing_streak(self.mind.db.conn, role)
            except Exception:  # noqa: BLE001
                self.log.debug("could not measure %s's turns", role, exc_info=True)
                continue
            if streak["turns"] < threshold:
                self._not_thinking.pop(role, None)
                continue
            if self._not_thinking.get(role) == streak["since"]:
                continue                     # this streak is already on the record
            self._not_thinking[role] = streak["since"]
            self.log.error("%s has failed %d turns in a row (%s)", role,
                           streak["turns"], streak["stop_reason"])
            repairs = [t for t in self._role_repairs.get(role, [])
                       if time.time() - t < 3600.0]
            allowed = int(self.cfg.scheduler.role_repairs_per_hour)
            self._emit_not_thinking(role, streak, repaired=len(repairs) < allowed)
            if len(repairs) >= allowed:
                self.log.error("%s stays unwell: %d repairs already used this "
                               "hour", role, len(repairs))
                continue
            repairs.append(time.time())
            self._role_repairs[role] = repairs
            self._restart_child(role, (f"failed {streak['turns']} turns in a row "
                                       f"({streak['stop_reason']})"))

    def _wake_on_conditions(self) -> None:
        """Wake the inward mind for what nobody else will tell it.

        Event-driven, so the pressure gate never holds it back: that gate
        exists to stop a *discretionary* review adding load, and pressure is
        the reason to wake, not a reason to stay quiet.
        """
        sched = self.cfg.scheduler
        if not sched.condition_wakes or self.mind is None:
            return
        streaks = {r: mailbox.failing_streak(self.mind.db.conn, r)["turns"]
                   for r in ("ego", "id")}
        found = conditions.detect(
            "id",
            failures=self.pulse.failures_last("last_5m"),
            pressure=self.homeostasis.last_pressure(),
            streaks=streaks,
            failure_threshold=int(sched.failure_wake_threshold),
            pressure_level=str(sched.pressure_wake_level),
            failure_turns=int(sched.role_failure_threshold_turns))
        now = time.time()
        cooldown = float(sched.condition_wake_cooldown_seconds)
        for condition in found:
            when = self._condition_woke.get(("id", condition.key))
            if when is not None and now - when < cooldown:
                continue
            if mailbox.pending_of_kind(self.mind.db.conn, "id", "attention",
                                       condition.key):
                continue
            self._condition_woke[("id", condition.key)] = now
            self.log.info("waking id: %s", condition.text)
            try:
                self.methods()["role_enqueue_trigger"](
                    role="id", kind="attention", source="harness",
                    source_ref=condition.key,
                    summary=condition.text[:200],
                    payload={"condition": condition.key,
                             "message": conditions.render(condition),
                             "measured": condition.measured},
                    ambient=True)
            except Exception:  # noqa: BLE001
                self.log.debug("could not wake id for %s", condition.key,
                               exc_info=True)

    def _emit_not_thinking(self, role: str, streak: dict[str, Any], *,
                           repaired: bool) -> None:
        def body(m: Mutation) -> None:
            m.emit(EventKind.ROLE_NOT_THINKING, {
                "role": role, "consecutive_failed_turns": streak["turns"],
                "stop_reason": streak["stop_reason"], "since": streak["since"],
                "repair": "restarting the role" if repaired else
                          "repairs for this hour are spent; left running and unwell"})

        try:
            self.mind.writer.apply(body, actor="supervisor", bump_version=False)
        except Exception:  # noqa: BLE001
            self.log.exception("could not record that %s is not thinking", role)

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
        if name in ("ego", "id"):
            # Whatever turn it was in the middle of, it is not going to close
            # it. One open turn per role is a database constraint, so leaving
            # the row behind would block every future turn for this role: it
            # would come back, heartbeat, report healthy, and never think
            # again while its mailbox filled. Its unconsumed triggers go back
            # on the queue for the incarnation that replaces it.
            self._recover_role_turns(
                name, reason=f"{name} was restarted mid-turn: {reason}"[:200])
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
    def _start_api(self) -> None:
        """Bring up the HTTP front door.

        Two adapters share the listener and nothing else: each holds its own
        credential and therefore its own method table. The operator session
        token is printed once, because a console nobody can log into is not a
        console.
        """
        try:
            self.api = ApiServer(self.cfg)
            host, port = self.api.start()
            self.log.info("operator console: http://%s:%s/", host, port)
            self.log.info("operator session token: %s",
                          self.api.operator_session)
            self.log.info("external API key: %s", self.api.clients.any_key())
        except Exception:  # noqa: BLE001
            self.log.exception("could not start the HTTP api; the mind runs "
                               "without an external interface")
            self.api = None

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
            from . import (ego_api, harness_api, id_api, io_api,
                           operator_api, prompt_api, supervisor_api, turn_api)

            methods = supervisor_api.build(self)
            methods.update(harness_api.build(self))
            methods.update(id_api.build(self))
            methods.update(ego_api.build(self))
            methods.update(io_api.build(self))
            methods.update(prompt_api.build(self))
            methods.update(turn_api.build(self))
            methods.update(operator_api.build(self))
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
