"""Context homeostasis: the Harness keeps the mind's contexts healthy.

A long-running mind fills its KV pool. That is not merely a capacity problem:
with ``kv_unified=True`` the pool is shared, so occupancy taxes *every* decode,
including sessions that have nothing to do with the one hogging space
(measured: 1.94x slowdown at 77% occupancy, fully recovered on retirement --
see ``docs/BENCHMARKS.md`` §2). Left alone, a mind gets slower and then stops.

## Who is allowed to do what

**The model never touches KV.** There is no tool, no verb and no code path by
which Ego, Id or a neuocyte manipulates a cache directly. Id observes pressure
and may *request* rejuvenation; the Harness decides whether to honour it, does
the work, and issues the receipt. A request is a proposal, and the refusal path
is a normal outcome rather than an error.

## What rejuvenation actually does

The honest minimum, which is what is implemented:

1. **Checkpoint.** Publish the session's exact token prefix to the durable
   content store. Nothing is lost from the *record*.
2. **Retire.** Close the backend session. Its cells are reclaimed once no other
   sequence owns them.
3. **Rebirth.** Open a replacement session and reconstitute it from the
   checkpoint.

Reconstitution has three possible modes and they are not interchangeable:

``exact``
    Replay the whole recorded token prefix. Semantically perfect, and useless
    for relieving pressure, because the context ends up the same size.
``rebuild``
    Reconstruct the substrate and keep cognition selectively, in whole
    parts: the governed prompt rendered fresh, every environment block
    removed (the next turn of a new session is given the current declaration
    in full), settled turns dropped whole, and turns still owed an answer
    kept whole -- an oversized result in one is shown as a bounded
    projection naming the exact stored copy, never deleted. No message is
    cut. See ``reconstitution``.
``summarise``
    Ask a model to compress the context. **Not implemented.** It is a different
    behaviour from rebuilding, not a better version of it, and calling it
    reconstitution would be a lie about what the mind now contains.

``rebuild`` is the default. There used to be a positional ``trim`` -- keep a
verbatim head and tail, drop what lay between -- and live it cut Id's
declaration off mid-line and left references to text it had removed: states
the conversation could never have reached by itself. It is gone rather than
kept as an option, because the rule is that nothing is token-spliced.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import framing
from . import reconstitution as rc
from .results import issue_result, issued_digest
from .tools import DEFAULT_RESULT_BUDGET_TOKENS, project_result
from .errors import CapabilityUnsupported, InvalidInput, NotFound
from .ids import new_id
from .logging_setup import get_logger
from .store.events import EventKind
from .store.writer import Mutation

PRESSURE_LEVELS = ("nominal", "elevated", "high", "critical")
# The smallest projection a rebuild will make of a result in owed work, once
# the role's ordinary delivery bound was not enough.
REBUILD_RESULT_FLOOR_TOKENS = 128
RECONSTITUTION_MODES = ("exact", "rebuild", "summarise")


@dataclass(slots=True)
class HomeostasisConfig:
    """Thresholds are fractions of the total shared KV pool."""

    elevated: float = 0.55
    high: float = 0.70
    critical: float = 0.85
    # A role is a candidate for rejuvenation once its own context passes this
    # fraction of the per-role budget.
    role_context_high: float = 0.75
    # A rebuild keeps whole cognition up to this fraction of the role's own
    # context budget, leaving the rest for the declaration the next turn is
    # given and for the turn itself.
    rebuild_keep_fraction: float = 0.40
    min_seconds_between_rejuvenations: float = 120.0
    max_rejuvenations_per_hour: int = 12
    auto_rejuvenate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class ContextReport:
    pool_tokens_used: int
    pool_capacity: int
    occupancy: float
    pressure: str
    sessions: list[dict[str, Any]] = field(default_factory=list)
    measured_at: float = 0.0
    backend_available: bool = True
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class ContextHomeostasis:
    """Owned by the supervisor. Deterministic: no model is consulted anywhere."""

    def __init__(self, cfg: HomeostasisConfig, *, mind: Any,
                 inference: Callable[[], Any], log_name: str = "homeostasis") -> None:
        self.cfg = cfg
        self.mind = mind
        self._inference = inference
        self.log = get_logger(log_name)
        self._history: list[float] = []
        self._last: dict[str, float] = {}

    # ------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------
    def last_pressure(self) -> tuple[str | None, float]:
        """The most recent measured pressure, and how old it is in seconds.

        Cached rather than measured, so a caller deciding a few times a minute
        whether to hold something back does not take an RPC to the very
        service it is worried about.

        `None` means nothing has been measured yet, or the last attempt found
        the backend unavailable. A caller must read that as *unknown*, never
        as pressure: an absent measurement is not evidence, and holding back
        cognition because a monitor was down would stop the organism thinking
        for a reason that has nothing to do with its resources.
        """
        report = getattr(self, "_last_report", None)
        if report is None or not report.backend_available:
            return None, float("inf")
        return report.pressure, max(0.0, time.time() - report.measured_at)

    def measure(self) -> ContextReport:
        """Read occupancy from the inference service. Never estimates."""
        try:
            raw = self._inference().call("context_report")
        except Exception as exc:  # noqa: BLE001
            self._last_report = ContextReport(
                pool_tokens_used=0, pool_capacity=0, occupancy=0.0,
                pressure="nominal", measured_at=time.time(),
                backend_available=False, detail=f"inference unreachable: {exc!r}")
            return self._last_report
        used = int(raw.get("pool_tokens_used", 0))
        cap = int(raw.get("pool_capacity", 0)) or 1
        occ = used / cap
        self._last_report = ContextReport(
            pool_tokens_used=used, pool_capacity=cap, occupancy=occ,
            pressure=self.classify(occ), sessions=raw.get("sessions", []),
            measured_at=time.time(), backend_available=True,
            detail=raw.get("detail", ""))
        return self._last_report

    def classify(self, occupancy: float) -> str:
        c = self.cfg
        if occupancy >= c.critical:
            return "critical"
        if occupancy >= c.high:
            return "high"
        if occupancy >= c.elevated:
            return "elevated"
        return "nominal"

    def assess(self) -> dict[str, Any]:
        """Measure, then say what the Harness would do about it.

        Assessment never acts. It is safe for Id to call as often as it likes.
        """
        report = self.measure()
        recommendations: list[dict[str, Any]] = []
        if report.backend_available:
            for s in report.sessions:
                role = s.get("role", "")
                budget = max(1, int(s.get("budget_tokens") or report.pool_capacity))
                # Measured the way this session's allowance is written. A role
                # is judged on everything it holds; a worker that inherited a
                # prefix is judged on what it added, because comparing its
                # total against an allowance written for its growth is how a
                # forked worker gets declared over budget the moment it is
                # born.
                held = int(s.get("budgeted_tokens", s.get("n_past", 0)) or 0)
                frac = held / budget
                if frac >= self.cfg.role_context_high or report.pressure in ("high", "critical"):
                    recommendations.append({
                        "session_id": s.get("session_id"), "role": role,
                        "n_past": s.get("n_past"),
                        "budgeted_tokens": held, "budget_tokens": budget,
                        "budget_basis": s.get("budget_basis", "total"),
                        "context_fraction": round(frac, 3),
                        "action": "rejuvenate" if role in ("ego", "id") else "retire",
                        "why": ("role context is large" if frac >= self.cfg.role_context_high
                                else f"pool pressure is {report.pressure}"),
                    })
        return {
            "report": report.to_dict(),
            "thresholds": self.cfg.to_dict(),
            "recommendations": recommendations,
            "rejuvenations_last_hour": self._recent_count(),
            "note": ("assessment performs no action; the Harness acts only through "
                     "rejuvenate(), and only it may touch a KV cache"),
        }

    # ------------------------------------------------------------------
    # act (Harness only)
    # ------------------------------------------------------------------
    def _recent_count(self) -> int:
        cutoff = time.time() - 3600
        self._history = [t for t in self._history if t > cutoff]
        return len(self._history)

    def admit_rejuvenation(self, role: str) -> tuple[bool, str]:
        """Rate limits, so a wedged Id cannot thrash the mind's contexts."""
        now = time.time()
        if self._recent_count() >= self.cfg.max_rejuvenations_per_hour:
            return False, (f"rate limit: {self.cfg.max_rejuvenations_per_hour} "
                           "rejuvenations per hour already used")
        last = self._last.get(role, 0.0)
        gap = now - last
        if gap < self.cfg.min_seconds_between_rejuvenations:
            return False, (f"{role} was rejuvenated {gap:.0f}s ago; minimum interval "
                           f"is {self.cfg.min_seconds_between_rejuvenations:.0f}s")
        return True, "admitted"

    def request_rejuvenation(self, *, role: str, reason: str, requested_by: str = "id",
                             mode: str = "rebuild", operation_id: str | None = None
                             ) -> dict[str, Any]:
        """Id's entry point. A request, not a command.

        Recorded either way: an accepted request and a refused one are both
        facts about how the mind managed itself.
        """
        if role not in ("ego", "id"):
            raise InvalidInput("only ego and id have rejuvenable contexts", role=role)
        ok, detail = self.admit_rejuvenation(role)
        self._emit(EventKind.REJUVENATION_REQUESTED, {
            "role": role, "requested_by": requested_by, "reason": reason,
            "mode": mode, "admitted": ok, "detail": detail,
        }, actor=requested_by, operation_id=operation_id)
        if not ok:
            self._emit(EventKind.REJUVENATION_REFUSED,
                       {"role": role, "reason": detail}, actor="supervisor",
                       operation_id=operation_id)
            return {"performed": False, "role": role, "refused_because": detail,
                    "requested_by": requested_by}
        return self.rejuvenate(role=role, reason=reason, mode=mode,
                               requested_by=requested_by, operation_id=operation_id)

    def rejuvenate(self, *, role: str, reason: str, mode: str = "rebuild",
                   requested_by: str = "supervisor", operation_id: str | None = None
                   ) -> dict[str, Any]:
        """Checkpoint, retire, reborn. Performed by the Harness, receipted."""
        if mode not in RECONSTITUTION_MODES:
            raise InvalidInput("unknown reconstitution mode", mode=mode,
                               allowed=list(RECONSTITUTION_MODES))
        if mode == "summarise":
            raise CapabilityUnsupported(
                "summarising a context is a different behaviour from reconstituting "
                "it, and is not implemented; use 'rebuild', which keeps every "
                "surviving message whole",
                mode=mode)
        inf = self._inference()
        before = self.measure()

        checkpoint = self.checkpoint(role=role, operation_id=operation_id)
        tokens: list[int] = checkpoint["tokens"]
        old_session = checkpoint["session_id"]
        if mode == "rebuild":
            built = self._rebuild(role, old_session, tokens, inf)
            keep, plan = built["tokens"], built["plan"]
        else:
            keep = list(tokens)
            plan = {"mode": "exact", "placed": []}

        inf.call("close_session", session_id=old_session)
        self._emit(EventKind.SESSION_RETIRED, {
            "role": role, "session_id": old_session, "reason": reason,
            "n_past": len(tokens),
        }, actor="supervisor", operation_id=operation_id)

        # The same identity continuing, so the same ceiling. Without this a
        # reborn role would quietly fall back to the global default, and the
        # symptom would be a role that worked until the first time it was
        # rejuvenated.
        role_cfg = getattr(self.mind.cfg, role, None)
        budget = int(getattr(role_cfg, "max_context_tokens", 0) or 0) or None
        new = inf.call("open_session", role=role,
                       context_budget_tokens=budget, budget_basis="total")
        if keep:
            inf.call("restore_prefix", session_id=new["session_id"], tokens=keep,
                     snapshot_id=checkpoint.get("snapshot_id"))

        # Durable, and before anything else can read the old handle. This is
        # the copy the Harness itself reads on the next rejuvenation, and
        # leaving it stale made that rejuvenation checkpoint a session that no
        # longer exists.
        try:
            self.mind.work.set_session_handle(
                agent_id=role, session_handle=new["session_id"],
                reason=f"rejuvenated: {reason}"[:200])
        except Exception:  # noqa: BLE001
            self.log.exception(
                "could not record %s's new session handle %s; the next "
                "rejuvenation would work from a closed session", role,
                new["session_id"])
            raise
        # Told to the role here rather than by the caller, because every
        # path that replaces a session has to do it and only one of the three
        # did. Live, Id used `id_request_rejuvenation` on its own context at
        # 77% -- correctly -- and was never told the new handle: it went on
        # calling a session the Harness had closed, and every turn for the
        # next day and a half failed with "unknown inference session". Asking
        # for help was the one way it could wedge itself.
        told = self._hand_over(role, new["session_id"], reason)
        # Where each carried turn now sits, against the new handle. Its own
        # row keeps the coordinates it was measured under (I94); without these
        # the next rebuild would find every carried turn `unknown`, keep it
        # all, and the context could only ever grow.
        self._record_carried_spans(new["session_id"], plan.get("placed") or [])

        after = self.measure()
        self._history.append(time.time())
        self._last[role] = time.time()

        result = {
            "performed": True,
            "role": role,
            "requested_by": requested_by,
            "reason": reason,
            "mode": mode,
            "old_session_id": old_session,
            "new_session_id": new["session_id"],
            "checkpoint_snapshot_id": checkpoint.get("snapshot_id"),
            "tokens_before": len(tokens),
            "tokens_after": len(keep),
            "dropped_tokens": len(tokens) - len(keep),
            "occupancy_before": round(before.occupancy, 4),
            "occupancy_after": round(after.occupancy, 4),
            "pressure_before": before.pressure,
            "pressure_after": after.pressure,
            # Whether the role itself knows. False is recoverable -- its next
            # heartbeat adopts the recorded handle -- but it is a fact about
            # this rejuvenation and is recorded as one.
            "role_told": told,
            "reconstitution": (
                "rebuilt from whole parts: the governed prompt rendered fresh, "
                "every environment block removed so the current declaration "
                "is given at the next turn, settled turns dropped whole, and "
                "turns still owed an answer kept whole with any oversized "
                "result shown as a projection of its stored copy; no message "
                "was cut, nothing is summarised, and the checkpoint holds "
                "every original token"
                if mode == "rebuild" else
                "the full recorded token prefix, replayed exactly"),
            **({k: v for k, v in plan.items() if k != "placed"}
               if mode == "rebuild" else {}),
        }
        self._emit(EventKind.REJUVENATION_PERFORMED, result, actor="supervisor",
                   operation_id=operation_id)
        self._emit(EventKind.SESSION_REBORN, {
            "role": role, "session_id": new["session_id"],
            "tokens_restored": len(keep),
        }, actor="supervisor", operation_id=operation_id)
        self.log.info("rejuvenated %s: %d -> %d tokens, occupancy %.1f%% -> %.1f%%",
                      role, len(tokens), len(keep), before.occupancy * 100,
                      after.occupancy * 100)
        return result

    def _hand_over(self, role: str, session_id: str, reason: str) -> bool:
        """Tell the role which session it now has. Never fails the rebuild."""
        source = getattr(self, "hand_over", None)
        if not callable(source):
            return False
        try:
            return bool(source(role, session_id, reason=reason))
        except Exception:  # noqa: BLE001
            self.log.warning("could not hand %s its new session %s", role,
                             session_id, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # rebuild: whole parts, never offsets
    # ------------------------------------------------------------------
    def _markers(self, inf: Any) -> dict[str, Any]:
        """The chat template's message framing, read from the template itself.

        Rendered rather than hard-coded, so the boundaries are the ones this
        model's sessions really have.
        """
        probe = "\x1f"
        rendered = inf.call("apply_chat_template",
                            messages=[{"role": "user", "content": probe}],
                            add_assistant=False)
        head, _, tail = rendered.partition(probe)
        first = inf.call("tokenize", text=head, add_special=False,
                         parse_special=True)
        start_id = int(first[0])
        start_text = inf.call("detokenize", tokens=[start_id], special=True)
        return {"start_id": start_id, "start_text": start_text, "end_text": tail}

    def _messages(self, tokens: Sequence[int], inf: Any,
                  mk: dict[str, Any]) -> list[rc.Message]:
        out = []
        for a, b in rc.split_messages(tokens, mk["start_id"]):
            piece = list(tokens[a:b])
            text = inf.call("detokenize", tokens=piece, special=True)
            role, kind, terminated, _ = rc.classify(
                text, start_text=mk["start_text"], end_text=mk["end_text"])
            out.append(rc.Message(start=a, end=b, tokens=piece, text=text,
                                  role=role, kind=kind, terminated=terminated))
        return out

    def _governed_tokens(self, role: str, inf: Any) -> list[int] | None:
        """The governed prompt, rendered the way a role primes its session."""
        source = getattr(self, "governed_prompt", None)
        text = source(role) if callable(source) else None
        if not text:
            return None
        return list(inf.call(
            "render_tokens",
            messages=[{"role": "system", "content": text}],
            add_assistant=False))

    def _rebuild(self, role: str, session_id: str, tokens: Sequence[int],
                 inf: Any) -> dict[str, Any]:
        from . import mailbox

        mk = self._markers(inf)
        messages = self._messages(tokens, inf, mk)
        system = messages[0] if messages and messages[0].kind == "system" else None
        rest = messages[1:] if system else messages

        units = rc.group_units(rest)
        conn = self.mind.db.conn
        rc.attribute(units, mailbox.session_spans(conn, role, session_id),
                     mailbox.owed_lineages(conn, role),
                     mailbox.active_turns(conn, role))
        # An assistant header nothing was generated into: a generation the
        # Harness refused. It carries nothing, and keeping it would leave the
        # next message nested inside an unopened reply. Removed after the
        # turns are placed, because the turn that ended there counts it.
        dropped_prompts = 0
        if units and rc.is_empty_generation_prompt(units[-1].messages[-1]) \
                and len(units[-1].messages) > 1:
            units[-1].messages.pop()
            dropped_prompts = 1

        # Substrate out of every kept opening. Re-rendered from its own text,
        # so what remains of the message is byte-for-byte what it said.
        environments_removed = 0
        environment_tokens = 0
        for u in units:
            for m in u.messages:
                if m.kind != "opening":
                    continue
                text, changed = rc.strip_environment(m.text)
                if changed:
                    new = self._retokenize(text, mk, inf)
                    environment_tokens += m.n - len(new)
                    environments_removed += 1
                    m.tokens, m.text, m.rewritten = new, text, True

        governed = self._governed_tokens(role, inf)
        if governed is not None:
            head, system_source = governed, "reconstructed"
        elif system is not None:
            # No binding to render from -- a partial bootstrap. The message
            # the incarnation primed with is whole and is its governed prompt.
            head, system_source = list(system.tokens), "verbatim"
        else:
            head, system_source = [], "absent"

        role_cfg = getattr(self.mind.cfg, role, None)
        budget = int(getattr(role_cfg, "max_context_tokens", 0) or 0) or len(tokens)
        target = int(budget * self.cfg.rebuild_keep_fraction)
        chosen = rc.plan(units, system_tokens=len(head), target=target)

        # Owed work is never cut, but an oversized result in it can be shown
        # the way a live call would show it: a bounded projection naming the
        # exact stored copy. The continuation keeps "I called X, here is what
        # it returned, and the rest is retrievable" instead of losing X.
        projected: list[dict[str, Any]] = []
        if not chosen["reached_target"]:
            total = chosen["kept_tokens"]
            delivery = (int(role_cfg.tool_result_budget_tokens)
                        if role_cfg is not None else DEFAULT_RESULT_BUDGET_TOKENS)
            for bound in (delivery, REBUILD_RESULT_FLOOR_TOKENS):
                for i, j in rc.shrinkable(units, chosen["kept_units"]):
                    if total <= target:
                        break
                    done = self._project_message(role, units[i].messages[j],
                                                 budget=bound, inf=inf, mk=mk)
                    if done:
                        total -= done["saved"]
                        projected.append(done)
                if total <= target:
                    break
            chosen["kept_tokens"] = total
            chosen["reached_target"] = total <= target

        body, placed = rc.assemble(units, chosen)
        base = len(head)
        return {
            "tokens": head + body,
            "plan": {
                "system_prompt": system_source,
                "system_prompt_changed": (system is not None and governed is not None
                                          and list(system.tokens) != governed),
                "messages_before": len(messages),
                "units_before": len(units),
                "units_kept": len(chosen["kept_units"]),
                "units_unknown": sum(1 for u in units if u.status == "unknown"),
                "units_owed": sum(1 for u in units if u.status == "owed"),
                "environments_removed": environments_removed,
                "environment_tokens_removed": environment_tokens,
                "empty_generation_prompts_removed": dropped_prompts,
                "dropped_units": chosen["dropped_units"],
                "results_projected": projected,
                "target_tokens": target,
                "reached_target": chosen["reached_target"],
                "placed": [{"turn_ids": u.turn_ids, "start": base + a, "end": base + b}
                           for u, a, b in placed if u.turn_ids],
            },
        }

    def _retokenize(self, text: str, mk: dict[str, Any], inf: Any) -> list[int]:
        """A message's text back to tokens, with the frame the only structure.

        A rebuild decodes tokens it already holds and tokenizes the text
        again. If the body of a message contains the *characters* of a chat
        marker -- which is exactly what a client's text now becomes -- reading
        the whole message back with specials parsed would turn those
        characters into a real boundary on the way in. So the header and the
        terminator are tokenized as the template's, and everything between
        them as what it is. See I132.
        """
        segs = framing.message_segments(
            text, start_text=mk["start_text"], end_text=mk["end_text"])
        return list(inf.call("tokenize_segments",
                             segments=framing.as_payload(segs)))

    def _project_message(self, role: str, m: rc.Message, *, budget: int,
                         inf: Any, mk: dict[str, Any]) -> dict[str, Any] | None:
        """Re-render one tool-result message as a bounded projection.

        From the exact result, always: a message that already holds a
        projection is re-projected from the stored copy its reference names,
        so "of 17" still means seventeen. A refusal or plain-text result has
        no structure to project and is left as it is.
        """
        parts = rc.result_body(m.text)
        if parts is None:
            return None
        before, name, body, after = parts
        try:
            payload = json.loads(body)
        except ValueError:
            return None
        if (isinstance(payload, dict) and payload.get("complete") is False
                and payload.get("result_ref")):
            ref = str(payload["result_ref"])
            digest = issued_digest(self.mind, ref, role=role)
            if digest is None:
                return None
            payload = json.loads(self.mind.blobs.get(digest).decode("utf-8"))
        else:
            ref = issue_result(self.mind, body, role=role, tool=name,
                               actor="supervisor")[:16]

        def count(text: str) -> int:
            # A projection is content being measured, so it is measured the
            # way it will be tokenized: as text, not as possible structure.
            return len(inf.call("tokenize", text=text, add_special=False,
                                parse_special=False))

        view = project_result(payload, budget_tokens=budget, count=count, ref=ref)
        text = before + view["text"] + after
        tokens = self._retokenize(text, mk, inf)
        if len(tokens) >= m.n:
            return None
        saved = m.n - len(tokens)
        record = {"tool": name, "tokens_before": m.n, "tokens_after": len(tokens),
                  "saved": saved, "result_ref": ref}
        m.tokens, m.text, m.rewritten = tokens, text, True
        return record

    def _record_carried_spans(self, session_id: str,
                              placed: Sequence[dict[str, Any]]) -> None:
        if not placed:
            return

        def body(m: Mutation) -> None:
            for p in placed:
                for turn_id in p["turn_ids"]:
                    m.conn.execute(
                        "INSERT OR REPLACE INTO turn_spans(turn_id, session_handle,"
                        " token_start, token_end) VALUES (?, ?, ?, ?)",
                        (turn_id, session_id, int(p["start"]), int(p["end"])))

        self.mind.writer.apply(body, actor="supervisor", bump_version=False,
                               mutation_id=f"homeo:spans:{new_id('h')}")

    def checkpoint(self, *, role: str, operation_id: str | None = None
                   ) -> dict[str, Any]:
        """Persist the role's exact token prefix before anything is discarded."""
        inf = self._inference()
        agents = {a["agent_id"]: a for a in self.mind.work.live_agents()}
        agent = agents.get(role)
        session_id = (agent or {}).get("session_handle")
        if not session_id:
            raise NotFound("role has no live inference session", role=role)
        info = inf.call("session_tokens", session_id=session_id)
        tokens = list(info["tokens"])
        snapshot_id = None
        if role == "ego" and tokens:
            text = inf.call("detokenize", tokens=tokens, special=True)
            caps = inf.call("capabilities")
            snapshot_id, _version, _receipt = self.mind.work.publish_snapshot(
                actor="ego", model_generation=caps.get("model_generation", ""),
                token_count=len(tokens), tokens=tokens, text=text,
                kv_mode=caps.get("kv_mode", "recomputed"), backend_handle=session_id,
                operation_id=operation_id)
        else:
            # Id's private context is never published as a shared snapshot; it
            # is still checkpointed to content-addressed storage so a rebirth
            # can reconstitute it.
            blob = self.mind.blobs.put_json(tokens)
            self._emit(EventKind.CONTEXT_MEASURED,
                       {"role": role, "checkpoint_blob": blob,
                        "token_count": len(tokens)},
                       actor="supervisor", operation_id=operation_id)
        return {"role": role, "session_id": session_id, "tokens": tokens,
                "snapshot_id": snapshot_id, "token_count": len(tokens)}

    def retire_session(self, *, session_id: str, reason: str,
                       operation_id: str | None = None) -> dict[str, Any]:
        """Close one backend session. Safe for neuocyte sessions at any time."""
        self._inference().call("close_session", session_id=session_id)
        self._emit(EventKind.SESSION_RETIRED,
                   {"session_id": session_id, "reason": reason},
                   actor="supervisor", operation_id=operation_id)
        return {"retired": session_id, "reason": reason}

    # ------------------------------------------------------------------
    def tick(self) -> dict[str, Any] | None:
        """Called by the scheduler. Acts only above the critical threshold.

        Deliberately conservative: rejuvenation costs a prefill and loses live
        context, so it happens when the alternative is a mind that is measurably
        degrading, not merely a full-ish pool.
        """
        if not self.cfg.auto_rejuvenate:
            return None
        report = self.measure()
        if not report.backend_available:
            return None
        if report.pressure != "critical":
            return None
        self._emit(EventKind.CONTEXT_PRESSURE, {
            "occupancy": round(report.occupancy, 4), "pressure": report.pressure,
            "pool_tokens_used": report.pool_tokens_used,
            "pool_capacity": report.pool_capacity,
        }, actor="supervisor")
        biggest = None
        for s in report.sessions:
            if s.get("role") in ("ego", "id"):
                if biggest is None or s.get("n_past", 0) > biggest.get("n_past", 0):
                    biggest = s
        if biggest is None:
            return None
        ok, detail = self.admit_rejuvenation(biggest["role"])
        if not ok:
            return {"performed": False, "refused_because": detail}
        return self.rejuvenate(role=biggest["role"],
                               reason=f"pool occupancy {report.occupancy:.0%} is critical",
                               requested_by="supervisor")

    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: dict[str, Any], *, actor: str,
              operation_id: str | None = None) -> None:
        def body(m: Mutation) -> None:
            m.emit(kind, payload)

        try:
            self.mind.writer.apply(body, actor=actor, operation_id=operation_id,
                                   bump_version=False,
                                   mutation_id=f"homeo:{kind}:{new_id('h')}")
        except Exception:  # noqa: BLE001
            self.log.exception("could not record %s", kind)
