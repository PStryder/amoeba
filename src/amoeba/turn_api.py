"""The Harness side of persistent-role turns.

Three audiences, three sets of verbs, and they are deliberately separate:

* the **role process** claims a turn and closes it (`role_claim_turn`,
  `role_complete_turn`);
* the **Harness itself** queues triggers when something happens
  (`role_enqueue_trigger`), from work completion, external input, startup and
  the heartbeat;
* the **Operator** looks at what is queued and what has been thought
  (`role_mailbox`, `role_turns`).

None of this is cognition. The scheduler is deterministic substrate: it
decides *when* a role gets another bounded turn, never what the role should
conclude. There is no scheduler neuocyte and no arbiter agent, and adding one
would put a model in the control plane the whole design keeps it out of.

The role process holds `role_claim_turn` and `role_complete_turn` in its scope
and the model does not: claiming your own next turn is not a cognitive act, and
a mind that could would be scheduling itself.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import TYPE_CHECKING, Any

from . import mailbox
from .errors import InvalidInput, NotFound
from .scopes import model_facing_verbs
from .tools import bounded_tool_result
from .store.writer import Mutation

if TYPE_CHECKING:
    from .supervisor import Supervisor

# What the role process may call. Not model-facing.
ROLE_TURN_VERBS = ("role_claim_turn", "role_complete_turn",
                   "role_abandon_turn", "role_tool_invoke")

# What the Operator may look at.
OPERATOR_TURN_VERBS = ("role_mailbox", "role_turns", "role_turn",
                       "operator_message_role")


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None

    # ==================================================================
    # The role process: claim, then close
    # ==================================================================
    def role_claim_turn(*, role: str, incarnation: int | None = None,
                        profile_ref: str | None = None,
                        prompt_sha256: str | None = None,
                        config_sha256: str | None = None,
                        model_generation: str = "") -> dict[str, Any]:
        """Open a bounded turn, if there is anything to think about.

        Everything that makes the turn explicable is frozen here in one
        transaction: the bound profile, a freshly built environment manifest,
        and the trigger bundle. Returns ``{"turn": None}`` when the mailbox is
        empty, which is the normal state of an idle organism rather than an
        error.

        The environment is built *inside* this call rather than by the role,
        so the manifest and the bundle are fixed at the same instant and the
        turn record cannot reference an environment from slightly before or
        after its own inputs.
        """
        if role not in mailbox.ROLES:
            raise InvalidInput("unknown role", role=role,
                               allowed=list(mailbox.ROLES))
        if not mailbox.pending_count(mind.db.conn, role):
            return {"turn": None, "reason": "nothing queued"}

        env = sup.methods()["role_environment"](
            role=role, incarnation=incarnation, profile_ref=profile_ref,
            prompt_sha256=prompt_sha256, config_sha256=config_sha256,
            trigger="turn start")

        def body(m: Mutation) -> dict[str, Any] | None:
            return mailbox.claim(
                m, mind, role=role, incarnation=incarnation,
                profile_ref=profile_ref, profile_sha256=prompt_sha256,
                environment_sha256=env["environment_sha256"],
                environment_blob=env["environment_blob"],
                model_generation=model_generation)

        receipt, claimed = mind.writer.apply(body, actor=role,
                                             bump_version=False)
        if claimed is None:
            return {"turn": None, "reason": "nothing queued"}
        return {"turn": claimed, "environment": env,
                "receipt_id": receipt.receipt_id}

    def role_complete_turn(*, turn_id: str, stop_reason: str,
                           tool_call_count: int = 0,
                           result: dict[str, Any] | None = None
                           ) -> dict[str, Any]:
        """Close a turn, consume its triggers, and continue if the Harness says so.

        Consumption is here rather than at claim time so that a role which
        died mid-turn leaves its inputs recoverable: being handed to a mind
        that then crashed is not the same as having been thought about.

        Whether a non-terminal stop earns another turn is decided by the
        Harness from the stop reason, without the model having to ask. A
        thought cut off by an output ceiling cannot be relied on to request
        its own continuation, because being cut off is what stopped it.
        """
        def body(m: Mutation) -> dict[str, Any]:
            return mailbox.complete(
                m, mind, turn_id=turn_id, stop_reason=stop_reason,
                tool_call_count=tool_call_count, result=result,
                max_continuations=sup.cfg.scheduler.max_continuations)

        receipt, out = mind.writer.apply(body, actor="harness",
                                         bump_version=False)
        sup.note_turn_finished(out.get("stop_reason"), turn_id=turn_id)
        if stop_reason == "context_pressure":
            out["rejuvenation"] = _rejuvenate(turn_id)
        return {**out, "receipt_id": receipt.receipt_id}

    def _rejuvenate(turn_id: str) -> dict[str, Any]:
        """Reclaim a role's context after a turn ended under pressure.

        Performed by the Harness, between turns. `context_rejuvenate` is
        Harness-initiated and appears in no role scope: a role that decided it
        needed more room could not simply take it, and it cannot rewrite its
        own context at all. It reports the pressure as a stop reason and the
        Harness answers.

        Between turns, never during one: rejuvenation replaces the inference
        session, and doing that mid-generation would discard the reasoning in
        progress. The continuation the stop reason earned then runs against
        the fresh session.
        """
        row = mind.db.conn.execute(
            "SELECT role FROM role_turns WHERE turn_id = ?",
            (turn_id,)).fetchone()
        if row is None:
            return {"performed": False, "reason": "unknown turn"}
        role = row["role"]
        try:
            done = sup.methods()["context_rejuvenate"](
                role=role, reason="context pressure at a turn boundary",
                mode="trim")
        except Exception as exc:  # noqa: BLE001
            sup.log.warning("rejuvenation for %s failed: %s", role, exc)
            return {"performed": False, "reason": str(exc)[:200]}
        session = done.get("session_id") or done.get("new_session_id")
        sup.hand_over_session(role, session, reason="context pressure")
        return {"performed": True, "role": role,
                "session_id": session,
                "note": ("identity, incarnation, profile binding and mailbox "
                         "are untouched; a replacement session is not a new "
                         "mind")}

    def role_abandon_turn(*, turn_id: str, reason: str = "abandoned"
                          ) -> dict[str, Any]:
        """Give a turn up and requeue what it never consumed."""
        receipt, out = mind.writer.apply(
            lambda m: mailbox.abandon(m, mind, turn_id=turn_id, reason=reason),
            actor="harness", bump_version=False)
        return {**out, "receipt_id": receipt.receipt_id}

    def role_tool_invoke(*, turn_id: str, name: str,
                         arguments: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
        """Execute one capability on behalf of a role, fenced to its turn.

        The fence the neuocytes always had and the roles did not. A role used
        to call its effectors directly on its own credential, carrying no turn
        identity at all -- so a turn that hung, was declared stale and had its
        inputs handed to a replacement could still wake up and act. Its
        *result* was refused, because `complete` rejects a turn that is no
        longer running, but its side effects were not: it could still request
        work, post findings and record conclusions into an organism that had
        moved on without it.

        `turn_id` is the capability, exactly as a neuocyte's fencing token is.
        It is minted by the Harness, handed only to the role that claimed that
        turn, and exposed nowhere a role can read -- the mailbox and turn
        views are operator-only. So there is no `role` argument to forge:
        which role is asking is derived from the turn, and a turn that is no
        longer `running` buys nothing.

        The verb must also still be one this role is offered, so the
        capability boundary is unchanged: this narrows what a role may do, and
        widens nothing.
        """
        row = mind.db.conn.execute(
            "SELECT role, status, operation_id FROM role_turns"
            " WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            return {"accepted": False, "result": None,
                    "reason": "no such turn"}
        if row["status"] != "running":
            return {"accepted": False, "result": None,
                    "reason": (f"turn {turn_id} is {row['status']}; the "
                               "organism has moved on and this turn can no "
                               "longer act")}
        role = row["role"]
        if name not in model_facing_verbs(role):
            return {"accepted": False, "result": None,
                    "reason": (f"{name!r} is not a capability offered to "
                               f"{role}")}
        handler = sup.methods().get(name)
        if handler is None:
            return {"accepted": False, "result": None,
                    "reason": f"{name!r} is not a dispatchable verb"}
        args = dict(arguments or {})
        # What a role does during a turn is accountable to the operation that
        # caused the turn. The role cannot supply this -- `operation_id` is
        # stripped from model-supplied arguments like every other authority
        # argument -- so the Harness supplies it from the turn itself.
        #
        # It matters beyond bookkeeping: work delegated here comes back as a
        # trigger whose lineage is the work's operation, and if that is empty
        # the result is untagged and will not enter the continuation that
        # delegated it.
        try:
            accepts = inspect.signature(handler).parameters
        except (TypeError, ValueError):  # a builtin or C callable
            accepts = {}
        takes_anything = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in accepts.values())
        if row["operation_id"] and "operation_id" not in args:
            if "operation_id" in accepts or takes_anything:
                args["operation_id"] = row["operation_id"]
        # A verb that scopes itself to the turn asking for it gets the turn,
        # from the fence rather than from the model. `turn_id` is an authority
        # argument, so anything the model supplied under that name was already
        # stripped before this.
        if "turn_id" in accepts:
            args["turn_id"] = turn_id
        try:
            result = handler(**args)
        except Exception as exc:  # noqa: BLE001
            # Reported back as a failed call rather than killing the turn: the
            # model may well be able to proceed without it.
            return {"accepted": False, "result": None,
                    "reason": f"{type(exc).__name__}: {exc}"[:500]}
        # Bounded here rather than in the role process: this is the side
        # that can store what does not fit, and a digest named by a notice has
        # to be a digest something actually holds.
        def _store(text: str) -> str:
            digest = mind.blobs.put(text.encode("utf-8"))
            mind.writer.apply(
                lambda m: m.register_blob(digest, len(text.encode("utf-8")),
                                          "application/json",
                                          "tool_result_full"),
                actor=role, bump_version=False)
            return digest

        bounded = bounded_tool_result(result, store=_store)
        return {"accepted": True, "result": result, "reason": None,
                "result_text": bounded["text"],
                "result_truncated": bounded["truncated"],
                "result_chars": bounded["chars"],
                "result_sha256": bounded["sha256"]}

    # ==================================================================
    # Queueing: the Harness noticing that something happened
    # ==================================================================
    def role_enqueue_trigger(*, role: str, kind: str, source: str,
                             summary: str, source_ref: str | None = None,
                             payload: dict[str, Any] | None = None,
                             correlation_id: str | None = None,
                             causal_parent: str | None = None,
                             operation_id: str | None = None,
                             expects_answer: bool = False,
                             lineage: str | None = None,
                             ambient: bool = False) -> dict[str, Any]:
        """Record that something happened which a role may need to think about.

        Queueing is not waking and not consumption. The trigger becomes a
        cognitive input only when a turn bundles it, and a caller can see both
        states separately rather than having to guess which one it is looking
        at.

        `lineage` says whose interaction this belongs to and `ambient` says
        every turn may see it. Both are separate from `expects_answer`, which
        is only about who is owed a reply: a work result is owed no reply and
        still belongs to exactly one interaction. Leaving both unset means
        "owned by nobody", which is deliberately *not* the same as ambient --
        such a trigger waits for a turn that is not already serving somebody.
        """
        receipt, out = mind.writer.apply(
            lambda m: mailbox.enqueue(
                m, role=role, kind=kind, source=source, summary=summary,
                source_ref=source_ref, payload=payload,
                correlation_id=correlation_id, causal_parent=causal_parent,
                operation_id=operation_id, expects_answer=expects_answer,
                lineage=lineage, ambient=ambient),
            actor=source or "harness", operation_id=operation_id,
            bump_version=False)
        sup.note_trigger(role)
        return {**out, "receipt_id": receipt.receipt_id}

    def operator_message_role(*, role: str, message: str, kind: str = "notice",
                              operation_id: str | None = None) -> dict[str, Any]:
        """The Operator putting something in a role's mailbox.

        Input, not authority. It wakes the role and is attributable, and the
        role acts on it only through its own effectors — the Operator asking a
        question does not lend the role any power it did not have.
        """
        if not message.strip():
            raise InvalidInput("a message needs a body")
        # Addressed to the role itself rather than to any interaction it is
        # handling, so it is ambient by construction.
        return role_enqueue_trigger(
            role=role, kind="operator_message", source="operator",
            summary=message.strip()[:mailbox.MAX_SUMMARY],
            payload={"message": message[:8000], "kind": kind},
            operation_id=operation_id, ambient=True)

    # ==================================================================
    # Looking at it
    # ==================================================================
    def role_mailbox(*, role: str | None = None, limit: int = 50
                     ) -> dict[str, Any]:
        """What each persistent role has queued, and what it is doing now."""
        out: dict[str, Any] = {}
        for name in ([role] if role else list(mailbox.ROLES)):
            queued = mailbox.pending(mind.db.conn, name)
            running = mailbox.open_turn(mind.db.conn, name)
            last = mind.db.conn.execute(
                "SELECT turn_id, stop_reason, status, finished_at, trigger_count"
                " FROM role_turns WHERE role = ? AND status != 'running'"
                " ORDER BY started_at DESC LIMIT 1", (name,)).fetchone()
            out[name] = {
                "state": sup.role_activity(name),
                "queued": len(queued),
                "queued_kinds": sorted({t["kind"] for t in queued}),
                "next_triggers": [
                    {"trigger_id": t["trigger_id"], "kind": t["kind"],
                     "source": t["source"], "summary": t["summary"][:120],
                     "created_at": t["created_at"]}
                    for t in queued[:limit]],
                "current_turn": (
                    {"turn_id": running["turn_id"],
                     "started_at": running["started_at"],
                     "trigger_count": running["trigger_count"],
                     "profile_ref": running["profile_ref"],
                     "environment_sha256": running["environment_sha256"]}
                    if running else None),
                "last_turn": dict(last) if last else None,
                "next_heartbeat": sup.next_heartbeat(name),
            }
        return out

    def role_turns(*, role: str | None = None, limit: int = 25
                   ) -> dict[str, Any]:
        """Recent bounded turns, newest first."""
        sql = ("SELECT turn_id, role, incarnation, profile_ref,"
               " environment_sha256, bundle_id, trigger_kinds, trigger_count,"
               " started_at, finished_at, status, stop_reason,"
               " tool_call_count, parent_turn FROM role_turns")
        params: tuple[Any, ...] = ()
        if role:
            sql += " WHERE role = ?"
            params = (role,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        rows = [dict(r) for r in mind.db.conn.execute(
            sql, (*params, max(1, min(int(limit), 200))))]
        for r in rows:
            r["trigger_kinds"] = json.loads(r["trigger_kinds"] or "[]")
        return {"turns": rows}

    def role_turn(*, turn_id: str) -> dict[str, Any]:
        """One turn, with the exact inputs it saw.

        The bundle and the environment are read back from the content store
        rather than recomputed, so this stays true after the mailbox, the
        library and the world have all moved on.
        """
        row = mind.db.conn.execute(
            "SELECT * FROM role_turns WHERE turn_id = ?", (turn_id,)).fetchone()
        if row is None:
            raise NotFound("no such turn", turn_id=turn_id)
        turn = dict(row)
        turn["trigger_kinds"] = json.loads(turn["trigger_kinds"] or "[]")
        turn["triggers"] = [dict(r) for r in mind.db.conn.execute(
            "SELECT trigger_id, kind, source, source_ref, summary, status,"
            " created_at, claimed_at, consumed_at, deliveries"
            " FROM role_triggers WHERE turn_id = ? ORDER BY created_at",
            (turn_id,))]
        for key, blob in (("bundle", turn.get("bundle_blob")),
                          ("environment", turn.get("environment_blob"))):
            if blob:
                try:
                    turn[key] = mind.blobs.get_json(blob)
                except Exception:          # a missing blob is reportable
                    turn[key] = None
                    turn[f"{key}_error"] = "content not retrievable"
        return turn

    return {
        "role_claim_turn": role_claim_turn,
        "role_complete_turn": role_complete_turn,
        "role_abandon_turn": role_abandon_turn,
        "role_tool_invoke": role_tool_invoke,
        "role_enqueue_trigger": role_enqueue_trigger,
        "operator_message_role": operator_message_role,
        "role_mailbox": role_mailbox,
        "role_turns": role_turns,
        "role_turn": role_turn,
    }
