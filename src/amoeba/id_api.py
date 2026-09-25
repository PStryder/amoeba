"""Id's effectors: the authorised ways the inward half can act.

Id may **request, propose, challenge, investigate and escalate**. It does not
mutate authoritative state because it concluded something. Every verb here
follows the same shape:

    sense -> reason -> request/propose/escalate -> Harness validates, executes,
    receipts

The asymmetry is deliberate. Detecting a problem is exactly the situation in
which a mind is most likely to be wrong in an interesting way, so the moment
Id is most confident is the moment its conclusion should pass through
something that can refuse.

These verbs exist in Id's scope and in no other. Not because a check rejects
other callers -- because they are absent from every other method table (see
``scopes.py``). A neuocyte cannot name one, list one, or reach one.

**Attribution.** Every consequential act is receipted through the existing
writer, with `actor="id"`, and carries the `pulse_id` Id was looking at when it
decided, where one applies. That is what makes it possible to reconstruct not
just what Id did but what it was seeing when it decided to.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

from .errors import InvalidInput, NotFound
from .ids import new_id
from .store.events import EventKind
from .store.writer import Mutation

SEVERITIES = ("notice", "concern", "urgent")
TARGET_ROLES = ("ego", "id")

if TYPE_CHECKING:
    from .supervisor import Supervisor

FINDING_KINDS = ("anomaly", "contradiction", "risk", "observation")
"""Id's own taxonomy for what it raises. Not memory kinds: a finding is
an interpretation the organism now has on record as Id's, and conflating
the two would let the monitor author beliefs."""

MAX_TEXT = 4000


def _text(value: str, field: str, *, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidInput(f"{field} must be a non-empty string")
    return value.strip()[:limit]


def build(sup: "Supervisor") -> dict[str, Any]:  # noqa: C901
    mind = sup.mind
    assert mind is not None

    def _provenance(pulse_id: str | None, reason: str) -> dict[str, Any]:
        """Bind an action to the telemetry that prompted it.

        Cheap, and it is the difference between "Id cancelled some work" and
        "Id cancelled some work while looking at this state".
        """
        if not pulse_id:
            return {"pulse_id": None,
                    "note": "no pulse cited; this action is unanchored telemetry-wise"}
        cached = sup.pulse._cited.get(pulse_id)
        if cached is None:
            current = sup.pulse.capture(max_age_seconds=60.0)
            cached = sup.pulse.cite(
                pulse_id, state_version=current.get("state_version", -1),
                captured_at=current.get("captured_at", time.time()),
                by="id", reason=reason)
        return cached

    # ==================================================================
    # Provenance
    # ==================================================================
    def id_cite_pulse(*, pulse_id: str, state_version: int, captured_at: float,
                      reason: str) -> dict[str, Any]:
        """Record that a pulse was used as reasoning input.

        Telemetry is not an event stream -- recording every pulse would bury
        the log in observations nobody reads. Recording the ones actually cited
        keeps the property that matters: a consequential conclusion can be
        traced to the state its author was looking at.
        """
        return sup.pulse.cite(pulse_id, state_version=int(state_version),
                              captured_at=float(captured_at), by="id",
                              reason=_text(reason, "reason", limit=500))

    # ==================================================================
    # Raising something
    # ==================================================================
    def id_raise_finding(*, claim: str, kind: str = "anomaly",
                         confidence: float = 0.5,
                         evidence: Sequence[dict[str, Any]] = (),
                         pulse_id: str | None = None,
                         operation_id: str | None = None) -> dict[str, Any]:
        """Put a durable finding on the record.

        A finding is Id saying something, not the organism believing it. It
        lands as a maintained-memory item authored by Id with its evidence
        attached, so a later reader can see who claimed it and on what basis --
        and can disagree with it through the normal path.
        """
        if kind not in FINDING_KINDS:
            raise InvalidInput("unknown finding kind", kind=kind,
                               allowed=list(FINDING_KINDS))
        prov = _provenance(pulse_id, f"finding: {claim[:80]}")
        # Deliberately stored as an *interpretation*, not a belief. Id noticing
        # something is Id's reading of the state it was looking at; promoting
        # that straight to a belief would let the monitor quietly author the
        # organism's convictions. The finding's own taxonomy is kept on the
        # event, so findings stay queryable without bending memory kinds.
        memory_id, receipt = mind.memory.remember(
            kind="interpretation", claim=_text(claim, "claim"),
            confidence=max(0.0, min(1.0, float(confidence))),
            created_by="id",
            supporting=[*evidence,
                        {"note": f"Id finding ({kind}) from pulse "
                                 f"{prov.get('pulse_id')}"}],
            operation_id=operation_id)

        def record(m: Mutation) -> None:
            m.emit(EventKind.ID_FINDING_RAISED, {
                "memory_id": memory_id, "finding_kind": kind,
                "confidence": float(confidence), "raised_by": "id",
                "pulse_id": prov.get("pulse_id"),
                "note": "an interpretation by Id, not an organism belief"})

        mind.writer.apply(record, actor="id", operation_id=operation_id,
                          bump_version=False)
        return {"memory_id": memory_id, "finding_kind": kind,
                "memory_kind": "interpretation",
                "receipt_id": receipt.receipt_id, "provenance": prov,
                "note": "recorded as an Id interpretation; a claim, not a verdict"}

    def id_propose_memory_correction(*, memory_id: str, claim: str,
                                     confidence: float = 0.5,
                                     rationale: str = "",
                                     evidence: Sequence[dict[str, Any]] = (),
                                     pulse_id: str | None = None,
                                     operation_id: str | None = None
                                     ) -> dict[str, Any]:
        """Supersede a maintained belief, preserving what it replaced.

        Correction is supersession (I2): the earlier item keeps its identity
        and its evidence, and the new one records what it replaced. Id cannot
        edit a belief in place, so being wrong about a correction costs a
        traceable extra item rather than the original.
        """
        existing = mind.memory.get_memory(memory_id)
        prov = _provenance(pulse_id, f"correction of {memory_id}")
        new_memory, receipt = mind.memory.remember(
            kind=existing["kind"], claim=_text(claim, "claim"),
            confidence=max(0.0, min(1.0, float(confidence))),
            created_by="id", supersedes=memory_id,
            supporting=[*evidence,
                        {"note": _text(rationale or "correction proposed by Id",
                                       "rationale", limit=1000)}],
            operation_id=operation_id)
        return {"memory_id": new_memory, "supersedes": memory_id,
                "receipt_id": receipt.receipt_id, "provenance": prov}

    # ==================================================================
    # Asking for work to be done
    # ==================================================================
    def id_request_investigation(*, objective: str, replicas: int = 1,
                                 independent: bool = False,
                                 sandbox: bool = False,
                                 pulse_id: str | None = None,
                                 operation_id: str | None = None
                                 ) -> dict[str, Any]:
        """Ask for bounded investigation by Id neuocytes.

        ``independent=True`` admits each replica board-naive, so no replica can
        read another's post and echo it, and it is why the replicas are
        admitted as separate work items rather than one item doing the work
        twice. It isolates influence, which is not the same as producing
        evidence: replicas that go and observe different things corroborate
        each other, and replicas that only reason from the same inherited
        context do not, however many of them agree (I138).

        The Arbiter still decides. Id requests; it does not schedule.
        """
        if not 1 <= int(replicas) <= 4:
            raise InvalidInput("replicas must be between 1 and 4",
                               replicas=replicas)
        prov = _provenance(pulse_id, f"investigation: {objective[:80]}")
        admitted, refused = [], []
        for i in range(int(replicas)):
            out = sup.methods()["admit_work"](
                objective=_text(objective, "objective", limit=2000),
                work_class="maintenance", origin_actor="id",
                operation_id=operation_id,
                board_access="none" if independent else "read_write",
                sandbox_allowed=bool(sandbox))
            (admitted if out.get("admitted") else refused).append(out)
        return {"requested": int(replicas), "admitted": admitted,
                "refused": refused, "independent": bool(independent),
                "provenance": prov,
                # What the flag bought, stated as what it is. It used to say
                # agreement between board-naive replicas was independent
                # replication, which is the claim I138 exists to refuse: they
                # inherit the same context, so agreement can be one observation
                # restated. Only a distinct acquisition makes it corroboration,
                # and `board_corroboration` is where that is answered.
                "note": ("admitted board-naive, so no replica can echo "
                         "another's post; whether their agreement is "
                         "corroboration depends on what each one went and "
                         "observed, not on how many agree"
                         if independent else
                         "replicas can read the board; agreement may be social")}

    def id_request_work_intervention(*, work_id: str, action: str,
                                     reason: str, pulse_id: str | None = None
                                     ) -> dict[str, Any]:
        """Ask the Harness to intervene in a specific work item.

        ``cancel`` is the only intervention offered, and it is containment:
        it stops *future* work on the item without unwinding what already
        happened (I27). Id names an item and a reason; the Harness performs the
        transition and receipts it.

        Requeue is deliberately absent. Returning a leased item to the queue is
        lease expiry, which the supervision loop owns as its recovery path --
        a second actor forcing it would race the loop and bump the fencing
        token underneath a neuocyte that is still working. If an item needs
        redoing, asking for an investigation is the honest way to get it.

        Id cannot change scheduling *policy* at all: no verb here touches the
        Arbiter's limits.
        """
        if action != "cancel":
            raise InvalidInput(
                "the only supported intervention is cancel", action=action,
                allowed=["cancel"],
                hint="lease expiry is the supervisor's recovery path; use "
                     "id_request_investigation to have work redone")
        row = mind.work.get_work(work_id)
        if row is None:
            raise NotFound("no such work item", work_id=work_id)
        prov = _provenance(pulse_id, f"cancel {work_id}: {reason[:60]}")
        detail = _text(reason, "reason", limit=1000)
        out = sup.methods()["cancel_work"](
            work_id=work_id, actor="id", reason=f"[id] {detail}")
        return {"work_id": work_id, "action": "cancel", "reason": detail,
                "provenance": prov, **out}

    # ==================================================================
    # Homeostasis
    # ==================================================================
    def id_request_rejuvenation(*, target_role: str, reason: str, mode: str = "rebuild",
                                pulse_id: str | None = None,
                                operation_id: str | None = None) -> dict[str, Any]:
        """Request that a role's context be rejuvenated.

        Id never manipulates KV. It states which role and why; the Harness
        decides whether to act, applies its own rate limits, and performs the
        operation. A refusal is recorded as readily as an action.
        """
        if target_role not in TARGET_ROLES:
            raise InvalidInput("rejuvenation targets ego or id", target_role=target_role)
        prov = _provenance(pulse_id, f"rejuvenate {target_role}: {reason[:60]}")
        out = sup.methods()["request_rejuvenation"](
            role=target_role, reason=_text(reason, "reason", limit=1000),
            requested_by="id", mode=mode, operation_id=operation_id)
        return {**out, "provenance": prov}

    # ==================================================================
    # Proposing a change to cognition itself
    # ==================================================================
    def id_propose_prompt(*, target_role: str, prompt: str, rationale: str,
                          model_vars: dict[str, Any] | None = None,
                          evidence: Sequence[dict[str, Any]] = (),
                          pulse_id: str | None = None,
                          operation_id: str | None = None) -> dict[str, Any]:
        """Propose new doctrine for a constitutional root (``ego`` or ``id``).

        A real Prompt Library candidate: `ego@N` becomes a proposed `ego@N+1`
        that goes through validation, evaluation and Operator approval like
        any other version. Id cannot approve it, select it, or cascade it --
        those verbs are in no scope Id holds.

        This verb previously could not do that. While the library forbade
        every root version, not merely a new top-level namespace, it recorded
        Id's wording as a note and told the Operator to go and edit a file.
        The over-restriction is gone, so the verb is honest again.

        It is the role-shaped entry point; `id_propose_profile` is the general
        one and reaches any namespace, roots included. Both go through the same
        store, so there is one creation path, not two governance schemes.
        """
        if target_role not in TARGET_ROLES:
            raise InvalidInput("prompt proposals target ego or id", target_role=target_role)
        text = _text(prompt, "prompt", limit=20000)
        proposal_id = new_id("ppr")
        prov = _provenance(pulse_id, f"prompt proposal for {target_role}")
        from .promptlib.store import PromptStore
        from .resources import prompt_version

        store = PromptStore(mind)
        current = prompt_version(target_role, sup.cfg, mind)

        def body(m: Mutation) -> dict[str, Any]:
            # A root replaces rather than composes: it has no parent whose
            # text it could extend.
            created = store.create_runtime_version(
                m, namespace=target_role, prompt_mode="replace", prompt_text=text,
                model_vars=model_vars or {}, origin="id", created_by="id",
                rationale=_text(rationale, "rationale", limit=4000),
                state="candidate")
            # The legacy proposal event is still emitted, because the operator
            # console reads it and because a proposal is a distinct act from
            # the version write that carried it.
            m.emit(EventKind.PROMPT_PROPOSED, {
                "proposal_id": proposal_id, "role": target_role,
                "namespace": target_role, "version_id": created["version_id"],
                "local_version": created["local_version"],
                "candidate_sha256": created["local_sha256"],
                "current_sha256": current.sha256,
                "rationale": _text(rationale, "rationale", limit=4000),
                "evidence": list(evidence)[:20],
                "proposed_by": "id", "pulse_id": prov.get("pulse_id"),
                "note": ("a governed candidate; it changes nothing until the "
                         "Operator approves and selects it, and Id can do "
                         "neither")})
            return created

        receipt, created = mind.writer.apply(body, actor="id",
                                             operation_id=operation_id)
        return {"proposal_id": proposal_id, "role": target_role,
                "namespace": target_role, "version_id": created["version_id"],
                "local_version": created["local_version"],
                "profile_ref": str(store.ref_for(target_role, created["local_version"])),
                "candidate_sha256": created["local_sha256"],
                "current_sha256": current.sha256,
                "receipt_id": receipt.receipt_id, "provenance": prov,
                "status": "candidate",
                "note": ("the operator decides; Id cannot approve, select or "
                         "cascade a prompt version")}

    # ==================================================================
    # Talking to the rest of the organism
    # ==================================================================
    def id_message_ego(*, kind: str, message: str,
                       payload: dict[str, Any] | None = None,
                       pulse_id: str | None = None) -> dict[str, Any]:
        """Send Ego an explicit, provenance-bearing message.

        The side channel is transient and bounded by design (I22): this changes
        no state and Ego is free to ignore it. What it adds over the raw
        channel is attribution -- the message is stamped as Id's, with the
        pulse that prompted it, so Ego is never guessing where a nudge came
        from.
        """
        prov = _provenance(pulse_id, f"message to ego: {kind}")
        body = {"message": _text(message, "message", limit=2000),
                "from": "id", "pulse_id": prov.get("pulse_id"),
                **(payload or {})}
        out = sup.methods()["side_channel"](
            to_role="ego", kind=_text(kind, "kind", limit=64),
            payload=body, from_role="id")
        return {"delivered": True, "kind": kind, "provenance": prov,
                "transport": out,
                "note": "transient: this changed no durable state"}

    def id_escalate_to_operator(*, summary: str, severity: str = "notice",
                                evidence: Sequence[dict[str, Any]] = (),
                                recommended_action: str = "",
                                pulse_id: str | None = None,
                                operation_id: str | None = None) -> dict[str, Any]:
        """Escalate something Id is not authorised to remediate.

        The honest end of the effector set. When the right response is outside
        what Id may do -- or when Id is unsure enough that acting would be
        worse than asking -- this records the concern durably where the
        operator will find it, rather than leaving Id to either overstep or
        stay silent.
        """
        if severity not in SEVERITIES:
            raise InvalidInput("unknown severity", severity=severity,
                               allowed=list(SEVERITIES))
        escalation_id = new_id("esc")
        prov = _provenance(pulse_id, f"escalation: {summary[:80]}")

        def body(m: Mutation) -> None:
            m.emit(EventKind.OPERATOR_ESCALATION, {
                "escalation_id": escalation_id, "severity": severity,
                "summary": _text(summary, "summary"),
                "recommended_action": recommended_action[:2000],
                "evidence": list(evidence)[:20],
                "raised_by": "id", "pulse_id": prov.get("pulse_id"),
                "note": "Id is not authorised to remediate this autonomously"})

        receipt, _ = mind.writer.apply(body, actor="id", operation_id=operation_id)
        return {"escalation_id": escalation_id, "severity": severity,
                "receipt_id": receipt.receipt_id, "provenance": prov,
                "status": "raised"}

    return {
        "id_cite_pulse": id_cite_pulse,
        "id_raise_finding": id_raise_finding,
        "id_propose_memory_correction": id_propose_memory_correction,
        "id_request_investigation": id_request_investigation,
        "id_request_work_intervention": id_request_work_intervention,
        "id_request_rejuvenation": id_request_rejuvenation,
        "id_propose_prompt": id_propose_prompt,
        "id_message_ego": id_message_ego,
        "id_escalate_to_operator": id_escalate_to_operator,
    }
