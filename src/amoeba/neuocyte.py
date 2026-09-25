"""Disposable, Ego-derived neuocytes.

Lifecycle, in the order the design requires:

1. The supervisor publishes (or reuses) a consistent Ego snapshot taken at an
   inference boundary.
2. The neuocyte takes a reference on that snapshot so the published storage
   cannot be reclaimed while it is reading from it.
3. It instantiates a session from the snapshot -- by forking the physically
   shared prefix when the backend supports it, otherwise by recomputing the
   exact recorded token prefix.
4. It appends its own private instruction tail and lets that continuation
   evolve. The tail is private: the source Ego session and sibling neuocytes
   never see it.
5. It stays pinned to that snapshot and model generation for its whole life.
   It never reads a newer snapshot; a replacement neuocyte is forked from the
   newer one instead.
6. It commits findings as evidence/proposals through a durable receipt, with
   its fencing token, and retires.

Killing a neuocyte is always safe: its lease expires, the fencing token advances,
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
from .promptlib.model import fallback_output_ceiling
from .tools import parse_tool_calls, strip_tool_calls

# The closing format used to read "Reply with exactly three lines", stated
# after the tool protocol and contradicting it: one instruction says emit a
# tool call and nothing else, the other says reply in a fixed shape now. Live
# on 2026-09-24 the workers followed the second one every time -- six
# delegations, sandbox granted, `run_code` named in the objective, and not one
# tool call in the organism's entire history. They answered from context and
# reported that the context did not contain the answer, which was true and
# was not the task.
# The persistent roles that may delegate, and therefore the only lineages a
# worker can be born into. Not a list of who may call `admit_work`: it is the
# set of minds that *have* delegated cognition to inherit.
WORK_ORIGINS = ("ego", "id")

# And the kind of work each of them delegates. Maintenance is the inward
# mind's job: Ego noticing that something needs tending is a reason to ask
# Id, not a reason to do it itself with a worker of its own. Keeping the
# pairing here means there is no combination where a lineage runs work of a
# kind its own prompt does not describe.
WORK_OF_ROLE = {"ego": "user", "id": "maintenance"}

WORKER_INSTRUCTION = """You are a bounded neuocyte forked from the mind's Ego context.
You inherited the context above. Do one narrow task and stop.

Task: {objective}
{board}{tools}
If answering needs an action -- running code, reading a file, looking
something up -- call the tool for it first. The context will not contain a
result nobody has produced yet, and "the state does not say" is not an answer
to a task that asked you to find out.

When you have what you need, and only then, close with exactly three lines:
FINDING: <one sentence, the thing you actually determined>
CONFIDENCE: <a number between 0 and 1>
EVIDENCE: <what supports it -- a tool result you obtained, something in the
context above, or "none">"""

TOOLS_BLOCK = """
{prompt_block}

A tool call is a request, not an action: the Harness validates it, decides
whether you may make it, runs it, and returns the result to you. Call a tool
when you need its result; otherwise answer from what you already have.
"""

MESSAGE_BLOCK = """A message arrived about this task after it was assigned:
{messages}

This is additional context, not a replacement for your task. Your
original objective stands; weigh this alongside it.
"""

TOOL_RESULT_BLOCK = """<tool_result name="{name}">
{result}
</tool_result>
Continue. Use this result, or call another tool if you still need one."""

BOARD_BLOCK = """
What other neuocytes have already posted about this:
{posts}
"""

# A post's fate is part of what it means. The read receipt already froze this
# as "what was shown", while the rendering left it out -- so a worker weighed a
# finding whose author was fenced or whose work was cancelled exactly as it
# weighed a corroborated one, and the record said it had been told.
POST_FATE = "      ({fate}{note})"

SILENT_BLOCK = """
{count} earlier attempt(s) at this work posted nothing at all.
"""

ATTEMPT_BLOCK = """
This is attempt {attempt} of this work item.{previous}
"""

NO_BOARD_BLOCK = """
You have deliberately NOT been shown what other neuocytes found. Answer from the
context and the task alone, so that agreement with another neuocyte means
something.
"""

MAINTENANCE_INSTRUCTION = """You are a bounded maintenance neuocyte for an amoeba.
You were NOT given Ego's private context: you get a narrow task and references to
durable state. Do the task and stop.

Task: {objective}

Relevant durable state:
{state}

Reply with exactly three lines:
FINDING: <one sentence>
CONFIDENCE: <a number between 0 and 1>
EVIDENCE: <which state references support it, or "none">"""


def worker_lineage(item: dict[str, Any]) -> str:
    """Whose delegated cognition a work item instantiates.

    Derived from the persistent role that originated the work, and from
    nothing a caller supplied. `work_class` and `specialisation` are both
    model-authored, so neither may name a lineage: a specialisation can only
    ever narrow *within* one, because the lineage is always its prefix.

    Work admitted by anything that is not a persistent role has no lineage to
    inherit. Admission refuses it (see `WORK_ORIGINS`), so reaching here with
    one means the row predates that rule; it binds to the outward lineage,
    which is the narrower of the two -- an `ego.neuocyte` cannot audit Id's
    conclusions, and inheriting Id's cognition by accident is the failure
    this function exists to stop.
    """
    origin = (item.get("origin_actor") or "").strip()
    return origin if origin in WORK_ORIGINS else "ego"


class Neuocyte:
    # A class-level default so the generation path never depends on __init__
    # having run. None means no profile was bound and the built-in instruction
    # plus the engine's own sampling defaults are in use.
    profile: dict[str, Any] | None = None

    def __init__(self, cfg: Config, *, neuocyte_id: str) -> None:
        self.cfg = cfg
        self.neuocyte_id = neuocyte_id
        self.log = get_logger("neuocyte")
        # The neuocyte scope is the narrowest table there is: work lifecycle,
        # snapshots, the board, and tool_invoke. It contains no Id effector and
        # no operator verb, so there is nothing here to misuse -- not because a
        # check refuses, but because those verbs do not exist on this
        # connection.
        self.token = read_or_create_token(cfg.token_path)
        self.scope_token = read_or_create_token(cfg.scope_token_path("neuocyte"))
        self.sup = RpcClient(cfg.supervisor_host, cfg.supervisor_port,
                             self.scope_token, name=f"{neuocyte_id}->supervisor")
        self.inf = RpcClient(cfg.supervisor_host, cfg.inference_port, self.token,
                             name=f"{neuocyte_id}->inference")
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

        item = self.sup.call("lease_work", neuocyte_id=self.neuocyte_id, work_id=work_id)
        if not item:
            self.log.info("%s: nothing to lease", self.neuocyte_id)
            return 0
        work_id = item["work_id"]
        fencing_token = item["fencing_token"]
        self.pinned_state_version = item.get("pinned_state_ver")
        deadline = item.get("deadline") or (time.time() + self.cfg.arbiter.neuocyte_wall_seconds)
        budget = item.get("budget_tokens") or self.cfg.arbiter.neuocyte_token_budget

        # An Ego-derived worker inherits its ancestors' text physically, in
        # the forked context; a maintenance neuocyte starts from nothing and
        # is injected the whole resolved profile. bind_profile is told which,
        # so the binding records what was injected rather than assuming.
        profile = self._bind_profile(item, work_id=work_id)
        self.sup.call("register_agent", agent_id=self.neuocyte_id, role="neuocyte",
                      pid=os.getpid(), work_id=work_id,
                      model_generation=self.model_generation,
                      prompt_sha256=(profile or {}).get("prompt_sha256"),
                      profile_binding_id=(profile or {}).get("binding_id"))
        try:
            result = self._execute(item, caps=caps, budget=budget, deadline=deadline)
            self.sup.call("complete_work", work_id=work_id, neuocyte_id=self.neuocyte_id,
                          fencing_token=fencing_token, result=result,
                          pinned_state_ver=self.pinned_state_version)
            self.log.info("%s completed %s", self.neuocyte_id, work_id)
            return 0
        except Fenced as exc:
            self.log.warning("%s fenced on %s: %s", self.neuocyte_id, work_id, exc.message)
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
            self.sup.call("fail_work", work_id=work_id, neuocyte_id=self.neuocyte_id,
                          fencing_token=fencing_token, failure=message[:2000])
        except Exception:  # noqa: BLE001
            self.log.exception("could not report failure for %s", work_id)

    # ------------------------------------------------------------------
    def _bind_profile(self, item: dict[str, Any], *, work_id: str
                      ) -> dict[str, Any] | None:
        """Resolve this neuocyte's cognitive profile for this work item.

        Neuocytes had no profile at all before the Prompt Library: their
        instructions were module constants, so there was no way to specialise
        a worker, and no record of what any of them had been told. They now
        descend from the same family tree as the mind that spawned them --
        ``ego.neuocyte`` for work forked from Ego, ``id.neuocyte`` for
        maintenance -- which is what makes a specialisation like
        ``ego.neuocyte.research`` expressible at all.

        A missing profile is survivable: the built-in instruction still works,
        and refusing to do the work because the library was incomplete would
        be a worse failure than doing it on the baseline.
        """
        # Two separate questions, which used to be answered by one field.
        #
        #   work class    -- what KIND of work this is. Decides the execution
        #                    shape: maintenance runs from durable state,
        #                    user work forks a published context.
        #   worker lineage -- WHOSE delegated cognition is doing it. Decides
        #                    the profile, and belongs to the Harness.
        #
        # `work_class` decided both, and it is a model-authored argument to
        # `ego_request_work`. So Ego asking for maintenance-shaped work was
        # handed an `id.neuocyte` -- Id's cognition, instantiated by Ego,
        # with no Id involvement anywhere in the record. Observed live on
        # 2026-09-24. Lineage now comes from `origin_actor`, which the
        # Harness writes at admission and no caller can supply.
        lineage = worker_lineage(item)
        maintenance = item.get("work_class") == "maintenance"
        base = f"{lineage}.neuocyte"
        # Kept so the built-in path still has the right ceiling: a worker
        # the library could not bind is the same kind of worker.
        self.profile_namespace = base
        wanted = (item.get("specialisation") or "").strip()
        # The specialisation first, the base as the fallback. Asking for a
        # specialist that the library does not have, or has not approved, must
        # not fail the work: refusing a job because a prompt was missing is a
        # worse failure than doing it on the baseline.
        candidates = [f"{base}.{wanted}", base] if wanted else [base]

        bound = None
        fallback = None
        for namespace in candidates:
            try:
                bound = self.sup.call(
                    "bind_profile", namespace=namespace,
                    actor_id=self.neuocyte_id, actor_kind="neuocyte",
                    work_id=work_id, model_generation=self.model_generation,
                    # Only the Ego-derived path inherits a primed context.
                    # What is physically inherited follows the lineage too.
                    # A worker never inherits another role's prefix.
                    inherited_namespace=None if maintenance else lineage)
                break
            except Exception as exc:  # noqa: BLE001
                if namespace != candidates[-1]:
                    # Recorded rather than swallowed: a specialisation that has
                    # quietly stopped applying looks exactly like one that was
                    # never asked for, unless somebody says which happened.
                    fallback = f"{namespace} unavailable: {exc}"[:300]
                    self.log.warning("no approved profile for %s (%s); "
                                     "falling back to %s", namespace, exc, base)
                    continue
                self.log.warning("no prompt profile for %s (%s); using the "
                                 "built-in instruction", namespace, exc)
                self.profile = None
                self._report_profile(work_id, None,
                                     f"{namespace} unavailable: {exc}"[:300])
                return None
        self.profile = bound
        self._report_profile(work_id, bound, fallback)
        self.log.info("%s bound to %s", self.neuocyte_id, bound["profile_ref"])
        return bound

    def _report_profile(self, work_id: str, bound: dict | None,
                        fallback: str | None) -> None:
        """Tell the Harness which profile was actually used.

        Only when something other than the plain request happened. A work item
        that got what it asked for needs no annotation; one that silently got
        something else is the case this exists for.
        """
        if not fallback:
            return
        try:
            self.sup.call("record_profile_fallback", work_id=work_id,
                          neuocyte_id=self.neuocyte_id,
                          profile_ref=(bound or {}).get("profile_ref"),
                          reason=fallback)
        except Exception:  # noqa: BLE001
            self.log.debug("could not record the profile fallback",
                           exc_info=True)

    def _profile_block(self) -> str:
        """The profile text this neuocyte injects, as a prompt prefix."""
        if not self.profile:
            return ""
        text = (self.profile.get("inject_text") or "").strip()
        return text + "\n\n" if text else ""

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
        snapshot = self.sup.call("acquire_snapshot", holder=self.neuocyte_id,
                                 snapshot_id=item.get("snapshot_id"))
        self.snapshot_id = snapshot["snapshot_id"]
        self.ref_id = snapshot["ref_id"]

        if snapshot["model_generation"] != self.model_generation:
            # Cached tensors from another weight/tokenizer/positional
            # configuration are never reinterpreted.
            self.sup.call("release_snapshot_ref", ref_id=self.ref_id,
                          actor=self.neuocyte_id)
            self.ref_id = None
            raise CapabilityUnsupported(
                "snapshot belongs to a different model generation; refusing reuse",
                snapshot_generation=snapshot["model_generation"],
                current_generation=self.model_generation,
            )

        instantiation = self._instantiate_from_snapshot(snapshot, caps=caps)
        board_block, board_seen = self._board_context(item)
        tools_block, tool_names = self._tools_block(item)
        prompt = self._profile_block() + WORKER_INSTRUCTION.format(
            objective=item["objective"],
            board=self._attempt_context(item) + board_block,
            tools=tools_block)
        self.inf.call("ingest_messages", session_id=self.session_id,
                      messages=[{"role": "user", "content": prompt}],
                      add_assistant=True)
        out, tool_trace = self._generate_with_tools(
            item, budget=budget, deadline=deadline,
            max_turns=self.cfg.arbiter.max_tool_turns)
        parsed = _parse_finding(out["text"])
        post_id = self._publish_finding(item, parsed, out)
        return {
            "kind": "ego_derived_finding",
            "objective": item["objective"],
            "board_access": item.get("board_access", "read_write"),
            "board_posts_seen": board_seen,
            "board_post_id": post_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_version": snapshot["version"],
            "model_generation": self.model_generation,
            "pinned_state_version": self.pinned_state_version,
            "instantiation": instantiation,
            "finding": parsed["finding"],
            "confidence": parsed["confidence"],
            "evidence_note": parsed["evidence"],
            "raw_text": out["text"],
            "tools_offered": tool_names,
            "tool_calls": tool_trace,
            "tool_call_count": len(tool_trace),
            "stop_reason": out["stop_reason"],
            "work_messages_seen": out.get("messages_seen", []),
            "influenced_by_messages": bool(out.get("messages_seen")),
            "completion_tokens": out["tokens_spent"],
            "is_simulated": out.get("is_simulated", False),
            "neuocyte_id": self.neuocyte_id,
        }

    # -- the tool execution loop ----------------------------------------
    def _generate_with_tools(self, item: dict[str, Any], *, budget: int,
                             deadline: float, max_turns: int
                             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Generate, execute any requested tool call, resume, repeat.

        The loop lives here, but the *authority* does not. This process parses
        a request out of generated text and sends it to the Harness, which
        decides whether it is permitted and runs it. Nothing is executed in
        this process, so a neuocyte cannot reach a capability by deciding it
        may.

        Three independent bounds, because a model that keeps calling tools is
        an expected outcome rather than a bug: turns, the token budget, and the
        wall-clock deadline. Whichever binds first ends the loop, and the reason
        is recorded rather than silently swallowed.
        """
        trace: list[dict[str, Any]] = []
        messages_seen: list[dict[str, Any]] = []
        spent = 0
        out: dict[str, Any] = {}
        stop_reason = "answered"

        for turn in range(max_turns):
            # Collect any message addressed to this work item, at a turn
            # boundary. Nothing is pushed into this process and nothing reaches
            # the sandbox: a message is durable state the Harness hands over
            # when it is safe to read it.
            messages_seen.extend(self._collect_work_messages(item))
            remaining = budget - spent
            if remaining <= 0:
                stop_reason = "token_budget_exhausted"
                break
            if time.time() >= deadline:
                stop_reason = "deadline_reached"
                break

            # The profile's ceiling and sampling, narrowed by what is actually
            # left of the budget. The Arbiter's remaining-token figure always
            # wins: a profile cannot spend more than it was granted.
            settings = (self.profile or {}).get("backend_arguments") or {}
            out = self.inf.call(
                "generate", session_id=self.session_id,
                max_tokens=min(remaining, int(settings.get(
                    "max_tokens", fallback_output_ceiling(
                        getattr(self, "profile_namespace", "ego.neuocyte"))))),
                temperature=float(settings.get("temperature", 0.0)),
                deadline=deadline)
            spent += int(out.get("completion_tokens") or 0)

            requests = parse_tool_calls(out["text"], limit=1)
            if not requests:
                stop_reason = "answered"
                break
            if turn == max_turns - 1:
                # Asked for a tool on the last turn it had: do not execute
                # something whose result it will never get to use.
                stop_reason = "turn_limit_reached"
                trace.append({"turn": turn, "tool": requests[0].name,
                              "executed": False,
                              "reason": "turn limit reached before execution"})
                break

            req = requests[0]
            try:
                res = self.sup.call(
                    "tool_invoke", neuocyte_id=self.neuocyte_id,
                    work_id=item["work_id"], fencing_token=item["fencing_token"],
                    name=req.name, arguments=req.arguments, turn=turn)
            except Fenced:
                raise
            except Exception as exc:  # noqa: BLE001
                # A transport or handler failure is reported back to the model
                # as a failed tool call rather than killing the neuocyte: the
                # model may well be able to proceed without it.
                res = {"name": req.name, "accepted": False,
                       "reason": f"{type(exc).__name__}: {exc}", "result": None,
                       "error": None}

            trace.append({"turn": turn, "tool": req.name, "executed": True,
                          "accepted": res.get("accepted"),
                          "reason": res.get("reason"),
                          "error": res.get("error"),
                          "receipt_id": res.get("receipt_id"),
                          "duration_seconds": res.get("duration_seconds")})
            self._feed_tool_result(req.name, res)
        else:
            stop_reason = "turn_limit_reached"

        return ({**out, "stop_reason": stop_reason, "tokens_spent": spent,
                 "messages_seen": messages_seen}, trace)

    def _collect_work_messages(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Pick up messages sent to this work item since the last turn.

        A message is an addition, never a replacement: the original objective
        stays exactly as admitted, and the model is told plainly that this
        arrived afterwards so it can weigh it as a later clarification rather
        than as part of the brief.
        """
        try:
            res = self.sup.call("work_messages", work_id=item["work_id"],
                                neuocyte_id=self.neuocyte_id,
                                fencing_token=item["fencing_token"])
        except Exception:  # noqa: BLE001 - messages are optional, work is not
            self.log.debug("work message collection failed", exc_info=True)
            return []
        messages = res.get("messages") or []
        if not messages:
            return []
        rendered = "\n".join(
            f"- ({m['kind']} from {m['from_role']}) {m['body']}"
            for m in messages)
        self.inf.call(
            "ingest_messages", session_id=self.session_id,
            messages=[{"role": "user", "content": MESSAGE_BLOCK.format(
                messages=rendered)}], add_assistant=True)
        self.log.info("%s collected %d work message(s)", self.neuocyte_id,
                      len(messages))
        return [{"message_id": m["message_id"], "from_role": m["from_role"],
                 "kind": m["kind"]} for m in messages]

    def _feed_tool_result(self, name: str, res: dict[str, Any]) -> None:
        """Append the outcome to the session so the model can use it.

        A rejection is fed back just as faithfully as a result. The model is
        told it was refused and why, because hiding the refusal would leave it
        guessing at why its request vanished.
        """
        if res.get("accepted") and not res.get("error"):
            # Bounded by the Harness, which can store what does not fit and so
            # can name a digest that exists. Cutting it again here would undo
            # the notice that says it was cut.
            body = res.get("result_text")
            if body is None:
                body = json.dumps(res.get("result"), default=str)
        elif res.get("accepted"):
            body = f"the tool ran but failed: {res.get('error')}"
        else:
            body = f"refused: {res.get('reason')}"
        self.inf.call(
            "ingest_messages", session_id=self.session_id,
            messages=[{"role": "user",
                       "content": TOOL_RESULT_BLOCK.format(name=name, result=body)}],
            add_assistant=True)

    def _tools_block(self, item: dict[str, Any]) -> tuple[str, list[str]]:
        """Describe the tools this work item actually permits.

        Asked of the Harness rather than assembled here, so the prompt cannot
        advertise a capability the work row does not carry.
        """
        try:
            res = self.sup.call("tool_schemas", work_id=item["work_id"],
                                role="neuocyte")
        except Exception:  # noqa: BLE001
            self.log.debug("tool schema fetch failed", exc_info=True)
            return "", []
        block = res.get("prompt_block") or ""
        names = [t["name"] for t in res.get("tools", [])]
        return (TOOLS_BLOCK.format(prompt_block=block) if block else ""), names

    def _board_context(self, item: dict[str, Any]) -> tuple[str, list[str]]:
        """Show the neuocyte the board only if its work item permits it.

        A work item admitted with ``board_access="none"`` produces a
        board-naive neuocyte on purpose. That is what makes later agreement
        between two neuocytes evidence of independent replication rather than one
        of them having read the other. Reading is recorded against this neuocyte
        the moment it happens.
        """
        access = item.get("board_access", "read_write")
        if access == "none":
            return NO_BOARD_BLOCK, []
        try:
            res = self.sup.call("board_read", reader=self.neuocyte_id,
                                work_id=item["work_id"], limit=6,
                                post_types=["finding", "hypothesis", "challenge"])
        except Exception:  # noqa: BLE001
            self.log.debug("board read failed", exc_info=True)
            return NO_BOARD_BLOCK, []
        posts = res.get("posts", [])
        # A report, not a number: {count, attempts, truncated, ...}
        silent = int((res.get("silent_attempts") or {}).get("count") or 0)
        if not posts:
            # Silence is itself information: attempts that ran and posted
            # nothing were reported to nobody, so a worker could not tell "no
            # one has looked at this" from "three have and found nothing".
            if silent:
                return NO_BOARD_BLOCK + SILENT_BLOCK.format(count=silent), []
            return NO_BOARD_BLOCK, []
        lines = []
        for p in posts:
            lines.append(
                f"- [{p['post_id']}] ({p['post_type']}, {p['author']}) "
                f"{p['body'][:180]}")
            fate, note = p.get("attempt_fate"), p.get("work_note")
            if fate or note:
                lines.append(POST_FATE.format(
                    fate=fate or "fate unrecorded",
                    note=f": {note}" if note else ""))
        rendered = "\n".join(lines)
        block = BOARD_BLOCK.format(posts=rendered)
        if silent:
            block += SILENT_BLOCK.format(count=silent)
        return block, [p["post_id"] for p in posts]

    def _attempt_context(self, item: dict[str, Any]) -> str:
        """What this work has already cost, from the row rather than a guess.

        Durable and previously unsaid: a worker retrying after two failures was
        told neither that it was a retry nor what went wrong last time, and
        then asked to do better.
        """
        attempt = int(item.get("attempt") or 1)
        if attempt <= 1:
            return ""
        failure = (item.get("failure") or "").strip()
        previous = (f" The previous attempt ended: {failure[:300]}"
                    if failure else " No reason was recorded for the last one.")
        return ATTEMPT_BLOCK.format(attempt=attempt, previous=previous)

    def _publish_finding(self, item: dict[str, Any], parsed: dict[str, Any],
                         out: dict[str, Any]) -> str | None:
        """Publish to the board if permitted. The board is communication; the
        durable finding is committed separately through complete_work."""
        if item.get("board_access") != "read_write":
            return None
        try:
            # Output that did not have the shape of a finding is posted as
            # what it is. A note is still evidence and still readable; what it
            # is not is a claim the worker never made.
            res = self.sup.call(
                "board_post", author=self.neuocyte_id, author_kind="neuocyte",
                post_type="finding" if parsed.get("parsed") else "note",
                body=parsed["finding"],
                title=item["objective"][:120], work_id=item["work_id"],
                confidence=parsed["confidence"],
                model_generation=self.model_generation,
                snapshot_id=self.snapshot_id,
                evidence=[{"note": parsed["evidence"]}] if parsed.get("evidence") else [])
            return res.get("post_id")
        except Exception:  # noqa: BLE001
            self.log.debug("board post failed", exc_info=True)
            return None

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
                    prefix_len=snapshot["token_count"], role="neuocyte",
                    snapshot_id=snapshot["snapshot_id"],
                    context_budget_tokens=self.cfg.arbiter.ego_neuocyte_budget_tokens,
                    budget_basis=self.cfg.arbiter.ego_neuocyte_budget_basis,
                )
                self.session_id = sess["session_id"]
                return {"method": "forked_shared_prefix", "kv_mode": sess["kv_mode"],
                        "prefix_len": sess["prefix_len"],
                        "note": "KV cells physically shared with the source sequence"}
            except Exception as exc:  # noqa: BLE001
                self.log.warning("fork failed (%s); falling back to recomputation", exc)

        # The fork failed and the prefix will be recomputed. The allowance is
        # unchanged -- this is the same worker doing the same job, and a
        # performance fallback must not decide what it may think. The
        # recomputed prefix is charged in full physically, which is a
        # different question answered from backend state.
        sess = self.inf.call(
            "open_session", role="neuocyte",
            context_budget_tokens=self.cfg.arbiter.ego_neuocyte_budget_tokens,
            budget_basis=self.cfg.arbiter.ego_neuocyte_budget_basis)
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
        """Id maintenance neuocytes get references to state, never a snapshot of
        Id's private context."""
        state = self.sup.call("maintenance_context", objective=item["objective"])
        # Cold start: this worker inherits nothing, so its whole context is
        # its own and a total ceiling says what it means.
        sess = self.inf.call(
            "open_session", role="neuocyte",
            context_budget_tokens=self.cfg.arbiter.id_neuocyte_budget_tokens,
            budget_basis=self.cfg.arbiter.id_neuocyte_budget_basis)
        self.session_id = sess["session_id"]
        prompt = self._profile_block() + MAINTENANCE_INSTRUCTION.format(
            objective=item["objective"],
            state=json.dumps(state, indent=2, default=str)[:2500],
        )
        self.inf.call("ingest_messages", session_id=self.session_id,
                      messages=[{"role": "user", "content": prompt}],
                      add_assistant=True)
        out, tool_trace = self._generate_with_tools(
            item, budget=budget, deadline=deadline,
            max_turns=self.cfg.arbiter.max_tool_turns)
        parsed = _parse_finding(out["text"])
        return {
            "kind": "maintenance_finding",
            "tool_calls": tool_trace,
            "tool_call_count": len(tool_trace),
            "stop_reason": out["stop_reason"],
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
            "completion_tokens": out["tokens_spent"],
            "is_simulated": out.get("is_simulated", False),
            "neuocyte_id": self.neuocyte_id,
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
                              actor=self.neuocyte_id)
        except Exception:  # noqa: BLE001
            self.log.debug("snapshot ref release failed", exc_info=True)
        try:
            self.sup.call("retire_agent", agent_id=self.neuocyte_id, reason="task complete")
        except Exception:  # noqa: BLE001
            pass
        self.sup.close()
        self.inf.close()


def _parse_finding(text: str) -> dict[str, Any]:
    """What the worker said, and how much of it had the shape we asked for.

    Two things this must not do, both found by audit on 2026-09-24.

    It must not invent a confidence. Output with no `FINDING:` line was
    published as a finding at 0.5 -- so a model declining to answer, or
    answering in prose, became a half-confident claim on the blackboard with
    a number nobody stated. `confidence` is `None` when it was not given, and
    `parsed` says whether this had the shape of a finding at all.

    And it must not crash on a field that is present but empty.
    `CONFIDENCE:` with nothing after it indexed the first whitespace-split
    element before entering the `try`, which caught only `ValueError` anyway
    -- so a truncated but usable result raised `IndexError`, became a worker
    failure, and spent a retry.
    """
    clean = strip_tool_calls(text)
    finding, evidence = "", ""
    confidence: float | None = None
    for line in clean.splitlines():
        upper = line.upper()
        if upper.startswith("FINDING:"):
            finding = line.split(":", 1)[1].strip()
        elif upper.startswith("CONFIDENCE:"):
            words = line.split(":", 1)[1].strip().split()
            if words:
                try:
                    confidence = max(0.0, min(1.0, float(words[0].rstrip(".,"))))
                except ValueError:
                    confidence = None
        elif upper.startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()
    parsed = bool(finding)
    if not parsed:
        finding = clean.strip()[:400] or "(neuocyte produced no parsable finding)"
    return {"finding": finding, "confidence": confidence, "evidence": evidence,
            "parsed": parsed}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="amoeba.neuocyte")
    ap.add_argument("--config", default=os.environ.get("AMOEBA_CONFIG"))
    ap.add_argument("--neuocyte-id", default=None)
    ap.add_argument("--work-id", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    setup_logging(cfg, "neuocyte")
    neuocyte_id = args.neuocyte_id or new_id("nc")
    return Neuocyte(cfg, neuocyte_id=neuocyte_id).run(work_id=args.work_id)


if __name__ == "__main__":
    raise SystemExit(main())
