"""Who may call what, declared in one place.

The capability boundary between Id and a neuocyte is **architectural absence**,
not a permission check. A verb outside a caller's table does not exist for that
caller: the dispatcher never finds it, the error does not name it, and there is
no shared implementation carrying an ``if caller != id`` that could be reasoned
around or refactored away.

The scope a connection gets is decided by the secret it presents, so there is
no role, caller, actor or work-class field anywhere in the handshake for a
caller to forge. Presenting the neuocyte secret *is* being a neuocyte.

Three rules this file is built on:

1. **Derived from call sites.** Each scope lists what that caller actually
   calls, traced from its source. A verb nobody calls is capability nobody
   asked for, and it is the ones nobody noticed granting that matter.
2. **Neuocyte is the narrowest.** It holds exactly the lifecycle, snapshot,
   board and tool verbs the neuocyte process uses -- and nothing that reads or
   changes the mind's own state.
3. **Id-only is genuinely id-only.** Id's effectors appear in no other table,
   including Ego's. Ego and Id share a base because they are both long-lived
   role processes, not because they should share authority.

Residual limit, stated rather than implied: these are files in the state
directory, so a process already running as the Amoeba account can read any of
them. This is the same boundary documented in ``security.py`` -- it removes
accidental and model-driven capability, not a determined same-account process.
The genuinely untrusted execution environment is the compute sandbox, and that
provably cannot reach the state directory or the network at all.
"""

from __future__ import annotations

# What the neuocyte process calls, traced from neuocyte.py. Nothing here reads
# or mutates maintained state, the blob store, or another work item.
NEUOCYTE = (
    "register_agent", "retire_agent", "heartbeat",
    # Birth: resolve the profile this neuocyte is about to think with, and
    # freeze what it actually received. Reading its own profile is not
    # governing the library -- none of the read, evaluate or approve verbs
    # are in this table.
    "bind_profile",
    "lease_work", "complete_work", "fail_work",
    "acquire_snapshot", "release_snapshot_ref", "snapshot_tokens",
    "maintenance_context",
    "board_read", "board_post",
    "tool_invoke", "tool_schemas",
    # Saying that the requested specialisation was not available. An
    # annotation on its own work item -- narrower than `fail_work`, which it
    # already holds -- and the only component that knows, because it is the
    # one that tried to bind.
    "record_profile_fallback",
    # Collected at a turn boundary, never pushed into the process.
    "work_messages",
)

# Shared by the two long-lived role processes, traced from roles.py. Read-heavy
# and deliberately free of anything that decides.
ROLE_BASE = (
    "register_agent", "retire_agent", "heartbeat",
    "bind_profile",
    # The turn's authoritative environment. Process plumbing rather than a
    # cognitive act, so it is deliberately not model-facing: the role asks for
    # it on the model's behalf and hands the answer over.
    "role_environment",
    # Claiming and closing a bounded turn. Also not model-facing: scheduling
    # your own next thought is not a cognitive act, and a mind that could
    # would be scheduling itself.
    "role_claim_turn", "role_complete_turn", "role_abandon_turn",
    # Every capability a role model invokes goes through here, carrying
    # the turn it belongs to. A turn that is no longer running cannot
    # act, which is what stops a hung turn waking up after its inputs
    # were handed to a replacement.
    "role_tool_invoke",
    #
    # `role_enqueue_trigger` is deliberately NOT here. It takes `source` as an
    # argument, so a role holding it could write into the other role's mailbox
    # attributed to anyone -- Ego could queue an "operator says approve this"
    # message into Id's cognition. That is the forgery the rest of the system
    # rules out by deriving identity from the presented credential, and Ego is
    # the component most exposed to a confident user.
    #
    # Roles reach each other through `ego_message_id` / `id_message_ego`,
    # which attribute the sender themselves. Every legitimate caller of
    # `role_enqueue_trigger` is the Harness, holding the control token.
    "status", "health", "capabilities",
    "recall", "get_memory", "history", "provenance", "audit_dossier",
    "get_conclusion", "get_work", "queue_stats",
    "board_read", "board_get_post", "board_thread", "board_stats",
    "artifact_list",
    "context_report",
)

# Ego's own surface. Two things are deliberately gone from an earlier draft of
# this table: `remember`, because authoring a belief directly is not a proposal
# and Ego is the component most exposed to a confident user; and
# `ensure_snapshot`, which is the supervisor's scheduling helper and was
# capability nobody asked for.
EGO = ROLE_BASE + (
    "record_conclusion",
    "publish_ego_snapshot", "list_snapshots",
    "board_post",
    # senses
    "ego_work_view", "ego_artifact_evidence", "ego_resource_identities",
    # effectors
    "ego_request_work", "ego_work_message", "ego_request_cancellation",
    "ego_propose_memory", "ego_message_id", "ego_request_id_review",
    # The external loop. Both are scoped to the turn Ego is running rather
    # than to an argument it supplies, so neither is a route to another
    # client's files or another client's results.
    "ego_read_attachment", "ego_surface_result",
)

EGO_ONLY = (
    "ego_work_view", "ego_artifact_evidence", "ego_resource_identities",
    "ego_request_work", "ego_work_message", "ego_request_cancellation",
    "ego_propose_memory", "ego_message_id", "ego_request_id_review",
    "ego_read_attachment", "ego_surface_result",
)

# Id's senses beyond the shared base, plus its effectors. Everything in
# ID_ONLY appears in no other scope.
ID_SENSES = (
    "system_pulse", "verify_integrity", "disagreements",
    "board_independence", "board_corroboration",
    "context_assess", "sandbox_capabilities", "file_roots",
    "id_health",
    # How much the organism is storing, and how much of that is forgettable.
    # A sense: `store_prune` is deliberately absent from every scope, because
    # a mind that could prune its own turn history could remove the record of
    # what it did in the same motion.
    "store_footprint",
)

# The Prompt Library, split at the authority line. Id may read the whole family
# tree, evaluate a version and propose a new one; approving, selecting and
# cascading are Operator verbs that appear in no scope table at all. "Id may
# not promote" is therefore a fact about the dispatcher, not a rule Id is asked
# to respect.
PROMPT_READ = (
    "prompt_tree", "prompt_versions", "prompt_resolve", "prompt_diff",
    "explain_profile", "prompt_incarnations",
)

ID_ONLY = (
    "id_evaluate_prompt",
    "id_propose_profile",
    "id_cite_pulse",
    "id_raise_finding",
    "id_request_investigation",
    "id_propose_memory_correction",
    "id_request_rejuvenation",
    "id_propose_prompt",
    "id_request_work_intervention",
    "id_escalate_to_operator",
    "id_message_ego",
)

ID = ROLE_BASE + ID_SENSES + PROMPT_READ + ID_ONLY


# The external I/O adapter's scope: MCP and the HTTP API both connect with
# this and nothing else. It is the smallest table in the system on purpose --
# every verb here is semantic input or output, and there is no verb it could
# name that admits work, cancels it, governs an artifact or a prompt, edits
# state, or reaches a role's internal effectors.
#
# The adapter is trusted to bind `client_id` from the authenticated credential.
# An external client never holds this token; it holds an API key the adapter
# maps to an identity, so "my interactions" is a fact about who asked rather
# than a parameter anyone can set.
EXTERNAL_IO = (
    "io_capabilities", "io_attach_input", "io_submit", "io_status",
    "io_await", "io_output", "io_list", "io_result",
)


# ---------------------------------------------------------------------------
# What the role MODEL is offered, as opposed to what the role PROCESS may call
# ---------------------------------------------------------------------------
# A scope table is the dispatch authority of a *connection*. Some of what it
# contains is plumbing the process performs on the model's behalf --
# registering, heartbeating, binding a profile -- and offering those to the
# model would be inviting it to operate its own life support.
#
# These are the verbs the environment manifest advertises and the role tool
# loop will execute. Two rules hold, and both are tested:
#
#   * every model-facing verb is in that role's scope table, so a role is
#     never told it has a capability its credential cannot invoke;
#   * every *deliberate* role capability -- everything in EGO_ONLY and
#     ID_ONLY -- is model-facing, so an effector cannot exist in dispatch
#     while being invisible to the mind it was built for.
#
# The descriptions and argument schemas are not written here. They are derived
# from the real functions in the supervisor's method table, because a
# hand-maintained second list is exactly how a manifest starts lying.

_SHARED_MODEL_FACING = (
    # reading the record
    "recall", "get_memory", "history", "provenance", "audit_dossier",
    "get_conclusion", "get_work", "queue_stats",
    # the blackboard
    "board_read", "board_get_post", "board_thread", "board_stats",
    # work product and self-measurement
    "artifact_list", "context_report",
)

EGO_MODEL_FACING = _SHARED_MODEL_FACING + (
    "record_conclusion", "board_post",
) + EGO_ONLY

ID_MODEL_FACING = _SHARED_MODEL_FACING + ID_SENSES + PROMPT_READ + ID_ONLY

# Deliberately absent from both: register_agent, retire_agent, heartbeat and
# bind_profile (the process's own lifecycle), status/health/capabilities (the
# manifest carries the resource identities that matter), and Ego's snapshot
# publication, which is the Harness's scheduling concern rather than a
# cognitive act.

MODEL_FACING: dict[str, tuple[str, ...]] = {
    "ego": EGO_MODEL_FACING,
    "id": ID_MODEL_FACING,
}


def model_facing_verbs(role: str) -> tuple[str, ...]:
    """The verbs a role's model may be offered and may invoke.

    Ordered and de-duplicated so a manifest built twice from the same state is
    byte-identical -- the environment digest has to be stable or provenance
    becomes noise.
    """
    return tuple(sorted(set(MODEL_FACING.get(role, ()))))


def scope_tables() -> dict[str, tuple[str, ...]]:
    """The whole capability model, as data.

    ``operator`` is intentionally absent: it is the full table, held by the
    supervisor itself and the MCP facade, and is granted by the control token
    rather than by a scope entry.
    """
    return {"neuocyte": NEUOCYTE, "ego": EGO, "id": ID,
            "external_io": EXTERNAL_IO}


def id_only_verbs() -> frozenset[str]:
    return frozenset(ID_ONLY)


def ego_only_verbs() -> frozenset[str]:
    return frozenset(EGO_ONLY)


def external_io_verbs() -> frozenset[str]:
    return frozenset(EXTERNAL_IO)


def verbs_for(scope: str) -> frozenset[str]:
    return frozenset(scope_tables().get(scope, ()))
