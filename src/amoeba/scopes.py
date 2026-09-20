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
    "lease_work", "complete_work", "fail_work",
    "acquire_snapshot", "release_snapshot_ref", "snapshot_tokens",
    "maintenance_context",
    "board_read", "board_post",
    "tool_invoke", "tool_schemas",
    # Collected at a turn boundary, never pushed into the process.
    "work_messages",
)

# Shared by the two long-lived role processes, traced from roles.py. Read-heavy
# and deliberately free of anything that decides.
ROLE_BASE = (
    "register_agent", "retire_agent", "heartbeat",
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
)

EGO_ONLY = (
    "ego_work_view", "ego_artifact_evidence", "ego_resource_identities",
    "ego_request_work", "ego_work_message", "ego_request_cancellation",
    "ego_propose_memory", "ego_message_id", "ego_request_id_review",
)

# Id's senses beyond the shared base, plus its effectors. Everything in
# ID_ONLY appears in no other scope.
ID_SENSES = (
    "system_pulse", "verify_integrity", "disagreements",
    "board_independence", "board_corroboration",
    "context_assess", "sandbox_capabilities", "file_roots",
    "id_health",
)

ID_ONLY = (
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

ID = ROLE_BASE + ID_SENSES + ID_ONLY


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
