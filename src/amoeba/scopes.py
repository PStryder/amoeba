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

EGO = ROLE_BASE + (
    "remember", "record_conclusion",
    "publish_ego_snapshot", "ensure_snapshot", "list_snapshots",
    "board_post",
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


def scope_tables() -> dict[str, tuple[str, ...]]:
    """The whole capability model, as data.

    ``operator`` is intentionally absent: it is the full table, held by the
    supervisor itself and the MCP facade, and is granted by the control token
    rather than by a scope entry.
    """
    return {"neuocyte": NEUOCYTE, "ego": EGO, "id": ID}


def id_only_verbs() -> frozenset[str]:
    return frozenset(ID_ONLY)


def verbs_for(scope: str) -> frozenset[str]:
    return frozenset(scope_tables().get(scope, ()))
