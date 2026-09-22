"""Does each invariant's test actually fail when the invariant is broken?

A test that passes is weak evidence. A test that *fails when the guarantee is
removed* is the real thing. This applies one targeted mutation per invariant --
each a minimal edit that negates precisely that claim -- runs only the tests
named for it, and requires them to fail.

A mutation that leaves its tests green is the finding: that test does not
express the claim, whatever its name says.

Source is always restored, including on interrupt.

    .\\.venv\\Scripts\\python.exe scripts\\verify_invariants.py [--only I7,I8]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


@dataclass
class Mutation:
    invariant: str
    claim: str
    path: str
    old: str
    new: str
    tests: list[str]
    layer: str
    note: str = ""
    # Extra edits. A guarantee defended in depth cannot be negated by a single
    # change: removing one of two redundant checks leaves the other working,
    # and the tests correctly stay green. To ask "is this claim defended at
    # all?" every defence has to come out. Two shapes:
    #   (old, new)        -- another edit in this mutation's own file
    #   (path, old, new)  -- an edit in a different file
    # The second exists because a claim can be defended across modules -- the
    # external surface is defined by both an adapter allowlist and a credential
    # scope -- and a harness that could not express that would report SKIP,
    # which proves nothing while looking like success.
    also: list[tuple[str, ...]] = field(default_factory=list)


# Each mutation removes exactly one guarantee, at the layer that owns it.
MUTATIONS: list[Mutation] = [
    Mutation(
        "I1", "One writer: state, events and receipt commit atomically",
        "src/amoeba/store/writer.py",
        "        except BaseException:\n            try:\n                self.conn.rollback()",
        "        except BaseException:\n            try:\n                pass  # MUTANT: no rollback",
        ["test_crash_during_commit_leaves_no_half_applied_mutation"],
        layer="StateWriter (the transaction itself)",
    ),
    Mutation(
        "I2", "A correction supersedes; it never rewrites earlier evidence",
        "src/amoeba/store/memory_repo.py",
        '                m.sql(\n                    "UPDATE memory_items SET status = \'superseded\', updated_at = ?,"\n'
        '                    " state_version = ? WHERE memory_id = ?",',
        '                m.sql(\n                    "UPDATE memory_items SET claim = \'REWRITTEN\', status = \'superseded\', updated_at = ?,"\n'
        '                    " state_version = ? WHERE memory_id = ?",',
        ["test_correction_supersedes_and_preserves_contrary_evidence"],
        layer="MemoryRepo (where supersession is written)",
    ),
    Mutation(
        "I4", "A committed reference to missing content is an integrity failure",
        "src/amoeba/store/events.py",
        "    missing: list[dict[str, str]] = []\n    refs: list[tuple[str, str, str]] = []",
        "    return []  # MUTANT: never reports missing content\n"
        "    missing: list[dict[str, str]] = []\n    refs: list[tuple[str, str, str]] = []",
        ["test_missing_blob_is_detected_not_glossed_over"],
        layer="events.missing_content (the integrity check)",
    ),
    Mutation(
        "I5", "The hash chain detects mutation",
        "src/amoeba/store/events.py",
        '        if row["prev_hash"] != prev:',
        '        if False:  # MUTANT: linkage check 1 removed',
        ["test_a_broken_chain_link_is_detected_not_just_a_tampered_payload",
         "test_hash_chain_detects_tampering"],
        layer="events.verify_chain (both linkage checks)",
        note="this claim is defended twice over. chain_hash folds the "
             "predecessor into every event hash AND the prev_hash column is "
             "compared directly, so removing either alone leaves the other "
             "catching excision. Negating the claim needs both out.",
        also=[('        if expect != row["event_hash"]:',
               '        if False:  # MUTANT: linkage check 2 removed')],
    ),
    Mutation(
        "I6", "An acknowledged mutation survives restart",
        "src/amoeba/store/db.py",
        'conn.execute("PRAGMA synchronous=FULL")',
        'conn.execute("PRAGMA synchronous=OFF")  # MUTANT',
        ["test_durability_pragma_is_set_where_durability_is_configured"],
        layer="Database (durability pragma)",
        note="a clean close/reopen cannot observe fsync, so the property is "
             "asserted where it is configured instead",
    ),
    Mutation(
        "I7", "Replaying a mutation id does not re-apply it",
        "src/amoeba/store/writer.py",
        "        existing = self.receipt_for(mutation_id)\n        if existing is not None:\n            return existing, None",
        "        existing = None  # MUTANT: idempotency disabled\n        if existing is not None:\n            return existing, None",
        ["test_writer_itself_refuses_to_reapply_a_mutation_id"],
        layer="StateWriter (the idempotency check)",
        note="the work-queue test has its own short-circuit and cannot see this",
    ),
    Mutation(
        "I8", "A lease bumps the fencing token, so a stale result cannot commit",
        "src/amoeba/store/work_repo.py",
        '                token = int(row["fencing_token"]) + 1',
        '                token = int(row["fencing_token"])  # MUTANT: no bump',
        ["test_each_lease_advances_the_fencing_token"],
        layer="WorkRepo.lease (where the token advances)",
        note="expire_leases also bumps the token, which masked this",
    ),
    Mutation(
        "I11", "A finding pinned to an older state version is flagged",
        "src/amoeba/store/work_repo.py",
        "            if pinned_state_ver is not None and pinned_state_ver < m.prior_version:\n"
        "                stale_against = {\"pinned\": pinned_state_ver, \"current\": m.prior_version}",
        "            if False:\n"
        "                stale_against = {\"pinned\": pinned_state_ver, \"current\": m.prior_version}",
        ["test_findings_pinned_to_an_older_state_version_are_flagged"],
        layer="WorkRepo.complete (where staleness is recorded)",
    ),
    Mutation(
        "I16", "A referenced snapshot is never reclaimed",
        "src/amoeba/store/work_repo.py",
        "            if int(row[\"refcount\"]) > 0:\n                raise ResourceExhausted(",
        "            if False:\n                raise ResourceExhausted(",
        ["test_reclaim_only_unreferenced_and_superseded"],
        layer="WorkRepo.mark_snapshot_released (the refcount guard)",
    ),
    Mutation(
        "I19", "A requested budget is capped, not honoured blindly",
        "src/amoeba/arbiter.py",
        "        budget = min(\n            requested_budget_tokens or cfg.neuocyte_token_budget, cfg.neuocyte_token_budget\n        )",
        "        budget = requested_budget_tokens or cfg.neuocyte_token_budget  # MUTANT",
        ["test_requested_budget_is_capped_not_honoured_blindly"],
        layer="Arbiter.admit (where the cap is applied)",
    ),
    Mutation(
        "I19b", "Maintenance recursion is bounded",
        "src/amoeba/arbiter.py",
        "            if maintenance_depth > cfg.max_maintenance_depth:",
        "            if False:  # MUTANT: recursion unbounded",
        ["test_maintenance_recursion_is_bounded"],
        layer="Arbiter.admit (the depth guard)",
    ),
    Mutation(
        "I24", "physical_overlap_verified is never claimed without evidence",
        "src/amoeba/backends/deterministic.py",
        '            "physical_overlap_verified": False,',
        '            "physical_overlap_verified": True,  # MUTANT: overclaim',
        ["test_health_reports_capabilities_without_conflating_them"],
        layer="backend capabilities() (where the flag is published)",
        note="the other test asserts the real backend, not this one",
    ),
    Mutation(
        "I24b", "A simulated backend labels every result it produces",
        "src/amoeba/backends/deterministic.py",
        'SIM_PREFIX = "[SIMULATED]"',
        'SIM_PREFIX = ""  # MUTANT: label removed',
        ["test_simulated_backend_is_labelled_on_every_cognitive_result"],
        layer="DeterministicBackend (where the label is attached)",
    ),
    Mutation(
        "BOARD", "Agreement after reading is not independent replication",
        "src/amoeba/store/board_repo.py",
        "        informed_by = sorted(self.posts_read_by(author, at_or_before=now))",
        "        informed_by = []  # MUTANT: influence not recorded",
        ["test_agreement_after_reading_is_not_independent",
         "test_corroboration_separates_real_support_from_echo",
         "test_read_and_post_in_the_same_tick_still_counts_as_informed"],
        layer="BoardRepo.post (where the influence snapshot is taken)",
    ),
    Mutation(
        "BOARD2", "A read is recorded against the reader",
        "src/amoeba/store/board_repo.py",
        "        if record and posts:\n            self.record_read(",
        "        if False and posts:\n            self.record_read(",
        ["test_reading_is_recorded", "test_agreement_after_reading_is_not_independent"],
        layer="BoardRepo.read (where reads are logged)",
    ),
    Mutation(
        "HOMEO", "Summarising a context is refused, not silently accepted",
        "src/amoeba/homeostasis.py",
        '        if mode == "summarise":\n            raise CapabilityUnsupported(',
        '        if False:\n            raise CapabilityUnsupported(',
        ["test_summarising_is_refused_because_it_is_a_different_behaviour"],
        layer="ContextHomeostasis.rejuvenate (the refusal)",
    ),
    Mutation(
        "HOMEO2", "trim keeps a verbatim head and tail",
        "src/amoeba/homeostasis.py",
        "        return list(tokens[:head]) + (list(tokens[-tail:]) if tail else [])",
        "        return [0] * (head + tail)  # MUTANT: not verbatim",
        ["test_trim_keeps_a_verbatim_head_and_tail"],
        layer="ContextHomeostasis.apply_trim (where tokens are selected)",
    ),
    Mutation(
        "HOMEO3", "Occupancy charges a recomputed prefix in full",
        "src/amoeba/backends/deterministic.py",
        "        sess.shares_prefix = False      # recomputed, not shared",
        "        sess.shares_prefix = True       # MUTANT: pretends recomputed is shared",
        ["test_a_recomputed_prefix_is_charged_in_full"],
        layer="backend restore_prefix (where sharing is declared)",
    ),
    Mutation(
        "CANCEL", "A cancel arriving before generate() is honoured",
        "src/amoeba/backends/deterministic.py",
        "            if sess.cancel_requested:\n                sess.cancel_requested = False\n                return GenerationResult(",
        "            if False:\n                sess.cancel_requested = False\n                return GenerationResult(",
        ["test_a_cancel_arriving_before_generate_is_honoured_not_discarded",
         "test_cancel_flag_stops_generation_and_is_reported"],
        layer="backend generate() (where the flag is consumed)",
    ),
    Mutation(
        "CANCEL2", "A cancelled MCP call issues cancel_operation",
        "src/amoeba/mcp_api.py",
        "                abandon_on_cancel=True)",
        "                abandon_on_cancel=False)  # MUTANT: the 97d5179 bug",
        ["test_call_cancellable_issues_a_cancel_when_the_await_is_cancelled",
         "test_call_cancellable_still_propagates_the_cancellation"],
        layer="Facade.call_cancellable (where cancellation is received)",
        note="this is the exact defect the review found; it must be caught now",
    ),
    Mutation(
        "SANDBOX", "A path escaping the sandbox root is rejected",
        "src/amoeba/sandbox.py",
        "        if target != root and root not in target.parents:\n"
        "            raise InvalidInput(\"path escapes the sandbox root\", path=relpath)",
        "        if False:\n"
        "            raise InvalidInput(\"path escapes the sandbox root\", path=relpath)",
        ["test_path_traversal_is_rejected"],
        layer="SandboxManager.resolve_inside (the path check)",
    ),
    Mutation(
        "I29", "Hardening severs inheritance, so a parent cannot re-grant",
        "src/amoeba/security.py",
        '    cmd = [str(path), "/inheritance:r"]',
        '    cmd = [str(path)]  # MUTANT: inheritance left live',
        ["test_hardening_removes_every_ace_for_everyone",
         "test_hardening_removes_inheritance_so_the_parent_cannot_regrant",
         "test_children_lose_the_permissive_access_they_inherited"],
        layer="security.harden (the icacls invocation itself)",
        note="the grants still apply; only the severing is removed. If the "
             "tests stay green they are asserting our return value rather "
             "than the DACL the OS reports.",
    ),
    Mutation(
        "I30", "Hardening that would lock the account out is rolled back",
        "src/amoeba/security.py",
        '        _icacls([str(path), "/reset", "/Q"])',
        '        pass  # MUTANT: no rollback; directory left unopenable',
        ["test_hardening_rolls_back_rather_than_locking_the_account_out"],
        layer="security.harden (the post-hardening self-check)",
    ),
    Mutation(
        "I30b", "Hardening is a single atomic icacls call",
        "src/amoeba/security.py",
        "    r = _icacls(cmd)",
        '    _icacls([str(path), "/inheritance:r", "/Q"])  # MUTANT: two-step\n'
        "    r = _icacls([c for c in cmd if c != \"/inheritance:r\"])",
        ["test_hardening_is_one_atomic_icacls_call"],
        layer="security.harden (call structure)",
        note="reproduces the two-step form that locked the state tree out "
             "when the second call failed. The end state is identical, so "
             "only a test counting calls can catch it.",
    ),
    Mutation(
        "I31", "Every state directory is hardened, not just the root",
        "src/amoeba/security.py",
        "    for t in targets:\n        results.append(harden(t, log_name=log_name))",
        "    for t in targets[:1]:  # MUTANT: root only, trust inheritance\n"
        "        results.append(harden(t, log_name=log_name))",
        ["test_every_state_directory_is_hardened_including_ones_outside_the_root"],
        layer="security.harden_state_tree (which directories are covered)",
    ),
    Mutation(
        "I32", "A destroyed container loses its grant on the shared runtime",
        "src/amoeba/sandbox.py",
        '            revoke(self.runtime_dir, sb.container_sid, log_name="sandbox")',
        "            pass  # MUTANT: stale grant left on the runtime",
        ["test_destroying_a_sandbox_revokes_its_grant_on_the_shared_runtime"],
        layer="SandboxManager.destroy (teardown of the shared grant)",
    ),
    Mutation(
        "I32b", "Sandboxed code gets execute, not write, on its runtime",
        "src/amoeba/sandbox.py",
        '               container_rights="(OI)(CI)(RX)", log_name="sandbox")',
        '               container_rights="(OI)(CI)(F)", log_name="sandbox")',
        ["test_sandboxed_code_cannot_modify_its_own_runtime"],
        layer="SandboxManager.create (the rights granted to the container)",
        note="the injection path that persists across sandboxes. Asserted "
             "from inside the container, so the kernel decides, not the ACL "
             "string we wrote.",
    ),
    Mutation(
        "I33", "The audit states residual exposure rather than claiming safety",
        "src/amoeba/security.py",
        '        "owner_can_restore_access": True,',
        '        "owner_can_restore_access": False,  # MUTANT: overclaim',
        ["test_the_audit_does_not_claim_protection_it_does_not_have"],
        layer="security.audit_path (the report itself)",
        note="honesty is a guarantee here like any other. The failure mode "
             "this defends against is the report quietly becoming an "
             "assertion of safety once the hardening starts working.",
    ),
    Mutation(
        "I34", "Promotion materialises the reviewed blob, not the scratch file",
        "src/amoeba/harness_api.py",
        "        data = mind.blobs.get(digest)",
        '        data = sup.sandboxes.resolve_inside(  # MUTANT: read scratch\n'
        '            sup.sandboxes.get(row["sandbox_id"]), row["path"]).read_bytes()',
        ["test_promotion_promotes_the_reviewed_bytes_not_whatever_scratch_holds",
         "test_promotion_materialises_the_reviewed_bytes_not_whatever_scratch_holds"],
        layer="artifact_promote (where the promoted bytes come from)",
        note="restores the time-of-check/time-of-use surface. Reading scratch "
             "makes what lands depend on a file the neuocyte can still write, "
             "which is exactly what sourcing from an immutable blob removes.",
    ),
    Mutation(
        "I35", "Concurrent sandboxes inherit no handles from each other",
        "src/amoeba/sandbox.py",
        "            if not k32.UpdateProcThreadAttribute(\n"
        "                    attrs, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_HANDLE_LIST),",
        "            if False and k32.UpdateProcThreadAttribute(\n"
        "                    attrs, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_HANDLE_LIST),",
        ["test_a_sandbox_does_not_inherit_another_sandboxs_handles"],
        layer="_spawn (the process-creation attribute list)",
        note="restores unrestricted handle inheritance. Every ACL test still "
             "passes with this hole open, which is the point: it is a "
             "different guarantee and needs its own test.",
    ),
    Mutation(
        "I36", "Sandbox runs are not serialised by a manager-wide lock",
        "src/amoeba/sandbox.py",
        "        return self._spawn(sb, cmd, timeout=timeout)",
        "        with self._lock:  # MUTANT: serialise every run\n"
        "            return self._spawn(sb, cmd, timeout=timeout)",
        ["test_sandboxes_run_concurrently_rather_than_serialised"],
        layer="run_python (whether runs hold the manager lock)",
        note="negated by *adding* a defence rather than removing one. The "
             "claim is the absence of serialisation, so the mutation has to "
             "introduce it.",
    ),
    Mutation(
        "I23", "A role may only call tools its role permits",
        "src/amoeba/tools.py",
        "        if role not in spec.allowed_roles:",
        "        if False:  # MUTANT: role permissions ignored",
        ["test_role_permissions_are_enforced"],
        layer="ToolRegistry.execute (the role check)",
    ),
    Mutation(
        "I23b", "A neuocyte cannot grant itself a capability by asking",
        "src/amoeba/tools.py",
        "    if not sandbox_allowed:\n        return reg",
        "    if False:  # MUTANT: register sandbox tools regardless\n        return reg",
        ["test_a_neuocyte_cannot_grant_itself_the_sandbox",
         "test_sandbox_tools_are_absent_when_the_work_item_did_not_allow_them"],
        layer="build_neuocyte_registry (where the work row's grant is applied)",
        note="the load-bearing one. If a model can talk its way into a "
             "sandbox there is no boundary, so the gate must come from the "
             "work row and not from the request.",
    ),
    Mutation(
        "I23c", "No neuocyte tool lets a model name a sandbox",
        "src/amoeba/tools.py",
        '        params=[ToolParam("code", "string", "Python source to execute",\n'
        '                          required=True, max_length=20000)],',
        '        params=[ToolParam("code", "string", "Python source to execute",\n'
        '                          required=True, max_length=20000),\n'
        '                ToolParam("sandbox_id", "string", "MUTANT: model-named sandbox",\n'
        '                          max_length=64)],',
        ["test_no_neuocyte_tool_accepts_a_sandbox_id"],
        layer="the run_code tool schema (what the model can address)",
    ),
    Mutation(
        "I23d", "A fenced neuocyte cannot still run code",
        "src/amoeba/store/work_repo.py",
        '        if int(row["fencing_token"]) != int(fencing_token):',
        "        if False:  # MUTANT: stale tokens accepted",
        ["test_a_fenced_neuocyte_cannot_invoke_a_tool"],
        layer="WorkRepo.authorise_tool_call (the fencing check)",
    ),
    Mutation(
        "I23e", "The tool loop is bounded by its turn limit",
        "src/amoeba/neuocyte.py",
        "        for turn in range(max_turns):",
        "        for turn in range(10_000):  # MUTANT: turn limit removed",
        ["test_the_tool_loop_stops_at_the_turn_limit"],
        layer="Neuocyte._generate_with_tools (both turn bounds)",
        note="defended twice, for two different reasons that happen to "
             "overlap: range(max_turns) bounds the loop, and the "
             "turn == max_turns - 1 check refuses to execute a tool whose "
             "result the model would never get to read. Removing either alone "
             "leaves the other stopping the loop, so negating the claim needs "
             "both -- do not 'simplify' one away.",
        also=[("            if turn == max_turns - 1:",
               "            if False:  # MUTANT: last-turn guard removed")],
    ),
    Mutation(
        "I23e2", "The tool loop is bounded by the token budget",
        "src/amoeba/neuocyte.py",
        "            if remaining <= 0:",
        "            if False:  # MUTANT: budget ignored",
        ["test_the_tool_loop_stops_when_the_token_budget_is_exhausted"],
        layer="Neuocyte._generate_with_tools (the budget bound)",
    ),
    Mutation(
        "I23e3", "The tool loop is bounded by the wall-clock deadline",
        "src/amoeba/neuocyte.py",
        "            if time.time() >= deadline:",
        "            if False:  # MUTANT: deadline ignored",
        ["test_the_tool_loop_stops_at_the_deadline"],
        layer="Neuocyte._generate_with_tools (the deadline bound)",
    ),
    Mutation(
        "I37", "A path outside its root is refused, not clamped into it",
        "src/amoeba/filespace.py",
        "        if final != root_path and root_path not in final.parents:",
        "        if False:  # MUTANT: containment check removed",
        ["test_paths_that_leave_the_root_or_name_a_device_are_refused",
         "test_a_refused_path_is_never_silently_clamped"],
        layer="Filespace.resolve (the containment check)",
        note="the component checks catch the obvious `..` cases on their own, "
             "so this also removes them -- otherwise the claim looks defended "
             "while a resolved junction walks straight out.",
        also=[("        for part in parts:\n            _reject_component(part)",
               "        for part in parts:\n            pass  # MUTANT")],
    ),
    Mutation(
        "I37b", "A link is not a way out of a root",
        "src/amoeba/filespace.py",
        "            final = candidate.resolve()",
        "            final = candidate.absolute()  # MUTANT: links not followed",
        ["test_a_junction_pointing_out_of_the_root_is_refused"],
        layer="Filespace.resolve (full resolution before the check)",
        note="without resolving, the containment check compares a path that "
             "still looks inside the root while pointing elsewhere.",
    ),
    Mutation(
        "I37c", "A read-only root refuses writes",
        "src/amoeba/filespace.py",
        '        if need_write and root.mode != "read_write":',
        "        if False:  # MUTANT: mode ignored",
        ["test_a_read_only_root_refuses_writes"],
        layer="Filespace.resolve (the mode check)",
        note="write_bytes has its own check, so this also removes that one; "
             "the claim is defended in both places deliberately.",
        also=[('        if not resolved.writable:\n'
               '            raise FilespaceDenied("this root is read-only", root=resolved.root_name)\n'
               '        resolved.path.parent.mkdir(parents=True, exist_ok=True)',
               '        resolved.path.parent.mkdir(parents=True, exist_ok=True)')]),
    Mutation(
        "I39", "No neuocyte tool names a filespace root",
        "src/amoeba/tools.py",
        '        params=[ToolParam("path", "string", "relative path inside the sandbox",\n'
        '                          required=True, max_length=512),\n'
        '                ToolParam("rationale", "string", "why this is worth keeping",\n'
        '                          required=True, max_length=2000)],',
        '        params=[ToolParam("path", "string", "relative path inside the sandbox",\n'
        '                          required=True, max_length=512),\n'
        '                ToolParam("root", "string", "MUTANT: model-chosen destination",\n'
        '                          max_length=64),\n'
        '                ToolParam("rationale", "string", "why this is worth keeping",\n'
        '                          required=True, max_length=2000)],',
        ["test_a_neuocyte_has_no_tool_that_reaches_the_host_filesystem"],
        layer="the propose_artifact schema (what a model can address)",
        note="negated by adding, because the claim is the absence of a verb. "
             "Letting a neuocyte name the destination moves the decision to "
             "put bytes on disk from the promoter to the proposer.",
    ),
    Mutation(
        "I40", "A file outside every root cannot be handed in",
        "src/amoeba/filespace.py",
        '        raise FilespaceDenied(\n'
        '            "path is not inside any configured filespace root",\n'
        '            path=str(target), known_roots=sorted(self._roots))',
        "        name = sorted(self._roots)[0]  # MUTANT: accept anything\n"
        "        return ResolvedPath(name, target.name, target, True, target.exists())",
        ["test_attaching_a_file_outside_every_root_is_refused"],
        layer="Filespace.resolve_host_path (the allowlist check)",
    ),
    Mutation(
        "I41", "A receipt's digest is what the neuocyte can actually hash",
        "src/amoeba/harness_api.py",
        '        landed = _sandbox_manager().write_bytes(sandbox_id, dest, data)',
        '        landed = _sandbox_manager().write_file(\n'
        '            sandbox_id, dest, data.decode("utf-8", "replace"))  # MUTANT',
        ["test_an_attached_files_digest_is_what_the_neuocyte_can_hash",
         "test_the_durable_event_carries_the_same_digest",
         "test_attaching_a_binary_file_lands_the_exact_bytes"],
        layer="file_attach (the write into the sandbox, and the check on it)",
        note="reintroduces the exact defect this was found by: content routed "
             "through str, every non-UTF-8 byte replaced with U+FFFD, while "
             "the receipt keeps the source digest. Defended twice -- the "
             "byte-accurate write and the integrity check on what landed -- so "
             "both come out. What is left is the only check that matters: the "
             "neuocyte hashing its own bytes inside the container and getting "
             "the digest it was promised.",
        also=[('        if landed["sha256"] != digest:',
               "        if False:  # MUTANT: receipt no longer checked")],
    ),
    Mutation(
        "I41b", "A reported digest describes the bytes that were actually stored",
        "src/amoeba/sandbox.py",
        "        target.write_bytes(data)\n"
        '        return {"path": relpath, "bytes": len(data), "sha256": sha256_hex(data)}',
        "        target.write_bytes(data[:-1] if data else data)  # MUTANT: short write\n"
        '        return {"path": relpath, "bytes": len(data), "sha256": sha256_hex(data)}',
        ["test_a_digest_a_neuocyte_was_told_it_wrote_is_what_is_on_disk",
         "test_a_proposed_artifacts_digest_matches_what_the_sandbox_holds",
         "test_an_attached_files_digest_is_what_the_neuocyte_can_hash"],
        layer="SandboxManager.write_bytes (what is stored vs what is reported)",
        note="the receipt stays internally consistent and is still a lie: it "
             "reports the digest of what it was handed, not of what landed. "
             "Only hashing from inside the container can tell the difference, "
             "which is why the tests do it there.",
    ),
    Mutation(
        "I41c", "A read returns exact text or refuses; it never renders",
        "src/amoeba/filespace.py",
        '    try:\n        return head.decode("utf-8")\n    except UnicodeDecodeError as exc:',
        '    try:\n        return head.decode("utf-8", "replace")  # MUTANT: render it\n'
        '    except UnicodeDecodeError as exc:',
        ["test_a_non_text_read_is_refused_not_rendered",
         "test_a_host_file_read_refuses_non_text_too",
         "test_reading_a_non_text_file_is_refused"],
        layer="decode_exact_text (the only place bytes become a string)",
        note="restores the lossy rendering. It used to be allowed if flagged; "
             "it is not allowed at all now, because a flag does not stop a "
             "model reasoning about U+FFFD soup as though it were the file.",
    ),
    Mutation(
        "I41d", "Truncation does not make a text file look like binary",
        "src/amoeba/filespace.py",
        "    head = data\n    if truncated:",
        "    head = data\n    if False:  # MUTANT: no character-boundary trim",
        ["test_truncation_does_not_make_a_text_file_look_like_binary"],
        layer="decode_exact_text (the boundary trim before decoding)",
        note="the edge the strict rule introduces. A cap landing mid-sequence "
             "would refuse an ordinary UTF-8 file, which would make the rule "
             "look broken rather than strict.",
    ),
    Mutation(
        "I42", "Accepted work product exists outside the sandbox before it dies",
        "src/amoeba/harness_api.py",
        "            fs.write_bytes(resolved, data)\n            dest = resolved.path",
        "            dest = resolved.path  # MUTANT: promotion copies nothing out",
        ["test_destroying_a_sandbox_preserves_input_evidence_and_work_product"],
        layer="artifact_promote (the copy out of the compute sandbox)",
        note="negates the disposable-laboratory claim at its root. If "
             "promotion records acceptance without moving the bytes, the only "
             "copy dies with the scratch and 'safe to destroy' is false.",
    ),
    Mutation(
        "I42b", "Destroying a sandbox does not decide a proposal",
        "src/amoeba/harness_api.py",
        "        out = mgr.destroy(sandbox_id)",
        '        mind.db.conn.execute(  # MUTANT: teardown decides\n'
        '            "UPDATE artifacts SET status = \'lapsed\'"\n'
        '            " WHERE sandbox_id = ? AND status = \'proposed\'", (sandbox_id,))\n'
        '        mind.db.conn.commit()\n'
        "        out = mgr.destroy(sandbox_id)",
        ["test_a_proposal_stays_promotable_after_its_sandbox_is_destroyed",
         "test_destroying_a_sandbox_decides_nothing"],
        layer="sandbox_destroy (whether teardown touches proposal state)",
        note="reintroduces lapsing by *adding* it back, because the claim is "
             "now the absence of a coupling: sandbox lifetime driving a "
             "decision is the thing being ruled out.",
    ),
    Mutation(
        "I43", "Sandboxed code cannot reach the other three stores",
        "src/amoeba/sandbox.py",
        '        harden(scratch, container_sid=sid_str, container_rights="(OI)(CI)(F)",\n'
        '               log_name="sandbox")',
        '        harden(scratch, container_sid=sid_str, container_rights="(OI)(CI)(F)",\n'
        '               log_name="sandbox")\n'
        '        harden(self.root.parent, container_sid=sid_str,  # MUTANT: grant the\n'
        '               container_rights="(OI)(CI)(F)", log_name="sandbox")',
        ["test_sandboxed_code_cannot_reach_filespace_blobs_or_state",
         "test_sandboxed_code_cannot_reach_another_work_items_scratch"],
        layer="SandboxManager.create (what the container SID is granted)",
        note="negated by *adding* a grant, because the claim is the absence of "
             "reach. Giving the container the whole state tree opens the blob "
             "store, the database and every other work item's scratch at once.",
    ),
    Mutation(
        "I44", "The pulse carries the sections Id needs",
        "src/amoeba/pulse.py",
        '            "resources": resources,',
        "            # MUTANT: resource versions dropped",
        ["test_id_can_obtain_the_complete_bounded_pulse"],
        layer="PulseCollector._assemble (the pulse contract)",
        note="a missing section is the quiet failure: the call still works and "
             "Id simply stops being able to notice that cognition changed.",
    ),
    Mutation(
        "I44b", "The pulse states observations, not verdicts",
        "src/amoeba/pulse.py",
        '            "pending_decisions": pending,',
        '            "pending_decisions": pending,\n'
        '            "ego_unhealthy": True,  # MUTANT: a verdict',
        ["test_the_pulse_reports_observations_not_verdicts"],
        layer="PulseCollector._assemble (what the field names assert)",
        note="the exact shape being ruled out. A verdict here moves cognition "
             "into the Harness and leaves Id agreeing with a number it cannot "
             "inspect.",
    ),
    Mutation(
        "I44c", "The pulse reflects live failure state",
        "src/amoeba/pulse.py",
        "            label = FAILURE_KINDS.get(row[\"kind\"])",
        "            label = None  # MUTANT: failures never counted",
        ["test_the_pulse_moves_when_failures_happen"],
        layer="PulseCollector._drain_failures (the counter feed)",
        note="counters that never move look exactly like a system with no "
             "problems, which is the worst possible failure for a sense.",
    ),
    Mutation(
        "I44d", "A resource digest tracks the resource",
        "src/amoeba/resources.py",
        "    text = prompt_text(role, cfg, mind)\n"
        "    return ResourceVersion(\n"
        '        kind=f"prompt.{role}", sha256=sha256_hex(text.encode("utf-8")),',
        "    text = prompt_text(role, cfg, mind)\n"
        "    return ResourceVersion(\n"
        '        kind=f"prompt.{role}", sha256=sha256_hex(role.encode("utf-8")),',
        ["test_a_resource_version_changes_when_the_resource_does"],
        layer="resources.prompt_version (what the digest is over)",
        note="a digest that is stable regardless of the text is worse than no "
             "digest: it actively asserts nothing changed.",
    ),
    Mutation(
        "I45", "Id actions are attributed and carry their telemetry",
        "src/amoeba/id_api.py",
        '                "raised_by": "id", "pulse_id": prov.get("pulse_id"),\n'
        '                "note": "Id is not authorised to remediate this autonomously"})',
        '                "raised_by": "id",  # MUTANT: provenance dropped\n'
        '                "note": "Id is not authorised to remediate this autonomously"})',
        ["test_consequential_id_actions_are_receipted_and_attributed"],
        layer="id_escalate_to_operator (the provenance on the record)",
        note="without the cited pulse the record says what Id did but not what "
             "it was looking at, which is the half that makes it auditable.",
    ),
    Mutation(
        "I46", "Id-only verbs are absent from the neuocyte scope",
        "src/amoeba/scopes.py",
        '    "work_messages",\n)',
        '    "work_messages",\n'
        '    "id_raise_finding", "system_pulse",  # MUTANT: granted\n)',
        ["test_a_neuocyte_cannot_invoke_an_id_only_verb_by_name",
         "test_the_neuocyte_scope_contains_no_id_only_verb"],
        layer="scopes.NEUOCYTE (the table a neuocyte connection sees)",
        note="the isolation is this table. Granting from it is the whole "
             "failure, and it is a two-word edit -- which is exactly why it "
             "needs a test that dies.",
    ),
    Mutation(
        "I46b", "An unknown method does not enumerate the table",
        "src/amoeba/rpc.py",
        '                                  "details": {"scope": getattr(self, "scope", None)}}})',
        '                                  "details": {"known": sorted(visible)}}})',
        ["test_a_neuocyte_cannot_enumerate_the_methods_it_lacks"],
        layer="_Handler._dispatch (what a refusal discloses)",
        note="restores the discovery oracle: asking for a method that does not "
             "exist used to return the name of every method that does.",
    ),
    Mutation(
        "I47", "Ego cannot instantiate a worker or set execution conditions",
        "src/amoeba/scopes.py",
        '    "ego_read_attachment", "ego_surface_result",\n)\n\nEGO_ONLY',
        '    "ego_read_attachment", "ego_surface_result",\n'
        '    "admit_work", "lease_work",  # MUTANT: execution authority\n)\n\nEGO_ONLY',
        ["test_ego_requests_work_and_cannot_instantiate_a_worker"],
        layer="scopes.EGO (the table an Ego connection sees)",
        note="the boundary is this table. Granting admission or leasing turns "
             "Ego from a component that states intent into one that runs the "
             "scheduler.",
    ),
    Mutation(
        "I47b", "Ego cannot read live compute scratch",
        "src/amoeba/scopes.py",
        '    "record_conclusion",\n    "publish_ego_snapshot", "list_snapshots",',
        '    "record_conclusion", "sandbox_files", "sandbox_read",  # MUTANT\n'
        '    "publish_ego_snapshot", "list_snapshots",',
        ["test_ego_cannot_inspect_compute_sandbox_scratch"],
        layer="scopes.EGO (whether scratch is reachable at all)",
        note="turns half-written scratch into a communication channel: Ego "
             "could then consume something no neuocyte ever published.",
    ),
    Mutation(
        "I47c", "Board-naive work refuses mid-flight messages",
        "src/amoeba/ego_api.py",
        '        if row["board_access"] == "none":',
        "        if False:  # MUTANT: independence not protected",
        ["test_board_naive_work_refuses_mid_flight_messages"],
        layer="ego_work_message (the independence gate)",
        note="the quiet failure: the message lands, the work still looks "
             "board-naive, and later agreement is read as independent "
             "replication when it was an echo of Ego.",
    ),
    Mutation(
        "I47d", "Ego proposes maintained state rather than authoring it",
        "src/amoeba/scopes.py",
        '    "record_conclusion",\n    "publish_ego_snapshot",',
        '    "record_conclusion", "remember",  # MUTANT: direct authoring\n'
        '    "publish_ego_snapshot",',
        ["test_ego_proposes_memory_rather_than_authoring_it"],
        layer="scopes.EGO (belief authoring vs proposing)",
        note="Ego is the component most exposed to a confident user, so direct "
             "authoring is where an unchecked belief enters the organism.",
    ),
    Mutation(
        "I47e", "A message's author is the authenticated scope",
        "src/amoeba/ego_api.py",
        '                  (message_id, work_id, "ego", body_text, kind, time.time(),',
        '                  (message_id, work_id, kind, body_text, kind, time.time(),',
        ["test_ego_can_message_eligible_work_and_the_worker_collects_it"],
        layer="ego_work_message (where from_role comes from)",
        note="sourcing the author from anything the caller supplies makes "
             "attribution a claim rather than a fact.",
    ),
    Mutation(
        "I48", "The external surface holds no control verb",
        "src/amoeba/scopes.py",
        '    "io_capabilities", "io_attach_input", "io_submit", "io_status",\n'
        '    "io_await", "io_output", "io_list", "io_result",\n)',
        '    "io_capabilities", "io_attach_input", "io_submit", "io_status",\n'
        '    "io_await", "io_output", "io_list", "io_result",\n'
        '    "admit_work", "cancel_work",  # MUTANT: control granted\n)',
        ["test_an_external_client_cannot_reach_a_control_verb",
         "test_the_adapter_and_the_credential_agree_on_the_surface"],
        layer="the external surface, in both places that define it",
        note="defended twice, deliberately. The adapter dispatches only what "
             "io_api.EXTERNAL_VERBS lists, and the credential can only reach "
             "what scopes.EXTERNAL_IO grants -- two independent lists, so one "
             "mistake does not open the door. Widening either alone leaves the "
             "other refusing, which is why negating this claim needs both.",
        also=[("src/amoeba/io_api.py",
               '    "io_attach_input", "io_result", "io_list",\n)',
               '    "io_attach_input", "io_result", "io_list",\n'
               '    "admit_work", "cancel_work",  # MUTANT\n)')],
    ),
    Mutation(
        "I48b", "The adapter dispatches nothing outside its own surface",
        "src/amoeba/http_api.py",
        "            if method not in external_allowed:",
        "            if False:  # MUTANT: dispatch anything the caller names",
        ["test_discovery_then_calling_the_exact_operator_verb_anyway"],
        layer="_external_rpc (the adapter's own allowlist)",
        note="the important half of this mutation is the *shape* of the "
             "failure. The call still fails -- the credential cannot name "
             "those verbs either -- but it fails downstream with a different "
             "code, and the test is written to reject exactly that: a route "
             "that exists and is rejected later is not an absent route.",
    ),
    Mutation(
        "I48c", "MCP holds the external credential, not the control token",
        "src/amoeba/mcp_api.py",
        '        self.token = read_or_create_token(cfg.scope_token_path("external_io"))',
        "        self.token = read_or_create_token(cfg.token_path)  # MUTANT",
        ["test_the_mcp_adapter_offers_only_the_io_surface"],
        layer="mcp Facade (which credential the adapter connects with)",
        note="restores the defect this work was built to fix: every MCP client "
             "reaching the full method table.",
    ),
    Mutation(
        "I48d", "Caller-supplied identity fields are discarded",
        "src/amoeba/http_api.py",
        '            for forged in ("client_id", "actor", "role", "caller", "scope",',
        '            for forged in ():  # MUTANT: honour what the caller claims\n'
        '                pass\n'
        '            for _ignored in ("client_id", "actor", "role", "caller", "scope",',
        ["test_spoofed_identity_fields_buy_nothing"],
        layer="_external_rpc (where identity is bound)",
        note="lets a client name itself, which turns 'my interactions' from a "
             "fact about the credential into a parameter anyone can set.",
    ),
    Mutation(
        "I48e", "A credential is required even on loopback",
        "src/amoeba/http_api.py",
        "            client_id = self._external_client()\n"
        "            if client_id is None:",
        "            client_id = self._external_client() or \"default\"\n"
        "            if False:",
        ["test_a_credential_is_required_even_on_loopback"],
        layer="_external_rpc (authentication)",
        note="treats binding to 127.0.0.1 as authentication, which it is not: "
             "it says nothing about other local processes or a hostile page in "
             "the user's browser.",
    ),
    Mutation(
        "I48f", "The console reaches state only through the Harness",
        "src/amoeba/operator_api.py",
        "    def _interaction_summary() -> dict[str, Any]:",
        "    import sqlite3  # MUTANT: a second path to the database\n"
        "    def _interaction_summary() -> dict[str, Any]:",
        ["test_the_dashboard_never_touches_the_database_or_filesystem"],
        layer="the console modules (whether they can reach past the Harness)",
        note="a console with its own database handle is a second writer with "
             "none of the invariants the first one enforces. The guarantee is "
             "about what the code *can* reach, so the test reads the source "
             "and the mutation puts the capability back.",
    ),
    Mutation(
        "I42c", "A proposal's bytes are preserved as evidence when it is made",
        "src/amoeba/harness_api.py",
        "        digest = mind.blobs.put(data)",
        "        digest = sha256_hex(data)  # MUTANT: hash but do not preserve",
        ["test_an_undecided_proposal_keeps_its_evidence_and_its_pending_status",
         "test_a_proposal_stays_promotable_after_its_sandbox_is_destroyed"],
        layer="artifact_propose (content-addressing at propose time)",
        note="now load-bearing rather than merely honest: without the "
             "blob there is nothing to promote once the scratch is gone, "
             "so the proposal state machine would be coupled to sandbox "
             "lifetime again by the back door.",
    ),
    Mutation(
        "I37d", "A hard link is not a way to reach a file outside a root",
        "src/amoeba/filespace.py",
        "        if exists and not self.cfg.allow_multiply_linked:\n"
        "            links = _link_count(final)\n"
        "            if links > 1:",
        "        if False:\n"
        "            links = _link_count(final)\n"
        "            if links > 1:",
        ["test_a_hard_link_into_the_root_cannot_be_used_to_read_outside_it"],
        layer="Filespace.resolve (the link-count check)",
        note="the leak this was found by. Path containment passes and is "
             "correct about the path; only the file's link count says the "
             "record has another name the root does not cover.",
    ),
    Mutation(
        "I37e", "One file has one identity key, whatever the caller called it",
        "src/amoeba/filespace.py",
        "            canonical = final.relative_to(root_path).as_posix()",
        '            canonical = "/".join(parts)  # MUTANT: caller spelling',
        ["test_case_variants_resolve_to_one_identity",
         "test_version_history_is_not_split_by_how_the_path_was_spelled"],
        layer="Filespace.resolve (where the identity key is derived)",
        note="not a containment failure: the path lands on exactly the right "
             "file. The damage is to the audit trail, where two spellings "
             "produce two version histories and a supersession under one is "
             "invisible from the other.",
    ),
    Mutation(
        "I38", "Prior content is preserved before an overwrite",
        "src/amoeba/harness_api.py",
        "            if sup.cfg.filespace.snapshot_before_overwrite:\n"
        "                existing = resolved.path.read_bytes()\n"
        "                prior_sha = mind.blobs.put(existing)\n"
        "                prior_bytes = len(existing)",
        "            if False:  # MUTANT: overwrite destroys\n"
        "                existing = resolved.path.read_bytes()\n"
        "                prior_sha = mind.blobs.put(existing)\n"
        "                prior_bytes = len(existing)",
        ["test_an_overwrite_supersedes_and_the_prior_version_is_restorable"],
        layer="file_write (the snapshot before the write)",
    ),
    Mutation(
        "I38b", "A delete keeps its content recoverable",
        "src/amoeba/harness_api.py",
        "        if resolved.path.is_file() and sup.cfg.filespace.snapshot_before_overwrite:",
        "        if False:  # MUTANT: delete destroys",
        ["test_a_delete_keeps_the_content_recoverable"],
        layer="file_delete (the snapshot before the unlink)",
    ),

    # -- Prompt Library: the versioned cognitive family tree ------------
    Mutation(
        "I49", "Runtime cannot establish a new top-level namespace",
        "src/amoeba/promptlib/store.py",
        "        if len(parts) == 1 and not self.versions(namespace):",
        "        if False:  # MUTANT: create_version may establish a root",
        ["test_runtime_cannot_invent_a_new_root"],
        layer="create_version (the existence guard)",
        also=[("src/amoeba/promptlib/model.py",
               "    if require_root and parts[0] not in ROOTS:",
               "    if False:  # MUTANT: any top-level name is a root")],
        note="Defended twice and independently: validate_namespace refuses an "
             "unknown root name, and create_version refuses a top-level "
             "namespace that does not exist yet. Both must go. Negating this "
             "must NOT negate I49b -- versioning an existing root stays legal "
             "under this mutation.",
    ),
    Mutation(
        "I49b", "An existing root may receive governed new versions",
        "src/amoeba/promptlib/store.py",
        "        if len(parts) == 1 and not self.versions(namespace):",
        "        if len(parts) == 1:  # MUTANT: no root may ever be versioned",
        ["test_runtime_can_propose_a_new_version_of_an_existing_root",
         "test_id_can_propose_root_doctrine_through_governance"],
        layer="create_version (the over-restriction this replaced)",
        note="The inverse claim. Reintroducing the blanket root refusal must "
             "kill these tests while leaving I49's test green, which is what "
             "makes the two guarantees genuinely separate.",
    ),
    Mutation(
        "I50", "An edited prompt file is a candidate, never an override",
        "src/amoeba/promptlib/bootstrap.py",
        '            state="candidate")',
        '            state="production_approved")\n'
        "        store.select(m, namespace=namespace,  # MUTANT: file overrides\n"
        '                     version_id=created["version_id"],\n'
        '                     purpose="production", selected_by=BOOTSTRAP_ACTOR)',
        ["test_edited_prompt_file_becomes_a_candidate_not_an_override"],
        layer="bootstrap.ingest (the delta branch)",
    ),
    Mutation(
        "I51", "A child pins an exact parent version",
        "src/amoeba/promptlib/store.py",
        '            ns, version = node["parent_namespace"], int(node["parent_version"])',
        '            ns = node["parent_namespace"]  # MUTANT: follow today\'s selection\n'
        "            _sel = self.selected(ns)\n"
        '            version = int(_sel["local_version"]) if _sel else int(node["parent_version"])',
        ["test_child_pins_an_exact_parent_version"],
        layer="pinned_chain (where the stored parent binding is walked)",
    ),
    Mutation(
        "I52", "A lineage reference resolves exactly or not at all",
        "src/amoeba/promptlib/store.py",
        "        if actual != ref.versions:",
        "        if False:  # MUTANT: reinterpret against today's ancestry",
        ["test_a_lineage_reference_resolves_to_one_thing_or_nothing"],
        layer="PromptStore.resolve_ref (where a requested lineage is verified)",
    ),
    Mutation(
        "I53", "Selection changes the next incarnation, not a living one",
        "src/amoeba/promptlib/resolver.py",
        "    def resolve_ref(self, ref: ProfileRef) -> ResolvedProfile:\n"
        '        """Resolve an explicitly requested historical lineage."""\n'
        "        return resolve_chain(self.store.resolve_ref(ref))",
        "    def resolve_ref(self, ref: ProfileRef) -> ResolvedProfile:\n"
        "        # MUTANT: re-resolve against whatever is selected now\n"
        "        return self.resolve_selected(ref.namespace)",
        ["test_selection_does_not_change_a_running_mind",
         "test_historical_lineage_still_resolves_after_the_tree_moves"],
        layer="Resolver.resolve_ref (resolution of an explicit lineage)",
    ),
    Mutation(
        "I54", "A cascade moves the pin and copies definitions unchanged",
        "src/amoeba/promptlib/cascade.py",
        '            prompt_text=source["prompt_text"],',
        '            prompt_text=source["prompt_text"] + " MUTANT",',
        ["test_cascade_copies_local_definitions_unchanged"],
        layer="cascade.apply_plan (the rebase itself)",
    ),
    Mutation(
        "I55", "Id may propose; approving is a verb Id does not have",
        "src/amoeba/scopes.py",
        "ID = ROLE_BASE + ID_SENSES + PROMPT_READ + ID_ONLY",
        "ID = ROLE_BASE + ID_SENSES + PROMPT_READ + ID_ONLY + (\n"
        '    "operator_prompt_state", "operator_prompt_select",\n'
        '    "operator_prompt_cascade")  # MUTANT: Id can approve',
        ["test_id_may_propose_but_the_approval_verbs_are_absent"],
        layer="scopes.ID (the method table Id's credential resolves to)",
    ),
    Mutation(
        "I56", "The prompt library is absent from the external surface",
        "src/amoeba/scopes.py",
        'EXTERNAL_IO = (\n    "io_capabilities",',
        'EXTERNAL_IO = (\n    "prompt_tree",  # MUTANT: external clients read cognition\n'
        '    "io_capabilities",',
        ["test_the_prompt_library_is_absent_from_the_external_surface"],
        layer="scopes.EXTERNAL_IO (the adapter's whole method table)",
        also=[("src/amoeba/io_api.py", "EXTERNAL_VERBS = (",
               'EXTERNAL_VERBS = (\n    "prompt_tree",')],
        note="Defended in depth: the adapter allowlist and the credential "
             "scope are independent lists, so both must be widened.",
    ),
    Mutation(
        "I57", "An incarnation binding freezes bytes, not a pointer",
        "src/amoeba/prompt_api.py",
        "                   resolved.prompt_sha256, resolved.config_sha256,",
        '                   "", resolved.config_sha256,  # MUTANT: no frozen digest',
        ["test_binding_freezes_resolved_bytes_not_a_pointer"],
        layer="bind_profile (the row written at birth)",
    ),

    # -- Role environment: profile / environment / turn input ----------
    Mutation(
        "I58", "A role is never offered a capability it cannot invoke",
        "src/amoeba/scopes.py",
        'EGO_MODEL_FACING = _SHARED_MODEL_FACING + (\n    "record_conclusion", "board_post",\n) + EGO_ONLY',
        'EGO_MODEL_FACING = _SHARED_MODEL_FACING + (\n    "record_conclusion", "board_post",\n'
        ') + EGO_ONLY + ID_ONLY  # MUTANT: advertise Id effectors to Ego',
        ["test_model_facing_verbs_are_a_subset_of_the_roles_scope",
         "test_role_environments_do_not_leak_across_roles"],
        layer="scopes.MODEL_FACING (what the manifest advertises)",
    ),
    Mutation(
        "I59", "A role can actually execute what its environment offers",
        "src/amoeba/roles.py",
        '            out = self.sup.call(\n                "role_tool_invoke", turn_id=self.current_turn_id, name=name,\n                arguments=self._sanitise(name, arguments))',
        '            out = {"accepted": False}  # MUTANT: never actually run it',
        ["test_ego_can_invoke_an_advertised_sense",
         "test_id_can_invoke_an_advertised_sense_and_effector"],
        layer="RoleProcess._invoke (the role tool loop's execution step)",
        note="The state before this work: roles parsed tool calls and reported "
             "them without executing, so a manifest would have advertised "
             "capabilities the model could not use.",
    ),
    Mutation(
        "I60", "Authority-shaped arguments cannot widen authority",
        "src/amoeba/roles.py",
        "        clean = {k: v for k, v in (arguments or {}).items()\n"
        "                 if k in declared and k not in AUTHORITY_ARGUMENTS}\n"
        "        for field in BOUND_IDENTITY_ARGUMENTS:\n"
        "            if field in declared:\n"
        "                clean[field] = self.role",
        "        clean = dict(arguments or {})  # MUTANT: trust the model",
        ["test_authority_shaped_arguments_cannot_widen_authority",
         "test_undeclared_arguments_are_dropped"],
        layer="RoleProcess._sanitise (where identity is bound, not accepted)",
    ),
    Mutation(
        "I61", "One turn sees one environment",
        "src/amoeba/roles.py",
        "        env_block = self._begin_turn(trigger or user_text[:200])",
        "        env_block = \"\"  # MUTANT: no environment is built for the turn",
        ["test_the_environment_is_built_once_per_turn",
         "test_the_environment_reaches_the_context_before_the_turn_input"],
        layer="RoleProcess._turn (the per-turn freeze)",
    ),
    Mutation(
        "I62", "A changing environment does not require rewriting doctrine",
        "src/amoeba/role_env.py",
        "        if not namespace.startswith(role + \".\"):",
        "        if True:  # MUTANT: no profile is ever discoverable",
        ["test_a_new_profile_becomes_visible_without_editing_doctrine"],
        layer="role_env._profile_inventory (what a role can discover)",
    ),
    Mutation(
        "I63", "Configuration cannot silently rewrite constitutional doctrine",
        "src/amoeba/config.py",
        "    raise ValueError(\n"
        "        f\"[{role}].{RETIRED_PROMPT_KEY} is no longer honoured:",
        "    section.pop(RETIRED_PROMPT_KEY, None)  # MUTANT: silently ignore\n"
        "    return\n"
        "    raise ValueError(\n"
        "        f\"[{role}].{RETIRED_PROMPT_KEY} is no longer honoured:",
        ["test_a_configured_system_prompt_is_refused_not_honoured"],
        layer="config._reject_retired_prompt (the loud refusal)",
        also=[("src/amoeba/roles.py",
               "        return (self.profile_prompt if self.profile_prompt is not None\n"
               "                else self.system_prompt)",
               "        base = (self.profile_prompt if self.profile_prompt is not None\n"
               "                else self.system_prompt)\n"
               "        return base + getattr(self.role_cfg, \"system_prompt\", \"\")")],
        note="Defended on both sides: the loader refuses the key and the role "
             "composes only what the library resolved. Reintroducing the "
             "append alone would not restore the bypass, because the field no "
             "longer loads.",
    ),
    Mutation(
        'I64', 'A persistent role runs one bounded turn at a time',
        'src/amoeba/mailbox.py',
        '    if open_turn(mind.db.conn, role) is not None:',
        '    if False:  # MUTANT: a second concurrent turn may open',
        ['test_a_role_cannot_have_two_turns_at_once'],
        layer='mailbox.claim, plus the unique index behind it',
        also=[('src/amoeba/store/db.py', 'CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_turn_per_role', 'CREATE INDEX IF NOT EXISTS ux_one_open_turn_per_role')],
        note='Defended twice: the claim path refuses and the database refuses, so the index is dropped to a non-unique one in the same mutation.',
    ),
    Mutation(
        'I65', 'Input arriving during a turn waits for the next boundary',
        'src/amoeba/mailbox.py',
        '           1 if ambient else 0, time.time(), m.prior_version + 1))\n    m.emit(EventKind.ROLE_TRIGGER_QUEUED, {',
        '           1 if ambient else 0, time.time(), m.prior_version + 1))\n    _open = m.conn.execute(  # MUTANT: inject into the running turn\n        "SELECT turn_id FROM role_turns WHERE role = ? AND status = \'running\'",\n        (role,)).fetchone()\n    if _open:\n        m.sql("UPDATE role_triggers SET status = \'claimed\', turn_id = ?"\n              " WHERE trigger_id = ?", (_open["turn_id"], trigger_id))\n        m.sql("UPDATE role_turns SET trigger_count = trigger_count + 1"\n              " WHERE turn_id = ?", (_open["turn_id"],))\n    m.emit(EventKind.ROLE_TRIGGER_QUEUED, {',
        ['test_input_arriving_during_a_turn_waits_for_the_next_one', 'test_queued_is_not_seen'],
        layer='mailbox.enqueue (the mutant attaches a new trigger to the open turn)',
        note="The honest negation of 'the bundle is frozen' is letting a later trigger join the turn already running, which is exactly what this does.",
    ),
    Mutation(
        'I66', 'Turn-end reasons are first class',
        'src/amoeba/mailbox.py',
        '    if stop_reason not in STOP_REASONS:',
        "    stop_reason = 'model_stop'  # MUTANT: collapse every ending\n    if stop_reason not in STOP_REASONS:",
        ['test_stop_reasons_are_recorded_distinctly', 'test_a_non_terminal_stop_schedules_a_continuation'],
        layer='mailbox.complete (what a turn records as its ending)',
    ),
    Mutation(
        'I67', 'The Harness continues an interrupted thought',
        'src/amoeba/mailbox.py',
        '    elif continuing:',
        '    elif False:  # MUTANT: a truncated thought is simply dropped',
        ['test_a_non_terminal_stop_schedules_a_continuation'],
        layer='mailbox.complete (the continuation decision)',
        note="Continuation is the Harness's call, not the model's: a thought cut off by an output ceiling cannot ask for its own continuation.",
    ),
    Mutation(
        'I68', 'The continuation chain is bounded',
        'src/amoeba/mailbox.py',
        '    exhausted = depth >= max(0, int(max_continuations))',
        '    exhausted = False  # MUTANT: continue forever',
        ['test_the_continuation_chain_is_bounded'],
        layer='mailbox.complete (the bound on repeated continuation)',
        note='Found by running it: every turn truncated, each scheduled a successor, and the organism burned its context until inference refused the prompt.',
    ),
    Mutation(
        'I69', 'A role that dies mid-turn does not swallow its inputs',
        'src/amoeba/mailbox.py',
        '            m.sql("UPDATE role_triggers SET status = \'queued\', bundle_id = NULL,"\n                  " turn_id = NULL, claimed_at = NULL, answer_status = NULL,"\n                  " answer_sha256 = NULL, answered_by_turn = NULL"\n                  " WHERE trigger_id = ?", (r["trigger_id"],))\n            requeued.append(r["trigger_id"])',
        '            m.sql("UPDATE role_triggers SET status = \'consumed\'"  # MUTANT\n                  " WHERE trigger_id = ?", (r["trigger_id"],))\n            requeued.append(r["trigger_id"])',
        ['test_a_role_that_dies_mid_turn_does_not_swallow_its_inputs'],
        layer='mailbox.abandon (recovery of claimed-but-unconsumed triggers)',
    ),
    Mutation(
        'I70', 'A turn records exactly what caused it',
        'src/amoeba/mailbox.py',
        '    bundle_blob = m.put_json(body, schema="amoeba.trigger_bundle/1")',
        '    bundle_blob = None  # MUTANT: the bundle is not recoverable',
        ['test_a_turn_records_what_caused_it'],
        layer='mailbox.claim (content-addressing the exact bundle)',
        note='A digest whose content cannot be recovered is not provenance.',
    ),
    Mutation(
        'I71', 'The scheduler is substrate, never a cognitive component',
        'src/amoeba/scopes.py',
        'EGO_MODEL_FACING = _SHARED_MODEL_FACING + (\n    "record_conclusion", "board_post",\n) + EGO_ONLY',
        'EGO_MODEL_FACING = _SHARED_MODEL_FACING + (\n    "record_conclusion", "board_post",\n    "role_claim_turn", "role_enqueue_trigger",  # MUTANT: schedule yourself\n) + EGO_ONLY',
        ['test_the_turn_verbs_are_not_offered_to_the_model'],
        layer='scopes (what the role model is offered)',
        note='A mind that could claim its own next turn would be scheduling itself.',
    ),
    Mutation(
        "I72", "Cognition happens only in claimed turns",
        "src/amoeba/supervisor_api.py",
        '        result = sup.client("ego").call("publish_snapshot", operation_id=operation_id)',
        '        result = sup.client("ego").call("converse", message="bypass")  # MUTANT',
        ["test_no_verb_generates_cognition_outside_the_mailbox"],
        layer="the verbs that reach a role process (the mailbox bypass)",
        note="Reintroduces the pattern that existed before the turn model: a "
             "verb making a role think outside a claimed turn, against the "
             "session a turn may already hold.",
    ),
    Mutation(
        "I73", "A role is never wedged by a turn it did not close",
        "src/amoeba/mailbox.py",
        '    if role is not None:',
        "    if False:  # MUTANT: recovery cannot target one role",
        ["test_recovery_can_target_one_role",
         "test_a_turn_nobody_closed_does_not_wedge_the_role"],
        layer="mailbox.recover / expire_stale_turns",
        also=[('    cutoff = time.time() - max(1.0, float(max_seconds))',
               "    cutoff = 0.0  # MUTANT: no turn is ever stale")],
        note="Two defences, because there are two ways to leave a turn "
             "open: the process dies and is restarted, or it hangs and is "
             "not. Both must go. The wedge this prevents presents as a "
             "healthy role that never thinks again.",
    ),
    Mutation(
        "I74", "A role cannot forge attribution in a mailbox",
        "src/amoeba/scopes.py",
        '    "role_claim_turn", "role_complete_turn", "role_abandon_turn",',
        '    "role_claim_turn", "role_complete_turn", "role_abandon_turn",\n    "role_enqueue_trigger",  # MUTANT: a role picks its own source',
        ["test_a_role_cannot_forge_attribution_in_a_mailbox"],
        layer="scopes.ROLE_BASE (what a role credential resolves to)",
        note="Ego holding this could queue an operator-attributed "
             "instruction into Id's cognition.",
    ),
    Mutation(
        "I75", "A turn that is no longer running cannot act",
        "src/amoeba/turn_api.py",
        '        if row["status"] != "running":',
        "        if False:  # MUTANT: a swept turn may still act",
        ["test_the_harness_refuses_a_capability_from_a_turn_that_is_not_running",
         "test_a_turn_that_is_no_longer_running_cannot_act"],
        layer="role_tool_invoke (the turn fence on every capability)",
        note="Being refused at commit was never enough: the result could "
             "not land, but the side effects could.",
    ),
    Mutation(
        "I76", "A role reads the request, not a preview of it",
        "src/amoeba/mailbox.py",
        '        body = trigger_body(t, blobs)',
        '        body = t.get("summary") or ""  # MUTANT: preview only',
        ["test_a_turn_shows_the_request_not_a_preview",
         "test_an_oversized_body_says_that_it_was_truncated"],
        layer="mailbox.render_bundle (what the model is handed)",
    ),
    Mutation(
        "I77", "An adverse audit is never recorded as a favourable one",
        "src/amoeba/supervisor_api.py",
        '            if len(found) == 1:',
        "                found = {v for v in VERDICTS if v in candidate}\n                if found:  # MUTANT: substring match, first wins",
        ["test_an_adverse_audit_is_not_recorded_as_a_favourable_one"],
        layer="_parse_audit (how a verdict is read)",
        note="`supported` is a substring of `unsupported`, so the adverse verdict inverts and the disagreement is never opened.",
    ),
    Mutation(
        "I78", "An effector whose target is a role stays callable",
        "src/amoeba/id_api.py",
        '    def id_propose_prompt(*, target_role: str, prompt: str, rationale: str,',
        "    def id_propose_prompt(*, role: str, prompt: str, rationale: str,"
        "  # MUTANT: the target collides with the authority argument",
        ["test_id_can_actually_call_the_effectors_that_name_a_target"],
        layer="id_api (the parameter naming the target role)",
    ),
    Mutation(
        "I79", "Complete means answered",
        "src/amoeba/io_api.py",
        '            settled = state in ("completed", "incomplete")',
        "            settled = True  # MUTANT: call it complete regardless",
        ["test_an_unanswered_interaction_is_not_reported_complete",
         "test_an_answered_interaction_carries_its_answer"],
        layer="io_api._run (when an interaction is called complete)",
    ),
    Mutation(
        "I80", "An answer belongs to the request that asked for it",
        "src/amoeba/mailbox.py",
        """        parent = conn.execute(
            "SELECT parent_turn FROM role_turns WHERE turn_id = ?",
            (current,)).fetchone()
        current = parent["parent_turn"] if parent else None""",
        """        current = None  # MUTANT: the answer belongs to the turn, not the chain""",
        ["test_a_continued_thought_answers_the_request_that_started_it"],
        layer="mailbox.awaiting_answer (walking back to the request that asked)",
        note="A thought split across continuation turns still answers the question that started it; stopping at the current turn leaves the original request waiting forever.",
    ),
    Mutation(
        "I81", "No result satisfies an interaction merely because triggers shared a turn",
        "src/amoeba/mailbox.py",
        """        if t["expects_answer"]:
            if request_taken:
                deferred.append(t)
                continue
            request_taken = True""",
        """        if t["expects_answer"]:
            request_taken = True  # MUTANT: two questions, one answer""",
        ["test_two_requests_never_share_one_turn",
         "test_a_continuation_does_not_adopt_a_new_request"],
        layer="mailbox.claim (admitting at most one answer-bearing request)",
        note="The evil twin: both requests are marked answered by whichever reply the turn produces, so one caller is handed a reply to somebody else's question.",
    ),
    Mutation(
        "I82", "Reply routing and information ownership are different questions",
        "src/amoeba/mailbox.py",
        """        if turn_lineage and t["lineage"] != turn_lineage:""",
        """        if False:  # MUTANT: any evidence rides into any turn""",
        ["test_evidence_from_another_interaction_does_not_ride_along",
         "test_unowned_evidence_does_not_default_to_everyone"],
        layer="mailbox.claim (scoping supporting evidence to the turn's lineage)",
        note="The evil cousin: nobody is owed a reply for a work result, so reply routing alone lets one interaction's evidence inform another's answer with every other rule still satisfied.",
        also=[("""        if t["ambient"] or t["kind"] == "continuation":""",
               """        if True:  # MUTANT: ambience is not a decision any more""")],
    ),
    Mutation(
        "I83", "A thought that ends without an answer says so",
        "src/amoeba/mailbox.py",
        '                sha, state = None, "unanswerable"',
        '                sha, state = None, "answered"  # MUTANT: silence is a reply',
        ["test_a_thought_that_ends_without_an_answer_says_so"],
        layer="mailbox.complete (what a terminal stop with no output records)",
        note="Reporting silence as an answer is the failure this tier exists to remove; the caller stops waiting and believes nothing was the reply.",
    ),
    Mutation(
        "I84", "An existing database gains the columns a release adds",
        "src/amoeba/store/db.py",
        """        self._migrate_turn_ownership()
        self.conn.executescript(SCHEMA_SQL)""",
        """        self.conn.executescript(SCHEMA_SQL)  # MUTANT: no migration""",
        ["test_a_database_from_the_previous_release_gains_the_ownership_columns"],
        layer="Database.initialize (upgrading a database written by an older release)",
        note="CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so without this the organism starts, heartbeats, and fails on its first mailbox read.",
    ),
    Mutation(
        "I85", "Work delegated during a turn is accountable to that turn",
        "src/amoeba/turn_api.py",
        '                args["operation_id"] = row["operation_id"]',
        '                pass  # MUTANT: the turn keeps its operation to itself',
        ["test_a_turns_operation_reaches_what_the_role_does"],
        layer="role_tool_invoke (supplying the turn's operation to what it does)",
        note="Defended in two places, so both come out: the injection here and the operation a continuation inherits.",
        also=[("src/amoeba/mailbox.py",
               '            operation_id=row["operation_id"],',
               '            operation_id=None,  # MUTANT: delegated anonymously')],
    ),
    Mutation(
        "I86", "A result the model cannot see whole says so",
        "src/amoeba/tools.py",
        '    return {"text": text[:budget] + notice, "truncated": True,',
        '    return {"text": text[:budget], "truncated": True,  # MUTANT: cut in silence',
        ["test_a_truncated_tool_result_says_so_and_names_where_the_rest_is",
         "test_the_neuocyte_feeds_back_the_bounded_text_it_was_given"],
        layer="tools.bounded_tool_result (saying that a result was cut)",
        note="Defended in two places, so both come out: the notice itself, and the cognitive path rendering the bounded text rather than re-serializing the whole result.",
        also=[("src/amoeba/neuocyte.py",
               '            body = res.get("result_text")',
               '            body = None  # MUTANT: ignore what the Harness bounded')],
    ),
    Mutation(
        "I87", "Evidence is never pruned; the working set is",
        "src/amoeba/retention.py",
        '    for trigger_id in doomed["triggers"]:',
        '    m.sql("DELETE FROM events WHERE seq = (SELECT MIN(seq) FROM events)")  # MUTANT\n    for trigger_id in doomed["triggers"]:',
        ["test_pruning_touches_no_evidence_table",
         "test_the_event_chain_still_verifies_after_pruning"],
        layer="retention.prune (what housekeeping is allowed to delete)",
        note="The honest negation of 'evidence is never pruned' is pruning some, so the mutant deletes old events -- which frees almost nothing and destroys the ability to verify the rest of the chain.",
    ),
    Mutation(
        "I88", "Nothing still in use is forgotten",
        "src/amoeba/retention.py",
        '        "   AND NOT (expects_answer = 1 AND answer_status IS NULL)",',
        '        "   AND 1 = 1",  # MUTANT: old enough is reason enough',
        ["test_a_request_still_owed_an_answer_is_never_old_enough"],
        layer="retention.prunable (deciding what may be forgotten)",
        note="Defended in three places and all three come out: a request still owed an answer, a turn still running, and a turn a continuation still points at. Two of these tests originally passed with their guard removed -- each was being saved by the referencing-trigger check instead of the one it was named after.",
        also=[("        if children:\n            continue\n",
               "        if False:\n            continue\n"),
              ("\" WHERE status != 'running' AND COALESCE(finished_at, started_at) < ?\"",
               "\" WHERE COALESCE(finished_at, started_at) < ?\"")],
    ),
    Mutation(
        "I89", "Ego chooses what crosses the external boundary, never whose",
        "src/amoeba/supervisor_api.py",
        '        if row is None or not mine or row["interaction_id"] != mine:',
        "        if row is None:  # MUTANT: any client's attachment will do",
        ["test_ego_cannot_read_an_attachment_from_another_request"],
        layer="ego_read_attachment (scoping an input to the turn's interaction)",
        note="Verified live rather than against a table: the claim is about what the dispatcher refuses when asked directly for another client's input id.",
    ),
    Mutation(
        "I90", "Batching changes throughput, never outcomes",
        "src/amoeba/inference_service.py",
        "                # A backend that cannot batch is not an error; it is a backend\n                # that cannot batch. Everything still runs, one at a time.\n                pass",
        "                raise  # MUTANT: a backend that cannot batch fails its callers",
        ["test_a_backend_that_cannot_batch_still_serves_everyone",
         "test_one_broken_session_does_not_fail_the_others"],
        layer="InferenceService._run_group (what happens when a batch cannot be served)",
        note="Defended twice and both come out: the fallback for a backend without batching, and the per-request isolation that keeps one bad session from failing everyone decoded beside it.",
        also=[("                self.log.debug(\"batched generate failed; running the group \"\n                               \"one at a time\", exc_info=True)",
               "                raise  # MUTANT: one caller's failure becomes everyone's")],
    ),
    Mutation(
        "I91", "A specialisation nothing can be born into does not exist",
        "src/amoeba/neuocyte.py",
        '        candidates = [f"{base}.{wanted}", base] if wanted else [base]',
        "        candidates = [base]  # MUTANT: the specialisation is ignored",
        ["test_a_requested_specialisation_is_bound",
         "test_maintenance_work_specialises_under_id"],
        layer="Neuocyte._bind_profile (which profile a worker is born with)",
        note="Defended twice and both come out: trying the specialisation at all, and recording the fallback when it is unavailable. Silence would make a specialisation that stopped applying indistinguishable from one nobody requested.",
        also=[("        self._report_profile(work_id, bound, fallback)",
               "        pass  # MUTANT: fall back in silence")],
    ),
    Mutation(
        "I92", "A role is woken by what happens to work it originated, and by nothing else",
        "src/amoeba/waking.py",
        '    if author is not None and author == owner:\n        return None',
        "    pass  # MUTANT: a role wakes itself by posting about its own work",
        ["test_a_role_posting_about_its_own_work_does_not_wake_itself"],
        layer="waking.wake_owner_of_work (who a recorded event concerns)",
        note="Defended twice and both come out: the ownership rule that decides whether anybody is woken at all, and the author check that stops a role notifying itself in a spiral.",
        also=[('    if not row or row.get("origin_actor") not in mailbox.ROLES:',
               '    if not row:  # MUTANT: work nobody owns wakes somebody anyway')],
    ),
    Mutation(
        "I93", "Context is reclaimed by dropping finished work, not by cutting the middle out",
        "src/amoeba/mailbox.py",
        '        if row["lineage"] in owed:\n            continue',
        "        if False:  # MUTANT: evict a thought that is still owed an answer\n            continue",
        ["test_a_lineage_still_owed_an_answer_is_never_evictable",
         "test_a_turn_with_no_measured_span_is_never_evictable"],
        layer="mailbox.settled_spans (which context is finished with)",
        note="Defended in several places and three come out together: the lineage still owed an answer, the span that was never measured, and the span measured in a previous session whose offsets now describe live context.",
        also=[('"   AND token_start IS NOT NULL AND token_end IS NOT NULL"\n            "   AND token_end > token_start"\n',
               ""),
              ('" WHERE role = ? AND session_handle = ? AND status != \'running\'"',
               '" WHERE role = ? AND ? IS NOT NULL AND status != \'running\'"')],
    ),
    Mutation(
        "I94", "The recorded session handle follows the live one",
        "src/amoeba/homeostasis.py",
        "            self.mind.work.set_session_handle(",
        "            _skipped = (lambda **kw: None)(  # MUTANT: leave it stale",
        ["test_a_second_rejuvenation_does_not_resurrect_what_the_first_dropped",
         "test_a_rejuvenated_role_keeps_its_identity_and_gains_a_new_handle"],
        layer="ContextHomeostasis.rejuvenate (recording the session that now exists)",
        note="Only the second rejuvenation can see this: the first reads a handle that is still correct. Defended alongside the span filter, which is what stops the stale coordinates from being readable at all.",
        also=[("src/amoeba/mailbox.py",
               '" WHERE role = ? AND session_handle = ? AND status != \'running\'"',
               '" WHERE role = ? AND ? IS NOT NULL AND status != \'running\'"')],
    ),
    Mutation(
        "I95", "What became of an attempt travels with what it said",
        "src/amoeba/store/board_repo.py",
        "        if current is not None and author_token < current:",
        "        if False:  # MUTANT: join on the work item, not the attempt",
        ["test_a_later_attempt_s_success_does_not_launder_a_fenced_attempt_s_post"],
        layer="BoardRepo._work_provenance (whose fate a post reports)",
        note="The mutant is the plausible wrong implementation, not the absent one: joining the post to `work_items.status` passes every other test here and still renders a fenced attempt's finding as `done`. A work item can fail attempt 1 and complete on attempt 2, and the naive join launders the dead attempt's post through the later success. Defended alongside attaching the provenance at all, and alongside drawing 'unfinished' at the right line -- calling everything that is not done unfinished would discount a finding whose retry may yet corroborate it.",
        also=[('        item.update(self._work_provenance(item.get("work_id"),\n                                          item.get("author_fencing_token")))',
               "        pass  # MUTANT: no provenance at all"),
              ('    WORK_IN_FLIGHT = ("queued", "leased")',
               '    WORK_IN_FLIGHT = ()  # MUTANT: a retry cannot corroborate'),
              ('                             rendered={p["post_id"]: p for p in posts})',
               "                             rendered=None)  # MUTANT: forget what was shown"),
              ('        unfinished_support = [f["post_id"] for f in support_provenance\n                              if f["attempt_unfinished"]]',
               '        unfinished_support = []  # MUTANT: a dead supporter looks alive')],
    ),
    Mutation(
        "I96", "A session carries its own allowance, and carries it unchanged",
        "src/amoeba/arbiter.py",
        "        ceiling = int(budget_tokens) if budget_tokens else cfg.max_prompt_tokens",
        "        ceiling = cfg.max_prompt_tokens  # MUTANT: one global ceiling for everyone",
        ["test_ids_ceiling_is_independent_of_egos",
         "test_a_role_session_is_bounded_by_its_own_ceiling_not_a_global_one"],
        layer="Arbiter.clamp_inference (whose ceiling bounds a generation)",
        note="The mutant is the old design, not an absent one: a single global scalar passes anything that does not compare two actors against each other, which is why the defending tests give Ego and Id the same occupancy and require different answers.",
    ),
    Mutation(
        "I97", "What a session may think and what it costs are different questions",
        "src/amoeba/backends/llama_engine.py",
        '        return (self.private_tokens if self.budget_basis == "private_growth"\n                else self.n_past)',
        "        return self.n_past  # MUTANT: the basis is ignored",
        ["test_a_forked_worker_is_not_charged_for_the_prefix_it_inherited",
         "test_a_shared_prefix_is_charged_once_but_a_recomputed_one_in_full",
         "test_the_service_judges_a_forked_worker_on_its_growth_not_its_total"],
        layer="SessionState.budgeted_tokens (which tokens an allowance counts)",
        note="Defended in two places and both come out: the basis that decides what counts against the allowance, and the charge that decides what the pool is told. Collapsing the charge alone makes a recomputed prefix free; collapsing the basis alone refuses a forked worker on its first generate.",
        also=[("        return self.private_tokens if self.shares_prefix else self.n_past",
               "        return self.private_tokens  # MUTANT: a recomputed prefix is free")],
    ),
    Mutation(
        "I98", "Admission spends a pool it has measured",
        "src/amoeba/arbiter.py",
        '        kv = self.kv_admission(work_class=work_class, snapshot=snapshot)',
        '        kv = {"admit": True}  # MUTANT: admit without looking at the pool',
        ["test_admission_refuses_when_the_pool_has_no_room"],
        layer="Arbiter.admit (consulting the KV pool before starting work)",
        note="This check did not exist before the pass that added it; the mutant restores the previous behaviour, in which admission read no token figure at all and overcommit was prevented only by the configured numbers happening to be small.",
    ),
    Mutation(
        "I99", "Only the discretionary turn yields to pressure",
        "src/amoeba/supervisor.py",
        "        if waited >= ceiling:",
        "        if False:  # MUTANT: defer the review for as long as pressure lasts",
        ["test_a_deferred_heartbeat_eventually_runs_anyway"],
        layer="Supervisor._heartbeat_deferred_for_pressure (the bound on deferral)",
        note="Defended in three places and all three come out: the ceiling that guarantees the review still happens, the reset that keeps the clock measuring one episode, and the rule that unknown pressure proceeds. The ceiling is the one that matters most -- without it, sustained pressure silently ends Id's self-examination at the moment it is most worth having.",
        also=[('            self._heartbeat_deferred_since.pop(role, None)\n            return None\n\n        since',
               "            return None\n\n        since"),
              ("        if level is None or level not in PRESSURE_LEVELS:\n            return None",
               '        if level is None:\n            level = "critical"  # MUTANT: unknown means pressure')],
    ),
    Mutation(
        "I100", "A claim can stop being made",
        "src/amoeba/ego_api.py",
        '        if row.get("produced_by") != "ego":',
        "        if False:  # MUTANT: withdraw anybody's claim",
        ["test_only_the_author_may_withdraw_a_claim"],
        layer="ego_withdraw_conclusion (who may stop making a claim)",
        note="Defended alongside the standing guard: an auditor able to edit the record it audits is not an auditor, and a claim already replaced must not acquire a second account of how it ended.",
        also=[("src/amoeba/store/memory_repo.py",
               '            if row["standing"] != "active":',
               "            if False:  # MUTANT: withdraw anything, twice")],
    ),
    Mutation(
        "I101", "A disagreement ends because the record moved, not because somebody said so",
        "src/amoeba/store/memory_repo.py",
        '            if live is not None:',
        "            if False:  # MUTANT: open a rival dispute about the same claim",
        ["test_a_repeated_contradiction_does_not_open_a_second_dispute"],
        layer="MemoryRepo.open_disagreement / withdraw_conclusion (one live dispute, ended by the record)",
        note="Defended twice and both come out: the deduplication that stops one unresolved issue becoming twenty rows, and the closure that stops a dispute outliving the claim it is about.",
        also=[('            self._close_disputes_in(\n                m, subject_kind="conclusion", subject_id=conclusion_id,\n                resolution="retracted", actor=actor,\n                detail=f"the claim was withdrawn by {actor}")',
               "            pass  # MUTANT: the dispute outlives the claim")],
    ),
    Mutation(
        "I102", "An adjudicator changing its mind is not the ground moving",
        "src/amoeba/supervisor_api.py",
        "    if opened_on and opened_on == basis:",
        "    if False:  # MUTANT: a reversal settles it, whatever the evidence",
        ["test_a_reversal_on_an_unchanged_basis_does_not_settle_the_dispute",
         "test_a_reversal_on_an_unchanged_basis_is_recorded_not_discarded"],
        layer="_settle_if_the_ground_moved (whether a supporting audit closes a dispute)",
        note="The mutant is the tempting wrong implementation: close whenever a later audit says supported. It passes every other test here, and lets Id open a dispute and then self-certify it away against identical evidence -- which is what separating Ego from Id exists to prevent.",
    ),
    Mutation(
        "I103", "A refused request does not poison the next one",
        "src/amoeba/http_api.py",
        "            self._drain()",
        "            pass  # MUTANT: the unread body stays in the socket",
        ["test_a_refused_request_does_not_poison_the_next_one",
         "test_several_refusals_in_a_row_leave_the_connection_usable"],
        layer="Handler._send (whether a response consumes the body it did not read)",
        note="Defended twice, because the fix has two halves and each fails alone. Removing the drain breaks the first refusal; leaving the drain but not clearing its state per request breaks the second one instead, since a single handler instance serves the whole connection. The second mutant is the more instructive: it looks fixed until somebody mistypes a token twice.",
        also=[("src/amoeba/http_api.py",
               "            self._body_consumed = False",
               "            pass  # MUTANT: last request's state decides this one")],
    ),
    Mutation(
        "I104", "The operator console is valid JavaScript",
        "src/amoeba/dashboard.py",
        " +\n",
        "\n",
        ["test_the_dashboard_script_has_no_python_string_concatenation",
         "test_the_dashboard_script_parses_as_javascript"],
        layer="DASHBOARD_HTML (whether the embedded script parses at all)",
        note="The mutant restores the Python habit that broke the console: drop one `+` between wrapped string literals. It is invisible in Python, fatal in JavaScript, and takes the whole page down rather than the one line it appears on.",
    ),
    Mutation(
        "I105", "The Operator contributes to the room and cannot speak as either mind in it",
        "src/amoeba/operator_api.py",
        '            for role in ("ego", "id"):',
        '            for role in ("ego",):  # MUTANT: only one mind hears it',
        ["test_the_operator_speaks_to_the_room_and_both_minds_hear_it",
         "test_an_author_nobody_can_be_held_to_is_refused"],
        layer="operator_backchannel / Room.post (who an Operator message reaches, and who it can claim to be)",
        note="The mutant is the shape the verb had before: one recipient, chosen by the caller. It passes every other test here, and it is how a room quietly becomes a private wire -- one mind acting on something the other never heard. The second anchor removes the authorship check, so an entry can be recorded that nobody can be held to.",
        also=[("src/amoeba/room.py",
               "        if author not in AUTHORS:",
               "        if False:  # MUTANT: any author will do")],
    ),
    Mutation(
        "I106", "The live view may forget; the record may not",
        "src/amoeba/room.py",
        "        self._entries: deque[dict[str, Any]] = deque(maxlen=maxlen)",
        "        self._entries: deque[dict[str, Any]] = deque()  # MUTANT: unbounded",
        ["test_the_room_is_bounded",
         "test_a_room_message_is_durably_recorded_even_though_the_view_is_not"],
        layer="Room (a viewport that is allowed to forget, and must not be the record)",
        note="An unbounded viewport is a leak wearing a feature's clothes, and a viewport that never forgets starts being treated as history. The second test holds the other end: durable causal recording is not this buffer's job and must survive without it.",
        also=[("src/amoeba/supervisor_api.py",
               '        if from_role in ("ego", "id"):',
               "        if False:  # MUTANT: peer messages never reach the room")],
    ),
    Mutation(
        "I107", "An answer belongs to its interaction, whole",
        "src/amoeba/mailbox.py",
        "        chain.append(current)\n        if current == admitting_turn:\n            break",
        "        chain.append(current)\n        break  # MUTANT: only the turn that ended it",
        ["test_an_answer_spanning_three_turns_arrives_whole_and_in_order",
         "test_each_piece_appears_exactly_once"],
        layer="mailbox.assemble_answer (which turns an answer is made of)",
        note="The mutant is the live defect exactly: the answer is whichever turn ended the thought. A reply cut off three times arrived as its last quarter, opening 'Continuing from where the previous analysis left off'. Separately verified to die for dropping the first piece, duplicating one, reversing their order, answering on every bounded turn, and assembling from the wrong interaction.",
    ),
    Mutation(
        "I108", "Only a concluded thought is a finished answer",
        "src/amoeba/mailbox.py",
        '                state = "answered" if finished else "incomplete"',
        '                state = "answered"  # MUTANT: stopped is as good as finished',
        ["test_running_out_of_continuations_is_incomplete_not_complete",
         "test_running_out_of_continuations_reaches_the_client_as_incomplete"],
        layer="mailbox.complete / _await_turn (what a terminal answer is called)",
        note="Defended at both ends: where the answer is recorded, and where it is read back for a caller. Either one alone turns a thought the continuation limit cut off into 'the reply', which is what the limit used to do while keeping only the last fragment.",
        also=[("src/amoeba/supervisor_api.py",
               '                return {"status": "completed" if finished else "incomplete",',
               '                return {"status": "completed",  # MUTANT')],
    ),
    Mutation(
        "I109", "A continuation resumes the message it continues, only where the session holds it",
        "src/amoeba/mailbox.py",
        "    if parent and len(admitted) == 1:",
        "    if parent:  # MUTANT: resume even with evidence aboard",
        ["test_a_continuation_carrying_evidence_asks_visibly_instead",
         "test_a_three_turn_answer_reaches_the_operator_whole"],
        layer="mailbox.claim / RoleProcess._turn (whether a continuation resumes or asks)",
        note="Both halves: the Harness must not offer a resume when evidence rides with the continuation, since a resumed generation has nowhere to show it; and the role must actually resume when offered, or every piece of a long answer opens with its own preamble and pays for the environment again. The second anchor makes the role refuse every resume.",
        also=[("src/amoeba/roles.py",
               "        resumed = environment is not None and self._can_resume(resume)",
               "        resumed = False  # MUTANT: always ask visibly")],
    ),
    Mutation(
        "I110", "Each mind has its own output ceiling, and nothing silently lowers it",
        "src/amoeba/promptlib/model.py",
        '    "ego": 3072,',
        '    "ego": 384,  # MUTANT: the hardcoded number Ego actually ran with',
        ["test_the_fallback_ceilings_match_the_shipped_headers",
         "test_the_backstop_clamps_no_governed_ceiling"],
        layer="FALLBACK_OUTPUT_CEILINGS / ArbiterConfig.max_completion_tokens",
        note="The two ways Ego's ceiling was fictional: a fallback of 384 that applied because the Ego root stated none, and a global 512 backstop that would have clamped a governed 3072. Separately verified to die for restoring the 4000-character cut on an answer.",
        also=[("src/amoeba/config.py",
               "    max_completion_tokens: int = 3072",
               "    max_completion_tokens: int = 512  # MUTANT")],
    ),
    Mutation(
        "I111", "A generation is admitted only if its whole allowance fits",
        "src/amoeba/arbiter.py",
        "        if prompt_tokens + capped > ceiling:",
        "        if prompt_tokens > ceiling:  # MUTANT: reserve nothing",
        ["test_a_generation_is_admitted_only_if_its_allowance_fits"],
        layer="Arbiter.clamp_inference (whether the room a ceiling promises exists)",
        note="The mutant is the old admission rule: check the prompt alone. A session a few hundred tokens short of its budget is then admitted with a 3072-token allowance it cannot hold, and rejuvenation happens after the overrun instead of before the turn.",
    ),
]


def run_tests(names: list[str]) -> tuple[bool, str]:
    expr = " or ".join(names)
    r = subprocess.run(
        [PY, "-m", "pytest", "tests", "-q", "-p", "no:randomly", "-x",
         "--timeout=300", "-k", expr],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    return r.returncode == 0, (r.stdout or "")[-400:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated invariant ids")
    args = ap.parse_args()
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    muts = [m for m in MUTATIONS if not wanted or m.invariant in wanted]

    backup = Path(tempfile.mkdtemp(prefix="inv_backup_"))
    touched = {m.path for m in muts}
    for m in muts:
        for entry in m.also:
            if len(entry) == 3:
                touched.add(entry[0])
    for rel in touched:
        dest = backup / rel.replace("/", "__")
        shutil.copy2(ROOT / rel, dest)

    results = []
    try:
        for m in muts:
            target = ROOT / m.path
            original = target.read_text(encoding="utf-8")
            if m.old not in original:
                results.append((m, "SKIP", "mutation anchor not found"))
                print(f"SKIP {m.invariant}: anchor not found in {m.path}")
                continue
            mutated = original.replace(m.old, m.new, 1)
            # (path, old, new) entries edit another file; restore every one of
            # them afterwards, including on failure.
            elsewhere: dict[str, tuple[str, str]] = {}
            for entry in m.also:
                if len(entry) == 3:
                    rel, old, new = entry
                    other = ROOT / rel
                    text = elsewhere.get(rel, (other.read_text(encoding="utf-8"),))[0] \
                        if rel in elsewhere else other.read_text(encoding="utf-8")
                    if old not in text:
                        mutated = None
                        results.append((m, "SKIP",
                                        f"secondary anchor not found in {rel}"))
                        print(f"SKIP {m.invariant}: secondary anchor not found "
                              f"in {rel}")
                        break
                    elsewhere[rel] = (text, text.replace(old, new, 1))
                else:
                    old, new = entry
                    if old not in mutated:
                        results.append((m, "SKIP",
                                        f"secondary anchor not found: {old[:40]}"))
                        print(f"SKIP {m.invariant}: secondary anchor not found")
                        mutated = None
                        break
                    mutated = mutated.replace(old, new, 1)
            if mutated is None:
                continue
            target.write_text(mutated, encoding="utf-8")
            for rel, (_orig, new_text) in elsewhere.items():
                (ROOT / rel).write_text(new_text, encoding="utf-8")
            try:
                passed, tail = run_tests(m.tests)
            finally:
                target.write_text(original, encoding="utf-8")
                for rel, (orig_text, _new) in elsewhere.items():
                    (ROOT / rel).write_text(orig_text, encoding="utf-8")
            if passed:
                results.append((m, "WEAK", tail))
                print(f"WEAK {m.invariant:8s} tests PASSED with the guarantee removed "
                      f"-> {m.tests}")
            else:
                results.append((m, "GOOD", ""))
                print(f"GOOD {m.invariant:8s} {m.claim}")
    finally:
        for rel in touched:
            shutil.copy2(backup / rel.replace("/", "__"), ROOT / rel)
        shutil.rmtree(backup, ignore_errors=True)
        print("\nsource restored")

    good = [r for r in results if r[1] == "GOOD"]
    weak = [r for r in results if r[1] == "WEAK"]
    skipped = [r for r in results if r[1] == "SKIP"]
    print(f"\n{'='*70}")
    print(f"{len(good)} invariants defended, {len(weak)} WEAK, {len(skipped)} skipped")
    for m, _, _ in weak:
        print(f"  WEAK {m.invariant}: {m.claim}")
        print(f"       layer: {m.layer}")
        if m.note:
            print(f"       note: {m.note}")
    return 1 if weak else 0


if __name__ == "__main__":
    raise SystemExit(main())
