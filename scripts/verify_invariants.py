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
    # Extra (old, new) pairs applied to the same file. A guarantee defended in
    # depth cannot be negated by a single edit: removing one of two redundant
    # checks leaves the other working, and the tests correctly stay green. To
    # ask "is this claim defended at all?" every defence has to come out.
    also: list[tuple[str, str]] = field(default_factory=list)


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
            for old, new in m.also:
                if old not in mutated:
                    results.append((m, "SKIP", f"secondary anchor not found: {old[:40]}"))
                    print(f"SKIP {m.invariant}: secondary anchor not found")
                    mutated = None
                    break
                mutated = mutated.replace(old, new, 1)
            if mutated is None:
                continue
            target.write_text(mutated, encoding="utf-8")
            try:
                passed, tail = run_tests(m.tests)
            finally:
                target.write_text(original, encoding="utf-8")
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
