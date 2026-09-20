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
