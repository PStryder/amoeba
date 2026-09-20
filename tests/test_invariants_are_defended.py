"""The standard, made enforceable.

Every important architectural claim needs at least one test whose failure
condition directly expresses that claim, at the layer where the guarantee
lives.

Three things are checkable cheaply and are checked here:

1. Every invariant in ARCHITECTURE.md names at least one test that exists.
   A claim whose test was renamed away is a claim with no evidence.
2. Every mutation in the verification harness names tests that exist, and
   every mutation anchor still matches its source. An anchor that has drifted
   silently turns into a SKIP, and a skipped mutation proves nothing.
3. The harness covers the invariants where the guarantee is cheap to negate.

What is *not* checked here is whether removing a guarantee actually makes its
tests fail -- that needs source mutation and a real test run, which belongs in
`scripts/verify_invariants.py` rather than in the suite it mutates. This file
guards the bookkeeping; that script produces the evidence.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "ARCHITECTURE.md"
HARNESS = ROOT / "scripts" / "verify_invariants.py"


def _declared_tests() -> dict[str, str]:
    found: dict[str, str] = {}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                found[node.name] = f.name
    return found


def _invariants() -> list[tuple[str, str, list[str]]]:
    text = ARCH.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n(?=\*\*I\d+\.)", text):
        m = re.match(r"\*\*(I\d+)\.\s*(.+?)\*\*", block, re.S)
        if not m:
            continue
        tail = block.split("→", 1)[1] if "→" in block else ""
        named = [n for n in re.findall(r"`([\w.]+)`", tail) if n.startswith("test_")]
        out.append((m.group(1), " ".join(m.group(2).split()), named))
    return out


def test_architecture_declares_invariants():
    inv = _invariants()
    assert len(inv) >= 25, f"only {len(inv)} invariants parsed; has the format changed?"


@pytest.mark.parametrize("iid,claim,named", _invariants(),
                         ids=[i[0] for i in _invariants()])
def test_every_invariant_names_a_test_that_exists(iid, claim, named):
    """A claim with no named, existing test is a claim with no evidence."""
    tests = _declared_tests()
    assert named, f"{iid} ({claim}) names no test"
    real = [n for n in named if n in tests]
    ghosts = [n for n in named if n not in tests]
    assert real, (
        f"{iid} ({claim}) names only tests that do not exist: {ghosts}. "
        "Either the test was renamed or the claim is unevidenced.")
    assert not ghosts, (
        f"{iid} names tests that no longer exist: {ghosts}. Update "
        "docs/ARCHITECTURE.md in the same change that renames a test.")


def _harness_mutations() -> list[dict]:
    """Read the mutation table without importing it (it has side effects)."""
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    muts = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Mutation"):
            continue
        args = [a for a in node.args]
        kw = {k.arg: k.value for k in node.keywords}
        try:
            entry = {
                "invariant": ast.literal_eval(args[0]),
                "path": ast.literal_eval(args[2]),
                "old": ast.literal_eval(args[3]),
                "tests": ast.literal_eval(args[5]),
                "also": ast.literal_eval(kw["also"]) if "also" in kw else [],
            }
        except (IndexError, ValueError):
            continue
        muts.append(entry)
    return muts


def test_the_mutation_harness_is_parseable_and_populated():
    muts = _harness_mutations()
    assert len(muts) >= 20, f"only {len(muts)} mutations found in the harness"


@pytest.mark.parametrize("mut", _harness_mutations(),
                         ids=[m["invariant"] for m in _harness_mutations()])
def test_every_mutation_anchor_still_matches_its_source(mut):
    """A drifted anchor turns a mutation into a SKIP, and a SKIP proves nothing.

    This is the failure mode that would quietly hollow out the verification:
    refactor the source, the anchor stops matching, the harness reports SKIP,
    and the invariant looks fine because nothing said otherwise.
    """
    src = (ROOT / mut["path"]).read_text(encoding="utf-8")
    assert mut["old"] in src, (
        f"{mut['invariant']}: mutation anchor no longer present in "
        f"{mut['path']}. Update scripts/verify_invariants.py so the guarantee "
        "is still actually negated.")
    for old, _new in mut["also"]:
        assert old in src, (
            f"{mut['invariant']}: secondary anchor missing from {mut['path']}")


@pytest.mark.parametrize("mut", _harness_mutations(),
                         ids=[m["invariant"] for m in _harness_mutations()])
def test_every_mutation_names_tests_that_exist(mut):
    tests = _declared_tests()
    missing = [t for t in mut["tests"] if t not in tests]
    assert not missing, (
        f"{mut['invariant']}: mutation targets tests that do not exist: "
        f"{missing}")


def test_the_cheaply_negatable_invariants_are_covered_by_the_harness():
    """Claims whose guarantee is a few lines of pure logic should be verified.

    Invariants that need a GPU, a live process stack or a power cut are out of
    scope for source mutation and are listed here explicitly, so the exemption
    is a decision on the record rather than an omission.
    """
    covered = {m["invariant"].rstrip("b") for m in _harness_mutations()}
    exempt = {
        "I3":  "history/memory separation is asserted directly and has no guard to remove",
        "I9":  "needs a live process stack",
        "I10": "needs a live process stack",
        "I12": "needs a live process stack",
        "I13": "needs the GPU engine",
        "I14": "needs the GPU engine",
        "I15": "needs the GPU engine",
        "I17": "needs a live process stack",
        "I18": "needs a live process stack",
        "I20": "needs a live process stack",
        "I21": "needs a live process stack",
        "I22": "needs a live process stack",
        "I23": "the execution loop is not wired; there is no guarantee to negate",
        "I25": "needs a live process stack",
        "I26": "needs a live process stack",
        "I27": "needs a live process stack",
        "I28": "needs a live process stack",
    }
    declared = {iid for iid, _, _ in _invariants()}
    unaccounted = declared - covered - set(exempt)
    assert not unaccounted, (
        f"invariants neither mutation-verified nor explicitly exempt: "
        f"{sorted(unaccounted)}. Add a mutation to scripts/verify_invariants.py "
        "or record why it cannot be negated cheaply.")
