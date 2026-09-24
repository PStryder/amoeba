"""The tool that checks every guarantee, and had none of its own.

Four defects were found in `scripts/verify_invariants.py` on 2026-09-24, all
of one kind -- it reported success without having checked anything:

1. an `also` entry naming the primary's own file was read from disk unmutated
   and written back afterwards, silently undoing the primary, so I122, I124
   and I125 never tested their primary mutants at all;
2. every mutant was applied together and the tests run once, so one lethal
   primary carried every inert secondary to GOOD -- an `also` entry rewriting
   `SCAN_MULTIPLE = 20` as the same line with a comment was reported GOOD;
3. a rewrite dropped `if __name__ == "__main__"`, so the script defined its
   functions, ran nothing and exited 0;
4. a non-unique anchor mutated whichever match came first -- I133's primary
   edited `io_attach_input` while its tests were about `io_submit`.

Nothing would have caught any of them, because nothing tested this file. These
do. They are deliberately about the machinery and not about any invariant: a
test here must fail if the verifier stops verifying.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_invariants.py"


@pytest.fixture()
def verifier():
    spec = importlib.util.spec_from_file_location("verify_invariants", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_invariants"] = module
    spec.loader.exec_module(module)
    return module


def _mutation(verifier, **kw):
    fields = {"invariant": "IX", "claim": "a claim", "path": "a.py",
              "old": "OLD", "new": "NEW", "tests": ["test_x"],
              "layer": "a layer"}
    fields.update(kw)
    return verifier.Mutation(**fields)


# ---------------------------------------------------------------------------
# Every mutant is its own trial
# ---------------------------------------------------------------------------
def test_each_mutant_is_tried_on_its_own(verifier):
    """Applied together, one lethal mutant answers for every inert one."""
    m = _mutation(verifier, also=[("b.py", "B_OLD", "B_NEW"), ("C_OLD", "C_NEW")])
    trials = verifier.trials_for(m)

    assert len(trials) == 3, "the primary and both secondaries are three trials"
    assert trials[0][1] == {"a.py": ("OLD", "NEW")}
    assert trials[1][1] == {"b.py": ("B_OLD", "B_NEW")}
    # A two-tuple means the primary's own file.
    assert trials[2][1] == {"a.py": ("C_OLD", "C_NEW")}


def test_a_secondary_naming_the_primarys_file_is_its_own_trial(verifier):
    """The first defect: read unmutated and written back, undoing the primary.

    With one mutant per trial there is nothing to undo, but the entry must
    still resolve to the right file.
    """
    m = _mutation(verifier, also=[("a.py", "A2_OLD", "A2_NEW")])
    assert verifier.trials_for(m)[1][1] == {"a.py": ("A2_OLD", "A2_NEW")}


# ---------------------------------------------------------------------------
# An anchor must name one place
# ---------------------------------------------------------------------------
def test_an_ambiguous_anchor_is_refused(verifier, tmp_path, monkeypatch):
    """The fourth defect: it mutated whichever match came first."""
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("x = 1\nSAME\ny = 2\nSAME\n", encoding="utf-8")

    problem = verifier.apply_trial({"a.py": ("SAME", "MUTANT")})

    assert isinstance(problem, str)
    assert "2 places" in problem
    assert (tmp_path / "a.py").read_text(encoding="utf-8").count("SAME") == 2, (
        "a refused trial still edited the file")


def test_a_missing_anchor_is_refused(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("nothing here\n", encoding="utf-8")
    assert "not found" in verifier.apply_trial({"a.py": ("ABSENT", "X")})


def test_a_trial_that_fails_partway_leaves_nothing_edited(verifier, tmp_path,
                                                          monkeypatch):
    """One file applied, the next refused: the first must not stay mutated."""
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("GOOD_ANCHOR\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("no anchor here\n", encoding="utf-8")

    problem = verifier.apply_trial({"a.py": ("GOOD_ANCHOR", "MUTATED"),
                                    "b.py": ("MISSING", "X")})

    assert isinstance(problem, str)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "GOOD_ANCHOR\n"


def test_a_trial_edits_and_can_be_put_back(verifier, tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("before ANCHOR after\n", encoding="utf-8")

    originals = verifier.apply_trial({"a.py": ("ANCHOR", "MUTANT")})

    assert isinstance(originals, dict)
    assert "MUTANT" in (tmp_path / "a.py").read_text(encoding="utf-8")
    for rel, text in originals.items():
        (tmp_path / rel).write_text(text, encoding="utf-8")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "before ANCHOR after\n"


# ---------------------------------------------------------------------------
# It has to actually run
# ---------------------------------------------------------------------------
def test_the_script_runs_when_it_is_run():
    """The third defect: a rewrite dropped the entry point.

    The script imported cleanly, defined everything, verified nothing and
    exited 0. Asserted by running it, because that is the only thing the
    defect was visible to.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source


def test_a_run_that_checks_nothing_is_an_error():
    """Exit 0 must mean "mutants died", never "no mutants ran"."""
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--only", "I_DOES_NOT_EXIST"],
        capture_output=True, text=True, timeout=120,
        cwd=str(SCRIPT.resolve().parents[1]))

    assert done.returncode != 0, done.stdout
    assert "no such invariant" in done.stdout


def test_every_declared_invariant_has_tests_named_for_it(verifier):
    """A mutant with no tests cannot fail, so it would always be GOOD."""
    for m in verifier.MUTATIONS:
        assert m.tests, f"{m.invariant} declares no tests"
        assert m.old != m.new, f"{m.invariant}'s mutation changes nothing"


def test_no_invariant_is_declared_twice(verifier):
    seen = [m.invariant for m in verifier.MUTATIONS]
    assert len(seen) == len(set(seen)), "an invariant id is declared twice"
