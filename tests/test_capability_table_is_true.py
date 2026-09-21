"""The capability table is a claim, so it gets the same treatment as one.

`test_invariants_are_defended` exists because a claim in ARCHITECTURE.md whose
test was renamed away is a claim with no evidence. The table in
IMPLEMENTATION.md is the same kind of claim -- "this works, here is what
proves it" -- and had no such guard, so it drifted: a verb count that predated
the demotion, a cancellation row contradicting the invariant whose test is
named `test_disconnecting_does_not_cancel_anything`, and two rows about
context trimming that disagreed about whether it existed.

Four things are cheaply checkable and checked here:

1. every test named in an evidence cell exists;
2. every test *file* named in an evidence cell exists;
3. every invariant cited resolves to one ARCHITECTURE.md actually declares;
4. two rows citing the same invariant agree about its status.

What is deliberately *not* checked is prose accuracy. A sentence can be stale
in ways no parser will catch, and pretending otherwise would be worse than
admitting the limit: this narrows the ways the table can lie, and does not
make it honest by itself.

Counts are handled by not having them. A row that said "10 cognitive verbs"
was wrong the moment the surface changed, so the row now points at
`scopes.EXTERNAL_IO` instead of restating its length -- the fix for a number
that can drift is usually to delete the number.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs" / "IMPLEMENTATION.md"
ARCH = ROOT / "docs" / "ARCHITECTURE.md"


def _rows() -> list[tuple[str, str, str]]:
    """(capability, status, evidence) for every row of every table."""
    out = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if set(cells[0]) <= set("-: ") or cells[0].lower() in ("capability", "area"):
            continue
        out.append((cells[0], cells[1], " ".join(cells[2:])))
    return out


def _declared_tests() -> set[str]:
    found: set[str] = set()
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                found.add(node.name)
    return found


def _declared_invariants() -> set[str]:
    return set(re.findall(r"\*\*(I\d+[a-z]?\d*)\.", ARCH.read_text(encoding="utf-8")))


def test_the_table_parses_and_is_populated():
    rows = _rows()
    assert len(rows) >= 40, f"only {len(rows)} rows parsed; has the format changed?"


@pytest.mark.parametrize("row", _rows(), ids=[r[0][:48] for r in _rows()])
def test_every_test_named_in_the_table_exists(row):
    """A row citing a test that was renamed away is a row with no evidence."""
    capability, _status, evidence = row
    named = [n for n in re.findall(r"`([\w./]+)`", evidence)
             if n.startswith("test_") and not n.endswith(".py")]
    missing = [n for n in named if n not in _declared_tests()]
    assert not missing, (
        f"{capability!r} names tests that do not exist: {missing}. Update "
        "docs/IMPLEMENTATION.md in the same change that renames a test.")


@pytest.mark.parametrize("row", _rows(), ids=[r[0][:48] for r in _rows()])
def test_every_test_file_named_in_the_table_exists(row):
    capability, _status, evidence = row
    files = [n for n in re.findall(r"`([\w./]+\.py)`", evidence)
             if "test" in n]
    missing = [n for n in files if not (ROOT / n).exists()]
    assert not missing, f"{capability!r} names test files that do not exist: {missing}"


@pytest.mark.parametrize("row", _rows(), ids=[r[0][:48] for r in _rows()])
def test_every_invariant_cited_by_the_table_is_declared(row):
    """A row pointing at an invariant nobody wrote is a dangling promise."""
    capability, _status, evidence = row
    cited = set(re.findall(r"\bI\d+[a-z]?\d*\b", evidence))
    # `I/O` is not an invariant, and neither is a bare year or version.
    cited = {c for c in cited if re.fullmatch(r"I\d+[a-z]?\d*", c)}
    missing = sorted(cited - _declared_invariants())
    assert not missing, (
        f"{capability!r} cites invariants ARCHITECTURE.md does not declare: "
        f"{missing}")


def test_rows_citing_one_invariant_agree_about_it():
    """Two rows describing the same guarantee must not disagree.

    This is the shape the context-trimming drift took: one row said trimming
    was implemented and tested, another said it was not implemented and that
    a long-running Ego would eventually overflow. Both cited the same ground.
    A reader cannot act on a table that contradicts itself, and neither can a
    reviewer.
    """
    claims: dict[str, set[str]] = {}
    for capability, status, evidence in _rows():
        for inv in re.findall(r"\bI\d+[a-z]?\d*\b", evidence):
            claims.setdefault(inv, set()).add(status)
    disagreements = {k: sorted(v) for k, v in claims.items() if len(v) > 1}
    assert not disagreements, (
        f"rows citing the same invariant disagree about its status: "
        f"{disagreements}")


def test_the_external_surface_is_not_restated_as_a_number():
    """A count typed by hand is wrong the moment the surface changes.

    The table once said "10 cognitive verbs" and, further down, "8 I/O tools".
    Both were describing `scopes.EXTERNAL_IO`, and one of them predated the
    demotion. Where a row states the size of the external surface it must
    agree with the scope, which is the thing that decides it.
    """
    from amoeba import scopes

    actual = len(scopes.EXTERNAL_IO)
    words = {"eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    for capability, _status, evidence in _rows():
        text = f"{capability} {evidence}"
        # Only rows about the external surface. Plenty of rows count verbs
        # that have nothing to do with it -- Id's effectors, for one -- and
        # holding those to the size of `external_io` would be nonsense.
        if not re.search(r"\bexternal\b|\bMCP\b|io_\*|EXTERNAL_IO", text, re.I):
            continue
        # Up to two words may sit between the number and the noun: the
        # stale row said "10 cognitive verbs", and requiring the noun to
        # follow directly walked straight past the exact claim this
        # exists to catch.
        for m in re.finditer(r"\b(\d+|" + "|".join(words) + r")\s+"
                             r"(?:[\w/*+-]+\s+){0,2}"
                             r"(?:tools|verbs)\b", text, re.I):
            # "was 23 tools ... now 8" states history, and history does not
            # drift. Only a present-tense count has to match the scope.
            before = text[max(0, m.start() - 24):m.start()].lower()
            if re.search(r"\b(was|were|from|previously|used to)\b", before):
                continue
            token = m.group(1).lower()
            claimed = words.get(token, None)
            if claimed is None:
                claimed = int(token)
            assert claimed == actual, (
                f"{capability!r} claims {claimed} external verbs; "
                f"scopes.EXTERNAL_IO has {actual}. Point at the scope rather "
                "than restating its length.")


def test_no_document_offers_the_blackboard_to_an_external_client():
    """The drift that read as a capability guarantee rather than progress.

    BLACKBOARD.md described an MCP surface of `board_read`, `board_post` and
    `board_corroboration` -- "contribute as a peer" -- long after the
    demotion left `external_io` holding eight `io_*` verbs and no board verb
    at all. A reader would have concluded an outside model can write to the
    swarm's working surface.
    """
    from amoeba import scopes

    board_verbs = {v for v in scopes.EXTERNAL_IO if v.startswith("board_")}
    assert not board_verbs, (
        f"the external scope now contains board verbs: {sorted(board_verbs)}. "
        "If that is intended, this test and BLACKBOARD.md both need rewriting "
        "-- deliberately, because it changes who can write to the board.")

    doc = (ROOT / "docs" / "BLACKBOARD.md").read_text(encoding="utf-8")
    offending = [line.strip() for line in doc.splitlines()
                 if re.search(r"`board_\w+`", line)
                 and re.search(r"\bMCP\b", line)]
    assert not offending, (
        "BLACKBOARD.md offers a board verb over MCP:\n  "
        + "\n  ".join(offending))
