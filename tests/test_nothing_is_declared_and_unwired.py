"""Things that exist and are reachable from nothing.

This is the defect class that kept recurring, and it is invisible to every
other check here. Each instance was complete in isolation, tested at its own
layer, and wired to nothing:

* interaction lineage took no argument on the only production path into the
  mailbox, so every real trigger was untagged;
* `artifact_event` and `board_event` were declared trigger kinds that nothing
  ever emitted;
* `vram_free_bytes` and the session counts were collected, displayed, and read
  by no admission decision;
* a specialist neuocyte profile could be authored, approved and advertised,
  and neuocyte birth bound the base namespace regardless;
* `max_context_tokens` reached the dashboard and never the control loop;
* `neuocyte_max_age_seconds` sat in config.toml saying 900 while the real
  bound said 180.

A mutation harness cannot find these: there is no guarantee to negate, because
nothing was ever claimed. So they are found structurally instead.

What this does *not* prove is that a reference is reached at runtime. A first
attempt at recording external denials added an emit inside the HTTP adapter --
a separate process with no store access, which would have found no `mind` and
quietly done nothing, while this check went green because the name now
appeared in the source. The check narrows the ways the code can be unwired; it
does not establish that a path executes, and a new emit deserves a test that
reads the event back.

Exemptions are listed with a reason rather than silently skipped. An entry
here is a decision on the record -- which is the difference between "we know
and this is why" and "nobody noticed".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "amoeba"
SOURCES = {p: p.read_text(encoding="utf-8") for p in sorted(SRC.rglob("*.py"))}
ALL_SRC = "\n".join(SOURCES.values())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_EXEMPT: dict[str, str] = {
    # Nothing yet. Every setting in the file is read by something, and an
    # addition that is not should either be wired or justified here.
}


def _config_fields() -> set[str]:
    tree = ast.parse(SOURCES[SRC / "config.py"])
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                name = item.target.id
                if not name.startswith("_"):
                    out.add(name)
    return out


def test_every_configuration_setting_is_read_by_something():
    """A setting nobody reads is a promise to an operator that nothing keeps.

    `max_context_tokens` reached the dashboard and never the control loop, so
    a reader had no way to tell which of two numbers governed.
    `neuocyte_max_age_seconds` said 900 while behaviour said 180. Both looked
    exactly like settings that worked.
    """
    others = "\n".join(t for p, t in SOURCES.items() if p.name != "config.py")
    dead = []
    for field in sorted(_config_fields()):
        if field in CONFIG_EXEMPT:
            continue
        if not re.search(rf"\.{field}\b", others):
            dead.append(field)
    assert not dead, (
        f"configuration settings nothing reads: {dead}. Wire them, delete "
        "them, or add them to CONFIG_EXEMPT with a reason.")


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------
EVENT_EXEMPT: dict[str, str] = {
    "ARTIFACT_LAPSED": "retired; the constant stays so historical events still "
                       "name a known kind, and events.py says so",
    "MCP_CALL": "the external surface records interactions, not transport calls",
    "MCP_ERROR": "as above",
    "INFERENCE_REQUEST": "one event per generation would drown the log; "
                         "INFERENCE_ERROR is emitted and is the informative half",
    "INFERENCE_RESULT": "as above",
    "BACKEND_LOADED": "process lifecycle, logged by the inference service",
    "BACKEND_UNLOADED": "as above",
    "PULSE_CITED": "reserved; nothing cites a pulse yet",
    "RESOURCE_DECISION": "admission refusals are recorded as WORK_REJECTED by "
                         "the path that refuses them",
    "DISAGREEMENT_RESOLVED": "there is no resolution path yet -- a known "
                             "functional gap, tracked, not an oversight here",
    "EXTERNAL_DENIED": "the HTTP adapter runs as a separate process and holds "
                       "no store access -- it reaches the Harness only over "
                       "RPC, so it cannot write an event. Recording a probe "
                       "needs a Harness verb the adapter may call, which is a "
                       "design decision rather than a wiring fix",
}


def _event_kinds() -> list[str]:
    return re.findall(r"^    ([A-Z][A-Z0-9_]+) = ",
                      SOURCES[SRC / "store" / "events.py"], re.M)


@pytest.mark.parametrize("kind", _event_kinds())
def test_every_event_kind_is_emitted_or_explicitly_not(kind):
    """A declared event nothing emits is a record that does not exist.

    `ROLE_TOOL_INVOKED` was declared while a role's tool calls emitted nothing
    at all, and a neuocyte's emitted three -- so the disposable worker's
    capability use was on the record and the persistent mind's was not.
    """
    if kind in EVENT_EXEMPT:
        return
    assert re.search(rf"EventKind\.{kind}\b", ALL_SRC), (
        f"EventKind.{kind} is declared and never emitted. Emit it, delete it, "
        "or add it to EVENT_EXEMPT with a reason.")


# ---------------------------------------------------------------------------
# Trigger kinds
# ---------------------------------------------------------------------------
def _trigger_kinds() -> list[str]:
    src = SOURCES[SRC / "mailbox.py"]
    block = src[src.index("TRIGGER_KINDS = ("):src.index("STOP_REASONS")]
    return re.findall(r'^    "(\w+)",', block, re.M)


@pytest.mark.parametrize("kind", _trigger_kinds())
def test_every_trigger_kind_is_enqueued_somewhere(kind):
    """A trigger kind nothing produces is a wake that never happens.

    `artifact_event` and `board_event` were declared from the beginning and
    emitted by nothing, so "Ego is event-driven" was true of a narrower set of
    events than the architecture claimed.
    """
    assert re.search(rf'kind="{kind}"', ALL_SRC), (
        f"trigger kind {kind!r} is declared and never enqueued")


# ---------------------------------------------------------------------------
# Constitutional text
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["ego", "id"])
def test_the_fallback_prompt_matches_the_governed_one(role):
    """Two copies of constitutional text, and nothing keeping them in step.

    `roles.py` carries `EGO_SYSTEM`/`ID_SYSTEM` as the text for a mind whose
    library has nothing selected -- a partial bootstrap. That is a fair reason
    to keep them, and it makes them a second copy of governed doctrine: edit
    the prompt through versioning, candidacy and approval, and the constant
    silently keeps the old wording. A partially bootstrapped mind would then
    be primed with superseded doctrine and nothing would say so.

    The constant is the opening of the shipped file, so that is asserted
    rather than trusted.
    """
    from amoeba import roles

    constant = (roles.EGO_SYSTEM if role == "ego" else roles.ID_SYSTEM).strip()
    shipped = (SRC / "promptlib" / "prompts" / f"{role}.md").read_text(
        encoding="utf-8")
    body = shipped.split("---", 1)[1].strip()
    assert body.startswith(constant), (
        f"the {role} fallback prompt has drifted from the governed one.\n"
        "Update the constant in roles.py in the same change that edits "
        f"prompts/{role}.md, or a partial bootstrap primes superseded "
        "doctrine.")
