"""Ingesting the shipped cognitive baseline at startup.

The files under ``promptlib/prompts`` are the authored starting point of the
family tree, and the **only** origin of a root. They are not the library: once
a namespace exists in the database, the database is authoritative and a file is
a proposal.

Four outcomes per file, decided by comparing the file's local definition
against what the library already holds:

``baseline``
    The namespace does not exist yet. Version 1 is created, approved and
    selected -- otherwise nothing could run on a fresh state directory. This is
    the one moment a file establishes rather than proposes.

``matched``
    The file's definition is byte-identical to the *selected* version. Nothing
    is written but the match itself, so the log can show the shipped baseline
    was still in force.

``present``
    Some version has this exact definition, but it is not the one running --
    the file was reverted, or its candidate was never approved. Nothing is
    created, which keeps restarts idempotent.

``delta``
    No version has this definition. A new **candidate** is created and
    selection is left alone.

That last case is the whole point. Editing a prompt file and restarting must
not quietly change how the organism thinks: the change becomes a governed
candidate that somebody has to approve, exactly like one Id proposed. The
previously selected version keeps running until then.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import InvalidInput
from ..store.events import EventKind
from ..store.writer import Mutation
from .model import (MODEL_VARS, PROMPT_MODES, ROOTS, depth, validate_namespace)
from .store import PromptStore, canonical_local

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
HEADER_SEPARATOR = "---"
BOOTSTRAP_ACTOR = "bootstrap"


def parse_prompt_file(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Parse a bootstrap prompt file.

    A small ``key: <json>`` header, a line containing only ``---``, then the
    prompt body verbatim. Header keys are ``mode``, ``rationale`` and the model
    variable names -- anything else is refused rather than ignored, so a typo
    like ``temprature`` fails loudly instead of silently configuring nothing.
    """
    # Line endings are a transport artifact, not part of what a prompt says,
    # and a profile version's identity is the digest of its text. A CRLF
    # checkout must therefore not look like an edit.
    #
    # `load_prompt_files` reads through `Path.read_text`, whose universal
    # newlines already translate CRLF, so the shipped-file path is covered
    # before this line runs. It is here for every *other* caller -- anything
    # that decodes bytes itself and parses the result -- where a CRLF document
    # would otherwise not match the header separator at all and be rejected as
    # malformed. Cheap, and it keeps "what is a prompt file" one answer rather
    # than one-per-reader.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    head, sep, body = text.partition("\n" + HEADER_SEPARATOR + "\n")
    if not sep:
        raise InvalidInput(
            f"{source}: expected a header, a line containing only "
            f"{HEADER_SEPARATOR!r}, then the prompt body")

    mode = "replace"
    rationale = ""
    variables: dict[str, Any] = {}
    for lineno, raw in enumerate(head.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, colon, value = line.partition(":")
        if not colon:
            raise InvalidInput(f"{source}:{lineno}: expected 'key: value'",
                               line=line[:80])
        key, value = key.strip(), value.strip()
        if key == "mode":
            mode = json.loads(value) if value.startswith('"') else value
            if mode not in PROMPT_MODES:
                raise InvalidInput(f"{source}:{lineno}: unknown mode {mode!r}",
                                   allowed=list(PROMPT_MODES))
        elif key == "rationale":
            rationale = json.loads(value) if value.startswith('"') else value
        elif key in MODEL_VARS:
            try:
                variables[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise InvalidInput(
                    f"{source}:{lineno}: {key} must be a JSON value",
                    given=value[:60]) from exc
        else:
            raise InvalidInput(
                f"{source}:{lineno}: unknown header key {key!r}",
                known=["mode", "rationale", *sorted(MODEL_VARS)])

    # The body is taken verbatim apart from a trailing newline, because the
    # prompt's bytes are its identity and "roughly this text" is not a baseline.
    return {"prompt_mode": mode, "prompt_text": body.rstrip("\n"),
            "model_vars": variables, "rationale": rationale}


def load_prompt_files(directory: Path | None = None) -> list[dict[str, Any]]:
    """Read every ``<namespace>.md`` file, ordered shallowest first.

    Depth order matters: a child cannot pin a parent version that does not
    exist yet, so parents must be ingested first. The filename *is* the
    namespace, so a file cannot claim an ancestry its name does not have.
    """
    directory = Path(directory or PROMPT_DIR)
    if not directory.is_dir():
        raise InvalidInput("no bootstrap prompt directory", path=str(directory))
    out = []
    for path in sorted(directory.glob("*.md")):
        namespace = path.stem
        validate_namespace(namespace)
        parsed = parse_prompt_file(path.read_text(encoding="utf-8"),
                                   source=path.name)
        parsed["namespace"] = namespace
        parsed["source"] = path.name
        out.append(parsed)
    known = {p["namespace"] for p in out}
    for missing in ROOTS:
        if missing not in known:
            raise InvalidInput(
                f"bootstrap is missing root {missing!r}; roots have no other "
                "origin", directory=str(directory))
    for parsed in out:
        ns = parsed["namespace"]
        parent = ns.rpartition(".")[0]
        if parent and parent not in known:
            raise InvalidInput(
                f"{parsed['source']}: parent {parent!r} has no bootstrap file",
                hint="a namespace's ancestry must exist before it does")
    out.sort(key=lambda p: (depth(p["namespace"]), p["namespace"]))
    return out


def ingest(m: Mutation, store: PromptStore, directory: Path | None = None
           ) -> dict[str, Any]:
    """Compare the shipped files against the library and record the outcome."""
    results: list[dict[str, Any]] = []
    for parsed in load_prompt_files(directory):
        namespace = parsed["namespace"]
        local_sha = canonical_local(parsed["prompt_mode"], parsed["prompt_text"],
                                    parsed["model_vars"])
        versions = store.versions(namespace)
        selected = store.selected(namespace)

        if not versions:
            created = store.create_version(
                m, namespace=namespace, prompt_mode=parsed["prompt_mode"],
                prompt_text=parsed["prompt_text"],
                model_vars=parsed["model_vars"], origin="bootstrap",
                created_by=BOOTSTRAP_ACTOR,
                rationale=parsed["rationale"]
                          or f"shipped baseline from {parsed['source']}",
                state="production_approved", _allow_root=True)
            store.select(m, namespace=namespace,
                         version_id=created["version_id"], purpose="production",
                         selected_by=BOOTSTRAP_ACTOR)
            m.emit(EventKind.PROMPT_BOOTSTRAP_BASELINE, {
                "namespace": namespace, "version_id": created["version_id"],
                "local_sha256": local_sha, "source": parsed["source"]})
            results.append({"namespace": namespace, "outcome": "baseline",
                            **created})
            continue

        # Compare against every version, not only the selected one. Restarting
        # must be idempotent: a file whose definition the library already holds
        # creates nothing, or an unselected namespace would accumulate one
        # identical candidate per restart.
        same = next((v for v in versions if v["local_sha256"] == local_sha), None)

        if same is not None:
            is_running = bool(selected and selected["version_id"] == same["version_id"])
            m.emit(EventKind.PROMPT_BOOTSTRAP_MATCHED, {
                "namespace": namespace, "version_id": same["version_id"],
                "local_version": same["local_version"],
                "local_sha256": local_sha, "source": parsed["source"],
                "selected": is_running,
                "selected_version_id": selected["version_id"] if selected else None})
            results.append({
                "namespace": namespace,
                "outcome": "matched" if is_running else "present",
                "version_id": same["version_id"],
                "local_version": same["local_version"],
                "state": same["state"]})
            continue

        # A definition the library has never seen becomes a candidate. The
        # selected version keeps running.
        created = store.create_version(
            m, namespace=namespace, prompt_mode=parsed["prompt_mode"],
            prompt_text=parsed["prompt_text"], model_vars=parsed["model_vars"],
            origin="bootstrap", created_by=BOOTSTRAP_ACTOR,
            rationale=parsed["rationale"]
                      or f"{parsed['source']} differs from the selected version",
            state="candidate", _allow_root=True)
        m.emit(EventKind.PROMPT_BOOTSTRAP_DELTA, {
            "namespace": namespace, "version_id": created["version_id"],
            "local_version": created["local_version"],
            "file_sha256": local_sha,
            "selected_version_id": selected["version_id"] if selected else None,
            "selected_sha256": selected["local_sha256"] if selected else None,
            "source": parsed["source"],
            "note": ("the file is a candidate, not an override; the selected "
                     "version keeps running until somebody approves this one")})
        results.append({"namespace": namespace, "outcome": "delta", **created})

    return {"ingested": results,
            "counts": {outcome: sum(1 for r in results if r["outcome"] == outcome)
                       for outcome in ("baseline", "matched", "present", "delta")}}
