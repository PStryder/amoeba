"""Acceptance tests 5, 6, 12: provenance, history-vs-memory, crash during commit."""

from __future__ import annotations

from pathlib import Path

import sqlite3

import pytest

from amoeba.errors import Fenced, IntegrityError, InvalidInput, NotFound, StaleVersion
from amoeba.store.events import EventKind, missing_content, verify_chain
from amoeba.store.writer import Mutation


# ---------------------------------------------------------------------------
# Test 5: event/receipt provenance resolves; missing content is detected.
# ---------------------------------------------------------------------------
def test_provenance_chain_resolves_end_to_end(mind):
    op_id, receipt, replayed = mind.memory.open_operation(
        kind="ego_converse", actor="ego", request={"message": "hello"}
    )
    assert not replayed and receipt.outcome == "committed"

    work_id, _ = mind.work.admit(
        objective="answer hello", work_class="user", origin_actor="ego",
        operation_id=op_id,
    )
    item = mind.work.lease(neuocyte_id="wk1")
    assert item["work_id"] == work_id
    mind.work.complete(work_id=work_id, neuocyte_id="wk1",
                       fencing_token=item["fencing_token"],
                       result={"finding": "hello back"}, operation_id=op_id)
    cid, _ = mind.memory.record_conclusion(
        claim="hello back", produced_by="ego", operation_id=op_id,
        evidence=[{"note": "work result"}], model_identity="gen_test",
    )
    mind.memory.update_operation(operation_id=op_id, status="completed", actor="ego",
                                 result={"answer": "hello back"})

    prov = mind.provenance(operation_id=op_id)
    kinds = [e["kind"] for e in prov["events"]]
    # input -> work -> result -> conclusion -> mutation, all under one operation
    assert EventKind.INPUT_RECEIVED in kinds
    assert EventKind.WORK_ADMITTED in kinds
    assert EventKind.WORK_COMPLETED in kinds
    assert EventKind.CONCLUSION_RECORDED in kinds
    assert EventKind.OUTPUT_EMITTED in kinds
    assert prov["hash_chain_ok"] is True
    assert prov["unresolved_content"] == []
    assert prov["conclusions"][0]["conclusion_id"] == cid
    assert prov["work_items"][0]["work_id"] == work_id
    assert len(prov["receipts"]) >= 3


def test_missing_blob_is_detected_not_glossed_over(mind, cfg):
    op_id, _, _ = mind.memory.open_operation(
        kind="ego_converse", actor="ego", request={"message": "x" * 5000}
    )
    row = mind.db.conn.execute(
        "SELECT request_blob FROM operations WHERE operation_id = ?", (op_id,)
    ).fetchone()
    sha = row["request_blob"]
    assert mind.blobs.exists(sha)

    # Simulate content loss under a committed reference.
    mind.blobs.path_for(sha).unlink()

    report = mind.verify_integrity(deep=True)
    assert report["missing_content_count"] >= 1
    assert any(m["sha256"] == sha and m["reason"] == "absent"
               for m in report["missing_content"])
    with pytest.raises(IntegrityError):
        mind.blobs.get(sha)


def test_corrupt_blob_is_detected(mind):
    sha = mind.blobs.put_text("original content")
    mind.blobs.path_for(sha).write_bytes(b"tampered")
    assert mind.blobs.verify(sha) is False
    with pytest.raises(IntegrityError):
        mind.blobs.get(sha)


def test_hash_chain_detects_tampering(mind):
    for i in range(5):
        mind.writer.apply(lambda m, i=i: m.emit("test.event", {"i": i}), actor="tester")
    assert verify_chain(mind.db.conn)[0] is True

    # Rewrite a payload without recomputing hashes.
    mind.db.conn.execute(
        "UPDATE events SET payload_inline = ? WHERE kind = 'test.event'"
        " AND seq = (SELECT MIN(seq) FROM events WHERE kind='test.event')",
        ('{"i":999}',),
    )
    mind.db.conn.commit()
    ok, bad = verify_chain(mind.db.conn)
    assert ok is False and bad is not None


def test_hash_chain_caveat_is_stated(mind):
    report = mind.verify_integrity()
    assert "administrator" in report["caveat"]


# ---------------------------------------------------------------------------
# Test 6: raw history stays separate from maintained memory.
# ---------------------------------------------------------------------------
def test_contradictory_history_does_not_become_belief(mind):
    # Two contradictory raw events.
    mind.writer.apply(
        lambda m: m.emit(EventKind.INPUT_RECEIVED, {"text": "the port is 8080"}),
        actor="user")
    mind.writer.apply(
        lambda m: m.emit(EventKind.INPUT_RECEIVED, {"text": "the port is 9090"}),
        actor="user")

    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind = ?", (EventKind.INPUT_RECEIVED,)
    ).fetchone()["n"] == 2
    # Nothing was promoted into memory.
    assert mind.memory.recall(scope="all") == []


def test_correction_supersedes_and_preserves_contrary_evidence(mind):
    first, _ = mind.memory.remember(
        kind="belief", claim="the port is 8080", confidence=0.6, created_by="ego",
        supporting=[{"note": "user said so"}],
    )
    second, _ = mind.memory.remember(
        kind="belief", claim="the port is 9090", confidence=0.85, created_by="id",
        supporting=[{"note": "observed listening socket"}],
        opposing=[{"note": "user originally said 8080"}],
        supersedes=first,
    )
    old = mind.memory.get_memory(first)
    new = mind.memory.get_memory(second)
    assert old["status"] == "superseded"
    assert new["status"] == "active" and new["supersedes"] == first
    assert new["version"] == old["version"] + 1
    # Contrary evidence survives on the new interpretation.
    assert len(new["evidence"]["opposing"]) == 1
    # The old claim is still retrievable, not rewritten.
    assert old["claim"] == "the port is 8080"
    # Default recall shows only the active interpretation.
    active = mind.memory.recall(query="port")
    assert [m["memory_id"] for m in active] == [second]
    assert len(mind.memory.recall(query="port", scope="all")) == 2


def test_recall_matches_salient_terms_not_whole_phrase(mind):
    mind.memory.remember(kind="belief",
                         claim="The inference backend keeps one resident weight set.",
                         confidence=0.9, created_by="operator")
    hits = mind.memory.recall(query="How many copies of the model weights are resident?")
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# Test 12: duplicate results, stale findings, crash during commit.
# ---------------------------------------------------------------------------
def test_duplicate_commit_is_idempotent(mind):
    work_id, _ = mind.work.admit(objective="o", work_class="user", origin_actor="ego")
    item = mind.work.lease(neuocyte_id="wk1")
    token = item["fencing_token"]
    r1 = mind.work.complete(work_id=work_id, neuocyte_id="wk1", fencing_token=token,
                            result={"finding": "a"})
    r2 = mind.work.complete(work_id=work_id, neuocyte_id="wk1", fencing_token=token,
                            result={"finding": "a"})
    assert r1.receipt_id == r2.receipt_id
    assert r2.replayed is True
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind = ?", (EventKind.WORK_COMPLETED,)
    ).fetchone()["n"] == 1


def test_stale_worker_result_is_fenced(mind):
    work_id, _ = mind.work.admit(objective="o", work_class="user", origin_actor="ego")
    first = mind.work.lease(neuocyte_id="wk1", lease_seconds=0.0)
    # The lease expires and a replacement neuocyte takes it.
    expired = mind.work.expire_leases(now=first["lease_expires"] + 1)
    assert work_id in expired
    second = mind.work.lease(neuocyte_id="wk2")
    assert second["fencing_token"] > first["fencing_token"]

    with pytest.raises(Fenced):
        mind.work.complete(work_id=work_id, neuocyte_id="wk1",
                           fencing_token=first["fencing_token"],
                           result={"finding": "stale"})
    # The replacement can still commit.
    mind.work.complete(work_id=work_id, neuocyte_id="wk2",
                       fencing_token=second["fencing_token"],
                       result={"finding": "fresh"})
    assert mind.work.get_work(work_id)["result"]["finding"] == "fresh"


def test_findings_pinned_to_an_older_state_version_are_flagged(mind):
    work_id, _ = mind.work.admit(objective="o", work_class="user", origin_actor="ego")
    item = mind.work.lease(neuocyte_id="wk1")
    pinned = item["pinned_state_ver"]
    # State moves on while the neuocyte is running.
    mind.memory.remember(kind="belief", claim="something changed", confidence=0.5,
                         created_by="id")
    mind.work.complete(work_id=work_id, neuocyte_id="wk1",
                       fencing_token=item["fencing_token"],
                       result={"finding": "x"}, pinned_state_ver=pinned)
    ev = mind.db.conn.execute(
        "SELECT payload_inline FROM events WHERE kind = ?", (EventKind.WORK_COMPLETED,)
    ).fetchone()
    assert '"stale_against"' in ev["payload_inline"]
    assert '"pinned"' in ev["payload_inline"]


def test_crash_during_commit_leaves_no_half_applied_mutation(mind):
    before_version = mind.state_version()
    before_events = mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM events").fetchone()["n"]

    class Boom(RuntimeError):
        pass

    def body(m: Mutation) -> None:
        m.sql("INSERT INTO memory_items(memory_id, kind, claim, confidence, status,"
              " version, created_by, created_at, updated_at, state_version)"
              " VALUES ('mem_x','belief','half written',0.5,'active',1,'ego',0,0,1)")
        m.emit("test.partial", {"x": 1})
        raise Boom("fault injected between state change and commit")

    with pytest.raises(Boom):
        mind.writer.apply(body, actor="tester")

    assert mind.state_version() == before_version
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM events").fetchone()["n"] == before_events
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id='mem_x'"
    ).fetchone()["n"] == 0
    # No receipt was issued for a mutation that did not commit.
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM receipts").fetchone()["n"] == 0
    assert verify_chain(mind.db.conn)[0] is True


def test_acknowledged_mutation_survives_restart(cfg):
    from amoeba.mind import Mind

    m1 = Mind(cfg)
    mem_id, receipt = m1.memory.remember(kind="belief", claim="durable claim",
                                         confidence=0.8, created_by="ego")
    version = receipt.result_version
    m1.close()

    m2 = Mind(cfg)
    assert m2.memory.get_memory(mem_id)["claim"] == "durable claim"
    assert m2.state_version() == version
    assert m2.writer.receipt_for(receipt.mutation_id).receipt_id == receipt.receipt_id
    m2.close()


def test_optimistic_concurrency_rejects_stale_expected_version(mind):
    stale = mind.state_version() - 1
    with pytest.raises(StaleVersion):
        mind.writer.apply(lambda m: m.emit("x", {}), actor="t", expect_version=stale)


def test_blob_deduplication(mind):
    a = mind.blobs.put_text("identical context block")
    b = mind.blobs.put_text("identical context block")
    assert a == b
    files = list(mind.cfg.blob_dir.rglob("*.blob"))
    assert len(files) == 1


def test_lease_can_target_a_specific_work_item(mind):
    """The dispatcher spawns a neuocyte FOR an item; the lease must honour that.

    Without targeting, a neuocyte claims the queue head instead, hands it back,
    and burns one of that item's retries for nothing.
    """
    a, _ = mind.work.admit(objective="first", work_class="user", origin_actor="ego")
    b, _ = mind.work.admit(objective="second", work_class="user", origin_actor="ego")

    got = mind.work.lease(neuocyte_id="wk2", work_id=b)
    assert got["work_id"] == b
    assert mind.work.get_work(a)["status"] == "queued"
    assert mind.work.get_work(a)["attempt"] == 0

    # A neuocyte targeting an already-leased item gets nothing rather than
    # stealing a different one.
    assert mind.work.lease(neuocyte_id="wk3", work_id=b) is None
    assert mind.work.get_work(a)["status"] == "queued"


def test_concurrent_writers_do_not_break_the_hash_chain(mind):
    """Regression: the supervisor serves RPC from multiple threads.

    A SQLite connection has exactly one transaction. Two threads inside
    apply() would interleave BEGIN/commit, read the same hash-chain tip, and
    publish each other's half-finished work. The visible symptom was a broken
    chain under live load, surfacing as a failed audit dossier.
    """
    import threading

    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def writer(n: int) -> None:
        try:
            barrier.wait(timeout=10)
            for i in range(10):
                mind.memory.remember(
                    kind="belief", claim=f"claim from thread {n} #{i}",
                    confidence=0.5, created_by=f"t{n}",
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == [], errors
    ok, bad = verify_chain(mind.db.conn)
    assert ok is True, f"hash chain broke at {bad}"

    # Every mutation committed exactly once, and versions are a dense sequence.
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM memory_items").fetchone()["n"] == 60
    versions = [r["result_version"] for r in mind.db.conn.execute(
        "SELECT result_version FROM receipts ORDER BY result_version")]
    assert versions == sorted(set(versions))
    assert mind.state_version() == max(versions)


def test_concurrent_heartbeats_do_not_commit_someone_elses_transaction(mind):
    """heartbeat() is the one write outside apply(); it must share the lock."""
    import threading

    stop = threading.Event()
    errors: list[BaseException] = []

    def beat() -> None:
        try:
            while not stop.is_set():
                mind.work.heartbeat("ego")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    mind.work.register_agent(agent_id="ego", role="ego")
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        for i in range(25):
            mind.memory.remember(kind="belief", claim=f"under heartbeat {i}",
                                 confidence=0.5, created_by="ego")
    finally:
        stop.set()
        t.join(timeout=10)

    assert errors == [], errors
    assert verify_chain(mind.db.conn)[0] is True
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM memory_items").fetchone()["n"] == 25


# ---------------------------------------------------------------------------
# Invariant tests written at the layer that owns the guarantee.
#
# Each of these exists because a mutation of the guarantee left the previously
# named test green: the old test passed for a different reason than its name
# claimed. scripts/verify_invariants.py re-checks that by removing each
# guarantee and requiring these to fail.
# ---------------------------------------------------------------------------
def test_writer_itself_refuses_to_reapply_a_mutation_id(mind):
    """I7 at the StateWriter layer.

    test_duplicate_commit_is_idempotent goes through WorkRepo.complete, which
    carries its *own* receipt_for short-circuit -- so it stayed green with the
    writer's idempotency check removed. The guarantee belongs to the writer, so
    it is asserted against the writer directly.
    """
    calls: list[int] = []

    def body(m):
        calls.append(1)
        m.sql("INSERT INTO memory_items(memory_id, kind, claim, confidence, status,"
              " version, created_by, created_at, updated_at, state_version)"
              " VALUES ('mem_idem','belief','once',0.5,'active',1,'t',0,0,1)")
        m.emit("test.idempotent", {"n": len(calls)})

    first, _ = mind.writer.apply(body, actor="t", mutation_id="same-key")
    second, _ = mind.writer.apply(body, actor="t", mutation_id="same-key")

    assert calls == [1], "the body ran twice for one mutation id"
    assert second.receipt_id == first.receipt_id
    assert second.replayed is True
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind = 'test.idempotent'"
    ).fetchone()["n"] == 1
    assert mind.db.conn.execute(
        "SELECT COUNT(*) AS n FROM memory_items WHERE memory_id = 'mem_idem'"
    ).fetchone()["n"] == 1


def test_each_lease_advances_the_fencing_token(mind):
    """I8 at the lease layer.

    test_stale_worker_result_is_fenced expires the lease first, and
    expire_leases bumps the token too -- so it stayed green with the bump
    removed from lease(). Leasing is what must advance the token, so lease it
    twice with no expiry in between.
    """
    work_id, _ = mind.work.admit(objective="o", work_class="user", origin_actor="ego")
    first = mind.work.lease(neuocyte_id="nc1")
    assert first["work_id"] == work_id
    token_a = first["fencing_token"]

    # Hand it back without expiring, then lease again.
    mind.work.fail(work_id=work_id, neuocyte_id="nc1", fencing_token=token_a,
                   failure="gave up", requeue=True)
    second = mind.work.lease(neuocyte_id="nc2")
    token_b = second["fencing_token"]

    assert token_b > token_a, "a fresh lease did not advance the fencing token"
    with pytest.raises(Fenced):
        mind.work.complete(work_id=work_id, neuocyte_id="nc1",
                           fencing_token=token_a, result={"stale": True})
    mind.work.complete(work_id=work_id, neuocyte_id="nc2", fencing_token=token_b,
                       result={"fresh": True})
    assert mind.work.get_work(work_id)["result"] == {"fresh": True}


def test_a_broken_chain_link_is_detected_not_just_a_tampered_payload(mind):
    """I5, the half the existing test did not cover.

    test_hash_chain_detects_tampering rewrites a payload, which the recomputed
    event hash catches. It stayed green with the prev_hash link check removed,
    because that check defends a different attack: removing or reordering
    events while leaving each one internally consistent.
    """
    for i in range(5):
        mind.writer.apply(lambda m, i=i: m.emit("chain.test", {"i": i}), actor="t")
    assert verify_chain(mind.db.conn)[0] is True

    # Excise a middle event. Every surviving row still hashes correctly on its
    # own; only the prev_hash linkage reveals the hole.
    row = mind.db.conn.execute(
        "SELECT seq, event_id FROM events WHERE kind = 'chain.test'"
        " ORDER BY seq LIMIT 1 OFFSET 2").fetchone()
    mind.db.conn.execute("DELETE FROM events WHERE seq = ?", (row["seq"],))
    mind.db.conn.commit()

    ok, bad = verify_chain(mind.db.conn)
    assert ok is False, "excising an event left the chain looking intact"
    assert bad is not None


def test_durability_pragma_is_set_where_durability_is_configured(cfg):
    """I6 at the layer the guarantee actually lives.

    test_acknowledged_mutation_survives_restart proves data survives a *clean*
    close and reopen. It cannot observe fsync, and stayed green with
    synchronous=OFF -- so it does not express crash durability. That property
    is configured by a pragma, so it is asserted where it is configured.
    """
    from amoeba.store.db import Database

    db = Database(cfg.db_path)
    try:
        mode = db.conn.execute("PRAGMA synchronous").fetchone()[0]
        # 2 == FULL. Anything lower does not fsync on commit.
        assert mode == 2, f"synchronous is {mode}, not FULL; commits may not fsync"
        journal = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal"
    finally:
        db.close()


def test_nothing_writes_to_the_store_outside_the_single_writer():
    """One writer, or the atomicity claim (I1) is only mostly true.

    A raw `conn.commit()` anywhere outside `StateWriter` can commit another
    mutation's half-built transaction, because the writer owns that
    connection. This caught a real instance: the incarnation stamp in
    `register_agent` did exactly that, and the visible symptom -- bindings
    with a NULL incarnation -- was the harmless half of the problem.

    Reads through `conn.execute` are fine and deliberately allowed; it is
    committing and mutating that must go through the writer.
    """
    import re

    root = Path(__file__).resolve().parents[1] / "src" / "amoeba"
    offenders = []
    for path in root.rglob("*.py"):
        # store/ is the writer's own home; that is where commits belong.
        if "store" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".commit()" in stripped:
                offenders.append(f"{path.name}:{lineno} {stripped[:70]}")
            if re.search(r"conn\.execute\(\s*$", stripped):
                continue
            m = re.search(r"conn\.execute\(\s*[\"']([A-Z]+)", stripped)
            if m and m.group(1) not in ("SELECT", "PRAGMA"):
                offenders.append(f"{path.name}:{lineno} {stripped[:70]}")
    assert not offenders, "direct store writes outside the writer:\n" + \
        "\n".join(offenders)


# ---------------------------------------------------------------------------
# An existing database survives the upgrade that gave requests their answers.
# ---------------------------------------------------------------------------
OWNERSHIP_COLUMNS = ("expects_answer", "lineage", "ambient", "answer_sha256",
                     "answer_status", "answered_at", "answered_by_turn")


def _make_previous_release_shape(path: Path) -> None:
    """Undo, in a real database, exactly what this release added.

    Building the old shape by subtraction rather than by a pasted copy of the
    old DDL: a copy would drift the moment the live schema changes, and the
    test would then be comparing one piece of history against another.
    """
    raw = sqlite3.connect(path)
    try:
        for index in ("ix_trigger_lineage", "ix_trigger_awaiting"):
            raw.execute(f"DROP INDEX IF EXISTS {index}")
        for column in OWNERSHIP_COLUMNS:
            raw.execute(f"ALTER TABLE role_triggers DROP COLUMN {column}")
        raw.execute("ALTER TABLE role_turns DROP COLUMN lineage")
        raw.commit()
    finally:
        raw.close()


def test_a_database_from_the_previous_release_gains_the_ownership_columns(tmp_path):
    """Upgrading in place must not leave the organism unable to think.

    The schema is applied with CREATE TABLE IF NOT EXISTS, which does nothing
    to a table that already exists. Without a migration, a database written
    before answers belonged to requests keeps the old `role_triggers` and the
    first mailbox read fails on a missing column -- an Amoeba that starts,
    heartbeats, and cannot take a single turn.
    """
    from amoeba.store.db import Database

    path = tmp_path / "state.db"
    Database(path).close()
    _make_previous_release_shape(path)

    with sqlite3.connect(path) as raw:
        old = {row[1] for row in raw.execute("PRAGMA table_info(role_triggers)")}
    assert not (old & set(OWNERSHIP_COLUMNS)), "the old shape was not built"

    db = Database(path)
    try:
        now = {row[1] for row in db.conn.execute("PRAGMA table_info(role_triggers)")}
        assert set(OWNERSHIP_COLUMNS) <= now
        assert "lineage" in {r[1] for r in db.conn.execute("PRAGMA table_info(role_turns)")}
        # The indexes name the new columns, so they are the part that fails
        # first if the migration runs after the schema script instead of
        # before it.
        indexes = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"ix_trigger_lineage", "ix_trigger_awaiting"} <= indexes
    finally:
        db.close()


def test_the_upgrade_does_not_invent_answers_for_old_requests(tmp_path):
    """A trigger that predates the column is owed nothing, not owed an answer.

    `expects_answer` defaults to 0 and `lineage` to NULL, which reads as
    "nobody is waiting on this, and it belongs to no interaction". The other
    default would have the first turn after an upgrade hand an unrelated
    answer to a stranger.
    """
    from amoeba.store.db import Database

    path = tmp_path / "state.db"
    db = Database(path)
    db.conn.execute(
        "INSERT INTO role_triggers(trigger_id, target_role, kind, source,"
        " summary, status, created_at, state_version)"
        " VALUES ('t-old', 'ego', 'message', 'operator', 'hi', 'queued', 1.0, 1)")
    db.conn.commit()
    db.close()

    _make_previous_release_shape(path)
    db = Database(path)
    try:
        row = db.conn.execute(
            "SELECT expects_answer, lineage, ambient, answer_status"
            " FROM role_triggers WHERE trigger_id = 't-old'").fetchone()
    finally:
        db.close()
    assert tuple(row) == (0, None, 0, None)
