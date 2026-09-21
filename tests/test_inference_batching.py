"""The batching scheduler, tested where the scheduling actually lives.

`generate_batched` existed for a long time and nothing on the live path called
it, so none of this had ever run outside a benchmark. What is asserted here is
the *scheduling* -- grouping, ordering, isolation of failures, the fallback --
and deliberately not throughput: no test in this file measures speed, because
the backends it uses do not model batching and a number taken from them would
be a number about a loop.

The fakes record what the backend was asked to do, which is the only thing the
service is responsible for. Whether a fused decode is faster than a serial one
is a property of llama.cpp and belongs to `bench/concurrency.py`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from amoeba.backends.llama_engine import GenerationResult, SessionState
from amoeba.config import Config
from amoeba.errors import BackendUnavailable, CapabilityUnsupported
from amoeba.inference_service import InferenceService


def _session(session_id: str = "s") -> SessionState:
    """A real session, not a stand-in.

    The first version of this was a two-line class with an `n_past` attribute,
    which was all the service needed then. When the service began asking a
    session for its budget and basis, the stand-in silently lacked them and
    four batching tests failed on an AttributeError that had nothing to do
    with batching. A fake that reimplements the thing under test agrees with
    itself; the real object agrees with the system.
    """
    return SessionState(session_id=session_id, role="ego", seq_id=0,
                        tokens=list(range(10)),
                        context_budget_tokens=16384, budget_basis="total")


class _FakeBackend:
    """Records every call, and blocks until released so batches can form."""

    backend_kind = "fake"
    is_simulated = True
    model_generation = "fake-1"

    def __init__(self, *, can_batch: bool = True) -> None:
        self.can_batch = can_batch
        self.batched_calls: list[list[str]] = []
        self.single_calls: list[str] = []
        self.gate = threading.Event()
        self.entered = threading.Semaphore(0)
        self.fail_sessions: set[str] = set()

    def get_session(self, session_id: str) -> SessionState:
        return _session(session_id)

    def _result(self, session_id: str) -> GenerationResult:
        return GenerationResult(
            session_id=session_id, text=f"answer for {session_id}", tokens=[1],
            finish_reason="stop", prompt_tokens=10, completion_tokens=3,
            time_to_first_token=0.0, total_seconds=0.0)

    def generate(self, session_id: str, **kwargs: Any) -> GenerationResult:
        self.single_calls.append(session_id)
        self.entered.release()
        self.gate.wait(timeout=10.0)
        if session_id in self.fail_sessions:
            raise BackendUnavailable(f"{session_id} is broken")
        return self._result(session_id)

    def generate_batched(self, requests, **kwargs: Any
                         ) -> dict[str, GenerationResult]:
        if not self.can_batch:
            raise CapabilityUnsupported("this backend does not batch")
        ids = [r["session_id"] for r in requests]
        self.batched_calls.append(ids)
        for _ in ids:
            self.entered.release()
        self.gate.wait(timeout=10.0)
        if self.fail_sessions.intersection(ids):
            raise BackendUnavailable("a session in this batch is broken")
        return {sid: self._result(sid) for sid in ids}


def _service(tmp_path, backend: _FakeBackend) -> InferenceService:
    cfg = Config()
    cfg.state_dir = tmp_path / "state"
    cfg.runtime_dir = tmp_path / "runtime"
    cfg.models_dir = tmp_path / "models"
    cfg.ensure_dirs()
    svc = InferenceService(cfg)
    svc.backend = backend
    return svc


def _call_many(svc, session_ids, results, errors):
    threads = []
    for sid in session_ids:
        def run(sid=sid):
            try:
                results[sid] = svc.generate(session_id=sid, max_tokens=8)
            except Exception as exc:  # noqa: BLE001
                errors[sid] = exc
        # Daemon on purpose: a mutation that kills the dispatcher leaves
        # these blocked forever, and a non-daemon thread would hang the
        # whole run instead of failing one test.
        t = threading.Thread(target=run, name=f"caller-{sid}", daemon=True)
        t.start()
        threads.append(t)
    return threads


# ---------------------------------------------------------------------------
def test_a_lone_generation_is_not_delayed_waiting_for_company(tmp_path):
    """The idle path costs nothing.

    The usual aggregator waits a few milliseconds hoping for a second request
    and charges that to the first one whether or not it arrives. This one
    never waits, so a single caller on an idle machine sees the backend
    immediately and is told the batch held only itself.
    """
    backend = _FakeBackend()
    backend.gate.set()
    svc = _service(tmp_path, backend)

    out = svc.generate(session_id="s1", max_tokens=8)
    assert out["batch_size"] == 1
    assert out["text"] == "answer for s1"
    assert backend.batched_calls == [], "one request was sent as a batch"
    assert backend.single_calls == ["s1"]


def test_concurrent_generations_are_decoded_together(tmp_path):
    """Contention is what produces a batch, and the only thing that does.

    The first caller is held inside the backend while the others queue behind
    it. When it returns, everything that piled up goes in one group -- which
    is the behaviour the live path never had, because every caller used to
    queue behind the engine lock one at a time.
    """
    backend = _FakeBackend()
    svc = _service(tmp_path, backend)
    results: dict[str, Any] = {}
    errors: dict[str, Any] = {}

    first = _call_many(svc, ["s1"], results, errors)
    assert backend.entered.acquire(timeout=5.0), "the first call never started"

    rest = _call_many(svc, ["s2", "s3", "s4"], results, errors)
    # Let them reach the queue before the dispatcher is free again.
    deadline = time.time() + 5.0
    while svc._pending.qsize() < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert svc._pending.qsize() == 3, "the later callers never queued"

    backend.gate.set()
    for t in first + rest:
        t.join(timeout=10.0)

    assert not errors, errors
    assert backend.batched_calls, "nothing was ever batched"
    grouped = backend.batched_calls[0]
    assert set(grouped) == {"s2", "s3", "s4"}
    for sid in ("s2", "s3", "s4"):
        assert results[sid]["batch_size"] == 3
        assert results[sid]["text"] == f"answer for {sid}"
    assert results["s2"]["queue_delay_seconds"] >= 0.0


def test_a_backend_that_cannot_batch_still_serves_everyone(tmp_path):
    """A backend without batching is not an error, it is a backend.

    The fallback matters more than the fast path: every deployment that is not
    llama.cpp goes through it, and a scheduler that only works on one engine
    is a scheduler nobody can change engines under.
    """
    backend = _FakeBackend(can_batch=False)
    svc = _service(tmp_path, backend)
    results: dict[str, Any] = {}
    errors: dict[str, Any] = {}

    first = _call_many(svc, ["s1"], results, errors)
    assert backend.entered.acquire(timeout=5.0)
    rest = _call_many(svc, ["s2", "s3"], results, errors)
    deadline = time.time() + 5.0
    while svc._pending.qsize() < 2 and time.time() < deadline:
        time.sleep(0.01)
    backend.gate.set()
    for t in first + rest:
        t.join(timeout=10.0)

    assert not errors, errors
    assert set(results) == {"s1", "s2", "s3"}
    assert all(r["batch_size"] == 1 for r in results.values())
    assert sorted(backend.single_calls) == ["s1", "s2", "s3"]


def test_one_broken_session_does_not_fail_the_others(tmp_path):
    """An error lands on the request that caused it.

    Without this, batching would make the organism strictly less reliable: one
    caller with a bad session would take down every other caller that happened
    to be decoding alongside it, and the failure would be reported to people
    who did nothing wrong.
    """
    backend = _FakeBackend()
    backend.fail_sessions = {"s3"}
    svc = _service(tmp_path, backend)
    results: dict[str, Any] = {}
    errors: dict[str, Any] = {}

    first = _call_many(svc, ["s1"], results, errors)
    assert backend.entered.acquire(timeout=5.0)
    rest = _call_many(svc, ["s2", "s3", "s4"], results, errors)
    deadline = time.time() + 5.0
    while svc._pending.qsize() < 3 and time.time() < deadline:
        time.sleep(0.01)
    backend.gate.set()
    for t in first + rest:
        t.join(timeout=10.0)

    assert set(errors) == {"s3"}, f"the wrong callers failed: {sorted(errors)}"
    assert isinstance(errors["s3"], BackendUnavailable)
    for sid in ("s1", "s2", "s4"):
        assert results[sid]["text"] == f"answer for {sid}"


def test_the_deterministic_backend_batches_without_claiming_to_measure(cfg):
    """The stand-in is honest about what it is.

    It used to refuse, which left the live batching path untestable anywhere
    but on a GPU. It now runs -- as a loop -- and says so in the only place a
    future reader will look before citing a number from it.
    """
    from amoeba.backends.deterministic import DeterministicBackend

    backend = DeterministicBackend()
    backend.load()
    opened = [backend.open_session(role="ego", session_id=sid)
              for sid in ("a", "b")]
    ids = [o["session_id"] if isinstance(o, dict) else sid
           for o, sid in zip(opened, ("a", "b"))]

    out = backend.generate_batched(
        [{"session_id": sid} for sid in ids], max_tokens=8)
    assert set(out) == set(ids)
    for sid in ids:
        assert out[sid].session_id == sid

    doc = DeterministicBackend.generate_batched.__doc__ or ""
    assert "models no batching" in doc
    assert "measures nothing" in doc
