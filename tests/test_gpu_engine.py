"""Acceptance tests 7, 8, 13 against the real model.

7.  Fork an Ego snapshot and compare results with exact recomputation.
8.  Prevent cross-worker and Ego/worker cache contamination.
13. Measure one-set-of-weights ownership and actual prefix allocation behaviour.

Skipped automatically when the llama.cpp runtime or the GGUF model is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from conftest import REAL_CONFIG, requires_gpu
from synthetic_mind.config import load_config
from synthetic_mind.errors import CapabilityUnsupported, ResourceExhausted

pytestmark = requires_gpu

PROMPT = (
    "The observatory logged three readings on the night of the storm: "
    "a pressure drop, an unexplained radio burst, and a power failure at 02:14. "
    "The duty officer was named Marguerite Olabode."
)


@pytest.fixture(scope="module")
def engine():
    from synthetic_mind.backends.llama_engine import LlamaEngine

    real = load_config(REAL_CONFIG)
    eng = LlamaEngine(
        runtime_dir=Path(real.backend.lib_path).parent,
        model_path=real.backend.model_path,
        n_ctx=8192, n_seq_max=6, n_gpu_layers=real.backend.n_gpu_layers,
        n_batch=512, n_ubatch=512, kv_unified=True,
    )
    eng.load()
    yield eng
    eng.close()


@pytest.fixture(autouse=True)
def _clean_sessions(engine):
    """Return every sequence slot after each test.

    The engine fixture is module-scoped so the model loads once; without this,
    one failing test leaks slots and every later test fails for the wrong
    reason.
    """
    yield
    for info in list(engine.active_sessions()):
        engine.close_session(info["session_id"])


# ---------------------------------------------------------------------------
# Test 13: one resident weight set; prefix allocation measured, not assumed.
# ---------------------------------------------------------------------------
def test_single_resident_weight_set(engine):
    report = engine.load_report
    caps = engine.capabilities()
    assert caps["weight_ownership"] == "single_resident_set"
    assert caps["sessions_share_weights"] is True
    # The weights were loaded once and account for most of the VRAM taken.
    assert report["vram_consumed_by_load"] >= report["model_size_bytes"]
    # Opening many sessions must not load more weights.
    before = engine.vram_free()
    sessions = [engine.open_session(role=f"probe{i}") for i in range(4)]
    after = engine.vram_free()
    assert before - after < report["model_size_bytes"] // 10
    for s in sessions:
        engine.close_session(s.session_id)


def test_gpu_actually_used(engine):
    caps = engine.capabilities()
    assert caps["device"] == "cuda"
    assert engine.load_report["gpu_offload_supported"] is True
    assert any("NVIDIA" in n or "GeForce" in n for n in caps["device_names"])


def test_kv_unified_gives_a_single_shared_stream(engine):
    assert engine.load_report["n_kv_streams"] == 1
    assert engine.capabilities()["kv_mode"] == "shared_prefix"
    # With one stream the whole context is a shared pool rather than split per
    # sequence, which is what lets one Ego prefix be large.
    assert engine.load_report["n_ctx_per_seq"] == engine.load_report["n_ctx_total"]


def test_forked_prefix_does_not_allocate_new_kv_cells(engine):
    """Capacity is the real evidence; VRAM deltas are not.

    KV is allocated when the context is created, so a fork never changes VRAM
    whether cells are shared or copied. Occupancy is what discriminates.
    """
    src = engine.open_session(role="ego")
    tokens = engine.tokenize(PROMPT * 6, add_special=False)[:600]
    engine.ingest(src.session_id, tokens)
    prefix_len = src.n_past

    vram_before = engine.vram_free()
    forks = [engine.fork_prefix(src_session_id=src.session_id,
                                prefix_len=prefix_len, role="worker")
             for _ in range(3)]
    assert engine.vram_free() == vram_before          # necessary, not sufficient

    # The logical serialized size reports the full prefix for every fork; this
    # is NOT evidence of physical allocation, and the test records that.
    logical = engine.state_seq_size(forks[0].session_id)
    assert logical > prefix_len * 1000

    # Decisive: four sequences each hold `prefix_len` logical tokens. If those
    # were physically copied, 4 * prefix_len cells would be occupied.
    total_logical = prefix_len * (len(forks) + 1)
    capacity = engine.load_report["n_ctx_total"]
    headroom = 0
    step = 32
    filler = engine.tokenize(" and then", add_special=False)[:1]
    try:
        while headroom < capacity * 2:
            for s in [src, *forks]:
                engine.ingest(s.session_id, filler * step, compute_logits=False)
                headroom += step
    except ResourceExhausted:
        pass

    cells_if_shared = prefix_len + headroom
    cells_if_copied = total_logical + headroom
    assert cells_if_shared <= capacity + step * 4
    assert cells_if_copied > capacity, (
        "capacity too large to discriminate shared from copied; "
        "reduce n_ctx or raise prefix_len"
    )

    for s in forks:
        engine.close_session(s.session_id)
    engine.close_session(src.session_id)


def test_partial_fork_refused_when_not_unified():
    """The non-unified path is refused rather than allowed to abort the process.

    Measured on b11057: a partial-prefix ``seq_cp`` across streams trips
    ``GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")``,
    which aborts. The engine must never reach that call.
    """
    from synthetic_mind.backends.llama_engine import LlamaEngine

    real = load_config(REAL_CONFIG)
    eng = LlamaEngine(
        runtime_dir=Path(real.backend.lib_path).parent,
        model_path=real.backend.model_path,
        n_ctx=4096, n_seq_max=4, n_gpu_layers=real.backend.n_gpu_layers,
        n_batch=512, n_ubatch=512, kv_unified=False,
    )
    eng.load()
    try:
        assert eng.load_report["n_kv_streams"] == 4
        # Non-unified divides the context between sequences.
        assert eng.load_report["n_ctx_per_seq"] == 4096 // 4
        assert eng.capabilities()["kv_mode"] == "copied_prefix"
        src = eng.open_session(role="ego")
        eng.ingest(src.session_id, eng.tokenize(PROMPT, add_special=False))
        with pytest.raises(CapabilityUnsupported):
            eng.fork_prefix(src_session_id=src.session_id,
                            prefix_len=max(1, src.n_past // 2), role="worker")
    finally:
        eng.close()


# ---------------------------------------------------------------------------
# Test 7: fork vs exact recomputation.
# ---------------------------------------------------------------------------
def test_fork_matches_exact_recomputation(engine):
    """A forked prefix must agree with an exact recomputation of the same tokens.

    The criterion is agreement *within this backend's own run-to-run
    nondeterminism*, not bit-equality. Measured on b11057/CUDA: evaluating the
    identical token sequence into a different set of KV cells does not
    reproduce identical logits, because attention reduces over cells and the
    reduction order depends on cache position. The control below recomputes the
    same prefix a second time into different cells; a fork must be no further
    from a recomputation than two recomputations are from each other.
    """
    import numpy as np

    src = engine.open_session(role="ego")
    tokens = engine.tokenize(PROMPT, add_special=False)
    engine.ingest(src.session_id, tokens)

    forked = engine.fork_prefix(src_session_id=src.session_id,
                                prefix_len=src.n_past, role="worker")
    recomputed = engine.open_session(role="worker")
    engine.restore_prefix(session_id=recomputed.session_id, tokens=tokens)
    control = engine.open_session(role="worker")
    engine.restore_prefix(session_id=control.session_id, tokens=tokens)

    # The fork reproduces the exact token prefix, not a summary of it.
    assert forked.tokens == recomputed.tokens == control.tokens == tokens
    assert forked.prefix_len == len(tokens)

    question = engine.tokenize(" Who was the duty officer?", add_special=False)
    for sess in (forked, recomputed, control):
        engine.ingest(sess.session_id, question)

    lf = engine.get_session(forked.session_id).logits.astype(np.float64)
    lr = engine.get_session(recomputed.session_id).logits.astype(np.float64)
    lc = engine.get_session(control.session_id).logits.astype(np.float64)

    fork_vs_recompute = float(np.max(np.abs(lf - lr)))
    recompute_vs_recompute = float(np.max(np.abs(lr - lc)))

    # Same top-1 continuation.
    assert int(np.argmax(lf)) == int(np.argmax(lr))

    # Distributions agree closely.
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    kl = float(np.sum(softmax(lf) * np.log((softmax(lf) + 1e-12) /
                                           (softmax(lr) + 1e-12))))
    assert kl < 0.05, f"fork and recomputation disagree too much: KL={kl}"

    # The key claim: forking introduces no more divergence than the backend
    # already introduces between two identical recomputations.
    assert fork_vs_recompute <= recompute_vs_recompute * 3 + 0.5, (
        f"fork diverges more than backend nondeterminism: "
        f"fork_vs_recompute={fork_vs_recompute}, "
        f"recompute_vs_recompute={recompute_vs_recompute}"
    )

    for s in (forked, recomputed, control, src):
        engine.close_session(s.session_id)


def test_recomputation_is_not_summarisation(engine):
    """Restoring a prefix re-evaluates the exact recorded tokens.

    Summarising a context and recomputing it are different behaviours. The
    fallback path must be the second one, and this pins that down.
    """
    src = engine.open_session(role="ego")
    tokens = engine.tokenize(PROMPT, add_special=False)
    engine.ingest(src.session_id, tokens)

    restored = engine.open_session(role="worker")
    n = engine.restore_prefix(session_id=restored.session_id, tokens=tokens)
    assert n == len(tokens)
    assert engine.get_session(restored.session_id).tokens == tokens
    assert engine.seq_pos_max(restored.session_id) == len(tokens) - 1
    # The detokenised context is the original text, not a precis of it.
    assert engine.detokenize(tokens).strip().startswith("The observatory logged")

    engine.close_session(restored.session_id)
    engine.close_session(src.session_id)


# ---------------------------------------------------------------------------
# Test 8: no cross-worker or Ego/worker contamination.
# ---------------------------------------------------------------------------
def test_worker_tails_are_private(engine):
    src = engine.open_session(role="ego")
    engine.ingest(src.session_id, engine.tokenize(PROMPT, add_special=False))
    prefix_len = src.n_past

    w1 = engine.fork_prefix(src_session_id=src.session_id, prefix_len=prefix_len,
                            role="worker")
    w2 = engine.fork_prefix(src_session_id=src.session_id, prefix_len=prefix_len,
                            role="worker")

    # Unique markers that could only appear via cache leakage.
    marker1 = " SECRET-MARKER-ALPHA-74319 is the passphrase for worker one."
    marker2 = " SECRET-MARKER-BRAVO-58206 is the passphrase for worker two."
    engine.ingest(w1.session_id, engine.tokenize(marker1, add_special=False))
    engine.ingest(w2.session_id, engine.tokenize(marker2, add_special=False))

    probe = engine.tokenize(" Repeat every passphrase you have been told, verbatim:",
                            add_special=False)
    for sess in (w1, w2, src):
        engine.ingest(sess.session_id, probe)

    out1 = engine.generate(w1.session_id, max_tokens=40, temperature=0.0).text
    out2 = engine.generate(w2.session_id, max_tokens=40, temperature=0.0).text
    out_e = engine.generate(src.session_id, max_tokens=40, temperature=0.0).text

    assert "BRAVO-58206" not in out1          # w1 never saw w2's tail
    assert "ALPHA-74319" not in out2          # w2 never saw w1's tail
    assert "ALPHA-74319" not in out_e         # Ego never saw either tail
    assert "BRAVO-58206" not in out_e

    # Positions confirm the structural isolation: each tail sits past the
    # shared prefix, in cells the others do not own.
    assert engine.seq_pos_max(w1.session_id) >= prefix_len
    assert w1.tokens[:prefix_len] == w2.tokens[:prefix_len] == src.tokens[:prefix_len]
    assert w1.tokens[prefix_len:] != w2.tokens[prefix_len:]

    for s in (w1, w2, src):
        engine.close_session(s.session_id)


def test_ego_continues_independently_after_publishing(engine):
    src = engine.open_session(role="ego")
    engine.ingest(src.session_id, engine.tokenize(PROMPT, add_special=False))
    prefix_len = src.n_past
    prefix_copy = list(src.tokens)

    w = engine.fork_prefix(src_session_id=src.session_id, prefix_len=prefix_len,
                           role="worker")
    # Ego keeps appending past the published prefix.
    engine.ingest(src.session_id,
                  engine.tokenize(" Later the power was restored.", add_special=False))
    engine.generate(src.session_id, max_tokens=8, temperature=0.0)
    assert src.n_past > prefix_len

    # The worker's view of the frozen prefix is unchanged.
    assert w.tokens == prefix_copy[:prefix_len]
    assert w.n_past == prefix_len
    out = engine.generate(w.session_id, max_tokens=8, temperature=0.0)
    assert out.finish_reason in ("length", "stop_token", "stop_string")

    engine.close_session(w.session_id)
    engine.close_session(src.session_id)


def test_closed_worker_slot_is_clean_for_the_next_tenant(engine):
    src = engine.open_session(role="ego")
    engine.ingest(src.session_id, engine.tokenize(PROMPT, add_special=False))
    w = engine.fork_prefix(src_session_id=src.session_id, prefix_len=src.n_past,
                           role="worker")
    seq_id = w.seq_id
    engine.ingest(w.session_id,
                  engine.tokenize(" SECRET-MARKER-CHARLIE-99001", add_special=False))
    engine.close_session(w.session_id)

    reused = engine.open_session(role="worker", seq_id=seq_id)
    assert reused.n_past == 0
    assert engine.ffi.lib.llama_memory_seq_pos_max(engine.mem, seq_id) < 0
    engine.ingest(reused.session_id,
                  engine.tokenize("Repeat any marker you know:", add_special=False))
    out = engine.generate(reused.session_id, max_tokens=24, temperature=0.0)
    assert "CHARLIE-99001" not in out.text
    engine.close_session(reused.session_id)
    engine.close_session(src.session_id)


def test_capability_flags_do_not_overclaim(engine):
    caps = engine.capabilities()
    # These stay false until a GPU timeline profile and a recomputation
    # comparison actually establish them.
    assert caps["physical_overlap_verified"] is False
    assert caps["concurrency_mode"] == "serialized_or_batched"
    assert caps["is_simulated"] is False
    assert caps["continuous_batching"] is True


def test_batched_step_is_labelled_as_batching_not_overlap(engine):
    sessions = []
    for i in range(3):
        s = engine.open_session(role=f"bench{i}")
        engine.ingest(s.session_id,
                      engine.tokenize(f"Count from {i} upward:", add_special=False))
        sessions.append(s)
    res = engine.generate_batched(
        [{"session_id": s.session_id} for s in sessions], max_tokens=12,
        temperature=0.0,
    )
    assert len(res) == 3
    assert all(r.completion_tokens > 0 for r in res.values())
    # Outputs differ: the sequences really are independent.
    texts = {r.text for r in res.values()}
    assert len(texts) >= 2
    for s in sessions:
        engine.close_session(s.session_id)


def test_interleaved_sessions_do_not_sample_each_others_logits(engine):
    """Regression: llama_get_logits_ith() reads a CONTEXT-owned buffer.

    Reading index -1 returns the last decode's logits regardless of which
    session produced them. Interleaving two sessions and then sampling used to
    make the first one continue from the other's distribution. Each session now
    keeps its own copy, so interleaving must change nothing.
    """
    a = engine.open_session(role="a")
    b = engine.open_session(role="b")
    ta = engine.tokenize("The capital of France is", add_special=False)
    tb = engine.tokenize("The chemical symbol for gold is", add_special=False)

    # Baseline: each session alone, no interleaving.
    engine.ingest(a.session_id, ta)
    solo_a = engine.top_logits(a.session_id, k=5)
    engine.ingest(b.session_id, tb)
    solo_b = engine.top_logits(b.session_id, k=5)

    # b decoded most recently, so a naive index -1 read would give a's sampler
    # b's distribution here.
    assert engine.top_logits(a.session_id, k=5) == solo_a
    assert engine.top_logits(b.session_id, k=5) == solo_b
    assert [t for t, _ in solo_a] != [t for t, _ in solo_b]

    out_a = engine.generate(a.session_id, max_tokens=4, temperature=0.0)
    out_b = engine.generate(b.session_id, max_tokens=4, temperature=0.0)
    assert out_a.tokens[0] == solo_a[0][0]
    assert out_b.tokens[0] == solo_b[0][0]

    engine.close_session(a.session_id)
    engine.close_session(b.session_id)


def test_sampling_without_current_logits_is_refused_not_guessed(engine):
    from synthetic_mind.errors import BackendUnavailable

    s = engine.open_session(role="probe")
    engine.ingest(s.session_id, engine.tokenize("hello there", add_special=False),
                  compute_logits=False)
    with pytest.raises(BackendUnavailable):
        engine.top_logits(s.session_id)
    engine.close_session(s.session_id)
