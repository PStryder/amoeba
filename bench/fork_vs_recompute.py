"""How closely does a forked prefix match an exact recomputation?

A fork reuses the source's physical KV cells. A recomputation re-evaluates the
same tokens into *different* cells. Attention reduces over cells, so the two
paths sum the same values in a different order. That is enough to produce tiny
floating-point differences, which can flip a greedy argmax at a near-tie.

This measures the actual difference instead of assuming bit-equality, and
reports how long greedy decoding stays identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amoeba.backends.llama_engine import LlamaEngine  # noqa: E402
from amoeba.config import load_config  # noqa: E402

PROMPTS = [
    "The observatory logged three readings on the night of the storm: a pressure "
    "drop, an unexplained radio burst, and a power failure at 02:14. The duty "
    "officer was named Marguerite Olabode.",
    "An amoeba keeps raw history as evidence and maintained beliefs as "
    "memory. The two must never be confused, because evidence does not decay and "
    "interpretations do.",
]
QUESTION = " In one sentence, what is the single most important detail above?"
MAX_TOKENS = 48


def run() -> list[dict]:
    real = load_config(ROOT / "config.toml")
    eng = LlamaEngine(
        runtime_dir=Path(real.backend.lib_path).parent,
        model_path=real.backend.model_path,
        n_ctx=8192, n_seq_max=6, n_gpu_layers=real.backend.n_gpu_layers,
        n_batch=512, n_ubatch=512, kv_unified=True,
    )
    eng.load()
    out = []
    try:
        for idx, prompt in enumerate(PROMPTS):
            src = eng.open_session(role="ego")
            tokens = eng.tokenize(prompt, add_special=False)
            eng.ingest(src.session_id, tokens)

            forked = eng.fork_prefix(src_session_id=src.session_id,
                                     prefix_len=src.n_past, role="neuocyte")
            recomp = eng.open_session(role="neuocyte")
            eng.restore_prefix(session_id=recomp.session_id, tokens=tokens)

            q = eng.tokenize(QUESTION, add_special=False)
            eng.ingest(forked.session_id, q)
            eng.ingest(recomp.session_id, q)

            lf = eng.get_session(forked.session_id).logits.astype(np.float64)
            lr = eng.get_session(recomp.session_id).logits.astype(np.float64)
            max_abs = float(np.max(np.abs(lf - lr)))
            cos = float(lf @ lr / (np.linalg.norm(lf) * np.linalg.norm(lr)))

            def softmax(x):
                e = np.exp(x - x.max())
                return e / e.sum()

            pf, pr = softmax(lf), softmax(lr)
            kl = float(np.sum(pf * np.log((pf + 1e-12) / (pr + 1e-12))))
            top_f = np.argsort(-lf)[:10].tolist()
            top_r = np.argsort(-lr)[:10].tolist()

            # CONTROL: a second, independent recomputation of the same tokens
            # into a different set of cells. If this also differs from the
            # first recomputation, the difference is a property of cache
            # position and kernel reduction order -- not of forking.
            control = eng.open_session(role="neuocyte")
            eng.restore_prefix(session_id=control.session_id, tokens=tokens)
            eng.ingest(control.session_id, q)
            lc = eng.get_session(control.session_id).logits.astype(np.float64)
            control_max_abs = float(np.max(np.abs(lr - lc)))
            control_identical_logits = bool(np.array_equal(lr, lc))
            fork_identical_logits = bool(np.array_equal(lf, lr))
            gc_ = eng.generate(control.session_id, max_tokens=MAX_TOKENS, temperature=0.0)

            gf = eng.generate(forked.session_id, max_tokens=MAX_TOKENS, temperature=0.0)
            gr = eng.generate(recomp.session_id, max_tokens=MAX_TOKENS, temperature=0.0)
            divergence = None
            for i, (a, b) in enumerate(zip(gf.tokens, gr.tokens)):
                if a != b:
                    divergence = i
                    break
            identical = divergence is None and len(gf.tokens) == len(gr.tokens)

            control_div = None
            for i, (a, b) in enumerate(zip(gr.tokens, gc_.tokens)):
                if a != b:
                    control_div = i
                    break

            out.append({
                "prompt_index": idx,
                "control_recompute_vs_recompute_max_abs_diff": control_max_abs,
                "control_recompute_logits_bit_identical": control_identical_logits,
                "control_recompute_first_divergence_step": control_div,
                "fork_vs_recompute_logits_bit_identical": fork_identical_logits,
                "prefix_tokens": len(tokens),
                "logits_max_abs_diff": max_abs,
                "logits_cosine": cos,
                "kl_divergence_nats": kl,
                "top1_same": top_f[0] == top_r[0],
                "top5_same_set": set(top_f[:5]) == set(top_r[:5]),
                "top10_same_order": top_f == top_r,
                "greedy_identical": identical,
                "greedy_first_divergence_step": divergence,
                "greedy_tokens_generated": [len(gf.tokens), len(gr.tokens)],
                "forked_text": gf.text.strip()[:220],
                "recomputed_text": gr.text.strip()[:220],
            })
            for s in (forked, recomp, control, src):
                eng.close_session(s.session_id)
    finally:
        eng.close()
    return out


def main() -> int:
    results = run()
    for r in results:
        print(f"\n--- prompt {r['prompt_index']} ({r['prefix_tokens']} prefix tokens) ---")
        for k in ("logits_max_abs_diff", "logits_cosine", "kl_divergence_nats",
                  "top1_same", "top5_same_set", "top10_same_order",
                  "greedy_identical", "greedy_first_divergence_step",
                  "fork_vs_recompute_logits_bit_identical",
                  "control_recompute_vs_recompute_max_abs_diff",
                  "control_recompute_logits_bit_identical",
                  "control_recompute_first_divergence_step"):
            print(f"  {k}: {r[k]}")
        print(f"  forked:      {r['forked_text']!r}")
        print(f"  recomputed:  {r['recomputed_text']!r}")
    outdir = ROOT / "bench" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fork_vs_recompute.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir / 'fork_vs_recompute.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
