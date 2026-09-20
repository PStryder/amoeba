"""Does a fuller unified KV pool slow down an unrelated single-session decode?

Serial aggregate throughput fell from ~163 to ~131 tok/s as the session count
rose, even though serial decode advances exactly one sequence at a time. Per
sequence nothing changed, so the candidate explanation is the shared pool:
with ``kv_unified=True`` there is one KV stream, and attention for a ubatch is
computed over the pool's used extent with masking. If that is right, resident
*idle* sessions should tax a decode that has nothing to do with them.

Controlled test: measure one session's decode rate with an empty pool, then
again with N other sessions resident and idle, then again after retiring them.
Same session, same tokens, same everything else.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthetic_mind.backends.llama_engine import LlamaEngine  # noqa: E402
from synthetic_mind.config import load_config  # noqa: E402

N_CTX = 32768
N_SEQ_MAX = 64
BALLAST_TOKENS = 400      # per idle session
MEASURE_TOKENS = 64
REPEATS = 5


def measure(eng: LlamaEngine, probe_id: str, prompt_tokens: list[int]) -> float:
    """Tokens/sec for one fresh decode of the probe session."""
    rates = []
    for _ in range(REPEATS):
        eng.reset_session(probe_id)
        eng.ingest(probe_id, prompt_tokens)
        r = eng.generate(probe_id, max_tokens=MEASURE_TOKENS, temperature=0.0)
        rates.append(r.completion_tokens / max(r.total_seconds, 1e-9))
    return statistics.median(rates)


def main() -> int:
    real = load_config(ROOT / "config.toml")
    eng = LlamaEngine(
        runtime_dir=Path(real.backend.lib_path).parent,
        model_path=real.backend.model_path,
        n_ctx=N_CTX, n_seq_max=N_SEQ_MAX, n_gpu_layers=real.backend.n_gpu_layers,
        n_batch=1024, n_ubatch=512, kv_unified=True,
    )
    eng.load()
    out: dict = {"n_ctx": N_CTX, "n_seq_max": N_SEQ_MAX,
                 "ballast_tokens_per_session": BALLAST_TOKENS,
                 "measure_tokens": MEASURE_TOKENS, "repeats": REPEATS,
                 "stages": []}
    try:
        probe = eng.open_session(role="probe")
        prompt = eng.tokenize(
            "Answer in a single short sentence: what does a fencing token prevent?",
            add_special=False)

        # warm up
        eng.ingest(probe.session_id, prompt)
        eng.generate(probe.session_id, max_tokens=8, temperature=0.0)

        filler = eng.tokenize(
            "The mind records evidence before it forms beliefs. ", add_special=False)
        ballast_tokens = (filler * (BALLAST_TOKENS // len(filler) + 2))[:BALLAST_TOKENS]

        def stage(name: str, idle: int) -> None:
            rate = measure(eng, probe.session_id, prompt)
            occupied = idle * BALLAST_TOKENS + eng.get_session(probe.session_id).n_past
            out["stages"].append({
                "stage": name, "idle_sessions": idle,
                "approx_cells_occupied": occupied,
                "pool_utilisation": occupied / N_CTX,
                "probe_tokens_per_s": rate,
            })
            print(f"{name:28s} idle={idle:3d} cells~{occupied:6d} "
                  f"({occupied/N_CTX:5.1%})  probe={rate:7.1f} tok/s")

        stage("empty pool", 0)

        ballast = []
        for target in (16, 32, 48, 63):
            while len(ballast) < target:
                s = eng.open_session(role="ballast")
                eng.ingest(s.session_id, ballast_tokens, compute_logits=False)
                ballast.append(s)
            stage(f"{target} idle sessions resident", target)

        for s in ballast:
            eng.close_session(s.session_id)
        stage("after retiring ballast", 0)

        base = out["stages"][0]["probe_tokens_per_s"]
        peak = out["stages"][-2]["probe_tokens_per_s"]
        recovered = out["stages"][-1]["probe_tokens_per_s"]
        out["verdict"] = {
            "empty_pool_tokens_per_s": base,
            "full_pool_tokens_per_s": peak,
            "slowdown_factor": base / max(peak, 1e-9),
            "recovered_after_retirement_tokens_per_s": recovered,
            "recovery_fraction": recovered / max(base, 1e-9),
            "conclusion": (
                "resident idle sessions tax unrelated decodes"
                if base / max(peak, 1e-9) > 1.05
                else "no measurable tax from resident idle sessions"
            ),
        }
        print("\n" + json.dumps(out["verdict"], indent=2))
    finally:
        eng.close()

    outdir = ROOT / "bench" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "kv_occupancy_tax.json").write_text(json.dumps(out, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {outdir / 'kv_occupancy_tax.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
