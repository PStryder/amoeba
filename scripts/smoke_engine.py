"""Milestone 2/3 smoke test: load one model, run isolated sessions, fork a prefix.

Run:
    .\.venv\Scripts\python.exe scripts\smoke_engine.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_mind.backends.llama_engine import LlamaEngine  # noqa: E402

RUNTIME = r"F:\hexylab\synthetic-mind-runtime\llama.cpp-b11057-win-cuda12.4"
MODEL = r"F:\hexylab\synthetic-mind-models\Qwen3-4B-Instruct-2507-Q5_K_M.gguf"


def main() -> int:
    logs: list[str] = []
    eng = LlamaEngine(
        runtime_dir=RUNTIME, model_path=MODEL,
        n_ctx=8192, n_seq_max=6, n_gpu_layers=-1, n_threads=8,
        kv_unified=True,
        log_sink=lambda lvl, txt: logs.append(txt.strip()),
    )
    report = eng.load()
    print("=== load report ===")
    for key in ("model_desc", "model_size_bytes", "model_n_params", "n_ctx_total",
                "n_ctx_per_seq", "n_ctx_train", "n_vocab", "kv_unified", "n_kv_streams",
                "gpu_offload_supported", "vram_consumed_by_load", "load_seconds"):
        print(f"  {key}: {report[key]}")
    print("  capabilities:", json.dumps(eng.capabilities(), indent=2))

    print("\n=== chat template present ===")
    tmpl = eng.chat_template()
    print("  has template:", bool(tmpl), "len:", len(tmpl or ""))

    # ---- one real generation, Ego role -------------------------------
    print("\n=== ego session ===")
    ego = eng.open_session(role="ego")
    prompt = eng.apply_chat_template([
        {"role": "system", "content": "You are Ego, the outward-facing half of a synthetic mind. Answer in one short sentence."},
        {"role": "user", "content": "Name the capital of France and one reason it matters."},
    ])
    toks = eng.tokenize(prompt, add_special=False, parse_special=True)
    print(f"  prompt tokens: {len(toks)}")
    eng.ingest(ego.session_id, toks)
    res = eng.generate(ego.session_id, max_tokens=64, temperature=0.0)
    print(f"  text: {res.text.strip()!r}")
    print(f"  finish={res.finish_reason} ttft={res.time_to_first_token*1000:.1f}ms "
          f"tok/s={res.completion_tokens/max(res.total_seconds,1e-9):.1f}")

    # ---- Id gets its own private session ------------------------------
    print("\n=== id session (separate context) ===")
    idsess = eng.open_session(role="id")
    idprompt = eng.apply_chat_template([
        {"role": "system", "content": "You are Id, the inward homeostatic half. Answer in one short sentence."},
        {"role": "user", "content": "What did the other half of this mind just say?"},
    ])
    eng.ingest(idsess.session_id, eng.tokenize(idprompt, parse_special=True))
    res_id = eng.generate(idsess.session_id, max_tokens=48, temperature=0.0)
    print(f"  text: {res_id.text.strip()!r}")
    print("  (isolation check: Id must NOT know about Paris unless it guesses)")

    # ---- fork the Ego prefix into two workers --------------------------
    print("\n=== fork ego prefix into 2 workers ===")
    prefix_len = ego.n_past
    vram_before = eng.vram_free()
    w1 = eng.fork_prefix(src_session_id=ego.session_id, prefix_len=prefix_len, role="worker")
    w2 = eng.fork_prefix(src_session_id=ego.session_id, prefix_len=prefix_len, role="worker")
    vram_after = eng.vram_free()
    print(f"  prefix_len={prefix_len}")
    print(f"  vram free before fork: {vram_before/2**20:.1f} MiB")
    print(f"  vram free after  fork: {vram_after/2**20:.1f} MiB")
    print(f"  delta: {(vram_before - vram_after)/2**20:.3f} MiB "
          f"(0 == physically shared prefix cells)")
    print(f"  state_seq_get_size(worker1) = {eng.state_seq_size(w1.session_id)} bytes "
          f"(LOGICAL size -- not evidence of physical allocation)")

    # workers append private tails
    for w, instruction in ((w1, " Now list one risk."), (w2, " Now list one opportunity.")):
        eng.ingest(w.session_id, eng.tokenize(instruction, parse_special=False))
        r = eng.generate(w.session_id, max_tokens=32, temperature=0.0)
        print(f"  worker {w.seq_id}: {r.text.strip()[:90]!r}")

    # ---- ego continues independently after publishing ------------------
    print("\n=== ego continues after snapshot ===")
    eng.ingest(ego.session_id, eng.tokenize(" And name its river.", parse_special=False))
    r = eng.generate(ego.session_id, max_tokens=32, temperature=0.0)
    print(f"  ego: {r.text.strip()[:120]!r}")
    print(f"  ego n_past={ego.n_past} w1 n_past={w1.n_past} w2 n_past={w2.n_past}")
    print(f"  seq_pos_max ego={eng.seq_pos_max(ego.session_id)} "
          f"w1={eng.seq_pos_max(w1.session_id)} w2={eng.seq_pos_max(w2.session_id)}")

    # ---- retire workers, reclaim -------------------------------------
    print("\n=== retire workers ===")
    eng.close_session(w1.session_id)
    eng.close_session(w2.session_id)
    print("  workers closed; ego prefix must survive")
    eng.ingest(ego.session_id, eng.tokenize(" One more word:", parse_special=False))
    r = eng.generate(ego.session_id, max_tokens=12, temperature=0.0)
    print(f"  ego still alive: {r.text.strip()[:80]!r}")

    if logs:
        print("\n=== backend warnings ===")
        for line in logs[:10]:
            print("  ", line)
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
