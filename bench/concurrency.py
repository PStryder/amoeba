"""Concurrency matrix: serial vs continuous batching, at 1/2/4/8 sessions.

What this measures and what it deliberately does not:

* **Serial** runs one session to completion before the next. One weight set,
  one sequence at a time.
* **Batched** advances every active session by one token per ``llama_decode``:
  a single fused kernel launch covering several independent sequences. This is
  continuous batching.
* **Independent overlapping execution** -- distinct inference streams whose
  kernels genuinely run at the same time on the device -- is NOT measured here
  and is not claimed. Establishing it needs a GPU timeline profile; this script
  reports whether such a profiler is even available, and otherwise records
  overlap as ``unknown``.

Throughput improvement is not evidence of overlap. Neither is asynchrony,
neither is a shared model pointer.

Usage:
    .\\.venv\\Scripts\\python.exe bench\\concurrency.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amoeba.backends.llama_engine import LlamaEngine  # noqa: E402
from amoeba.config import load_config  # noqa: E402

# Geometric sweep. Override with --counts to zoom in on a bend.
SESSION_COUNTS = (1, 2, 4, 8, 16, 32, 64)
MAX_TOKENS = 64
# n_ctx is the TOTAL shared KV pool. 64 sessions x (prompt + 64 completion)
# needs roughly 8k cells; 32768 leaves headroom and costs ~4.6 GiB at
# 147,470 bytes/token.
N_CTX = 32768
N_SEQ_MAX = 64

# Unequal request lengths, short and long contexts, distinct content so
# cross-session contamination would be visible in the output.
PROMPTS = [
    "List three properties of a durable append-only event log.",
    "In two sentences, explain why raw history is not the same as memory.",
    ("A supervisor owns admission control, leases and hard resource caps. "
     "A model may request work but cannot raise a cap. " * 6
     + " Summarise the paragraph above in one sentence."),
    "Name one failure mode of reference-counted cache reclamation.",
    "Why does a fencing token stop a replaced neuocyte from committing a result?",
    ("The observatory logged a pressure drop, a radio burst and a power failure "
     "at 02:14, and the duty officer was Marguerite Olabode. " * 6
     + " Who was the duty officer?"),
    "Give one reason a hash chain is not administrator-proof.",
    "State one difference between batching and independent kernel overlap.",
]
MAX_SESSIONS = 64
# One unique marker per session, so leakage between any two of the 64 shows up.
MARKERS = [f"ZEBRA-{i:04d}" for i in range(MAX_SESSIONS)]


def prompt_for(i: int) -> str:
    """Cycle the prompt set; the marker is what makes each session distinct."""
    return PROMPTS[i % len(PROMPTS)]


def profiler_available() -> dict:
    """Which profilers exist, and can any of them actually show kernel overlap?

    Only a *timeline* profiler can. Nsight Compute is a kernel profiler that
    serialises launches to collect per-kernel metrics, so its output can never
    show two kernels running at once -- finding it is not the same as being
    able to answer the question.
    """
    found = {t: shutil.which(t) for t in ("nsys", "ncu", "nvprof")}
    found = {k: v for k, v in found.items() if v}
    return {
        "tools_found": found,
        "nsight_systems_present": "nsys" in found,
        "timeline_profiling_possible": "nsys" in found,
        "note": (
            "Nsight Systems (nsys) produces the GPU timeline that could show "
            "concurrent kernels; it is not installed. Nsight Compute (ncu) "
            "serialises kernel launches by design and cannot demonstrate "
            "overlap. nvprof does not support this GPU's compute capability."
        ),
    }


def make_engine(real) -> LlamaEngine:
    eng = LlamaEngine(
        runtime_dir=Path(real.backend.lib_path).parent,
        model_path=real.backend.model_path,
        n_ctx=N_CTX, n_seq_max=N_SEQ_MAX, n_gpu_layers=real.backend.n_gpu_layers,
        n_batch=1024, n_ubatch=512, kv_unified=True,
    )
    eng.load()
    return eng


def build_sessions(eng: LlamaEngine, n: int) -> list:
    sessions = []
    for i in range(n):
        s = eng.open_session(role=f"bench{i}")
        text = f"[{MARKERS[i]}] {prompt_for(i)}"
        rendered = eng.apply_chat_template(
            [{"role": "user", "content": text}], add_assistant=True
        )
        eng.ingest(s.session_id, eng.tokenize(rendered, parse_special=True))
        sessions.append(s)
    return sessions


def check_isolation(results: dict, n: int) -> dict:
    """Any session echoing another session's marker is contamination."""
    leaks = []
    for i, (sid, res) in enumerate(results.items()):
        for j, marker in enumerate(MARKERS[:n]):
            if j != i and marker in res.text:
                leaks.append({"session_index": i, "saw_marker_of": j})
    return {"contamination_detected": bool(leaks), "leaks": leaks}


def run_serial(eng: LlamaEngine, n: int) -> dict:
    sessions = build_sessions(eng, n)
    vram_before = eng.vram_free()
    submitted = time.perf_counter()
    per_session = []
    results = {}
    t0 = time.perf_counter()
    for idx, s in enumerate(sessions):
        queue_delay = time.perf_counter() - submitted
        r = eng.generate(s.session_id, max_tokens=MAX_TOKENS, temperature=0.0)
        results[s.session_id] = r
        per_session.append({
            "index": idx,
            "queue_delay_s": queue_delay,
            "ttft_s": r.time_to_first_token,
            "latency_s": r.total_seconds,
            "completion_tokens": r.completion_tokens,
            "tokens_per_s": r.completion_tokens / max(r.total_seconds, 1e-9),
            "finish_reason": r.finish_reason,
        })
    makespan = time.perf_counter() - t0
    vram_after = eng.vram_free()
    total_tokens = sum(p["completion_tokens"] for p in per_session)
    out = {
        "mode": "serial",
        "sessions": n,
        "makespan_s": makespan,
        "total_completion_tokens": total_tokens,
        "aggregate_tokens_per_s": total_tokens / max(makespan, 1e-9),
        "per_session": per_session,
        "vram_free_before_bytes": vram_before,
        "vram_free_after_bytes": vram_after,
        "vram_used_by_run_bytes": max(0, vram_before - vram_after),
        "isolation": check_isolation(results, n),
        "execution_mode": "one sequence decoded at a time",
        "physical_overlap_verified": False,
    }
    for s in sessions:
        eng.close_session(s.session_id)
    return out


def run_batched(eng: LlamaEngine, n: int) -> dict:
    sessions = build_sessions(eng, n)
    vram_before = eng.vram_free()
    submitted = time.perf_counter()
    t0 = time.perf_counter()
    results = eng.generate_batched(
        [{"session_id": s.session_id, "max_tokens": MAX_TOKENS} for s in sessions],
        max_tokens=MAX_TOKENS, temperature=0.0,
    )
    makespan = time.perf_counter() - t0
    vram_after = eng.vram_free()
    per_session = []
    for idx, s in enumerate(sessions):
        r = results[s.session_id]
        per_session.append({
            "index": idx,
            "queue_delay_s": t0 - submitted,
            "ttft_s": r.time_to_first_token,
            "latency_s": r.total_seconds,
            "completion_tokens": r.completion_tokens,
            "tokens_per_s": r.completion_tokens / max(makespan, 1e-9),
            "finish_reason": r.finish_reason,
        })
    total_tokens = sum(p["completion_tokens"] for p in per_session)
    out = {
        "mode": "batched",
        "sessions": n,
        "makespan_s": makespan,
        "total_completion_tokens": total_tokens,
        "aggregate_tokens_per_s": total_tokens / max(makespan, 1e-9),
        "per_session": per_session,
        "vram_free_before_bytes": vram_before,
        "vram_free_after_bytes": vram_after,
        "vram_used_by_run_bytes": max(0, vram_before - vram_after),
        "isolation": check_isolation(results, n),
        "execution_mode": ("continuous batching: one fused llama_decode per step "
                           "covering all active sequences"),
        "physical_overlap_verified": False,
    }
    for s in sessions:
        eng.close_session(s.session_id)
    return out


def run_prefix_sharing_cost(eng: LlamaEngine) -> dict:
    """Snapshot/fork cost, and the KV a copying implementation would have needed."""
    src = eng.open_session(role="ego")
    long_prompt = (
        "A persistent mind keeps beliefs with supporting and opposing evidence. " * 40
    )
    tokens = eng.tokenize(long_prompt, add_special=False)
    t0 = time.perf_counter()
    eng.ingest(src.session_id, tokens)
    prefill = time.perf_counter() - t0
    prefix_len = src.n_past

    fork_times, recompute_times = [], []
    forks = []
    for _ in range(4):
        t = time.perf_counter()
        forks.append(eng.fork_prefix(src_session_id=src.session_id,
                                     prefix_len=prefix_len, role="neuocyte"))
        fork_times.append(time.perf_counter() - t)

    recompute = eng.open_session(role="neuocyte")
    t = time.perf_counter()
    eng.restore_prefix(session_id=recompute.session_id, tokens=tokens)
    recompute_times.append(time.perf_counter() - t)

    kv_bytes_per_token = eng.state_seq_size(src.session_id) / max(prefix_len, 1)
    saved_cells = prefix_len * len(forks)
    out = {
        "prefix_tokens": prefix_len,
        "prefill_s": prefill,
        "fork_seconds_mean": statistics.fmean(fork_times),
        "fork_seconds_max": max(fork_times),
        "recompute_seconds": recompute_times[0],
        "fork_speedup_vs_recompute": recompute_times[0] / max(statistics.fmean(fork_times), 1e-9),
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_cells_saved_by_sharing": saved_cells,
        "kv_bytes_saved_by_sharing": saved_cells * kv_bytes_per_token,
        "note": ("savings are what a copying implementation would additionally "
                 "have allocated; with kv_unified=True the fork adds no cells"),
    }
    for s in [*forks, recompute, src]:
        eng.close_session(s.session_id)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--counts", type=str, default="",
                    help="comma-separated session counts, e.g. 8,12,16,20,24")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for the output file, e.g. zoom")
    args = ap.parse_args()

    counts = (tuple(int(c) for c in args.counts.split(",") if c.strip())
              if args.counts else SESSION_COUNTS)
    if max(counts) > N_SEQ_MAX:
        raise SystemExit(f"counts exceed n_seq_max={N_SEQ_MAX}")

    real = load_config(ROOT / "config.toml")
    eng = make_engine(real)
    report = {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "devices": eng.devices,
        },
        "runtime": {
            "build": eng.abi_info.get("build_tag"),
            "model": eng.load_report["model_desc"],
            "model_generation": eng.model_generation,
            "model_size_bytes": eng.load_report["model_size_bytes"],
            "n_ctx_total": eng.load_report["n_ctx_total"],
            "n_seq_max": eng.load_report["n_seq_max"],
            "kv_unified": eng.load_report["kv_unified"],
            "vram_consumed_by_load": eng.load_report["vram_consumed_by_load"],
            "load_seconds": eng.load_report["load_seconds"],
        },
        "capabilities": eng.capabilities(),
        "protocol": {
            "max_tokens": MAX_TOKENS,
            "repeats": args.repeats,
            "greedy": True,
            "warmup_excluded": True,
            "prompt_lengths_unequal": True,
        },
        "profiler": profiler_available(),
        "runs": [],
    }

    # Warm up: first decode pays graph build and allocator costs.
    warm = eng.open_session(role="warmup")
    eng.ingest(warm.session_id, eng.tokenize("warm up", add_special=False))
    eng.generate(warm.session_id, max_tokens=8, temperature=0.0)
    eng.close_session(warm.session_id)

    # Randomised order across repeats so drift does not favour one mode.
    plan = []
    for rep in range(args.repeats):
        for n in counts:
            for mode in ("serial", "batched"):
                plan.append((rep, n, mode))
    import random

    random.Random(20260919).shuffle(plan)

    for rep, n, mode in plan:
        fn = run_serial if mode == "serial" else run_batched
        try:
            row = fn(eng, n)
        except Exception as exc:  # noqa: BLE001
            row = {"mode": mode, "sessions": n, "error": repr(exc)}
        row["repeat"] = rep
        report["runs"].append(row)
        agg = row.get("aggregate_tokens_per_s")
        print(f"rep{rep} {mode:8s} n={n}  "
              f"makespan={row.get('makespan_s', float('nan')):.3f}s  "
              f"agg={agg:.1f} tok/s" if agg else f"rep{rep} {mode} n={n} ERROR")

    report["prefix_sharing"] = run_prefix_sharing_cost(eng)
    report["vram_free_at_end"] = eng.vram_free()
    eng.close()

    # ---- summary ----
    summary = {}
    for mode in ("serial", "batched"):
        for n in counts:
            rows = [r for r in report["runs"]
                    if r.get("mode") == mode and r.get("sessions") == n
                    and "error" not in r]
            if not rows:
                continue
            aggs = [r["aggregate_tokens_per_s"] for r in rows]
            makespans = [r["makespan_s"] for r in rows]
            ttfts = [p["ttft_s"] for r in rows for p in r["per_session"]]
            lats = sorted(p["latency_s"] for r in rows for p in r["per_session"])
            summary[f"{mode}_n{n}"] = {
                "aggregate_tokens_per_s_mean": statistics.fmean(aggs),
                "aggregate_tokens_per_s_stdev": (statistics.stdev(aggs)
                                                 if len(aggs) > 1 else 0.0),
                "makespan_s_mean": statistics.fmean(makespans),
                "ttft_s_mean": statistics.fmean(ttfts),
                "latency_p50_s": lats[len(lats) // 2],
                "latency_p95_s": lats[min(len(lats) - 1, int(len(lats) * 0.95))],
                "per_session_tokens_per_s_mean": statistics.fmean(
                    p["tokens_per_s"] for r in rows for p in r["per_session"]),
                "contamination_detected": any(
                    r["isolation"]["contamination_detected"] for r in rows),
                "vram_used_by_run_bytes_max": max(
                    r["vram_used_by_run_bytes"] for r in rows),
            }
    report["summary"] = summary

    scaling = {}
    for mode in ("serial", "batched"):
        rows = [(n, summary[f"{mode}_n{n}"]) for n in counts
                if f"{mode}_n{n}" in summary]
        base = rows[0][1]["aggregate_tokens_per_s_mean"] if rows else 1.0
        prev_n, prev_agg = None, None
        curve = []
        for n, v in rows:
            agg = v["aggregate_tokens_per_s_mean"]
            entry = {
                "sessions": n,
                "aggregate_tokens_per_s": agg,
                "speedup_vs_n1": agg / max(base, 1e-9),
                "scaling_efficiency_vs_n1": (agg / max(base, 1e-9)) / n,
                "makespan_s": v["makespan_s_mean"],
                "ttft_s": v["ttft_s_mean"],
                "per_session_tokens_per_s": v["per_session_tokens_per_s_mean"],
                "latency_p95_s": v["latency_p95_s"],
            }
            if prev_agg is not None:
                # Marginal return on the last doubling: 1.0 = perfect scaling
                # across that step, 0.0 = completely flat.
                ratio = n / prev_n
                entry["gain_over_previous_step"] = agg / max(prev_agg, 1e-9)
                entry["marginal_efficiency_of_step"] = (
                    (agg / max(prev_agg, 1e-9) - 1.0) / max(ratio - 1.0, 1e-9)
                )
            curve.append(entry)
            prev_n, prev_agg = n, agg
        scaling[mode] = curve
    report["scaling"] = scaling
    report["claims"] = {
        "one_resident_weight_set": True,
        "sessions_share_weights": True,
        "continuous_batching_measured": True,
        "independent_kernel_overlap_verified": False,
        "independent_kernel_overlap_status": (
            "NOT ATTEMPTED, not merely unmeasured. This engine serialises every "
            "call into llama.cpp behind one lock because llama.cpp contexts are "
            "not thread safe, so there is no concurrent submission for a "
            "profiler to find. Separately, the tool that could verify overlap "
            "if it were implemented (Nsight Systems) is not installed. "
            "Throughput gains from batching are not evidence of overlap."
        ),
        "throughput_gain_source": (
            "continuous batching: more sequences per fused kernel launch, not "
            "more kernels running at once"
        ),
    }

    outdir = ROOT / "bench" / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"concurrency{('_' + args.tag) if args.tag else ''}.json"
    (outdir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    print(f"{'config':14s} {'agg tok/s':>12s} {'makespan s':>11s} "
          f"{'ttft s':>8s} {'p95 lat':>8s} {'leak':>5s}")
    for key, v in summary.items():
        print(f"{key:14s} {v['aggregate_tokens_per_s_mean']:12.1f} "
              f"{v['makespan_s_mean']:11.3f} {v['ttft_s_mean']:8.3f} "
              f"{v['latency_p95_s']:8.3f} "
              f"{'YES' if v['contamination_detected'] else 'no':>5s}")
    print("\n=== scaling curve (aggregate throughput) ===")
    for mode in ("serial", "batched"):
        print(f"\n{mode}:")
        print(f"  {'n':>4s} {'agg tok/s':>10s} {'x vs n=1':>9s} {'eff':>6s} "
              f"{'step gain':>10s} {'step eff':>9s} {'ttft ms':>8s} "
              f"{'per-sess':>9s}")
        for e in report["scaling"][mode]:
            step = e.get("gain_over_previous_step")
            seff = e.get("marginal_efficiency_of_step")
            print(f"  {e['sessions']:4d} {e['aggregate_tokens_per_s']:10.1f} "
                  f"{e['speedup_vs_n1']:9.2f} {e['scaling_efficiency_vs_n1']:6.2f} "
                  f"{(f'{step:.3f}' if step else '-'):>10s} "
                  f"{(f'{seff:.3f}' if seff is not None else '-'):>9s} "
                  f"{e['ttft_s']*1000:8.1f} {e['per_session_tokens_per_s']:9.1f}")

    ps = report["prefix_sharing"]
    print(f"\nprefix sharing: {ps['prefix_tokens']} tokens, "
          f"fork {ps['fork_seconds_mean']*1000:.3f} ms vs recompute "
          f"{ps['recompute_seconds']*1000:.1f} ms "
          f"({ps['fork_speedup_vs_recompute']:.0f}x), "
          f"KV avoided {ps['kv_bytes_saved_by_sharing']/2**20:.1f} MiB")
    print(f"\nwrote {outdir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
