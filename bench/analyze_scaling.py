"""Where does the batching curve bend, and what does each added session cost?

Aggregate throughput alone hides the answer: it keeps rising well past the
point where adding sessions stops being worthwhile. The decision-relevant
quantity is the *marginal* cost of one more session, in milliseconds of decode
step time. That is flat while the kernel is bound by reading the weights, and
turns up sharply once it is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "out"
MAX_TOKENS = 64


def rows(path: Path, mode: str) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["scaling"][mode], d


def analyse(path: Path, mode: str, label: str) -> None:
    curve, _ = rows(path, mode)
    print(f"\n=== {label} :: {mode} ===")
    col = "step ms" if mode == "batched" else "ms/token"
    print(f"{'n':>4s} {'agg tok/s':>10s} {col:>9s} "
          f"{'d(agg)/dn':>10s} {'d(col)/dn':>11s} {'ttft ms':>8s}")
    prev = None
    for e in curve:
        n = e["sessions"]
        if mode == "batched":
            # One decode step advances all n sessions, so makespan / tokens
            # generated per session IS the per-step time.
            step_ms = e["makespan_s"] / MAX_TOKENS * 1000.0
        else:
            # Serial advances one session per step, so the comparable quantity
            # is cost per token across the whole run.
            step_ms = 1000.0 / max(e["aggregate_tokens_per_s"], 1e-9)
        if prev is None:
            dagg = dstep = None
        else:
            dn = n - prev["n"]
            dagg = (e["aggregate_tokens_per_s"] - prev["agg"]) / dn
            dstep = (step_ms - prev["step_ms"]) / dn
        print(f"{n:4d} {e['aggregate_tokens_per_s']:10.1f} {step_ms:9.2f} "
              f"{(f'{dagg:+.1f}' if dagg is not None else '-'):>10s} "
              f"{(f'{dstep:+.3f}' if dstep is not None else '-'):>11s} "
              f"{e['ttft_s']*1000:8.1f}")
        prev = {"n": n, "agg": e["aggregate_tokens_per_s"], "step_ms": step_ms}


def find_knee(path: Path, mode: str) -> dict:
    """The knee is where the marginal ms-per-added-session jumps."""
    curve, _ = rows(path, mode)
    pts = [(e["sessions"], e["makespan_s"] / MAX_TOKENS * 1000.0) for e in curve]
    marginals = []
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        marginals.append((n0, n1, (s1 - s0) / (n1 - n0)))
    best = None
    for i in range(1, len(marginals)):
        before = sum(m[2] for m in marginals[:i]) / i
        after = sum(m[2] for m in marginals[i:]) / (len(marginals) - i)
        if before <= 0:
            continue
        ratio = after / before
        if best is None or ratio > best["ratio"]:
            best = {"knee_at_sessions": marginals[i][0],
                    "ms_per_session_before": before,
                    "ms_per_session_after": after,
                    "ratio": ratio}
    return best or {}


def main() -> int:
    geo = OUT / "concurrency_geo64.json"
    zoom = OUT / "concurrency_zoom.json"
    for path, label in ((geo, "geometric 1..64"), (zoom, "zoom 24..64")):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        for mode in ("batched", "serial"):
            analyse(path, mode, label)

    print("\n=== knee detection (batched) ===")
    for path, label in ((geo, "geometric"), (zoom, "zoom")):
        if not path.exists():
            continue
        k = find_knee(path, "batched")
        if k:
            print(f"{label:10s} knee at n={k['knee_at_sessions']}: "
                  f"{k['ms_per_session_before']:.3f} ms/session before -> "
                  f"{k['ms_per_session_after']:.3f} ms/session after "
                  f"({k['ratio']:.1f}x)")

    summary = {}
    for path, label in ((geo, "geometric"), (zoom, "zoom")):
        if path.exists():
            summary[label] = find_knee(path, "batched")
    (OUT / "scaling_analysis.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {OUT / 'scaling_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
