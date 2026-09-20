"""Drive a running mind through a full cycle over the control plane.

    .\\.venv\\Scripts\\python.exe scripts\\exercise.py --config config.test.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synthetic_mind.config import load_config  # noqa: E402
from synthetic_mind.rpc import RpcClient, read_or_create_token  # noqa: E402


def show(label: str, obj: object, limit: int = 700) -> None:
    text = json.dumps(obj, indent=2, default=str)
    print(f"\n--- {label} ---")
    print(text[:limit] + ("..." if len(text) > limit else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.test.toml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    token = read_or_create_token(cfg.token_path)
    c = RpcClient(cfg.supervisor_host, cfg.supervisor_port, token, timeout=300)
    c.connect(retries=10)

    caps = c.call("capabilities")
    print(f"backend={caps.get('backend_kind')} simulated={caps.get('is_simulated')} "
          f"kv_mode={caps.get('kv_mode')} concurrency={caps.get('concurrency_mode')}")

    # 1. seed maintained memory with supporting AND opposing evidence
    m1 = c.call("remember", kind="belief",
                claim="The inference backend keeps exactly one resident weight set.",
                confidence=0.9, created_by="operator",
                supporting=[{"note": "capabilities.weight_ownership"}],
                opposing=[{"note": "not verified under a rolling model upgrade"}],
                tags=["architecture"])
    show("remember", m1)

    # 2. Ego converses and produces an auditable conclusion
    turn = c.call("ego_converse",
                  message="How many copies of the model weights are resident?",
                  idempotency_key="exercise-turn-1")
    show("ego_converse", turn, 900)
    conclusion_id = (turn.get("result") or {}).get("conclusion_id")

    # 3. idempotency: the same key must not re-run the turn
    again = c.call("ego_converse",
                   message="How many copies of the model weights are resident?",
                   idempotency_key="exercise-turn-1")
    print(f"\nidempotent replay: replayed={again.get('replayed')} "
          f"same_operation={again.get('operation_id') == turn.get('operation_id')}")

    # 4. Id audits that conclusion through the record, without asking Ego
    if conclusion_id:
        audit = c.call("id_audit", conclusion_id=conclusion_id,
                       focus="is the claim supported by recorded evidence?")
        res = audit.get("result") or {}
        print(f"\naudit verdict={res.get('verdict')} "
              f"asked_ego_to_defend_itself={res.get('asked_ego_to_defend_itself')} "
              f"audit_id={res.get('audit_id')} disagreement={res.get('disagreement_id')}")
        show("audit limitations", audit.get("limitations"))

    # 5. investigation: publishes an Ego snapshot and admits worker work
    inv = c.call("ego_investigate",
                 question="What limits concurrent sessions on this machine?",
                 constraints="one paragraph")
    show("ego_investigate", inv, 900)

    # 6. wait for a worker to pick it up and commit a finding
    work = ((inv.get("result") or {}).get("work") or {})
    work_id = work.get("work_id")
    if work_id:
        for _ in range(60):
            item = c.call("get_work", work_id=work_id)
            if item["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(1)
        show("work item", {k: item[k] for k in
                           ("work_id", "status", "attempt", "fencing_token",
                            "snapshot_id", "failure")})
        if item.get("result"):
            r = item["result"]
            print(f"  worker instantiation: {r.get('instantiation', {}).get('method')}")
            print(f"  finding: {str(r.get('finding'))[:160]}")

    # 7. provenance chain for the conversational operation
    prov = c.call("provenance", operation_id=turn["operation_id"])
    print(f"\nprovenance: events={len(prov['events'])} "
          f"hash_chain_ok={prov['hash_chain_ok']} "
          f"unresolved_content={len(prov['unresolved_content'])} "
          f"receipts={len(prov['receipts'])} conclusions={len(prov['conclusions'])}")
    print("  kinds:", [e["kind"] for e in prov["events"]])

    # 8. raw history vs maintained memory
    recall = c.call("ego_recall", query="weight", limit=5)
    mem = (recall.get("result") or {}).get("memories", [])
    print(f"\nrecall returned {len(mem)} MAINTAINED items "
          f"(raw event count is {c.call('status')['event_count']})")
    for m in mem:
        print(f"  [{m['kind']}] {m['claim'][:70]} conf={m['confidence']} "
              f"support={len(m['evidence']['supporting'])} "
              f"oppose={len(m['evidence']['opposing'])}")

    # 9. maintenance admission and its bounds
    maint = c.call("id_maintenance", objective="check for stale unreferenced snapshots")
    show("id_maintenance", maint.get("result"))

    # 10. integrity
    integ = c.call("verify_integrity", deep=True)
    print(f"\nintegrity: hash_chain_ok={integ['hash_chain_ok']} "
          f"missing_content={integ['missing_content_count']} counts={integ['counts']}")

    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
