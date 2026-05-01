import json, sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
manifest = json.loads((sweep_dir / "sweep_manifest.json").read_text())

runs = []
for r in manifest["runs"]:
    cfg = r.get("config", {})
    env = cfg.get("environment", {})
    ov = cfg.get("oversight", {})
    run_id = r["run_id"]
    run_dir = sweep_dir.parent / run_id
    log_path = run_dir / "log.jsonl"

    entry = {
        "run_id": run_id[:8],
        "comm": cfg.get("communication_mode"),
        "seed": env.get("seed"),
        "audit_p": ov.get("audit_probability"),
        "status": r["status"],
    }

    if r["status"] in ("completed", "succeeded") and log_path.exists():
        lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        prices_a0 = [l["actions"][0] for l in lines]
        prices_a1 = [l["actions"][1] for l in lines]
        cum = lines[-1]["cumulative_rewards"]
        audit_flags = sum(1 for l in lines if l.get("audit_event") and l["audit_event"].get("flag"))
        penalties = sum(1 for l in lines if (l.get("audit_event") or {}).get("penalty_applied"))

        # message content sample
        msgs = []
        for l in lines:
            for m in l.get("messages", []):
                content = m.get("content", "").strip()
                if content and content.lower() not in ("", "none"):
                    msgs.append(content[:80])

        # elevation at end
        final_elev = lines[-1]["trajectory_signals"].get("reward_elevation", [None, None])

        entry.update({
            "prices_a0_first5": prices_a0[:5],
            "prices_a0_last5": prices_a0[-5:],
            "prices_a1_first5": prices_a1[:5],
            "prices_a1_last5": prices_a1[-5:],
            "cum": [round(c, 1) for c in cum],
            "audit_flags": audit_flags,
            "penalties": penalties,
            "final_elev": final_elev,
            "msg_samples": msgs[:4],
        })
    runs.append(entry)

for entry in runs:
    print(f"\n{'='*60}")
    print(f"Run {entry['run_id']} | comm={entry['comm']} | seed={entry['seed']} | audit_p={entry['audit_p']} | {entry['status']}")
    if "prices_a0_first5" in entry:
        print(f"  Prices A0: {entry['prices_a0_first5']} ... {entry['prices_a0_last5']}")
        print(f"  Prices A1: {entry['prices_a1_first5']} ... {entry['prices_a1_last5']}")
        print(f"  Cumulative: {entry['cum']}")
        print(f"  Audit flags: {entry['audit_flags']} | Penalties applied: {entry['penalties']}")
        e = entry['final_elev']
        if e and e[0] is not None:
            print(f"  Final elevation: [{e[0]:.2f}, {e[1]:.2f}]")
        print(f"  Message samples:")
        for msg in entry['msg_samples']:
            print(f"    - {msg}")
