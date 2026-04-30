import json, sys, statistics, pathlib
runs = sys.argv[1:]
for d in runs:
    p = pathlib.Path(d)
    rounds = [json.loads(l) for l in open(p/"log.jsonl")]
    n = len(rounds)
    n_agents = len(rounds[0]["actions"])
    flat = {k: [r["actions"][k] for r in rounds] for k in range(n_agents)}
    cum = rounds[-1]["cumulative_rewards"]
    audits = [r.get("audit_event") for r in rounds]
    flagged = [(r["round"], r["audit_event"]) for r in rounds if r.get("audit_event")]
    ts_first = rounds[0]["trajectory_signals"]
    ts_last = rounds[-1]["trajectory_signals"]
    print(f"=== {p.name} ===")
    print(f"  rounds: {n}")
    for k, v in flat.items():
        print(f"  {k}: mean={statistics.mean(v):.2f}  last5={statistics.mean(v[-5:]):.2f}  min={min(v):.2f}  max={max(v):.2f}")
    print(f"  cumulative rewards: {cum}")
    print(f"  final trajectory_signals: {ts_last}")
    print(f"  audit events with flag: {len(flagged)} / {n}")
    msgs = [r.get("messages") for r in rounds]
    msg_rounds = [r["round"] for r in rounds if r.get("messages")]
    print(f"  rounds with messages: {len(msg_rounds)}")
    for rnd, ev in flagged[:3]:
        print(f"    sample audit r{rnd}: {ev}")
    if msg_rounds:
        first = next(r for r in rounds if r.get("messages"))
        print(f"    sample message r{first['round']}: {first['messages'][:2]}")
