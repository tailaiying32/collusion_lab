import json, sys

run_configs = [
    ('592c1ab7-d853-4125-aab7-bec92de3afd5', 'seed=1'),
    ('8ecef0eb-bca2-4f39-b6be-d31e700f6f94', 'seed=1000'),
]

for run_id, label in run_configs:
    print(f'=== {label} ({run_id[:8]}) ===')
    with open(f'data/raw/{run_id}/log.jsonl') as f:
        lines = [json.loads(l) for l in f]
    for line in lines:
        r = line['round']
        a = line['actions']
        rw = line['rewards']
        elev = line['trajectory_signals'].get('reward_elevation', [None, None])
        msgs = line.get('messages', [])
        msg_texts = [m.get('content', '')[:70] for m in msgs if m.get('content')]
        elev_str = f'{elev[0]:.2f}' if elev and elev[0] is not None else '?'
        audit = line.get('audit_event') or {}
        flag = audit.get('flag', False)
        risk = audit.get('risk_score')
        risk_str = f' risk={risk:.2f}' if risk is not None else ''
        flag_str = ' [FLAG]' if flag else ''
        print(f'  R{r:02d} prices={a} rew=[{rw[0]:.3f},{rw[1]:.3f}] elev={elev_str}{risk_str}{flag_str}')
        for mt in msg_texts:
            print(f'       msg: {mt}')
    cum = lines[-1]['cumulative_rewards']
    print(f'  FINAL cumulative: [{cum[0]:.3f}, {cum[1]:.3f}]')
    audit_hits = sum(1 for l in lines if l.get('audit_event') and l['audit_event'].get('flag'))
    print(f'  Audit flags: {audit_hits}/{len(lines)}')
    print()
