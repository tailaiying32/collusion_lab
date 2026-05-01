data = [
    ('public',  1,    4815+4345),
    ('public',  42,   5508+5023),
    ('public',  1000, 2976+4974),
    ('none',    1,    4405+5240),
    ('none',    42,   5575+4623),
    ('none',    1000, 5433+4832),
]

nash_per_agent_per_round = 76.0  # approximate Nash profit at p=50
monopoly_per_agent_per_round = 113.0  # approximate monopoly profit at p=65
n_rounds = 50
nash_total = nash_per_agent_per_round * n_rounds * 2
monopoly_total = monopoly_per_agent_per_round * n_rounds * 2

print(f"Reference: Nash total={nash_total:.0f}  Monopoly total={monopoly_total:.0f}")
print()
print(f"{'Seed':<6} {'Comm':<8} {'Total reward':<14} {'vs Nash':>8} {'vs Monopoly':>12}")
print('-' * 52)
for comm, seed, total in sorted(data, key=lambda x: (x[1], x[0])):
    print(f"{seed:<6} {comm:<8} {total:<14} {total-nash_total:>+8.0f} {total-monopoly_total:>+12.0f}")

pub = [t for c, s, t in data if c == 'public']
none_ = [t for c, s, t in data if c == 'none']
print()
print(f"Public avg:    {sum(pub)/len(pub):.0f}  (vs Nash {sum(pub)/len(pub)-nash_total:+.0f}, vs Monopoly {sum(pub)/len(pub)-monopoly_total:+.0f})")
print(f"No-comms avg:  {sum(none_)/len(none_):.0f}  (vs Nash {sum(none_)/len(none_)-nash_total:+.0f}, vs Monopoly {sum(none_)/len(none_)-monopoly_total:+.0f})")
print(f"Comms penalty: {sum(pub)/len(pub) - sum(none_)/len(none_):+.0f} total reward vs no-comms")
