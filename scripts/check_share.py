import sys
sys.path.insert(0, "src")
from collusionlab.environments.pricing import CalvanoDemand

d = CalvanoDemand(n_agents=2, a=67.688, mu=8.461, a_0=0.0, c=33.844)
for label, p in [("Nash=50", 50), ("Mid=57", 57), ("Mono=65", 65), ("High=75", 75)]:
    qs = d.quantities([p, p])
    q = qs[0]                       # already = exp_i / denom (outside option included)
    total_firm = sum(qs)            # fraction going to any firm
    outside = 1.0 - total_firm      # fraction going to outside option
    active = q / total_firm         # share among actual buyers
    print(f"{label}: own_qty(displayed)={q*100:.1f}%  active_share={active*100:.1f}%  outside_option={outside*100:.1f}%")
