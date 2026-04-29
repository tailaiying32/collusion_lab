"""Calibrate Calvano logit demand parameters to a target integer price grid.

Phase 1.6: produces parameter values (a, mu, a_0, c) such that the symmetric Nash
and monopoly equilibria land at desired integer positions on the price grid, with
the largest margin from neighboring grid points so that small numerical noise
cannot flip which integer they round to.

The published Calvano et al. (2020) values (a=2, mu=0.25, a_0=0, c=1) yield
Nash ≈ 1.47 and monopoly ≈ 1.92 — unusable on an integer grid. Uniform scaling
(multiplying a, mu, c by the same constant k, leaving a_0=0) preserves the
demand structure and equilibrium ratios while shifting both equilibria up by k.

Run once after Phase 1's demand layer is in place; writes results into
``configs/base.yaml`` under ``environment.demand_params``.

    python -m scripts.calibrate_calvano

By default targets Nash in {6, 7, 8} and monopoly in {10, 11} for a 1..15 grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.environments.pricing.demand import CalvanoDemand  # noqa: E402

# Published Calvano values; used as the unscaled reference.
BASE_A = 2.0
BASE_MU = 0.25
BASE_A0 = 0.0
BASE_C = 1.0


def margin_to_grid_boundary(x: float) -> float:
    """How far x is from the nearest half-integer (the boundary between integers).

    Larger = more robust to numerical noise: round(x) won't flip across runs.
    """
    return abs((x - round(x))) - 0.0  # distance from integer
    # Actually we want distance to the nearest *half-integer* (boundary).
    # Recompute: the boundary nearest x is round(x) ± 0.5.


def grid_robustness(x: float) -> float:
    """Distance from x to the nearest integer-boundary (half-integer).

    Range [0, 0.5]. 0.5 means x sits exactly on an integer — maximally safe.
    """
    return 0.5 - abs(x - round(x))


def calibrate(
    n_agents: int,
    nash_targets: tuple[int, ...],
    monopoly_targets: tuple[int, ...],
    scale_lo: float = 2.0,
    scale_hi: float = 8.0,
    n_steps: int = 6001,
) -> dict:
    """Search uniform scale k ∈ [scale_lo, scale_hi] for the best parameters."""
    best = None
    for i in range(n_steps):
        k = scale_lo + (scale_hi - scale_lo) * i / (n_steps - 1)
        a = BASE_A * k
        mu = BASE_MU * k
        a_0 = BASE_A0 * k  # stays 0
        c = BASE_C * k
        try:
            demand = CalvanoDemand(n_agents=n_agents, a=a, mu=mu, a_0=a_0, c=c)
        except Exception:
            continue
        nash = demand.nash_price()
        monop = demand.monopoly_price()
        if round(nash) not in nash_targets:
            continue
        if round(monop) not in monopoly_targets:
            continue
        score = min(grid_robustness(nash), grid_robustness(monop))
        candidate = {
            "scale": k,
            "a": a,
            "mu": mu,
            "a_0": a_0,
            "c": c,
            "nash_price": nash,
            "monopoly_price": monop,
            "nash_int": round(nash),
            "monopoly_int": round(monop),
            "score": score,
        }
        if best is None or score > best["score"]:
            best = candidate
    if best is None:
        raise RuntimeError(
            "no scale produced (Nash, monopoly) in the requested target sets; "
            "widen the search or relax targets"
        )
    return best


def write_to_base_yaml(path: Path, params: dict) -> None:
    """Update configs/base.yaml in-place with the calibrated demand parameters."""
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env = config.setdefault("environment", {})
    env["demand_model"] = "calvano"
    env["demand_params"] = {
        "a": float(params["a"]),
        "mu": float(params["mu"]),
        "a_0": float(params["a_0"]),
        "c": float(params["c"]),
    }
    env["nash_price"] = int(params["nash_int"])
    env["monopoly_price"] = int(params["monopoly_int"])
    env["_calibration_note"] = (
        f"calibrated by scripts/calibrate_calvano.py: "
        f"continuous Nash={params['nash_price']:.4f}, "
        f"monopoly={params['monopoly_price']:.4f}, "
        f"scale={params['scale']:.4f}"
    )
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument(
        "--nash-targets",
        type=int,
        nargs="+",
        default=[6, 7, 8],
        help="acceptable integer Nash positions on the grid",
    )
    parser.add_argument(
        "--monopoly-targets",
        type=int,
        nargs="+",
        default=[10, 11],
        help="acceptable integer monopoly positions on the grid",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "base.yaml",
        help="path to base.yaml to update in place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the result but do not write base.yaml",
    )
    args = parser.parse_args()

    result = calibrate(
        n_agents=args.n_agents,
        nash_targets=tuple(args.nash_targets),
        monopoly_targets=tuple(args.monopoly_targets),
    )
    print("Calibrated Calvano parameters:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if not args.dry_run:
        write_to_base_yaml(args.config, result)
        print(f"\nWrote calibrated values to {args.config}")


if __name__ == "__main__":
    main()
