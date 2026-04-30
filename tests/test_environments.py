"""Phase 1 unit tests for the environment layer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from collusionlab.environments.base import (
    GameEnvironment,
    get_environment,
    get_environment_classes,
)
from collusionlab.environments.pricing import (  # noqa: F401  -- triggers registration
    BertrandDemand,
    CalvanoDemand,
    DemandModel,
    PricingConfig,
    PricingGame,
)


# ---------------------------------------------------------------------------
# Demand models
# ---------------------------------------------------------------------------


def test_calvano_published_equilibria_match_paper():
    """Published Calvano (2020) values give the published equilibria."""
    d = CalvanoDemand(n_agents=2, a=2.0, mu=0.25, a_0=0.0, c=1.0)
    assert math.isclose(d.nash_price(), 1.4729, abs_tol=1e-3)
    assert math.isclose(d.monopoly_price(), 1.9250, abs_tol=1e-3)


def test_calvano_config_auto_calibrates_equilibria():
    """PricingConfig auto-computes nash_price and monopoly_price from the demand model."""
    import yaml
    from collusionlab.environments.pricing.config import PricingConfig

    with (ROOT / "configs" / "base.yaml").open() as f:
        raw = yaml.safe_load(f)
    env_raw = {k: v for k, v in raw["environment"].items() if not k.startswith("_")}
    cfg = PricingConfig(**env_raw)

    params = env_raw["demand_params"]
    n = env_raw["n_agents"]
    d = CalvanoDemand(n_agents=n, **params)
    assert cfg.nash_price == round(d.nash_price())
    assert cfg.monopoly_price == round(d.monopoly_price())


def test_calvano_quantities_sum_below_one_due_to_outside_option():
    d = CalvanoDemand(n_agents=2, a=2.0, mu=0.25, a_0=0.0, c=1.0)
    q = d.quantities([1.5, 1.5])
    assert all(0.0 < qi < 1.0 for qi in q)
    assert sum(q) < 1.0  # outside option absorbs the rest
    # Symmetric prices give symmetric quantities.
    assert math.isclose(q[0], q[1], abs_tol=1e-12)


def test_calvano_nash_solution_is_near_fixed_point():
    d = CalvanoDemand(n_agents=2, a=2.0, mu=0.25, a_0=0.0, c=1.0)
    p = d.nash_price()
    assert abs(d._best_response(p) - p) < 1e-4


def test_demand_model_interface_satisfied():
    """Both demand models conform to the DemandModel ABC."""
    cd = CalvanoDemand(n_agents=2, a=2.0, mu=0.25, a_0=0.0, c=1.0)
    bd = BertrandDemand(n_agents=2, Q=1.0, c=1.0, reservation_price=10.0)
    for m in (cd, bd):
        assert isinstance(m, DemandModel)
        assert isinstance(m.quantities([2.0, 2.0]), list)
        assert isinstance(m.nash_price(), float)
        assert isinstance(m.monopoly_price(), float)
        assert isinstance(m.marginal_cost, float)


def test_bertrand_winner_takes_all():
    bd = BertrandDemand(n_agents=3, Q=1.0, c=1.0, reservation_price=10.0)
    # Cheapest firm takes all.
    q = bd.quantities([3.0, 5.0, 4.0])
    assert q == [1.0, 0.0, 0.0]
    # Tie at the minimum splits evenly.
    q = bd.quantities([3.0, 3.0, 5.0])
    assert q == [0.5, 0.5, 0.0]
    # All three tied: thirds.
    q = bd.quantities([4.0, 4.0, 4.0])
    assert all(math.isclose(qi, 1 / 3) for qi in q)
    # Above reservation: nobody buys.
    q = bd.quantities([11.0, 12.0, 13.0])
    assert q == [0.0, 0.0, 0.0]
    # Nash is marginal cost.
    assert bd.nash_price() == 1.0


# ---------------------------------------------------------------------------
# PricingGame
# ---------------------------------------------------------------------------


def _calibrated_config(**overrides) -> PricingConfig:
    import yaml

    with (ROOT / "configs" / "base.yaml").open() as f:
        env_cfg = yaml.safe_load(f)["environment"]
    env_cfg.pop("_calibration_note", None)
    env_cfg.update(overrides)
    return PricingConfig(**env_cfg)


def test_pricing_game_step_at_nash_gives_expected_profits():
    cfg = _calibrated_config()
    game = PricingGame(cfg)
    reset_obs = game.reset(seed=42)
    assert "nash_price" not in reset_obs
    assert "monopoly_price" not in reset_obs
    nash = cfg.nash_price
    rewards, obs, done = game.step([nash, nash])
    # Profits are positive (price above marginal cost) and symmetric.
    assert rewards[0] > 0
    assert math.isclose(rewards[0], rewards[1], abs_tol=1e-9)
    assert obs["prices"] == [nash, nash]
    assert obs["round"] == 1
    assert "nash_price" not in obs
    assert "monopoly_price" not in obs
    assert done is False  # 50 rounds in the calibrated config
    # Cumulative profits == per-round profits after one step.
    assert math.isclose(obs["cumulative_profits"][0], rewards[0])


def test_bertrand_default_nash_price_respects_constrained_grid_floor():
    cfg = _calibrated_config(
        demand_model="bertrand",
        demand_params={"Q": 1.0, "c": 1.0, "reservation_price": 10.0},
        nash_price=None,
        monopoly_price=None,
        price_min=3,
        price_max=15,
        profit_scale=1.0,
    )
    game = PricingGame(cfg)
    nash_profit, _ = game.reward_elevation_baseline()
    # At constrained Nash (price_min), ties split demand evenly across firms.
    assert nash_profit == pytest.approx((cfg.price_min - 1.0) * (1.0 / cfg.n_agents))


def test_pricing_game_reset_is_deterministic():
    cfg = _calibrated_config(n_rounds=5)
    g1 = PricingGame(cfg)
    g2 = PricingGame(cfg)
    g1.reset(seed=123)
    g2.reset(seed=123)
    actions_seq = [[5, 6], [7, 7], [8, 9], [10, 10], [6, 8]]
    out1 = [g1.step(a) for a in actions_seq]
    out2 = [g2.step(a) for a in actions_seq]
    assert out1 == out2
    assert g1.is_done() and g2.is_done()


def test_pricing_game_grid_enforcement_rejects_out_of_grid():
    cfg = _calibrated_config()
    game = PricingGame(cfg)
    game.reset(seed=0)
    with pytest.raises(ValueError, match="out of grid"):
        game.step([0, 5])  # below price_min
    with pytest.raises(ValueError, match="out of grid"):
        game.step([5, cfg.price_max + 1])  # above price_max
    with pytest.raises(ValueError, match="not an integer"):
        game.step([5.5, 7])  # non-integer


def test_pricing_game_step_errors_after_done():
    cfg = _calibrated_config(n_rounds=1)
    game = PricingGame(cfg)
    game.reset(seed=0)
    game.step([5, 5])
    assert game.is_done()
    with pytest.raises(RuntimeError):
        game.step([5, 5])


# ---------------------------------------------------------------------------
# parse_action / action_space
# ---------------------------------------------------------------------------


def test_parse_action_accepts_well_formed_strings():
    cfg = _calibrated_config()
    game = PricingGame(cfg)
    assert game.parse_action("7") == 7
    assert game.parse_action("  7  ") == 7
    assert game.parse_action("PRICE: 7") == 7
    assert game.parse_action("My final decision.\nPRICE: 9") == 9
    assert game.parse_action("Price: 12") == 12
    assert game.parse_action("Round 5 had price 12.\nPRICE: 9") == 9


def test_parse_action_rejects_invalid_with_range_in_message():
    cfg = _calibrated_config()
    game = PricingGame(cfg)
    with pytest.raises(ValueError) as ei:
        game.parse_action("")
    assert "empty" in str(ei.value).lower()
    with pytest.raises(ValueError) as ei:
        game.parse_action("definitely not a number")
    msg = str(ei.value)
    assert str(cfg.price_min) in msg and str(cfg.price_max) in msg
    with pytest.raises(ValueError, match="PRICE: <int>"):
        game.parse_action("The other firm priced at 7 last round. I'll raise to 9.")
    with pytest.raises(ValueError, match="out of grid"):
        game.parse_action("101")
    with pytest.raises(ValueError, match="out of grid"):
        game.parse_action("0")


def test_action_space_shape():
    cfg = _calibrated_config()
    game = PricingGame(cfg)
    space = game.action_space()
    for key in ("type", "min", "max", "description"):
        assert key in space
    assert space["min"] == cfg.price_min
    assert space["max"] == cfg.price_max
    assert isinstance(space["description"], str) and space["description"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_pricing():
    env_cls, cfg_cls = get_environment_classes("pricing")
    assert env_cls is PricingGame
    assert cfg_cls is PricingConfig


def test_get_environment_returns_instance():
    cfg = _calibrated_config()
    game = get_environment(cfg)
    assert isinstance(game, GameEnvironment)
    assert isinstance(game, PricingGame)
