"""The repeated pricing game.

A symmetric N-firm Bertrand-style game on a discrete integer price grid.
Demand is supplied by a `DemandModel` (Calvano logit by default). All state is
plain Python types so the runner can serialize round logs to JSONL directly.
"""

from __future__ import annotations

import random

from collusionlab.environments.base import GameEnvironment, register_environment
from collusionlab.environments.pricing.config import PricingConfig
from collusionlab.environments.pricing.demand import DemandModel, get_demand_model


class PricingGame(GameEnvironment):
    env_type = "pricing"

    def __init__(self, config: PricingConfig) -> None:
        self.config = config
        self.n_agents = config.n_agents
        self.demand: DemandModel = get_demand_model(
            config.demand_model, n_agents=config.n_agents, params=config.demand_params
        )
        self.price_grid: list[int] = list(range(config.price_min, config.price_max + 1))

        # Reference prices on the integer grid. Prefer config-frozen values (from
        # the calibration script); otherwise round the demand model's continuous
        # equilibria to the grid.
        self._nash_price: int = (
            config.nash_price
            if config.nash_price is not None
            else self._round_to_grid(self.demand.nash_price())
        )
        self._monopoly_price: int = (
            config.monopoly_price
            if config.monopoly_price is not None
            else self._round_to_grid(self.demand.monopoly_price())
        )

        self._round: int = 0
        self._cumulative_profits: list[float] = [0.0] * self.n_agents
        self._rng: random.Random | None = None
        self._done: bool = False

    # ------------------------------------------------------------------
    # GameEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, seed: int) -> dict:
        self._rng = random.Random(seed)
        self._round = 0
        self._cumulative_profits = [0.0] * self.n_agents
        self._done = False
        return {
            "round": 0,
            "prices": [],
            "quantities": [],
            "profits": [],
            "cumulative_profits": list(self._cumulative_profits),
            "nash_price": self._nash_price,
            "monopoly_price": self._monopoly_price,
        }

    def step(self, actions: list) -> tuple[list[float], dict, bool]:
        if self._done:
            raise RuntimeError("step() called after game is done")
        if len(actions) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} actions, got {len(actions)}"
            )
        prices = [self._validate_price(a) for a in actions]

        quantities = self.demand.quantities([float(p) for p in prices])
        profits = [
            (prices[i] - self.demand.marginal_cost) * quantities[i]
            for i in range(self.n_agents)
        ]
        self._cumulative_profits = [
            self._cumulative_profits[i] + profits[i] for i in range(self.n_agents)
        ]
        self._round += 1
        self._done = self._round >= self.config.n_rounds

        obs = {
            "round": self._round,
            "prices": list(prices),
            "quantities": list(quantities),
            "profits": list(profits),
            "cumulative_profits": list(self._cumulative_profits),
            "nash_price": self._nash_price,
            "monopoly_price": self._monopoly_price,
        }
        return list(profits), obs, self._done

    def action_space(self) -> dict:
        return {
            "type": "integer",
            "min": self.config.price_min,
            "max": self.config.price_max,
            "description": (
                f"integer price between {self.config.price_min} and "
                f"{self.config.price_max} inclusive"
            ),
        }

    def parse_action(self, raw: str) -> int:
        if raw is None:
            raise ValueError("empty action")
        text = str(raw).strip().strip(".,;:!?\"' \t\n")
        if not text:
            raise ValueError("empty action")
        # Pick the first integer-looking token; tolerates verbose model output.
        token = None
        for piece in text.replace(",", " ").split():
            piece = piece.strip(".,;:!?\"'")
            try:
                token = int(piece)
                break
            except ValueError:
                continue
        if token is None:
            raise ValueError(
                f"could not parse an integer from {raw!r}; "
                f"valid range is {self.config.price_min}..{self.config.price_max}"
            )
        return self._validate_price(token)

    def default_action(self) -> int:
        return self._nash_price

    def obs_keys(self) -> list[str]:
        return [
            "round",
            "prices",
            "quantities",
            "profits",
            "cumulative_profits",
            "nash_price",
            "monopoly_price",
        ]

    def is_done(self) -> bool:
        return self._done

    def system_prompt_vars(self, agent_id: int) -> dict:
        return {
            "agent_id": agent_id,
            "n_agents": self.n_agents,
            "n_rounds": self.config.n_rounds,
            "price_min": self.config.price_min,
            "price_max": self.config.price_max,
            "cost": self.demand.marginal_cost,
        }

    def reward_elevation_baseline(self) -> tuple[float, float]:
        cost = self.demand.marginal_cost
        nash_q = self.demand.quantities([float(self._nash_price)] * self.n_agents)[0]
        mono_q = self.demand.quantities([float(self._monopoly_price)] * self.n_agents)[0]
        nash_profit = (self._nash_price - cost) * nash_q
        mono_profit = (self._monopoly_price - cost) * mono_q
        return (nash_profit, mono_profit)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_price(self, p) -> int:
        if isinstance(p, bool) or not isinstance(p, int):
            # Reject floats and bools; the price grid is strictly integer.
            try:
                pi = int(p)
                if pi != p:
                    raise ValueError
                p = pi
            except (TypeError, ValueError):
                raise ValueError(
                    f"price {p!r} is not an integer; valid range is "
                    f"{self.config.price_min}..{self.config.price_max}"
                )
        if not (self.config.price_min <= p <= self.config.price_max):
            raise ValueError(
                f"price {p} out of grid [{self.config.price_min}, "
                f"{self.config.price_max}]"
            )
        return p

    def _round_to_grid(self, p: float) -> int:
        return max(
            self.config.price_min,
            min(self.config.price_max, int(round(p))),
        )


register_environment("pricing", PricingGame, PricingConfig)
