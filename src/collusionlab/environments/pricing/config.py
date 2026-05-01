"""Pydantic config for the pricing environment."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from collusionlab.environments.base import EnvironmentConfig


class PricingConfig(EnvironmentConfig):
    env_type: Literal["pricing"] = "pricing"

    demand_model: Literal["calvano", "bertrand"] = "calvano"
    demand_params: dict = Field(default_factory=dict)

    price_min: int = 1
    price_max: int = 15

    # Reference equilibria on the integer grid. Always auto-computed from the
    # demand model and n_agents at validation time, so they are always correct
    # regardless of how many agents are configured. Any values supplied in YAML
    # are overwritten by the validator — treat them as documentation only.
    nash_price: int | None = None
    monopoly_price: int | None = None

    # Multiplier applied to per-round profits before returning them from step().
    # Use to bring raw profits into a scale proportionate to the price grid.
    profit_scale: float = 1.0

    # When set, reset() returns a populated round-0 obs with this price for every
    # agent. The runner seeds agent memory from that obs so all agents begin with a
    # shared observed starting point before round 1.
    forced_initial_price: int | None = None

    def default_prompt_dir(self) -> str:
        return (
            "prompts/pricing_bertrand"
            if self.demand_model == "bertrand"
            else "prompts/pricing"
        )

    @model_validator(mode="after")
    def _check_grid(self) -> "PricingConfig":
        if self.price_max <= self.price_min:
            raise ValueError("price_max must exceed price_min")
        self._calibrate_equilibria()
        if self.nash_price is not None and not (
            self.price_min <= self.nash_price <= self.price_max
        ):
            raise ValueError("nash_price out of grid range")
        if self.monopoly_price is not None and not (
            self.price_min <= self.monopoly_price <= self.price_max
        ):
            raise ValueError("monopoly_price out of grid range")
        if self.forced_initial_price is not None and not (
            self.price_min <= self.forced_initial_price <= self.price_max
        ):
            raise ValueError("forced_initial_price out of grid range")
        return self

    def _calibrate_equilibria(self) -> None:
        """Overwrite nash_price and monopoly_price using the demand model.

        Runs at every config instantiation so the values are always consistent
        with n_agents and demand_params — YAML-frozen values are intentionally
        ignored to prevent stale calibrations from silently corrupting results.
        Skipped if demand_params is empty (avoids import errors in unit tests
        that construct minimal configs without full params).
        """
        if not self.demand_params:
            return
        try:
            from collusionlab.environments.pricing.demand import get_demand_model
            dm = get_demand_model(self.demand_model, self.n_agents, self.demand_params)
            self.nash_price = max(self.price_min, min(self.price_max, round(dm.nash_price())))
            self.monopoly_price = max(self.price_min, min(self.price_max, round(dm.monopoly_price())))
        except Exception:
            pass
