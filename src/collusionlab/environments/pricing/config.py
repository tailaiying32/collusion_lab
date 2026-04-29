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

    # Reference equilibria on the integer grid; produced by
    # `scripts/calibrate_calvano.py` and frozen into configs/base.yaml.
    # Optional because Bertrand resolves them analytically at instantiation.
    nash_price: int | None = None
    monopoly_price: int | None = None

    @model_validator(mode="after")
    def _check_grid(self) -> "PricingConfig":
        if self.price_max <= self.price_min:
            raise ValueError("price_max must exceed price_min")
        if self.nash_price is not None and not (
            self.price_min <= self.nash_price <= self.price_max
        ):
            raise ValueError("nash_price out of grid range")
        if self.monopoly_price is not None and not (
            self.price_min <= self.monopoly_price <= self.price_max
        ):
            raise ValueError("monopoly_price out of grid range")
        return self
