"""Pricing environment package.

Importing this package registers `PricingGame` under env_type "pricing" via
the side effect in `game.py`.
"""

from collusionlab.environments.pricing.config import PricingConfig
from collusionlab.environments.pricing.demand import (
    BertrandDemand,
    CalvanoDemand,
    DemandModel,
    LinearDifferentiatedDemand,
    get_demand_model,
)
from collusionlab.environments.pricing.game import PricingGame

__all__ = [
    "PricingConfig",
    "PricingGame",
    "DemandModel",
    "CalvanoDemand",
    "BertrandDemand",
    "LinearDifferentiatedDemand",
    "get_demand_model",
]
