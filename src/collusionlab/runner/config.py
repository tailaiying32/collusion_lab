"""ExperimentConfig — top-level pydantic schema for one experiment run.

The `environment` field is deserialized polymorphically using `env_type` as a
discriminator: the matching `EnvironmentConfig` subclass is looked up in the
environment registry. A new environment can plug in without touching this file.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from collusionlab.environments.base import (
    EnvironmentConfig,
    get_environment_classes,
)


class AgentConfig(BaseModel):
    backend: str
    model: str
    memory_window: int = Field(ge=1)
    temperature: float = 0.2

    # Free-form passthrough for backend-specific kwargs (e.g. `replies` for the
    # scripted test backend). Keys are forwarded to the backend constructor.
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class OversightConfig(BaseModel):
    mode: Literal["none", "audit-penalty"] = "none"
    audit_probability: float = 0.0
    penalty_factor: float = 1.0

    model_config = {"extra": "forbid"}


class ExperimentConfig(BaseModel):
    run_id: str | None = None
    env_type: str
    environment: EnvironmentConfig
    agents: list[AgentConfig]
    prompt_dir: str = "prompts/pricing"
    communication_mode: Literal["none", "public", "private"] = "none"
    oversight: OversightConfig = Field(default_factory=OversightConfig)
    output_dir: str = "data/raw"

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _coerce_environment(cls, data: Any) -> Any:
        # Polymorphic deserialization: pick the EnvironmentConfig subclass that
        # matches env_type. Pydantic's discriminated unions can't help us here
        # because the registry is dynamic.
        if not isinstance(data, dict):
            return data
        env_type = data.get("env_type")
        env = data.get("environment")
        if env_type and isinstance(env, dict):
            _, config_cls = get_environment_classes(env_type)
            data = dict(data)
            data["environment"] = config_cls(**env)
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> "ExperimentConfig":
        if self.environment.env_type != self.env_type:
            raise ValueError(
                f"env_type mismatch: top-level {self.env_type!r} vs "
                f"environment.env_type {self.environment.env_type!r}"
            )
        if len(self.agents) != self.environment.n_agents:
            raise ValueError(
                f"agents list length {len(self.agents)} does not match "
                f"environment.n_agents {self.environment.n_agents}"
            )
        return self

    def with_run_id(self) -> "ExperimentConfig":
        """Return a copy with run_id populated (auto-uuid if absent)."""
        if self.run_id:
            return self
        return self.model_copy(update={"run_id": str(uuid.uuid4())})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        # Strip free-form documentation keys.
        if isinstance(data.get("environment"), dict):
            data["environment"].pop("_calibration_note", None)
        return cls(**data)

    def to_yaml_dict(self) -> dict:
        """Plain-dict serialization suitable for round-tripping or manifest saving."""
        return self.model_dump(mode="json")
