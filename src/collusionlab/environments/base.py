"""Environment-agnostic base classes and registry for game environments.

A `GameEnvironment` is the deterministic, fully self-contained simulation of one
repeated-game instance. The runner only ever talks to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    """Base pydantic config for any environment. Subclasses add env-specific fields."""

    env_type: str
    n_agents: int = Field(ge=1)
    n_rounds: int = Field(ge=1)
    seed: int

    model_config = {"extra": "forbid"}


class GameEnvironment(ABC):
    """Abstract base class for all game environments.

    Subclasses set `env_type` as a class attribute and implement the methods below.
    `actions`, `obs_dict`, and parsed action values are plain Python types so the
    runner can serialize them to JSONL without environment-specific logic.
    """

    env_type: ClassVar[str]
    n_agents: int

    @abstractmethod
    def reset(self, seed: int) -> dict:
        """Reset to round 0 deterministically. Returns initial obs_dict.

        The returned dict has the same shape as `step()`'s obs_dict and is
        agent-facing. It should contain only information agents are allowed to
        observe directly in the game (e.g. past actions/rewards), not privileged
        reference anchors used for internal metrics.
        """

    @abstractmethod
    def step(self, actions: list) -> tuple[list[float], dict, bool]:
        """Advance one round. Returns (rewards, obs_dict, done).

        `obs_dict` is agent-facing and should exclude privileged/internal-only
        quantities (for example equilibrium reference prices).
        """

    @abstractmethod
    def action_space(self) -> dict:
        """Structured description of the valid action set.

        Used by both prompt rendering and `parse_action()`. Must contain at least
        a `description` field (human-readable, for prompts). Environment-specific
        keys (e.g. `type`, `min`, `max` for integer prices) are allowed.
        """

    @abstractmethod
    def parse_action(self, raw: str) -> Any:
        """Parse a model's raw text output into a validated action.

        Returns an action of the type `step()` expects. Raises `ValueError` with
        a human-readable message on invalid input; the agent retry loop feeds that
        message back to the model.
        """

    @abstractmethod
    def default_action(self) -> Any:
        """Competitive-baseline action used as the agent's fallback.

        Used by `LLMAgent` when its retry budget on `parse_action()` is exhausted.
        For pricing this is the integer-grid Nash price; other environments return
        whatever counts as default competitive play.
        """

    @abstractmethod
    def obs_keys(self) -> list[str]:
        """Agent-facing keys present in obs_dict each round."""

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def system_prompt_vars(self, agent_id: int) -> dict:
        """Env-specific placeholder values for `prompts/{env_type}/system.txt`.

        The runner spreads the returned dict into `template.format(**vars)`. Keeps the
        runner generic — it renders prompts without knowing which env is running.
        """

    def compute_extra_signals(
        self,
        actions: list,
        rewards: list[float],
        prev_actions: list | None,
        round_idx: int,
    ) -> dict:
        """Environment-specific per-round trajectory signals.

        Called by the runner after computing the base signals.  Returned dict
        is merged into ``trajectory_signals`` (append-only — must not
        overwrite the base keys).  Default: no extra signals.
        """
        return {}

    def reward_elevation_baseline(self) -> tuple[float, float] | None:
        """Per-agent (low, high) reward anchors for normalized reward elevation.

        Used by the runner to compute `trajectory_signals.reward_elevation` as
        `(reward - low) / (high - low)`. `low` is the competitive baseline (Nash),
        `high` is the joint-monopoly upper bound (symmetric, per agent). Returning
        `None` skips the signal for envs without a meaningful baseline.
        """
        return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[type[GameEnvironment], type[EnvironmentConfig]]] = {}


def register_environment(
    env_type: str,
    env_cls: type[GameEnvironment],
    config_cls: type[EnvironmentConfig],
) -> None:
    """Register an environment class + its config class under `env_type`."""
    if env_type in _REGISTRY:
        raise ValueError(f"env_type {env_type!r} already registered")
    _REGISTRY[env_type] = (env_cls, config_cls)


def get_environment_classes(
    env_type: str,
) -> tuple[type[GameEnvironment], type[EnvironmentConfig]]:
    if env_type not in _REGISTRY:
        raise KeyError(
            f"unknown env_type {env_type!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[env_type]


def get_environment(config: EnvironmentConfig) -> GameEnvironment:
    """Instantiate the environment associated with `config.env_type`."""
    env_cls, _ = get_environment_classes(config.env_type)
    return env_cls(config)


def registered_env_types() -> list[str]:
    return sorted(_REGISTRY)
