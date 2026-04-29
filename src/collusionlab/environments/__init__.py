"""Environment registry and built-in environments.

Importing this package side-effect-registers every built-in env so that
`get_environment_classes(env_type)` works without callers importing each env
subpackage by hand.
"""

from collusionlab.environments.base import (
    EnvironmentConfig,
    GameEnvironment,
    get_environment,
    get_environment_classes,
    register_environment,
    registered_env_types,
)

# Trigger built-in environment registration.
from collusionlab.environments import pricing  # noqa: F401

__all__ = [
    "EnvironmentConfig",
    "GameEnvironment",
    "get_environment",
    "get_environment_classes",
    "register_environment",
    "registered_env_types",
]
