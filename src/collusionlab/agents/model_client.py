"""ModelClient abstract base class and backend registry.

Each backend (OpenAI, Anthropic, ...) lives in its own module and registers itself
here. The runner reads only this interface and never imports a provider SDK directly.

Token accounting is a side effect: every successful `generate()` call increments
`input_tokens` and `output_tokens` on the client. The runner reads the cumulative
counters and `cost_estimate()` once at end of run for the manifest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelClient(ABC):
    """Provider-agnostic chat-style LLM client."""

    model_name: str

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate a single chat completion.

        `messages` follows the standard role/content shape used by OpenAI and
        Anthropic SDKs. Returns the assistant's text reply. Implementations must
        update `input_tokens`/`output_tokens` from the SDK response.
        """

    @abstractmethod
    def cost_estimate(self) -> float:
        """USD cost based on accumulated token counts and a per-model price table."""


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[ModelClient]] = {}


def register_backend(name: str, cls: type[ModelClient]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"backend {name!r} already registered")
    _REGISTRY[name] = cls


def get_model_client(backend: str, model_name: str, **kwargs) -> ModelClient:
    """Instantiate the registered backend `backend` with the given model name."""
    if backend not in _REGISTRY:
        # Lazy import: backends register on import.
        try:
            __import__(f"collusionlab.agents.backends.{backend}_client")
        except ImportError as e:
            raise KeyError(
                f"unknown backend {backend!r}; registered: {sorted(_REGISTRY)} "
                f"(import error: {e})"
            )
    if backend not in _REGISTRY:
        raise KeyError(
            f"backend module loaded but did not register {backend!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[backend](model_name=model_name, **kwargs)


def registered_backends() -> list[str]:
    return sorted(_REGISTRY)
