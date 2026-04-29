"""OpenAI chat-completions backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from collusionlab.agents.model_client import ModelClient, register_backend

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI


# USD per token. Update when OpenAI revises pricing.
OPENAI_PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
    "gpt-4o": {"input": 2.50e-6, "output": 10.00e-6},
    "gpt-4.1-mini": {"input": 0.40e-6, "output": 1.60e-6},
}


def _is_transient(exc: BaseException) -> bool:
    """Treat rate limits, timeouts, and connection errors as retryable."""
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "APIStatusError",
    }


class OpenAIModelClient(ModelClient):
    """Calls OpenAI's chat-completions API with tenacity retry + token accounting."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_attempts: int = 5,
        wait_min: float = 1.0,
        wait_max: float = 30.0,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> None:
        super().__init__(model_name=model_name)
        from openai import OpenAI  # local import to keep core deps clean

        self._client: OpenAI = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        # Build a per-instance retry decorator so backoff config is configurable.
        # Only transient errors (rate limits, timeouts, 5xx) trigger retries; everything
        # else propagates immediately.
        self._call = retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1.0, min=wait_min, max=wait_max),
            retry=retry_if_exception(_is_transient),
        )(self._raw_call)

    def _raw_call(self, messages: list[dict[str, str]], **kwargs):
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_tokens=kwargs.get("max_tokens", self._max_output_tokens),
        )

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self._call(messages, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        return response.choices[0].message.content or ""

    def cost_estimate(self) -> float:
        prices = OPENAI_PRICES.get(self.model_name)
        if prices is None:
            return 0.0
        return self.input_tokens * prices["input"] + self.output_tokens * prices["output"]


register_backend("openai", OpenAIModelClient)
