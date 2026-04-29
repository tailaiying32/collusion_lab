"""Anthropic messages-API backend."""

from __future__ import annotations

import os

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from collusionlab.agents.model_client import ModelClient, register_backend


# USD per token. Update when Anthropic revises pricing.
ANTHROPIC_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00e-6, "output": 5.00e-6},
    "claude-sonnet-4-6": {"input": 3.00e-6, "output": 15.00e-6},
    "claude-opus-4-7": {"input": 15.00e-6, "output": 75.00e-6},
}


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "APIStatusError",
    }


class AnthropicModelClient(ModelClient):
    """Calls Anthropic's messages API with tenacity retry + token accounting.

    The Anthropic API separates the system prompt from the message list. The
    `generate()` interface still takes a single OpenAI-style messages list; any
    leading system message is extracted and passed via the `system` parameter.
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_attempts: int = 5,
        wait_min: float = 1.0,
        wait_max: float = 30.0,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> None:
        super().__init__(model_name=model_name)
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        self._call = retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1.0, min=wait_min, max=wait_max),
            retry=retry_if_exception(_is_transient),
        )(self._raw_call)

    @staticmethod
    def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        system: str | None = None
        rest: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                # Concatenate multiple system messages with blank lines.
                system = m["content"] if system is None else f"{system}\n\n{m['content']}"
            else:
                rest.append(m)
        return system, rest

    def _raw_call(self, messages: list[dict[str, str]], **kwargs):
        system, rest = self._split_system(messages)
        params = {
            "model": self.model_name,
            "messages": rest,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_output_tokens),
        }
        if system is not None:
            params["system"] = system
        return self._client.messages.create(**params)

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self._call(messages, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        # Anthropic responses are a list of content blocks; concatenate text blocks.
        text_parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return "".join(text_parts)

    def cost_estimate(self) -> float:
        prices = ANTHROPIC_PRICES.get(self.model_name)
        if prices is None:
            return 0.0
        return self.input_tokens * prices["input"] + self.output_tokens * prices["output"]


register_backend("anthropic", AnthropicModelClient)
