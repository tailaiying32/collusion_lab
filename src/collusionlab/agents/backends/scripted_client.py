"""Scripted model client — deterministic test backend.

Returns a pre-scripted sequence of replies in order. Used by `tests/test_runner.py`
for the byte-identical reproducibility check, since real LLM calls are not
deterministic even at temperature=0. Selectable via `agents[i].backend = "scripted"`.
"""

from __future__ import annotations

from collusionlab.agents.model_client import ModelClient, register_backend


class ScriptedModelClient(ModelClient):
    """Returns replies from a pre-supplied list, one per call."""

    def __init__(
        self,
        model_name: str = "scripted",
        replies: list[str] | None = None,
        **_: object,
    ) -> None:
        super().__init__(model_name=model_name)
        self._replies: list[str] = list(replies or [])
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            raise RuntimeError("ScriptedModelClient ran out of replies")
        reply = self._replies.pop(0)
        # Rough token accounting: 4 chars per token. Lets cost_estimate stay 0
        # while still exercising the input_tokens / output_tokens counters.
        self.input_tokens += sum(len(m["content"]) for m in messages) // 4
        self.output_tokens += len(reply) // 4
        return reply

    def cost_estimate(self) -> float:
        return 0.0


register_backend("scripted", ScriptedModelClient)
