"""HeadAgency orchestrator that delegates to an LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm_client import LLMClient


@dataclass
class HeadAgencyConfig:
    base_prompt: str
    temperature: float = 0.7
    model: Optional[str] = None


class HeadAgency:
    """High-level interface used to interact with the LLM."""

    def __init__(self, llm_client: LLMClient, config: HeadAgencyConfig) -> None:
        if config.model:
            llm_client.set_model(config.model)
        self._llm = llm_client
        self._config = config

    def respond(self, user_message: str, *, context: Optional[str] = None) -> str:
        """Generate a response for *user_message*.

        Additional context can be supplied to append to the system prompt.
        """

        system_prompt = self._config.base_prompt
        if context:
            system_prompt = f"{system_prompt}\n\nContext:\n{context.strip()}"
        return self._llm.complete(
            system_prompt,
            user_message,
            temperature=self._config.temperature,
        )


__all__ = ["HeadAgency", "HeadAgencyConfig"]
