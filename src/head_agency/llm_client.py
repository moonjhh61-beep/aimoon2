"""Minimal OpenAI client wrapper used by HeadAgency."""

from __future__ import annotations

from typing import Optional


class LLMClient:
    """Thin wrapper around OpenAI's chat completion API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - guard for missing dependency
            raise ImportError(
                "The 'openai' package is required to use HeadAgency. Install it with 'pip install openai'."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system_prompt: str, user_message: str, *, temperature: float = 0.7) -> str:
        """Generate a response using the stored system prompt and the user message."""

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        message = response.choices[0].message
        content: Optional[str] = getattr(message, "content", None)
        if not content:
            raise RuntimeError("LLM response contained no content")
        return content.strip()

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        self._model = model


__all__ = ["LLMClient"]
