"""Minimal OpenAI client wrapper used by HeadAgency."""

from __future__ import annotations

import logging
from typing import Dict, Optional


class LLMClient:
    """Thin wrapper around OpenAI's chat completion API."""

    _COLOR_SCHEMES: Dict[str, Dict[str, str]] = {
        "HeadAgency": {
            "input": "\033[36m",  # Cyan
            "output": "\033[1;36m",  # Bold cyan for subtle contrast
        },
        "ResearchAgency": {
            "input": "\033[35m",  # Magenta
            "output": "\033[1;35m",  # Bold magenta
        },
        "BacktestAgency": {
            "input": "\033[33m",  # Yellow
            "output": "\033[1;33m",  # Bold yellow
        },
    }
    _DEFAULT_SCHEME: Dict[str, str] = {
        "input": "\033[32m",  # Green
        "output": "\033[1;32m",  # Bold green
    }
    _FIXED_TEMPERATURE_PREFIXES = ("gpt-5",)

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4.1",
        agency_name: str = "HeadAgency",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - guard for missing dependency
            raise ImportError(
                "The 'openai' package is required to use HeadAgency. Install it with 'pip install openai'."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._agency_name = agency_name
        self._logger = logging.getLogger(f"{__name__}.{agency_name}")

    def complete(self, system_prompt: str, user_message: str, *, temperature: float = 0.7) -> str:
        """Generate a response using the stored system prompt and the user message."""

        prefix = f"[{self._agency_name}]"
        self._logger.info("%s request meta | model=%s temperature=%s", prefix, self._model, temperature)
        self._log_with_color("input", "%s input/system\n%s", prefix, system_prompt)
        self._log_with_color("input", "%s input/user\n%s", prefix, user_message)

        request_temperature = self._normalise_temperature(temperature, prefix)
        request_kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        if request_temperature is not None:
            request_kwargs["temperature"] = request_temperature

        response = self._client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content: Optional[str] = getattr(message, "content", None)
        if not content:
            raise RuntimeError("LLM response contained no content")
        text = content.strip()
        self._log_with_color("output", "%s output\n%s", prefix, text)
        return text

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        self._model = model

    def _log_with_color(self, key: str, message: str, *args: object) -> None:
        scheme = self._COLOR_SCHEMES.get(self._agency_name, self._DEFAULT_SCHEME)
        color = scheme.get(key)
        extra = {"color_code": color} if color else None
        if extra:
            self._logger.info(message, *args, extra=extra)
        else:
            self._logger.info(message, *args)

    def _normalise_temperature(self, temperature: float, prefix: str) -> Optional[float]:
        if temperature is None:
            return None

        if any(self._model.startswith(candidate) for candidate in self._FIXED_TEMPERATURE_PREFIXES):
            if abs(temperature - 1.0) > 1e-6:
                self._logger.info(
                    "%s model enforces default temperature; overriding %.2f -> 1.0",
                    prefix,
                    temperature,
                )
            return 1.0

        return temperature


__all__ = ["LLMClient"]
