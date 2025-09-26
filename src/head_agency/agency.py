"""HeadAgency orchestrator that delegates to an LLM client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm_client import LLMClient
from .prompt_template import augment_context, render_template


@dataclass
class HeadAgencyConfig:
    system_template: str
    temperature: float = 0.7
    model: Optional[str] = None
    user_template: Optional[str] = None
    prompt_context: Dict[str, str] = field(default_factory=dict)
    progress_items: List[str] = field(default_factory=list)


class HeadAgency:
    """High-level interface used to interact with the LLM."""

    def __init__(self, llm_client: LLMClient, config: HeadAgencyConfig) -> None:
        if config.model:
            llm_client.set_model(config.model)
        self._llm = llm_client
        self._config = config
        self._update_progress_context()

    def respond(self, user_message: str, *, context: Optional[str] = None) -> str:
        """Generate a response for *user_message* using optional *context*."""

        prompt_context = augment_context(
            self._config.prompt_context,
            user_message=user_message,
            runtime_context=context,
        )
        system_prompt = render_template(
            self._config.system_template,
            prompt_context,
        )

        formatted_user = user_message
        if self._config.user_template:
            candidate = render_template(self._config.user_template, prompt_context).strip()
            if candidate:
                formatted_user = candidate

        return self._llm.complete(
            system_prompt,
            formatted_user,
            temperature=self._config.temperature,
        )

    def add_progress_entry(self, label: str, text: str) -> None:
        entry_text = text.strip()
        if not entry_text:
            return

        formatted = entry_text.replace("\n", "\n   ")
        index = len(self._config.progress_items) + 1
        entry = f"{index}. [{label}] {formatted}"
        self._config.progress_items.append(entry)
        self._update_progress_context()

    def _update_progress_context(self) -> None:
        section = "\n".join(self._config.progress_items)
        self._config.prompt_context["progress_section"] = section
        self._config.prompt_context["progress_count"] = str(len(self._config.progress_items))


__all__ = ["HeadAgency", "HeadAgencyConfig"]
