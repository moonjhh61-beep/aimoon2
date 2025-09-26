"""Configuration helpers for HeadAgency."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "head_agency_prompt.txt"
DEFAULT_KEY_PATH = PROJECT_ROOT / "secrets" / "llm_api_key.txt"
API_KEY_ENV_VAR = "HEAD_AGENCY_API_KEY"


def read_api_key(
    path: Optional[Path] = None,
    *,
    env_var: Optional[str] = None,
) -> str:
    """Return the API key stored in an environment variable or file."""

    env_name = env_var or API_KEY_ENV_VAR
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()

    key_path = path or DEFAULT_KEY_PATH
    if key_path.exists():
        with key_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped

    raise ValueError(
        f"No API key found. Set environment variable {env_name} or create {key_path}."
    )


def read_prompt(path: Optional[Path] = None) -> str:
    """Load the base prompt from *path*."""

    prompt_path = path or DEFAULT_PROMPT_PATH
    return prompt_path.read_text(encoding="utf-8").strip()


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_KEY_PATH",
    "API_KEY_ENV_VAR",
    "read_api_key",
    "read_prompt",
]
