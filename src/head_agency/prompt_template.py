"""Prompt template parsing and rendering utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


SECTION_PATTERN = re.compile(r"^===\s*(system|user)\s*===$", re.IGNORECASE)
VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


@dataclass
class PromptTemplateSections:
    """Container for system/user prompt templates extracted from a file."""

    system: str
    user: Optional[str] = None


def parse_prompt_template(raw_text: str) -> PromptTemplateSections:
    """Split *raw_text* into ``system`` and ``user`` templates.

    Files can declare sections using ``=== system ===`` and ``=== user ===`` markers.
    If no markers are found, the entire file is treated as the system template.
    """

    lines = raw_text.splitlines()
    sections: Dict[str, list[str]] = {}
    current = "system"
    buffer: list[str] = []

    for line in lines:
        match = SECTION_PATTERN.match(line.strip())
        if match:
            sections[current] = buffer
            current = match.group(1).lower()
            buffer = []
            continue
        buffer.append(line)

    sections[current] = buffer

    system_lines = sections.get("system")
    if system_lines is None:
        system = raw_text.strip()
    else:
        system = "\n".join(system_lines).strip()

    user_lines = sections.get("user")
    user = "\n".join(user_lines).strip() if user_lines is not None else None

    return PromptTemplateSections(system=system, user=user or None)


def render_template(template: str, context: Mapping[str, str]) -> str:
    """Substitute ``{{ variable }}`` placeholders with values from *context*."""

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return context.get(key, "")

    return VARIABLE_PATTERN.sub(replacer, template)


def load_user_request_text(
    *,
    inline: Optional[str],
    file_path: Optional[Path],
) -> Optional[str]:
    """Return the combined user request text from inline and file inputs."""

    parts: list[str] = []

    if inline:
        inline_text = inline.strip()
        if inline_text:
            parts.append(inline_text)

    if file_path:
        file_text = file_path.read_text(encoding="utf-8").strip()
        if file_text:
            parts.append(file_text)

    if not parts:
        return None

    return "\n\n".join(parts)


def build_request_context(
    user_request: Optional[str],
    *,
    heading: str,
) -> Dict[str, str]:
    """Return template context values derived from the user request."""

    request = (user_request or "").strip()
    has_request = bool(request)
    request_heading = heading if has_request else ""
    request_section = (
        f"{request_heading}\n{request}" if has_request else ""
    )

    return {
        "user_request": request,
        "user_request_heading": request_heading,
        "user_request_section": request_section,
        "has_user_request": "true" if has_request else "false",
        "progress_section": "",
        "progress_count": "0",
    }


def augment_context(
    base_context: Mapping[str, str],
    *,
    user_message: Optional[str] = None,
    runtime_context: Optional[str] = None,
) -> Dict[str, str]:
    """Return a copy of *base_context* with runtime values merged in."""

    merged = dict(base_context)
    if user_message is not None:
        merged["user_message"] = user_message
    if runtime_context:
        trimmed = runtime_context.strip()
        merged["runtime_context"] = trimmed
        merged["runtime_context_section"] = f"Context:\n{trimmed}"
    else:
        merged.setdefault("runtime_context", "")
        merged.setdefault("runtime_context_section", "")
    return merged


__all__ = [
    "PromptTemplateSections",
    "parse_prompt_template",
    "render_template",
    "load_user_request_text",
    "build_request_context",
    "augment_context",
]
