"""Utilities for parsing structured research plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResearchStep:
    """Single step in a research execution plan."""

    id: int
    name: str
    description: str
    data_config: Optional[Dict[str, Any]] = None

    def as_dict(self) -> dict:
        payload: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }
        if self.data_config:
            payload["data_config"] = self.data_config
        return payload


def _strip_code_fence(text: str) -> str:
    pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    match = pattern.match(text.strip())
    if match:
        return match.group(1)
    return text


def extract_research_plan(text: str) -> Optional[List[ResearchStep]]:
    """Parse JSON plan emitted by ResearchAgency."""

    candidate = _strip_code_fence(text).strip()

    def _loads(payload_text: str):
        try:
            return json.loads(payload_text)
        except json.JSONDecodeError:
            return None

    payload = _loads(candidate)
    if payload is None:
        array_match = re.search(r"\[.*\]", candidate, re.DOTALL)
        if array_match:
            payload = _loads(array_match.group(0))
        if payload is None:
            object_match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if object_match:
                payload = _loads(object_match.group(0))
    if payload is None:
        return None

    if isinstance(payload, dict) and "steps" in payload:
        steps_data = payload["steps"]
    elif isinstance(payload, list):
        steps_data = payload
    else:
        return None

    if not isinstance(steps_data, list):
        return None

    steps: List[ResearchStep] = []
    for item in steps_data:
        if not isinstance(item, dict):
            continue
        try:
            step = ResearchStep(
                id=int(item["id"]),
                name=str(item["name"]),
                description=str(item["description"]),
                data_config=item.get("data_config"),
            )
        except (KeyError, ValueError, TypeError):
            continue
        steps.append(step)

    if not steps:
        return None

    steps.sort(key=lambda s: s.id)
    return steps


def format_plan_json(steps: List[ResearchStep]) -> str:
    """Return a prettified JSON representation of *steps*."""

    return json.dumps({"steps": [step.as_dict() for step in steps]}, ensure_ascii=False, indent=2)


__all__ = ["ResearchStep", "extract_research_plan", "format_plan_json"]
