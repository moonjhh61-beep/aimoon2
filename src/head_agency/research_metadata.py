"""Helpers for tracking research agency instruction metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass(frozen=True)
class ResearchMetadataLabels:
    """Labels and formatting configuration for research instructions."""

    scope: str = "조사 범위"
    factors: str = "조사 펙터"
    timeframe: str = "조사 시간 단위"
    separator: str = " | "

    def ordered_fields(self) -> Tuple[Tuple[str, str], ...]:
        """Return the attribute order used when formatting metadata."""

        return (
            ("scope", self.scope),
            ("factors", self.factors),
            ("timeframe", self.timeframe),
        )


@dataclass
class ResearchMetadata:
    """Container for the instruction details passed to ResearchAgency."""

    scope: Optional[str] = None
    factors: Optional[str] = None
    timeframe: Optional[str] = None

    def has_data(self) -> bool:
        return any((self.scope, self.factors, self.timeframe))

    def format_header(self, labels: ResearchMetadataLabels) -> Optional[str]:
        parts = []
        for attr, label in labels.ordered_fields():
            value = getattr(self, attr)
            if value:
                parts.append(f"{label}: {value}")
        if not parts:
            return None
        return labels.separator.join(parts)


DEFAULT_RESEARCH_LABELS = ResearchMetadataLabels()


def extract_research_metadata(
    text: str,
    *,
    labels: ResearchMetadataLabels = DEFAULT_RESEARCH_LABELS,
) -> ResearchMetadata:
    """Parse the instruction block written by HeadAgency for ResearchAgency."""

    def find_value(label: str) -> Optional[str]:
        pattern = re.compile(rf"^\s*{re.escape(label)}\s*[:=：]\s*(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        value = matches[-1].group(1).strip()
        return value or None

    return ResearchMetadata(
        scope=find_value(labels.scope),
        factors=find_value(labels.factors),
        timeframe=find_value(labels.timeframe),
    )


__all__ = [
    "DEFAULT_RESEARCH_LABELS",
    "ResearchMetadata",
    "ResearchMetadataLabels",
    "extract_research_metadata",
]
