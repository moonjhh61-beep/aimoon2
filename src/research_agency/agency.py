"""ResearchAgency orchestrator."""

from __future__ import annotations

from head_agency.agency import HeadAgency, HeadAgencyConfig


ResearchAgencyConfig = HeadAgencyConfig


class ResearchAgency(HeadAgency):
    """Alias of ``HeadAgency`` for semantic clarity."""


__all__ = ["ResearchAgency", "ResearchAgencyConfig"]
