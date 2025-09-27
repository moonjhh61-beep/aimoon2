"""ResearchSummaryAgency orchestrator."""

from __future__ import annotations

from head_agency.agency import HeadAgency, HeadAgencyConfig


ResearchSummaryAgencyConfig = HeadAgencyConfig


class ResearchSummaryAgency(HeadAgency):
    """Alias of ``HeadAgency`` tailored for summarisation flows."""


__all__ = ["ResearchSummaryAgency", "ResearchSummaryAgencyConfig"]
