"""ResearchRunnerAgency orchestrator."""

from __future__ import annotations

from head_agency.agency import HeadAgency, HeadAgencyConfig


ResearchRunnerAgencyConfig = HeadAgencyConfig


class ResearchRunnerAgency(HeadAgency):
    """Alias of ``HeadAgency`` for runner-specific workflows."""


__all__ = ["ResearchRunnerAgency", "ResearchRunnerAgencyConfig"]
