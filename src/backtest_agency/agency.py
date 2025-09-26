"""BacktestAgency orchestrator."""

from __future__ import annotations

from head_agency.agency import HeadAgency, HeadAgencyConfig


BacktestAgencyConfig = HeadAgencyConfig


class BacktestAgency(HeadAgency):
    """Alias of ``HeadAgency`` tuned for backtesting workflows."""


__all__ = ["BacktestAgency", "BacktestAgencyConfig"]
