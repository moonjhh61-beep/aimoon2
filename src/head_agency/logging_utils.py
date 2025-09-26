"""Logging configuration helpers."""

from __future__ import annotations

import logging
import os
import sys


DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
COLOR_RESET = "\033[0m"
COLOR_MAP = {
    logging.DEBUG: "\033[35m",  # Magenta
    logging.INFO: "\033[36m",  # Cyan
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[41m",  # Red background
}


class ColorFormatter(logging.Formatter):
    """Formatter that wraps log lines with ANSI colours when enabled."""

    def __init__(self, fmt: str, *, use_color: bool) -> None:
        super().__init__(fmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self._use_color:
            return message

        prefix = COLOR_MAP.get(record.levelno)
        if not prefix:
            return message

        return f"{prefix}{message}{COLOR_RESET}"


def configure_logging() -> None:
    """Initialise root logging if no handlers are configured."""

    root = logging.getLogger()
    if root.handlers:
        return

    level_name = os.getenv("HEAD_AGENCY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    use_color = sys.stderr.isatty() and os.getenv("HEAD_AGENCY_LOG_COLOR", "1") != "0"
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(DEFAULT_FORMAT, use_color=use_color))

    root.setLevel(level)
    root.addHandler(handler)


__all__ = ["configure_logging"]
