"""Reusable CLI argument helpers."""

from __future__ import annotations

import argparse

from .config import API_KEY_ENV_VAR


def add_key_env_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = API_KEY_ENV_VAR,
) -> None:
    """Attach the ``--key-env`` option to *parser*."""

    parser.add_argument(
        "--key-env",
        type=str,
        default=default,
        help=f"Environment variable to read the API key from (default: {default})",
    )


__all__ = ["add_key_env_argument"]
