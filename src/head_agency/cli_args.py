"""Reusable CLI argument helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def add_user_request_arguments(
    parser: argparse.ArgumentParser,
    *,
    label: str = "user request",
) -> None:
    """Attach shared ``--request`` options to *parser*."""

    parser.add_argument(
        "--request",
        "--user-request",
        dest="user_request",
        type=str,
        default=None,
        help=f"Text describing the {label} to append to the system prompt.",
    )
    parser.add_argument(
        "--request-file",
        dest="user_request_file",
        type=Path,
        default=None,
        help=f"Read the {label} from a file and append it to the system prompt.",
    )


__all__ = ["add_key_env_argument", "add_user_request_arguments"]
