"""CLI entry point for HeadAgency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .agency import HeadAgency, HeadAgencyConfig
from .cli_args import add_key_env_argument
from .config import DEFAULT_KEY_PATH, DEFAULT_PROMPT_PATH, read_api_key, read_prompt
from .llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HeadAgency once or in interactive mode.")
    parser.add_argument("message", nargs="?", help="Optional single prompt to send to the agency")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"Path to the system prompt file (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=DEFAULT_KEY_PATH,
        help=f"Path to the fallback API key file (default: {DEFAULT_KEY_PATH})",
    )
    add_key_env_argument(parser)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to the LLM",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override (defaults to the client default)",
    )
    return parser


def create_agency(
    prompt_path: Path,
    key_path: Path,
    *,
    key_env: str,
    temperature: float,
    model: Optional[str],
) -> HeadAgency:
    prompt = read_prompt(prompt_path)
    api_key = read_api_key(key_path, env_var=key_env)
    llm_client = LLMClient(api_key=api_key)
    config = HeadAgencyConfig(base_prompt=prompt, temperature=temperature, model=model)
    return HeadAgency(llm_client, config)


def run_interactive(agency: HeadAgency) -> None:
    print("HeadAgency ready. Type your messages, or press Ctrl+C to exit.")
    try:
        while True:
            user_message = input("You > ").strip()
            if not user_message:
                continue
            response = agency.respond(user_message)
            print(f"HeadAgency > {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    agency = create_agency(
        prompt_path=args.prompt,
        key_path=args.key_file,
        key_env=args.key_env,
        temperature=args.temperature,
        model=args.model,
    )

    if args.message:
        print(agency.respond(args.message))
    else:
        run_interactive(agency)


if __name__ == "__main__":
    main()
