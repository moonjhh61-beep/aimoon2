"""CLI entry point for ResearchSummaryAgency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from head_agency.cli_args import add_key_env_argument, add_user_request_arguments
from head_agency.llm_client import LLMClient
from head_agency.prompt_template import (
    PromptTemplateSections,
    build_request_context,
    load_user_request_text,
    parse_prompt_template,
    render_template,
)
from head_agency.logging_utils import configure_logging

from .agency import ResearchSummaryAgency, ResearchSummaryAgencyConfig
from .config import (
    API_KEY_ENV_VAR,
    DEFAULT_KEY_PATH,
    DEFAULT_PROMPT_PATH,
    read_api_key,
    read_prompt,
)


REQUEST_HEADING = "### 요약 요청"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ResearchSummaryAgency once or in interactive mode.")
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
    add_key_env_argument(parser, default=API_KEY_ENV_VAR)
    add_user_request_arguments(parser, label="summary brief")
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
    user_request: Optional[str] = None,
    request_heading: str = REQUEST_HEADING,
) -> ResearchSummaryAgency:
    template_text = read_prompt(prompt_path)
    sections: PromptTemplateSections = parse_prompt_template(template_text)
    context = build_request_context(user_request, heading=request_heading)
    system_template = sections.system or ""
    render_template(system_template, context)
    api_key = read_api_key(key_path, env_var=key_env)
    llm_client = LLMClient(api_key=api_key, agency_name="ResearchSummaryAgency")
    config = ResearchSummaryAgencyConfig(
        system_template=system_template,
        temperature=temperature,
        model=model,
        user_template=sections.user,
        prompt_context=context,
    )
    return ResearchSummaryAgency(llm_client, config)


def run_interactive(agency: ResearchSummaryAgency) -> None:
    print("ResearchSummaryAgency ready. Type your messages, or press Ctrl+C to exit.")
    try:
        while True:
            user_message = input("You > ").strip()
            if not user_message:
                continue
            response = agency.respond(user_message)
            print(f"ResearchSummaryAgency > {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    user_request = load_user_request_text(
        inline=args.user_request,
        file_path=args.user_request_file,
    )
    agency = create_agency(
        prompt_path=args.prompt,
        key_path=args.key_file,
        key_env=args.key_env,
        temperature=args.temperature,
        model=args.model,
        user_request=user_request,
    )

    if args.message:
        print(agency.respond(args.message))
    else:
        run_interactive(agency)


if __name__ == "__main__":
    main()
