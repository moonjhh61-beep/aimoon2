"""CLI entry point for HeadAgency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from .agency import HeadAgency, HeadAgencyConfig
from .cli_args import add_key_env_argument, add_user_request_arguments
from .config import DEFAULT_KEY_PATH, DEFAULT_PROMPT_PATH, read_api_key, read_prompt
from .prompt_template import (
    PromptTemplateSections,
    build_request_context,
    load_user_request_text,
    parse_prompt_template,
    render_template,
)
from .llm_client import LLMClient
from .logging_utils import configure_logging


REQUEST_HEADING = "### 사용자 요청"
BRANCH_PATTERN = re.compile(r"선택한\s*branch\s*[:=]\s*(\d+)")


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
    add_user_request_arguments(parser, label="user request")
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
) -> HeadAgency:
    template_text = read_prompt(prompt_path)
    sections: PromptTemplateSections = parse_prompt_template(template_text)
    context = build_request_context(user_request, heading=request_heading)
    system_template = sections.system or ""
    api_key = read_api_key(key_path, env_var=key_env)
    llm_client = LLMClient(api_key=api_key)
    config = HeadAgencyConfig(
        system_template=system_template,
        temperature=temperature,
        model=model,
        user_template=sections.user,
        prompt_context=context,
    )
    # Pre-render the system template with static context to ensure template errors surface early.
    render_template(system_template, context)
    return HeadAgency(llm_client, config)


def handle_branch_result(
    response: str,
    *,
    agency: HeadAgency,
    original_message: str,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
    depth: int = 0,
    max_depth: int = 5,
) -> None:
    if depth >= max_depth:
        return

    match = BRANCH_PATTERN.search(response)
    if not match:
        return

    branch = match.group(1)
    if branch == "1":
        result = run_research_branch(
            original_message=original_message,
            head_response=response,
            key_file=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
        )
        label = "ResearchAgency"
    elif branch == "2":
        result = run_backtest_branch(
            original_message=original_message,
            head_response=response,
            key_file=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
        )
        label = "BacktestAgency"
    else:
        return

    if not result:
        return

    agency.add_progress_entry(label, result)

    follow_up = agency.respond(original_message)
    print(f"HeadAgency > {follow_up}\n")
    handle_branch_result(
        follow_up,
        agency=agency,
        original_message=original_message,
        key_file=key_file,
        key_env=key_env,
        temperature=temperature,
        model=model,
        user_request=user_request,
        depth=depth + 1,
        max_depth=max_depth,
    )


def run_research_branch(
    *,
    original_message: str,
    head_response: str,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
) -> Optional[str]:
    try:
        from research_agency.main import (
            REQUEST_HEADING as RESEARCH_REQUEST_HEADING,
            create_agency as create_research_agency,
        )
        from research_agency.config import DEFAULT_PROMPT_PATH as RESEARCH_PROMPT_PATH

        research_agency = create_research_agency(
            prompt_path=RESEARCH_PROMPT_PATH,
            key_path=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
            request_heading=RESEARCH_REQUEST_HEADING,
        )
        result = research_agency.respond(head_response, context=original_message)
        print("\n[ResearchAgency]\n" + result)
        return result
    except Exception as exc:  # pragma: no cover - runtime safety net
        print(
            f"[HeadAgency] Failed to run ResearchAgency branch: {exc}",
            file=sys.stderr,
        )
        return None


def run_backtest_branch(
    *,
    original_message: str,
    head_response: str,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
) -> Optional[str]:
    try:
        from backtest_agency.main import (
            REQUEST_HEADING as BACKTEST_REQUEST_HEADING,
            create_agency as create_backtest_agency,
        )
        from backtest_agency.config import DEFAULT_PROMPT_PATH as BACKTEST_PROMPT_PATH

        backtest_agency = create_backtest_agency(
            prompt_path=BACKTEST_PROMPT_PATH,
            key_path=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
            request_heading=BACKTEST_REQUEST_HEADING,
        )
        result = backtest_agency.respond(head_response, context=original_message)
        print("\n[BacktestAgency]\n" + result)
        return result
    except Exception as exc:  # pragma: no cover - runtime safety net
        print(
            f"[HeadAgency] Failed to run BacktestAgency branch: {exc}",
            file=sys.stderr,
        )
        return None


def run_interactive(
    agency: HeadAgency,
    *,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
) -> None:
    print("HeadAgency ready. Type your messages, or press Ctrl+C to exit.")
    try:
        while True:
            user_message = input("You > ").strip()
            if not user_message:
                continue
            response = agency.respond(user_message)
            print(f"HeadAgency > {response}\n")
            handle_branch_result(
                response,
                agency=agency,
                original_message=user_message,
                key_file=key_file,
                key_env=key_env,
                temperature=temperature,
                model=model,
                user_request=user_request,
            )
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
        response = agency.respond(args.message)
        print(response)
        handle_branch_result(
            response,
            agency=agency,
            original_message=args.message,
            key_file=args.key_file,
            key_env=args.key_env,
            temperature=args.temperature,
            model=args.model,
            user_request=user_request,
        )
    else:
        run_interactive(
            agency,
            key_file=args.key_file,
            key_env=args.key_env,
            temperature=args.temperature,
            model=args.model,
            user_request=user_request,
        )


if __name__ == "__main__":
    main()
