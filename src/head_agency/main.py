"""CLI entry point for HeadAgency."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from .research_metadata import DEFAULT_RESEARCH_LABELS, ResearchMetadata, extract_research_metadata
from .research_plan import ResearchStep, extract_research_plan, format_plan_json
from .execution import load_dataset, np, pd


REQUEST_HEADING = "### 사용자 요청"
BRANCH_PATTERN = re.compile(r"선택한\s*branch\s*[:=]\s*(\d+)")
JSON_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
PYTHON_BLOCK_PATTERN = re.compile(r"^```(?:python)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_json_response(text: str) -> Optional[dict]:
    """Attempt to parse *text* as JSON, handling fenced blocks."""

    stripped = text.strip()
    fence_match = JSON_BLOCK_PATTERN.match(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(stripped)
        return obj
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"[\[{]", stripped):
        try:
            obj, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        return obj

    return None


def _strip_code_block(code: str) -> str:
    """Remove fenced code blocks from *code* if present."""

    match = PYTHON_BLOCK_PATTERN.match(code.strip())
    if match:
        return match.group(1)
    return code


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of *value* into JSON-serialisable objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient='records')]

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            pass

    return str(value)


def _describe_scope_from_config(data_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Construct a human-readable scope string from a data config block."""

    if not data_config:
        return None

    parts: List[str] = []
    symbols = data_config.get("symbols")
    if symbols:
        if isinstance(symbols, (list, tuple)):
            symbols_text = ", ".join(str(sym) for sym in symbols)
        else:
            symbols_text = str(symbols)
        parts.append(f"심볼: {symbols_text}")

    start_date = data_config.get("start_date")
    end_date = data_config.get("end_date")
    if start_date or end_date:
        if start_date and end_date:
            parts.append(f"기간: {start_date}~{end_date}")
        elif start_date:
            parts.append(f"시작: {start_date}")
        else:
            parts.append(f"종료: {end_date}")

    data_type = data_config.get("data_type")
    if data_type:
        parts.append(f"데이터 타입: {data_type}")

    return " | ".join(parts) if parts else None




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
    parser.add_argument(
        "--max-depth",
        type=int,
        default=int(os.getenv("HEAD_AGENCY_MAX_DEPTH", "20")),
        help="Maximum recursive branch depth before stopping (env: HEAD_AGENCY_MAX_DEPTH)",
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
    llm_client = LLMClient(api_key=api_key, agency_name="HeadAgency")
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
    max_depth: int = 20,
) -> None:
    if depth >= max_depth:
        return

    match = BRANCH_PATTERN.search(response)
    if not match:
        return

    branch = match.group(1)
    if branch == "1":
        instruction = extract_research_metadata(response)
        research_output = run_research_branch(
            original_message=original_message,
            head_response=response,
            key_file=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
            instruction=instruction,
        )
        if not research_output:
            return

        agency.add_progress_entry("ResearchAgency", research_output.display_text)

        plan_steps = research_output.plan_steps or []
        if not plan_steps:
            plan_steps = [
                ResearchStep(
                    id=1,
                    name="기본",
                    description="리서치 결과를 기반으로 추가 실행 계획을 수립합니다.",
                    input="리서치 결과 텍스트",
                )
            ]

        if instruction and not instruction.scope:
            candidate_scope = _describe_scope_from_config(
                plan_steps[0].data_config if plan_steps else None
            )
            if candidate_scope:
                instruction.scope = candidate_scope

        step_results: List[dict] = []
        aggregated_snippets: List[str] = []
        for step in plan_steps:
            runner_output = run_research_runner_step(
                original_message=original_message,
                step=step,
                accumulated_results=step_results,
                data_config=step.data_config,
                instruction=instruction,
                key_file=key_file,
                key_env=key_env,
                temperature=temperature,
                model=model,
                user_request=user_request,
            )
            output_payload = None
            snippet = f"Step {step.id}: 실행 결과가 제공되지 않았습니다."
            if runner_output:
                output_payload = runner_output.data if runner_output.data is not None else runner_output.raw_text
                if runner_output.data:
                    candidate = runner_output.data.get("simulated_result") or runner_output.data.get("summary")
                    if isinstance(candidate, str) and candidate.strip():
                        snippet = f"Step {step.id}: {candidate.strip()}"
                    else:
                        snippet = f"Step {step.id}: {runner_output.raw_text.strip()}"
                elif runner_output.execution_result and isinstance(runner_output.execution_result, dict):
                    error_message = runner_output.execution_result.get("error")
                    if isinstance(error_message, str) and error_message.strip():
                        snippet = f"Step {step.id}: 오류 - {error_message.strip()}"
                    else:
                        snippet = f"Step {step.id}: {runner_output.raw_text.strip()}"
                else:
                    snippet = f"Step {step.id}: {runner_output.raw_text.strip()}"
            step_results.append({
                "step": step.as_dict(),
                "output": _json_safe(output_payload),
                "execution_stdout": runner_output.execution_stdout if runner_output else None,
                "execution_result": _json_safe(runner_output.execution_result) if runner_output else None,
            })
            aggregated_snippets.append(snippet)

        plan_json = research_output.plan_json if research_output.plan_json else format_plan_json(plan_steps)
        plan_payload = json.loads(plan_json) if plan_json else [step.as_dict() for step in plan_steps]
        summary_payload = json.dumps(
            {
                "plan": _json_safe(plan_payload),
                "runner_results": _json_safe(step_results),
                "combined_result": "\n".join(aggregated_snippets),
            },
            ensure_ascii=False,
            indent=2,
        )


        summary_result = run_research_summary_branch(
            original_message=original_message,
            summary_payload=summary_payload,
            key_file=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
        )
        if summary_result:
            agency.add_progress_entry("ResearchSummaryAgency", summary_result)

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
        return
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


@dataclass
class ResearchBranchOutput:
    display_text: str
    plan_steps: List[ResearchStep]
    plan_json: Optional[str]


@dataclass
class RunnerStepOutput:
    raw_text: str
    data: Optional[dict]
    execution_stdout: Optional[str] = None
    execution_result: Optional[dict] = None


ALLOWED_BUILTINS = {
    'abs': abs,
    'enumerate': enumerate,
    'bool': bool,
    'dict': dict,
    'float': float,
    'int': int,
    'list': list,
    'len': len,
    'max': max,
    'min': min,
    'print': print,
    'range': range,
    'set': set,
    'str': str,
    'round': round,
    'sum': sum,
    'sorted': sorted,
    '__import__': __import__,
    'tuple': tuple,
    'zip': zip,
    'Exception': Exception,
    'ValueError': ValueError,
    'KeyError': KeyError,
    'TypeError': TypeError,
}


def _coerce_to_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {str(i): item for i, item in enumerate(value)}
    return {'value': value}


def execute_runner_code(
    code: str,
    *,
    step_config: Optional[Dict[str, Any]] = None,
    step_payload: Optional[Dict[str, Any]] = None,
    previous_results: Optional[List[dict]] = None,
    instruction_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[dict]]:
    env_globals = {
        '__builtins__': ALLOWED_BUILTINS,
        'load_dataset': load_dataset,
        'pd': pd,
        'np': np,
        'step_config': step_config or {},
    }
    if step_payload:
        env_globals['current_step'] = step_payload
    elif step_config is not None:
        env_globals['current_step'] = {'data_config': step_config}
    if previous_results is not None:
        env_globals['previous_results'] = previous_results
    if instruction_metadata is not None:
        env_globals['instruction_metadata'] = instruction_metadata
    env_locals: dict[str, object] = {}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, env_globals, env_locals)
    except Exception as exc:  # pragma: no cover - safeguard
        trace = traceback.format_exc()
        error_payload = {'error': str(exc), 'traceback': trace}
        return buffer.getvalue() + '\nERROR: {}\n'.format(exc), error_payload

    result_obj = (
        env_locals.get('result')
        or env_globals.get('result')
        or env_locals.get('result_summary')
        or env_globals.get('result_summary')
    )
    result_dict = _coerce_to_dict(result_obj)
    return buffer.getvalue(), result_dict

def run_research_branch(
    *,
    original_message: str,
    head_response: str,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
    instruction: Optional[ResearchMetadata] = None,
) -> Optional[ResearchBranchOutput]:
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
        plan_steps = extract_research_plan(result) or []
        plan_json = format_plan_json(plan_steps) if plan_steps else None
        header = instruction.format_header(DEFAULT_RESEARCH_LABELS) if instruction else None
        output = f"{header}\n{result}" if header else result
        print("\n[ResearchAgency]\n" + output)
        return ResearchBranchOutput(
            display_text=output,
            plan_steps=plan_steps,
            plan_json=plan_json,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        print(
            f"[HeadAgency] Failed to run ResearchAgency branch: {exc}",
            file=sys.stderr,
        )
        return None


def run_research_runner_step(
    *,
    original_message: str,
    step: ResearchStep,
    accumulated_results: List[dict],
    data_config: Optional[Dict[str, Any]],
    instruction: Optional[ResearchMetadata],
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
) -> Optional[RunnerStepOutput]:
    try:
        from research_runner_agency.main import (
            REQUEST_HEADING as RUNNER_REQUEST_HEADING,
            create_agency as create_runner_agency,
        )
        from research_runner_agency.config import DEFAULT_PROMPT_PATH as RUNNER_PROMPT_PATH

        runner_agency = create_runner_agency(
            prompt_path=RUNNER_PROMPT_PATH,
            key_path=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
            request_heading=RUNNER_REQUEST_HEADING,
        )
        context_payload_dict = {
            "original_goal": original_message,
            "current_step": step.as_dict(),
            "previous_results": accumulated_results,
        }
        current_step_payload = step.as_dict()
        derived_scope = _describe_scope_from_config(data_config)
        if data_config:
            context_payload_dict["data_config"] = data_config
        instruction_payload = None
        if instruction and instruction.has_data():
            instruction_payload = {
                "scope": instruction.scope,
                "factors": instruction.factors,
                "timeframe": instruction.timeframe,
            }
        if derived_scope:
            if instruction_payload is None:
                instruction_payload = {}
            if not instruction_payload.get("scope"):
                instruction_payload["scope"] = derived_scope
        if instruction_payload:
            context_payload_dict["instruction_metadata"] = instruction_payload
        safe_context = _json_safe(context_payload_dict)
        context_payload = json.dumps(safe_context, ensure_ascii=False, indent=2)
        runner_message = (
            "아래 JSON은 현재 실행할 스텝과 이전 결과입니다.\n"
            "이 정보를 참고하여 코드를 생성하세요.\n" + context_payload
        )
        result_text = runner_agency.respond(runner_message, context=original_message)
        print("\n[ResearchRunnerAgency]\n" + result_text)
        parsed = _parse_json_response(result_text)
        parsed_payload = parsed if isinstance(parsed, dict) else None

        execution_stdout = None
        execution_result: Optional[Dict[str, Any]] = None
        if not parsed_payload:
            execution_result = {
                "error": "Runner response was not valid JSON; skipping execution.",
            }
        else:
            code = parsed_payload.get('code')
            if code:
                cleaned_code = _strip_code_block(code)
                execution_stdout, execution_result = execute_runner_code(
                    cleaned_code,
                    step_config=data_config or {},
                    step_payload=current_step_payload,
                    previous_results=accumulated_results,
                    instruction_metadata=instruction_payload,
                )
            else:
                execution_result = {
                    "error": "Runner response missing 'code' field; skipping execution.",
                }
        return RunnerStepOutput(
            raw_text=result_text,
            data=parsed_payload,
            execution_stdout=execution_stdout,
            execution_result=execution_result,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        print(
            f"[HeadAgency] Failed to run ResearchRunnerAgency branch: {exc}",
            file=sys.stderr,
        )
        return None




def run_research_summary_branch(
    *,
    original_message: str,
    summary_payload: str,
    key_file: Path,
    key_env: str,
    temperature: float,
    model: Optional[str],
    user_request: Optional[str],
) -> Optional[str]:
    try:
        from research_summary_agency.main import (
            REQUEST_HEADING as SUMMARY_REQUEST_HEADING,
            create_agency as create_summary_agency,
        )
        from research_summary_agency.config import (
            DEFAULT_PROMPT_PATH as SUMMARY_PROMPT_PATH,
        )

        summary_agency = create_summary_agency(
            prompt_path=SUMMARY_PROMPT_PATH,
            key_path=key_file,
            key_env=key_env,
            temperature=temperature,
            model=model,
            user_request=user_request,
            request_heading=SUMMARY_REQUEST_HEADING,
        )
        result = summary_agency.respond(summary_payload, context=original_message)
        print("\n[ResearchSummaryAgency]\n" + result)
        return result
    except Exception as exc:  # pragma: no cover - runtime safety net
        print(
            f"[HeadAgency] Failed to run ResearchSummaryAgency branch: {exc}",
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
    max_depth: int,
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
                max_depth=max_depth,
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
            max_depth=args.max_depth,
        )
    else:
        run_interactive(
            agency,
            key_file=args.key_file,
            key_env=args.key_env,
            temperature=args.temperature,
            model=args.model,
            user_request=user_request,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()
