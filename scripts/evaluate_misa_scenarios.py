from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.catalog import load_products
from mai_agent.config import get_settings
from mai_agent.graph import create_agent_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "tests" / "misa_scenarios.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "results" / "semantic_evaluation.json"
DEFAULT_REQUIREMENTS = ["R01", "R04", "R05", "R06", "R07"]

Verdict = Literal["pass", "fail", "invalid_test", "error"]


@tool
def grade_pass(reason: str) -> str:
    """Câu trả lời đáp ứng đúng ý nghĩa nghiệp vụ và an toàn."""
    return reason


@tool
def grade_fail(reason: str) -> str:
    """Câu trả lời sai, thiếu trọng yếu, bịa đặt hoặc vi phạm nghiệp vụ."""
    return reason


@tool
def grade_invalid_test(reason: str) -> str:
    """Test hoặc đáp án tham chiếu sai/mâu thuẫn dữ liệu thật hay thiếu tiền đề."""
    return reason


JUDGE_TOOLS = [grade_pass, grade_fail, grade_invalid_test]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy scenario và dùng Gemini chấm ngữ nghĩa câu trả lời của Mai."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--requirements",
        nargs="+",
        default=DEFAULT_REQUIREMENTS,
        help="Các requirement cần chạy; mặc định bỏ R02 và R03.",
    )
    parser.add_argument("--limit-per-requirement", type=int)
    parser.add_argument("--case-retries", type=int, default=3)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path = Path(file.name)
    temporary_path.replace(path)


def load_scenarios(
    path: Path,
    requirement_ids: set[str],
    limit_per_requirement: int | None,
) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    selected: list[dict[str, Any]] = []
    for requirement in payload.get("requirements", []):
        requirement_id = str(requirement.get("id", ""))
        if requirement_id not in requirement_ids:
            continue
        cases = requirement.get("cases", [])
        if limit_per_requirement is not None:
            cases = cases[:limit_per_requirement]
        for case in cases:
            if not isinstance(case, dict):
                continue
            selected.append(
                {
                    **case,
                    "requirement_id": requirement_id,
                    "requirement_name": requirement.get("name", ""),
                    "requirement_description": requirement.get("description", ""),
                }
            )
    return selected


def load_previous_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(item["case_id"]): item
        for item in payload.get("results", [])
        if isinstance(item, dict) and item.get("case_id")
    }


def user_turn_text(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if isinstance(turn, dict):
        return str(turn.get("content") or turn.get("message") or "").strip()
    return str(turn).strip()


def state_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reply": result.get("reply", ""),
        "session": result.get("session", {}),
        "handoff": result.get("handoff", {}),
        "active_order": result.get("active_order", {}),
    }


def run_case(case: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    graph = create_agent_graph(checkpointer=InMemorySaver())
    case_id = str(case.get("id") or case.get("case_id") or "unknown")
    config = {"configurable": {"thread_id": f"semantic-eval-{case_id}"}}
    user_id = f"semantic-eval-{case_id}"
    transcript: list[dict[str, str]] = []
    result: dict[str, Any] = {}

    turns = [*case.get("setup_turns", []), case.get("user_message", "")]
    for raw_turn in turns:
        turn = user_turn_text(raw_turn)
        if not turn:
            continue
        transcript.append({"role": "user", "content": turn})
        result = graph.invoke(
            {"user_id": user_id, "messages": [HumanMessage(content=turn)]},
            config=config,
        )
        reply = str(result.get("reply", "")).strip()
        if reply:
            transcript.append({"role": "assistant", "content": reply})

    return transcript, state_summary(result)


def is_transient_api_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".upper()
    return any(
        marker in message
        for marker in (
            "500 INTERNAL",
            "429",
            "RESOURCE_EXHAUSTED",
            "503",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "TIMEOUT",
            "TIMED OUT",
            "CONNECTION RESET",
        )
    )


def run_case_with_retries(
    case: dict[str, Any],
    *,
    retries: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return run_case(case)
        except Exception as error:  # noqa: BLE001 - phân loại lỗi API để retry
            last_error = error
            if not is_transient_api_error(error) or attempt == retries:
                raise
            delay = min(5 * 2 ** (attempt - 1), 30)
            print(
                f"  {case.get('id')}: lỗi tạm thời ở lần {attempt}/{retries}; "
                f"thử lại sau {delay}s: {error}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Không chạy được case sau {retries} lần: {last_error}")


def semantic_grade(
    case: dict[str, Any],
    transcript: list[dict[str, str]],
    final_state: dict[str, Any],
    *,
    retries: int,
) -> tuple[Verdict, str]:
    settings = get_settings()
    judge = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
        temperature=0,
    ).bind_tools(JUDGE_TOOLS, tool_choice="any")

    judge_input = {
        "requirement": {
            "id": case.get("requirement_id"),
            "name": case.get("requirement_name"),
            "description": case.get("requirement_description"),
        },
        "scenario": {
            "title": case.get("title"),
            "setup_turns": case.get("setup_turns", []),
            "user_message": case.get("user_message"),
            "reference_answer": case.get("reference_answer"),
            "expected_notes": case.get("expected", {}),
        },
        "actual_transcript": transcript,
        "actual_final_state": final_state,
        "authoritative_catalog": load_products(),
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = judge.invoke(
                [
                    SystemMessage(
                        content=(
                            "Bạn là giám khảo QA độc lập. Chấm theo NGỮ NGHĨA, không "
                            "so khớp nguyên văn. Bắt buộc gọi đúng một tool. Catalog "
                            "thật và đặc tả requirement có ưu tiên cao hơn đáp án tham "
                            "chiếu do model sinh. Gọi grade_pass nếu hành vi thực tế "
                            "đáp ứng ý định và không có lỗi nghiệp vụ trọng yếu. Gọi "
                            "grade_fail nếu bot sai, thiếu trọng yếu, bịa dữ liệu, làm "
                            "sai state hoặc vi phạm an toàn. Gọi grade_invalid_test nếu "
                            "reference_answer/expected mâu thuẫn catalog, scenario thiếu "
                            "tiền đề cần thiết hoặc bản thân test không thể chấm công "
                            "bằng. Không hạ điểm vì khác văn phong hay khác câu chữ. Lý "
                            "do phải ngắn gọn và nêu bằng chứng cụ thể."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(judge_input, ensure_ascii=False, default=str)
                    ),
                ]
            )
            if not response.tool_calls:
                raise RuntimeError("Gemini judge không trả tool call.")
            call = response.tool_calls[0]
            verdict_by_tool: dict[str, Verdict] = {
                "grade_pass": "pass",
                "grade_fail": "fail",
                "grade_invalid_test": "invalid_test",
            }
            verdict = verdict_by_tool.get(call["name"])
            if verdict is None:
                raise RuntimeError(f"Judge gọi tool không hợp lệ: {call['name']}")
            return verdict, str(call["args"].get("reason", "")).strip()
        except Exception as error:  # noqa: BLE001 - lỗi API cần retry và ghi báo cáo
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Không chấm được sau {retries} lần: {last_error}")


def build_report(
    selected_cases: list[dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_ids = {str(case.get("id")) for case in selected_cases}
    results = [
        result
        for case_id, result in results_by_id.items()
        if case_id in selected_ids
    ]
    counts = Counter(str(result.get("verdict", "error")) for result in results)
    valid_graded = counts["pass"] + counts["fail"]
    return {
        "selected_total": len(selected_cases),
        "completed": len(results),
        "remaining": len(selected_cases) - len(results),
        "passed": counts["pass"],
        "failed": counts["fail"],
        "invalid_test": counts["invalid_test"],
        "error": counts["error"],
        "semantic_pass_rate": (
            round(counts["pass"] * 100 / valid_graded, 2) if valid_graded else 0.0
        ),
        "results": sorted(results, key=lambda item: str(item.get("case_id", ""))),
    }


def main() -> None:
    args = parse_args()
    if args.limit_per_requirement is not None and args.limit_per_requirement <= 0:
        raise ValueError("limit-per-requirement phải lớn hơn 0.")
    if args.case_retries <= 0 or args.judge_retries <= 0:
        raise ValueError("case-retries và judge-retries phải lớn hơn 0.")

    selected_cases = load_scenarios(
        args.input,
        set(args.requirements),
        args.limit_per_requirement,
    )
    results_by_id = {} if args.no_resume else load_previous_results(args.output)

    for index, case in enumerate(selected_cases, start=1):
        case_id = str(case.get("id"))
        previous_result = results_by_id.get(case_id)
        if previous_result and previous_result.get("verdict") != "error":
            continue
        try:
            transcript, final_state = run_case_with_retries(
                case,
                retries=args.case_retries,
            )
            verdict, reason = semantic_grade(
                case,
                transcript,
                final_state,
                retries=args.judge_retries,
            )
            result = {
                "case_id": case_id,
                "requirement_id": case.get("requirement_id"),
                "title": case.get("title"),
                "verdict": verdict,
                "reason": reason,
                "transcript": transcript,
                "final_state": final_state,
            }
        except Exception as error:  # noqa: BLE001 - mỗi case phải được cô lập
            result = {
                "case_id": case_id,
                "requirement_id": case.get("requirement_id"),
                "title": case.get("title"),
                "verdict": "error",
                "reason": f"{type(error).__name__}: {error}",
            }

        results_by_id[case_id] = result
        report = build_report(selected_cases, results_by_id)
        atomic_write_json(args.output, report)
        print(
            f"[{index}/{len(selected_cases)}] {case_id}: {result['verdict']} - "
            f"{result['reason']}",
            flush=True,
        )
        if args.delay > 0:
            time.sleep(args.delay)

    report = build_report(selected_cases, results_by_id)
    atomic_write_json(args.output, report)
    print(
        "Hoàn tất: "
        f"pass={report['passed']}, fail={report['failed']}, "
        f"invalid={report['invalid_test']}, error={report['error']}, "
        f"pass_rate={report['semantic_pass_rate']}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
