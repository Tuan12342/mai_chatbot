#!/usr/bin/env python3
"""Tạo bộ acceptance S1, S3-S7 từ corpus chuẩn, loại bỏ hoàn toàn S2."""

import json
import re
from collections import Counter
from pathlib import Path

from mai_agent.acceptance import load_acceptance_cases

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DOCUMENT = (
    PROJECT_DIR / "data" / "350 câu hỏi & trả lời mẫu – Mai, chatbot OA Cosmetics.docx"
)
OUTPUT_JSON = PROJECT_DIR / "data" / "acceptance_questions_s1_s3_s7.json"
OUTPUT_MARKDOWN = PROJECT_DIR / "data" / "acceptance_questions_s1_s3_s7.md"
SCOPE_FILE = PROJECT_DIR / "data" / "acceptance_scope.json"
SELECTED_SECTIONS = {1, 3, 4, 5, 6, 7}
LEADING_PRECONDITION = re.compile(r"^\((?P<precondition>[^)]+)\)\s*(?P<message>.*)$")
SECTION_NAMES = {
    1: "RAG sản phẩm có kiểm soát",
    3: "Hồ sơ da và memory dài hạn",
    4: "Chốt đơn và sửa giữa chừng",
    5: "Khách nóng giận và handoff",
    6: "An toàn và human-in-the-loop",
    7: "Đa ngôn ngữ",
}


def _section_number(section: str) -> int:
    match = re.search(r"Mục\s+(\d+)", section)
    if not match:
        raise ValueError(f"Không xác định được số mục: {section}")
    return int(match.group(1))


def _split_expected_answer(expected_answer: str) -> tuple[str, str | None]:
    if "Hành động:" not in expected_answer:
        return expected_answer.strip(), None
    answer, expected_action = expected_answer.split("Hành động:", 1)
    return answer.strip(), expected_action.strip()


def build_cases() -> list[dict[str, object]]:
    scope = json.loads(SCOPE_FILE.read_text(encoding="utf-8"))
    excluded_cases = set(scope.get("excluded_cases", {}))
    selected: list[dict[str, object]] = []
    for source_case in load_acceptance_cases(SOURCE_DOCUMENT):
        section_number = _section_number(source_case.section)
        if section_number not in SELECTED_SECTIONS or source_case.case_id in excluded_cases:
            continue

        precondition_match = LEADING_PRECONDITION.match(source_case.question)
        precondition = (
            precondition_match.group("precondition").strip() if precondition_match else None
        )
        user_message = (
            precondition_match.group("message").strip()
            if precondition_match
            else source_case.question
        )
        expected_answer, expected_action = _split_expected_answer(source_case.expected_answer)
        selected.append(
            {
                "case_id": source_case.case_id,
                "requirement": f"S{section_number}",
                "requirement_name": SECTION_NAMES[section_number],
                "case_type": "scenario" if precondition else "standalone",
                "precondition": precondition,
                "user_message": user_message,
                "expected_answer": expected_answer,
                "expected_action": expected_action,
            }
        )
    return selected


def write_json(cases: list[dict[str, object]]) -> None:
    scope = json.loads(SCOPE_FILE.read_text(encoding="utf-8"))
    payload = {
        "source": SOURCE_DOCUMENT.name,
        "excluded_requirements": ["S2"],
        "excluded_cases": scope["excluded_cases"],
        "total_cases": len(cases),
        "cases_per_requirement": dict(
            sorted(Counter(str(case["requirement"]) for case in cases).items())
        ),
        "cases": cases,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown(cases: list[dict[str, object]]) -> None:
    lines = [
        f"# Bộ {len(cases)} câu hỏi acceptance — phạm vi đã triển khai",
        "",
        "S2 (tra đơn và tồn kho) đã được loại hoàn toàn.",
        "Các chức năng S4 chưa triển khai được loại theo `data/acceptance_scope.json`.",
        "",
    ]
    current_requirement = ""
    for case in cases:
        requirement = str(case["requirement"])
        if requirement != current_requirement:
            current_requirement = requirement
            lines.extend(
                [
                    f"## {requirement} — {case['requirement_name']}",
                    "",
                ]
            )
        lines.append(f"### {case['case_id']}")
        lines.append("")
        if case["precondition"]:
            lines.append(f"- Precondition: {case['precondition']}")
        lines.append(f"- Khách: {case['user_message']}")
        lines.append(f"- Kỳ vọng: {case['expected_answer']}")
        if case["expected_action"]:
            lines.append(f"- Hành động kỳ vọng: {case['expected_action']}")
        lines.append("")
    OUTPUT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = build_cases()
    counts = Counter(str(case["requirement"]) for case in cases)
    expected_counts = Counter({"S1": 50, "S3": 50, "S4": 22, "S5": 50, "S6": 50, "S7": 50})
    if len(cases) != 272 or counts != expected_counts:
        raise RuntimeError(f"Corpus không đúng phạm vi 272 ca: {counts}")
    write_json(cases)
    write_markdown(cases)
    print(f"Đã tạo {len(cases)} ca: {dict(sorted(counts.items()))}")
    print(OUTPUT_JSON)
    print(OUTPUT_MARKDOWN)


if __name__ == "__main__":
    main()
