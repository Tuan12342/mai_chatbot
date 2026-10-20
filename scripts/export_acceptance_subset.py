#!/usr/bin/env python3
"""Xuất một bộ acceptance JSON đã lọc, tương thích với run_acceptance.py."""

import argparse
import json
from collections import Counter
from pathlib import Path

from mai_agent.acceptance import load_acceptance_cases

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENT = (
    PROJECT_DIR / "data" / "350 câu hỏi & trả lời mẫu – Mai, chatbot OA Cosmetics.docx"
)
DEFAULT_SCOPE = PROJECT_DIR / "data" / "acceptance_scope.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-section",
        type=int,
        action="append",
        default=[],
        help="Loại toàn bộ mục, ví dụ --exclude-section 2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_acceptance_cases(args.document)
    scope = json.loads(args.scope.read_text(encoding="utf-8"))
    excluded_sections = {
        *scope.get("excluded_sections", []),
        *args.exclude_section,
    }
    excluded_cases = set(scope.get("excluded_cases", {}))

    selected = [
        case
        for case in cases
        if int(case.case_id.split("-")[0][1:]) not in excluded_sections
        and case.case_id not in excluded_cases
    ]
    counts = Counter(case.case_id.split("-")[0] for case in selected)
    payload = {
        "source": str(args.document),
        "scope": str(args.scope),
        "excluded_sections": sorted(excluded_sections),
        "total_cases": len(selected),
        "cases_per_section": dict(sorted(counts.items())),
        "cases": [
            {
                "section": case.section,
                "number": case.number,
                "case_id": case.case_id,
                "question": case.question,
                "expected_answer": case.expected_answer,
            }
            for case in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Đã xuất {len(selected)} ca: {dict(sorted(counts.items()))}")
    print(args.output)


if __name__ == "__main__":
    main()
