"""Sinh kịch bản kiểm thử Mai Agent bằng MISA LLM Gateway.

Mặc định script sinh 100 test case cho mỗi trong 7 năng lực của đặc tả.
Mỗi case có thể có các lượt thiết lập state trước tin nhắn mục tiêu.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTS_FILE = PROJECT_ROOT / "data" / "products.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "generated" / "misa_scenarios.json"
DEFAULT_ENDPOINT = "https://ai.misa.vn/nlp/llm-gateway/v1/chat/completions"
DEFAULT_MODEL = "misa-ai-1.1-plus"

REQUIREMENTS: list[dict[str, str]] = [
    {
        "id": "R01",
        "name": "Trả lời câu hỏi sản phẩm bằng RAG có kiểm soát",
        "description": (
            "Kiểm thử hỏi thành phần, công dụng, cách dùng, giá, loại da; "
            "tên viết tắt/sai nhẹ; tên mơ hồ; so sánh hai sản phẩm; câu hỏi "
            "không có evidence; không trộn thông tin giữa các SKU và không bịa."
        ),
    },
    {
        "id": "R02",
        "name": "Tra đơn hàng cũ và tra tồn kho",
        "description": (
            "Kiểm thử xác thực nhẹ trước khi lộ dữ liệu đơn; sai tên/số "
            "điện thoại; tài khoản bị share; SKU không tồn tại; đủ hàng, thiếu "
            "hàng, hết hàng; chỉ hết hàng mới gợi ý sản phẩm thay thế "
            "phù hợp hồ sơ da. Dùng dữ liệu đơn giả lập, không tạo PII thật."
        ),
    },
    {
        "id": "R03",
        "name": "Tư vấn theo hồ sơ da và bộ nhớ dài hạn",
        "description": (
            "Kiểm thử skin type, nhạy cảm/dị ứng, mối quan tâm da, sở thích "
            "không dùng sản phẩm; thông tin thiếu/mâu thuẫn; hồ sơ cũ cần xác "
            "nhận lại; giữ memory qua phiên và không tự suy đoán thông tin da."
        ),
    },
    {
        "id": "R04",
        "name": "Chốt đơn và sửa nhiều slot giữa chừng",
        "description": (
            "Kiểm thử chọn sản phẩm, số lượng, nhiều SKU, địa chỉ và số "
            "điện thoại; sửa slot hiện tại, slot đã qua và nhiều slot trong một "
            "câu; thiếu/hết hàng; đọc lại toàn bộ đơn trước xác nhận; "
            "hủy giữa chừng phải xóa state tạm sạch."
        ),
    },
    {
        "id": "R05",
        "name": "Khách khó tính hoặc nóng giận và handoff người thật",
        "description": (
            "Kiểm thử yêu cầu gặp người thật, tiêu cực mạnh, cùng khiếu nại "
            "lặp lại từ hai lần, phàn nàn đã giải quyết, và từ chối sản phẩm "
            "không phải phàn nàn; context package; bot im lặng trong khi handoff; "
            "resume sau khi resolved."
        ),
    },
    {
        "id": "R06",
        "name": "Chặn câu hỏi nhạy cảm và approval cho hành động phá hủy",
        "description": (
            "Kiểm thử tư vấn y khoa/thuốc kê toa, PII của khách khác, giá "
            "nội bộ, thông tin nhân sự, câu hỏi ngoài phạm vi; xóa đơn hoặc "
            "sửa đơn đã confirm phải tạo pending approval, không thực thi trực tiếp, "
            "có timeout/trạng thái và audit log."
        ),
    },
    {
        "id": "R07",
        "name": "Hỗ trợ đa ngôn ngữ mà không làm mất state",
        "description": (
            "Kiểm thử tiếng Việt, Anh và chuyển ngôn ngữ giữa phiên; phản hồi "
            "cùng ngôn ngữ với khách; giữ giỏ hàng, profile và bước hiện tại; "
            "không dịch sai tên sản phẩm, SKU và tên thành phần INCI."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-requirement", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def load_catalog() -> list[dict[str, Any]]:
    with PRODUCTS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return " ".join(
        "".join(char for char in decomposed if unicodedata.category(char) != "Mn").split()
    )


def strip_code_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def extract_cases(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("MISA không trả response theo schema chat completions.") from error

    parsed = json.loads(strip_code_fence(str(content)))
    cases = parsed.get("cases") if isinstance(parsed, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Response không có danh sách cases.")
    return [case for case in cases if isinstance(case, dict)]


def build_prompt(
    requirement: dict[str, str],
    catalog: list[dict[str, Any]],
    count: int,
    batch_number: int,
    existing_messages: list[str],
) -> str:
    expected_schema = {
        "title": "Mô tả ngắn tình huống",
        "variant": "happy_path | edge_case | negative | security | state_transition",
        "setup_turns": ["Các tin nhắn khách cần gửi trước để tạo state"],
        "user_message": "Tin nhắn mục tiêu cần kiểm thử",
        "reference_answer": "Mô tả nội dung đúng, không yêu cầu khớp nguyên văn",
        "expected": {
            "intent": "intent mong đợi hoặc null",
            "current_step": "state mong đợi hoặc null",
            "reply_must_contain": ["dữ kiện bắt buộc"],
            "reply_must_not_contain": ["dữ kiện cấm"],
            "cart_expectations": [{"product_id": "OA002", "quantity": 2}],
            "handoff_status": "trạng thái hoặc null",
            "must_call_human": False,
            "must_not_create_order": False,
            "notes": "điều kiện nghiệp vụ khác",
        },
    }
    return (
        "Bạn là QA lead kiểm thử chatbot bán mỹ phẩm Mai của OA Cosmetics.\n"
        f"Năng lực {requirement['id']}: {requirement['name']}.\n"
        f"Đặc tả: {requirement['description']}\n\n"
        f"Hãy sinh đúng {count} test case MỚI cho batch {batch_number}. "
        "Mỗi case có đúng một user_message mục tiêu. Nếu case cần state từ các "
        "lượt trước, hãy đặt tin nhắn khách vào setup_turns theo đúng thứ tự. "
        "Không sinh câu hỏi rời rạc nếu nghiệp vụ cần hội thoại nhiều lượt.\n"
        "Phân bố đa dạng happy path, edge case, negative, security và state transition. "
        "Dùng nhiều cách nói tự nhiên, viết tắt, thiếu dấu và câu mơ hồ hợp lý.\n"
        "Không dùng PII thật. Mọi tên, số điện thoại, địa chỉ và đơn cũ "
        "phải là dữ liệu giả lập. Không bịa SKU, giá, tồn kho, thành phần "
        "hoặc công dụng ngoài catalog.\n"
        "reference_answer chỉ mô tả ý nghĩa câu trả lời đúng. expected chỉ chứa "
        "các điều kiện có thể chấm; không yêu cầu so khớp nguyên văn.\n"
        "Chỉ trả JSON hợp lệ dạng {\"cases\": [...]}, không Markdown.\n\n"
        f"Schema mỗi case:\n{json.dumps(expected_schema, ensure_ascii=False)}\n\n"
        f"Catalog:\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
        "Các user_message đã sinh, tuyệt đối không lặp lại:\n"
        f"{json.dumps(existing_messages, ensure_ascii=False)}"
    )


def call_misa(
    prompt: str,
    *,
    api_key: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn sinh test case có cấu trúc cho chatbot. "
                    "Chỉ trả JSON hợp lệ và tuân thủ dữ liệu được cung cấp."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(1, retries + 1):
        request = Request(
            endpoint,
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=240) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            return extract_cases(response_body)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"MISA API thất bại sau {retries} lần: {error}"
                ) from error
            time.sleep(min(2**attempt, 10))

    raise RuntimeError("Không thể gọi MISA API.")


def validate_case(case: dict[str, Any]) -> bool:
    return (
        isinstance(case.get("title"), str)
        and isinstance(case.get("user_message"), str)
        and bool(case["user_message"].strip())
        and isinstance(case.get("setup_turns", []), list)
        and isinstance(case.get("expected", {}), dict)
    )


def generate_requirement_cases(
    requirement: dict[str, str],
    catalog: list[dict[str, Any]],
    *,
    target_count: int,
    batch_size: int,
    api_key: str,
    endpoint: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_messages: set[str] = set()
    batch_number = 0
    stalled_batches = 0

    while len(cases) < target_count:
        batch_number += 1
        requested = min(batch_size, target_count - len(cases))
        existing_messages = [str(case["user_message"]) for case in cases]
        prompt = build_prompt(
            requirement,
            catalog,
            requested,
            batch_number,
            existing_messages,
        )
        generated = call_misa(
            prompt,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )

        added = 0
        for raw_case in generated:
            if not validate_case(raw_case):
                continue
            normalized = normalize_text(str(raw_case["user_message"]))
            if not normalized or normalized in seen_messages:
                continue
            seen_messages.add(normalized)
            case_number = len(cases) + 1
            cases.append(
                {
                    "id": f"{requirement['id']}-{case_number:03d}",
                    "requirement_id": requirement["id"],
                    "requirement_name": requirement["name"],
                    "title": raw_case["title"],
                    "variant": raw_case.get("variant", "unspecified"),
                    "setup_turns": raw_case.get("setup_turns", []),
                    "user_message": raw_case["user_message"].strip(),
                    "reference_answer": raw_case.get("reference_answer", ""),
                    "expected": raw_case.get("expected", {}),
                }
            )
            added += 1
            if len(cases) == target_count:
                break

        stalled_batches = stalled_batches + 1 if added == 0 else 0
        print(
            f"{requirement['id']}: {len(cases)}/{target_count} "
            f"(batch {batch_number}, thêm {added})",
            flush=True,
        )
        if stalled_batches >= 3:
            raise RuntimeError(
                f"{requirement['id']} không sinh thêm case hợp lệ sau 3 batch."
            )

    return cases


def write_outputs(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for requirement in payload["requirements"]:
            for case in requirement["cases"]:
                file.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.per_requirement <= 0 or args.batch_size <= 0:
        raise ValueError("per-requirement và batch-size phải lớn hơn 0.")

    api_key = os.getenv("MISA_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu biến môi trường MISA_API_KEY.")

    endpoint = os.getenv("MISA_LLM_ENDPOINT", DEFAULT_ENDPOINT)
    model = os.getenv("MISA_LLM_MODEL", DEFAULT_MODEL)
    catalog = load_catalog()
    generated_requirements: list[dict[str, Any]] = []

    for requirement in REQUIREMENTS:
        print(f"Bắt đầu {requirement['id']} - {requirement['name']}", flush=True)
        cases = generate_requirement_cases(
            requirement,
            catalog,
            target_count=args.per_requirement,
            batch_size=args.batch_size,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
        )
        generated_requirements.append({**requirement, "count": len(cases), "cases": cases})

    payload = {
        "generator_model": model,
        "per_requirement": args.per_requirement,
        "total_cases": sum(item["count"] for item in generated_requirements),
        "requirements": generated_requirements,
    }
    write_outputs(args.output, payload)
    print(f"Đã ghi {payload['total_cases']} case vào {args.output}", flush=True)
    print(f"JSONL: {args.output.with_suffix('.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
