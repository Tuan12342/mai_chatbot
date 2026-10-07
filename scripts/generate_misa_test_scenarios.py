

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
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


class PermanentAPIError(RuntimeError):
    """Lỗi cấu hình/xác thực sẽ không tự hết nếu chỉ retry."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-requirement", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Không dùng lại các case đã lưu trong file output.",
    )
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


def parse_json_value(value: str) -> Any:
    """Đọc JSON kể cả khi model bọc code fence hoặc thêm chữ ở hai đầu."""
    text = strip_code_fence(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                return parsed
            except json.JSONDecodeError:
                continue
        raise original_error


def _content_as_text(content: Any) -> str:
    """Chuẩn hóa content dạng chuỗi hoặc danh sách content-part."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _case_collection(value: Any) -> list[dict[str, Any]]:
    """Tìm danh sách test case trong các vỏ JSON phổ biến của model."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []

    question_keys = {
        "user_message",
        "question",
        "câu_hỏi",
        "cau_hoi",
        "input",
        "query",
        "prompt",
    }
    if question_keys.intersection(value):
        return [value]

    for key in (
        "cases",
        "test_cases",
        "testCases",
        "scenarios",
        "test_scenarios",
        "questions",
        "items",
        "data",
        "result",
        "results",
    ):
        nested = value.get(key)
        cases = _case_collection(nested)
        if cases:
            return cases

    dictionary_items = [item for item in value.values() if isinstance(item, dict)]
    if dictionary_items and len(dictionary_items) == len(value):
        return dictionary_items
    return []


def _first_string(value: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def normalize_generated_case(raw_case: dict[str, Any]) -> dict[str, Any] | None:
    """Đưa nhiều cách đặt tên question/answer về schema duy nhất của project."""
    user_message = _first_string(
        raw_case,
        (
            "user_message",
            "question",
            "câu_hỏi",
            "cau_hoi",
            "input",
            "query",
            "prompt",
            "message",
        ),
    )
    if not user_message:
        return None

    reference_answer = _first_string(
        raw_case,
        (
            "reference_answer",
            "answer",
            "câu_trả_lời",
            "cau_tra_loi",
            "expected_answer",
            "expected_response",
            "ideal_answer",
            "response",
            "output",
        ),
    )
    expected_value = raw_case.get("expected", {})
    if isinstance(expected_value, str):
        reference_answer = reference_answer or expected_value.strip()
        expected: dict[str, Any] = {"notes": expected_value.strip()}
    elif isinstance(expected_value, dict):
        expected = expected_value
    else:
        expected = {}

    if not reference_answer:
        reference_answer = _first_string(expected, ("answer", "notes", "expected_reply"))
    if not reference_answer:
        return None

    setup_value = raw_case.get("setup_turns", raw_case.get("context", []))
    if isinstance(setup_value, str):
        setup_turns = [setup_value.strip()] if setup_value.strip() else []
    elif isinstance(setup_value, list):
        setup_turns = [str(item).strip() for item in setup_value if str(item).strip()]
    else:
        setup_turns = []

    title = _first_string(raw_case, ("title", "name", "description"))
    return {
        **raw_case,
        "title": title or user_message[:80],
        "variant": raw_case.get("variant", raw_case.get("type", "unspecified")),
        "setup_turns": setup_turns,
        "user_message": user_message,
        "reference_answer": reference_answer,
        "expected": expected,
    }


def _extract_plain_qa(text: str) -> list[dict[str, Any]]:
    """Fallback cho response dạng 'Câu hỏi: ... / Trả lời: ...'."""
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+[.)]\s*)?"
        r"(?:câu hỏi|question|user_message)\s*:\s*(?P<question>.+?)\s*\n"
        r"\s*(?:trả lời|answer|reference_answer)\s*:\s*(?P<answer>.+?)"
        r"(?=\n\s*(?:\d+[.)]\s*)?(?:câu hỏi|question|user_message)\s*:|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [
        {
            "title": match.group("question").strip()[:80],
            "user_message": match.group("question").strip(),
            "reference_answer": match.group("answer").strip(),
            "setup_turns": [],
            "expected": {},
        }
        for match in pattern.finditer(text)
    ]


def decode_api_response(raw_body: bytes) -> dict[str, Any]:
    """Hỗ trợ response JSON thường và OpenAI-compatible SSE streaming."""
    text = raw_body.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("MISA trả response rỗng.")
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("MISA không trả JSON object.")
        return parsed
    except json.JSONDecodeError:
        pass

    chunks: list[str] = []
    final_response: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        final_response = event
        choices = event.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        content = delta.get("content") or message.get("content")
        if isinstance(content, str):
            chunks.append(content)

    if chunks:
        return {"choices": [{"message": {"content": "".join(chunks)}}]}
    if final_response is not None:
        return final_response
    raise ValueError("Không đọc được response JSON hoặc SSE từ MISA.")


def extract_cases(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        content = response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("MISA không trả response theo schema chat completions.") from error

    raw_cases = _case_collection(content)
    text = _content_as_text(content)
    if not raw_cases:
        try:
            raw_cases = _case_collection(parse_json_value(text))
        except json.JSONDecodeError:
            pass
    if not raw_cases:
        raw_cases = _extract_plain_qa(text)

    cases = [
        normalized
        for raw_case in raw_cases
        if (normalized := normalize_generated_case(raw_case)) is not None
    ]
    if not cases:
        preview = " ".join(text.split())[:200]
        raise ValueError(
            "Response không chứa test case có đủ câu hỏi và câu trả lời. "
            f"Nội dung đầu: {preview!r}"
        )
    return cases


CATALOG_FIELDS_BY_REQUIREMENT: dict[str, tuple[str, ...]] = {
    "R01": ("id", "name", "skin_types", "ingredients", "benefits", "usage", "price_vnd"),
    "R02": ("id", "name", "category", "skin_types", "ingredients", "price_vnd", "stock"),
    "R03": ("id", "name", "skin_types", "ingredients", "benefits"),
    "R04": ("id", "name", "price_vnd", "stock"),
    "R05": ("id", "name"),
    "R06": ("id", "name"),
    "R07": ("id", "name", "ingredients"),
}

CASE_VARIANTS = (
    "happy_path",
    "edge_case",
    "negative",
    "security",
    "state_transition",
)

WRITING_STYLES = (
    "tiếng Việt tự nhiên có dấu",
    "tiếng Việt nói ngắn gọn",
    "tiếng Việt thiếu dấu hoặc viết tắt",
    "English",
    "hội thoại nhiều lượt",
    "câu diễn đạt mơ hồ nhưng thực tế",
)


def compact_catalog_for_batch(
    requirement_id: str,
    catalog: list[dict[str, Any]],
    batch_number: int,
    *,
    products_per_batch: int = 3,
) -> list[dict[str, Any]]:
    """Luân phiên một phần catalog để prompt nhỏ nhưng vẫn phủ hết sản phẩm."""
    if not catalog:
        return []
    fields = CATALOG_FIELDS_BY_REQUIREMENT[requirement_id]
    start = ((batch_number - 1) * products_per_batch) % len(catalog)
    selected = [
        catalog[(start + offset) % len(catalog)]
        for offset in range(min(products_per_batch, len(catalog)))
    ]
    return [{field: product[field] for field in fields if field in product} for product in selected]


def build_prompt(
    requirement: dict[str, str],
    catalog: list[dict[str, Any]],
    count: int,
    batch_number: int,
    existing_messages: list[str],
) -> str:
    catalog_sample = compact_catalog_for_batch(
        requirement["id"],
        catalog,
        batch_number,
    )
    target_variant = CASE_VARIANTS[(batch_number - 1) % len(CASE_VARIANTS)]
    writing_style = WRITING_STYLES[(batch_number - 1) % len(WRITING_STYLES)]
    first_case_number = len(existing_messages) + 1
    return (
        f"Sinh đúng {count} test case JSON cho chatbot mỹ phẩm. "
        f"Yêu cầu {requirement['id']}: {requirement['description']}\n"
        f"Đây là batch {batch_number}, bắt đầu từ case số {first_case_number}. "
        f"Ưu tiên variant={target_variant}; cách viết={writing_style}. "
        "Phải tạo tình huống mới, khác nội dung và mục đích cụ thể của các câu cũ.\n"
        "Mỗi case có một user_message tự nhiên và reference_answer mô tả cách xử lý đúng. "
        "Nếu cần hội thoại trước, ghi vào setup_turns. Dùng dữ liệu giả, không bịa dữ "
        "liệu sản phẩm. Chỉ trả JSON, không Markdown.\n"
        "JSON gốc phải có key cases là một mảng. Mỗi phần tử bắt buộc có: "
        "title (string), variant (string), setup_turns (array), user_message (string), "
        "reference_answer (string). Có thể thêm expected (object). Không được chép các "
        "cụm mô tả kiểu 'tin nhắn kiểm thử' hoặc 'cách bot phải xử lý' làm dữ liệu thật.\n"
        f"Sản phẩm dùng cho batch này: "
        f"{json.dumps(catalog_sample, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Không lặp các câu này: "
        f"{json.dumps(existing_messages[-10:], ensure_ascii=False, separators=(',', ':'))}"
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
                response_body = decode_api_response(response.read())
            return extract_cases(response_body)
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")[:500]
            if error.code in {400, 401, 403, 404}:
                raise PermanentAPIError(
                    f"MISA API trả HTTP {error.code}; kiểm tra endpoint, model và API key. "
                    f"Chi tiết: {error_body}"
                ) from error
            last_error: Exception = RuntimeError(
                f"HTTP {error.code}: {error_body or error.reason}"
            )
        except (
            URLError,
            TimeoutError,
            http.client.HTTPException,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error

        if attempt == retries:
            raise RuntimeError(
                f"MISA API thất bại sau {retries} lần: {last_error}"
            ) from last_error
        delay = min(2**attempt, 30) + random.random()
        print(
            f"  Lần gọi {attempt}/{retries} lỗi: {last_error}; "
            f"thử lại sau {delay:.1f}s.",
            flush=True,
        )
        time.sleep(delay)

    raise RuntimeError("Không thể gọi MISA API.")


def validate_case(case: dict[str, Any]) -> bool:
    placeholders = {
        "tin nhan kiem thu",
        "cau hoi kiem thu",
        "cach bot phai xu ly",
        "test question",
        "expected answer",
    }
    normalized_message = normalize_text(str(case.get("user_message", "")))
    normalized_answer = normalize_text(str(case.get("reference_answer", "")))
    return (
        isinstance(case.get("title"), str)
        and isinstance(case.get("user_message"), str)
        and bool(normalized_message)
        and normalized_message not in placeholders
        and isinstance(case.get("reference_answer"), str)
        and bool(normalized_answer)
        and normalized_answer not in placeholders
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
    initial_cases: list[dict[str, Any]] | None = None,
    save_progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    cases = list(initial_cases or [])[:target_count]
    seen_messages = {
        normalized
        for case in cases
        if (normalized := normalize_text(str(case.get("user_message", ""))))
    }
    batch_number = 0
    stalled_batches = 0
    current_batch_size = batch_size

    while len(cases) < target_count:
        batch_number += 1
        requested = min(current_batch_size, target_count - len(cases))
        existing_messages = [str(case["user_message"]) for case in cases]
        prompt = build_prompt(
            requirement,
            catalog,
            requested,
            batch_number,
            existing_messages,
        )
        try:
            effective_temperature = min(
                1.0,
                max(temperature, 0.5) + stalled_batches * 0.1,
            )
            generated = call_misa(
                prompt,
                api_key=api_key,
                endpoint=endpoint,
                model=model,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                retries=retries,
            )
        except PermanentAPIError:
            if save_progress is not None:
                save_progress(cases)
            raise
        except RuntimeError as error:
            current_batch_size = max(1, current_batch_size // 2)
            stalled_batches += 1
            print(
                f"{requirement['id']}: batch {batch_number} lỗi ({error}). "
                f"Đã giữ {len(cases)} case; batch kế tiếp dùng "
                f"{current_batch_size} case.",
                flush=True,
            )
            if save_progress is not None:
                save_progress(cases)
            time.sleep(min(5 + stalled_batches, 30))
            continue

        added = 0
        invalid = 0
        duplicated = 0
        for raw_case in generated:
            if not validate_case(raw_case):
                invalid += 1
                continue
            normalized = normalize_text(str(raw_case["user_message"]))
            if not normalized or normalized in seen_messages:
                duplicated += 1
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
        if added > 0:
            current_batch_size = min(batch_size, current_batch_size + 1)
        else:
            current_batch_size = max(1, current_batch_size // 2)
        print(
            f"{requirement['id']}: {len(cases)}/{target_count} "
            f"(batch {batch_number}, thêm {added}, trùng {duplicated}, "
            f"không hợp lệ {invalid})",
            flush=True,
        )
        if save_progress is not None:
            save_progress(cases)
        if stalled_batches and stalled_batches % 10 == 0:
            print(
                f"{requirement['id']}: {stalled_batches} batch liên tiếp chưa thêm "
                "được case; tiếp tục với batch nhỏ và giữ nguyên checkpoint.",
                flush=True,
            )

    return cases


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as file:
        file.write(content)
        temporary_path = Path(file.name)
    temporary_path.replace(path)


def write_outputs(output_path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        output_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )

    jsonl_path = output_path.with_suffix(".jsonl")
    lines = [
        json.dumps(case, ensure_ascii=False)
        for requirement in payload["requirements"]
        for case in requirement["cases"]
    ]
    _atomic_write_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))


def load_existing_results(output_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not output_path.exists():
        return {}
    try:
        with output_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Không đọc được checkpoint {output_path}: {error}. Bắt đầu mới.", flush=True)
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("requirements", []):
        requirement_id = item.get("id") or item.get("requirement_id")
        cases = item.get("cases", [])
        if isinstance(requirement_id, str) and isinstance(cases, list):
            results[requirement_id] = [case for case in cases if validate_case(case)]
    return results


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
    results_by_id = (
        {} if args.no_resume else load_existing_results(args.output)
    )

    def save_checkpoint() -> dict[str, Any]:
        requirements_payload = []
        for configured_requirement in REQUIREMENTS:
            saved_cases = results_by_id.get(configured_requirement["id"], [])
            if saved_cases:
                requirements_payload.append(
                    {
                        **configured_requirement,
                        "count": len(saved_cases),
                        "cases": saved_cases,
                    }
                )
        payload = {
            "generator_model": model,
            "per_requirement": args.per_requirement,
            "total_cases": sum(item["count"] for item in requirements_payload),
            "complete": all(
                len(results_by_id.get(requirement["id"], []))
                >= args.per_requirement
                for requirement in REQUIREMENTS
            ),
            "requirements": requirements_payload,
        }
        write_outputs(args.output, payload)
        return payload

    for requirement in REQUIREMENTS:
        existing_cases = results_by_id.get(requirement["id"], [])
        if len(existing_cases) >= args.per_requirement:
            print(
                f"Bỏ qua {requirement['id']}: đã có đủ "
                f"{args.per_requirement} case trong checkpoint.",
                flush=True,
            )
            continue
        print(f"Bắt đầu {requirement['id']} - {requirement['name']}", flush=True)

        def save_requirement_progress(
            cases: list[dict[str, Any]],
            requirement_id: str = requirement["id"],
        ) -> None:
            results_by_id[requirement_id] = list(cases)
            save_checkpoint()

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
            initial_cases=existing_cases,
            save_progress=save_requirement_progress,
        )
        results_by_id[requirement["id"]] = cases
        save_checkpoint()

    payload = save_checkpoint()
    print(f"Đã ghi {payload['total_cases']} case vào {args.output}", flush=True)
    print(f"JSONL: {args.output.with_suffix('.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
