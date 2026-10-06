"""Gọi MISA LLM Gateway từ terminal và in câu trả lời của model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENDPOINT = "https://ai.misa.vn/nlp/llm-gateway/v1/chat/completions"
DEFAULT_MODEL = "misa-ai-1.1-plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Xin chào!",
        help="Nội dung gửi tới model.",
    )
    parser.add_argument(
        "--system",
        default="Bạn là một trợ lý hữu ích.",
        help="System prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="In toàn bộ JSON response thay vì chỉ in câu trả lời.",
    )
    return parser.parse_args()


def call_misa(
    prompt: str,
    *,
    system_prompt: str,
    api_key: str,
    endpoint: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MISA API trả HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Không thể kết nối MISA API: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("MISA API không trả JSON hợp lệ.") from error


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Response không đúng schema Chat Completions.") from error
    return str(content)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("MISA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu MISA_API_KEY. Hãy chạy: "
            "read -rsp 'Nhập MISA API key: ' MISA_API_KEY; echo; export MISA_API_KEY"
        )

    response = call_misa(
        args.prompt,
        system_prompt=args.system,
        api_key=api_key,
        endpoint=os.getenv("MISA_LLM_ENDPOINT", DEFAULT_ENDPOINT),
        model=os.getenv("MISA_LLM_MODEL", DEFAULT_MODEL),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.raw:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(response_text(response))


if __name__ == "__main__":
    main()
