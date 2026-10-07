import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_misa_test_scenarios.py"
)
SPEC = importlib.util.spec_from_file_location("generate_misa_test_scenarios", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_extract_cases_accepts_fenced_json() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"cases": [{"title": "T", '
                    '"user_message": "U", "reference_answer": "A"}]}\n```'
                }
            }
        ]
    }

    case = generator.extract_cases(response)[0]
    assert case["title"] == "T"
    assert case["user_message"] == "U"


def test_decode_api_response_accepts_sse_chunks() -> None:
    events = [
        {"choices": [{"delta": {"content": '{"cases": ['}}]},
        {
            "choices": [
                {
                    "delta": {
                        "content": (
                            '{"title": "T", "user_message": "U", '
                            '"reference_answer": "A"}'
                        )
                    }
                }
            ]
        },
        {"choices": [{"delta": {"content": "]}"}}]},
    ]
    body = "\n".join(f"data: {json.dumps(event)}" for event in events)
    body += "\ndata: [DONE]\n"

    decoded = generator.decode_api_response(body.encode())

    case = generator.extract_cases(decoded)[0]
    assert case["title"] == "T"
    assert case["user_message"] == "U"


def test_extract_cases_accepts_test_cases_and_question_answer_keys() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "test_cases": [
                                {
                                    "question": "Da dầu nên dùng sản phẩm nào?",
                                    "answer": "Tư vấn sản phẩm phù hợp da dầu.",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    case = generator.extract_cases(response)[0]

    assert case["user_message"] == "Da dầu nên dùng sản phẩm nào?"
    assert case["reference_answer"] == "Tư vấn sản phẩm phù hợp da dầu."


def test_extract_cases_accepts_json_array() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        [
                            {
                                "user_message": "OA002 còn hàng không?",
                                "reference_answer": "Kiểm tra và trả đúng tồn kho.",
                            }
                        ],
                        ensure_ascii=False,
                    )
                }
            }
        ]
    }

    assert generator.extract_cases(response)[0]["user_message"] == "OA002 còn hàng không?"


def test_extract_cases_accepts_plain_question_answer() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Câu hỏi: Tôi muốn mua hai chai OA002\n"
                        "Trả lời: Kiểm tra tồn kho cho đúng số lượng 2."
                    )
                }
            }
        ]
    }

    case = generator.extract_cases(response)[0]

    assert case["user_message"] == "Tôi muốn mua hai chai OA002"
    assert case["reference_answer"] == "Kiểm tra tồn kho cho đúng số lượng 2."


def test_checkpoint_can_be_loaded_after_atomic_write(tmp_path) -> None:
    output = tmp_path / "cases.json"
    case = {
        "id": "R01-001",
        "title": "Test",
        "setup_turns": [],
        "user_message": "Sản phẩm này giá bao nhiêu?",
        "reference_answer": "Trả lời đúng giá từ catalog.",
        "expected": {},
    }
    payload = {
        "requirements": [{"id": "R01", "count": 1, "cases": [case]}]
    }

    generator.write_outputs(output, payload)

    assert generator.load_existing_results(output) == {"R01": [case]}
    assert output.with_suffix(".jsonl").exists()


def test_prompt_uses_small_rotating_catalog_sample() -> None:
    catalog = [
        {"id": f"OA{index:03d}", "name": f"Product {index}", "stock": index}
        for index in range(1, 11)
    ]

    first = generator.compact_catalog_for_batch("R04", catalog, 1)
    second = generator.compact_catalog_for_batch("R04", catalog, 2)
    prompt = generator.build_prompt(generator.REQUIREMENTS[3], catalog, 1, 1, [])

    assert len(first) == 3
    assert first != second
    assert "OA001" in prompt
    assert "OA010" not in prompt
    assert len(prompt) < 2_000


def test_transient_batch_error_keeps_progress_and_continues(monkeypatch) -> None:
    calls = 0
    checkpoints: list[int] = []

    def fake_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary error")
        return [
            {
                "title": "Valid case",
                "setup_turns": [],
                "user_message": "Cho tôi xem OA002",
                "reference_answer": "Cung cấp đúng thông tin OA002.",
                "expected": {},
            }
        ]

    monkeypatch.setattr(generator, "call_misa", fake_call)
    monkeypatch.setattr(generator.time, "sleep", lambda _seconds: None)

    cases = generator.generate_requirement_cases(
        generator.REQUIREMENTS[0],
        [],
        target_count=1,
        batch_size=5,
        api_key="test-key",
        endpoint="https://example.invalid",
        model="test-model",
        temperature=0,
        max_tokens=100,
        retries=1,
        save_progress=lambda saved: checkpoints.append(len(saved)),
    )

    assert len(cases) == 1
    assert calls == 2
    assert checkpoints == [0, 1]
