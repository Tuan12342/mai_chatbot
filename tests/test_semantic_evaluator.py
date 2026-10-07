import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_misa_scenarios.py"
SPEC = importlib.util.spec_from_file_location("evaluate_misa_scenarios", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_load_scenarios_filters_requirements_and_limit(tmp_path) -> None:
    input_path = tmp_path / "scenarios.json"
    input_path.write_text(
        json.dumps(
            {
                "requirements": [
                    {
                        "id": "R01",
                        "name": "One",
                        "description": "First",
                        "cases": [{"id": "R01-001"}, {"id": "R01-002"}],
                    },
                    {
                        "id": "R02",
                        "name": "Two",
                        "description": "Second",
                        "cases": [{"id": "R02-001"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = evaluator.load_scenarios(input_path, {"R01"}, 1)

    assert [case["id"] for case in cases] == ["R01-001"]
    assert cases[0]["requirement_description"] == "First"


def test_build_report_excludes_invalid_from_pass_rate() -> None:
    cases = [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}]
    results = {
        "A": {"case_id": "A", "verdict": "pass"},
        "B": {"case_id": "B", "verdict": "fail"},
        "C": {"case_id": "C", "verdict": "invalid_test"},
    }

    report = evaluator.build_report(cases, results)

    assert report["completed"] == 3
    assert report["remaining"] == 1
    assert report["semantic_pass_rate"] == 50.0


def test_transient_api_error_detection() -> None:
    assert evaluator.is_transient_api_error(
        RuntimeError("GoogleAPIError: 500 INTERNAL")
    )
    assert evaluator.is_transient_api_error(
        RuntimeError("429 RESOURCE_EXHAUSTED")
    )
    assert not evaluator.is_transient_api_error(NameError("selected_product"))


def test_run_case_retries_transient_error(monkeypatch) -> None:
    attempts = 0

    def fake_run_case(_case):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("500 INTERNAL")
        return [], {"reply": "ok"}

    monkeypatch.setattr(evaluator, "run_case", fake_run_case)
    monkeypatch.setattr(evaluator.time, "sleep", lambda _seconds: None)

    transcript, state = evaluator.run_case_with_retries(
        {"id": "R01-001"},
        retries=3,
    )

    assert attempts == 2
    assert transcript == []
    assert state == {"reply": "ok"}
