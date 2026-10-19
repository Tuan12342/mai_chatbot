import pytest

from mai_agent.vision import prompt as skin_prompt
from mai_agent.vision.prompt import SKIN_CONSULTATION_SYSTEM_PROMPT


def test_skin_prompt_locks_product_selection_and_evidence() -> None:
    prompt = SKIN_CONSULTATION_SYSTEM_PROMPT

    assert "không được tự chọn" in prompt
    assert "product_id` trùng chính xác `id" in prompt
    assert "Không thêm sản phẩm ngoài `selected_catalog_products`" in prompt
    assert "không có knowledge tương ứng" in prompt
    assert "routine_order_product_ids` là thứ tự" in prompt
    assert "không tự đổi thứ tự" in prompt
    assert "sku_validation_owner" in prompt
    assert "match_evidence.status" not in prompt


def test_skin_prompt_defines_refusal_and_response_structure() -> None:
    prompt = SKIN_CONSULTATION_SYSTEM_PROMPT

    assert "ĐIỀU KIỆN TỪ CHỐI" in prompt
    assert "match_decision.status" not in prompt
    assert "CẤU TRÚC CÂU TRẢ LỜI KHI MATCH" in prompt
    assert "không phải chẩn đoán" in prompt
    assert "KIỂM TRA THẦM TRƯỚC KHI TRẢ LỜI" in prompt
    assert "Chỉ xuất plain text" in prompt
    assert "ĐIỀU CHỈNH RIÊNG KHI care_mode=medical_review" in prompt
    assert "Chăm sóc hỗ trợ tạm thời" in prompt
    assert "nên sắp xếp gặp bác sĩ da liễu" in prompt
    assert "`noticeable` thành “nhẹ”" in prompt


def test_model_selection_accepts_only_candidate_skus(monkeypatch) -> None:
    monkeypatch.setattr(
        skin_prompt,
        "invoke_tool",
        lambda *_args, **_kwargs: {
            "name": "choose_skin_products",
            "args": {"product_ids": ["oa002"], "selection_reason": "Phù hợp ảnh."},
        },
    )

    selected, reason = skin_prompt.select_skin_products(
        {"candidate_products": [{"id": "OA002"}]},
        max_products=2,
    )

    assert selected == ["OA002"]
    assert reason == "Phù hợp ảnh."


def test_model_selection_rejects_hallucinated_sku(monkeypatch) -> None:
    monkeypatch.setattr(
        skin_prompt,
        "invoke_tool",
        lambda *_args, **_kwargs: {
            "name": "choose_skin_products",
            "args": {"product_ids": ["OA999"], "selection_reason": ""},
        },
    )

    with pytest.raises(RuntimeError, match="ngoài danh sách ứng viên"):
        skin_prompt.select_skin_products(
            {"candidate_products": [{"id": "OA002"}]},
            max_products=2,
        )
