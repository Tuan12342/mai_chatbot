from unittest.mock import patch

import pytest

from mai_agent.config import Settings
from mai_agent.vision.models import SkinObservation
from mai_agent.vision.recommendation import extract_confirmed_skin_type
from mai_agent.vision.service import SkinConsultationService


def _observation(**updates) -> SkinObservation:
    values = {
        "image_subject": "facial_skin",
        "face_count": 1,
        "facial_skin_visible": True,
        "relevance_confidence": 0.95,
        "relevance_reason": "Một khuôn mặt nhìn rõ.",
        "image_quality": "good",
        "lighting_quality": "good",
        "visible_shine": "t_zone",
        "visible_dryness": "none",
        "visible_redness": "mild",
        "visible_spots": "few",
        "confidence": 0.9,
    }
    values.update(updates)
    return SkinObservation.model_validate(values)


class FakeProvider:
    def __init__(self, observation: SkinObservation | None = None) -> None:
        self.observation = observation or _observation()
        self.analyze_calls = 0
        self.research_calls = 0

    def analyze_image(self, image_bytes: bytes, mime_type: str) -> SkinObservation:
        self.analyze_calls += 1
        return self.observation

    def research_skin_care(self, observation: SkinObservation, skin_type: str):
        self.research_calls += 1
        raise AssertionError("Luồng tư vấn ảnh không được gọi web search.")


class FakeObservationContext:
    def __init__(self, observation) -> None:
        self.observation = observation

    def __enter__(self):
        return self.observation

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class FakeLangfuseObservation:
    def __init__(self, attributes: dict) -> None:
        self.attributes = attributes
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations: list[FakeLangfuseObservation] = []

    def start_as_current_observation(self, **kwargs):
        observation = FakeLangfuseObservation(kwargs)
        self.observations.append(observation)
        return FakeObservationContext(observation)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="test-key",
        skin_vision_min_confidence=0.6,
    )


@pytest.fixture(autouse=True)
def _stub_product_selection(monkeypatch):
    def select(context, *, max_products):
        product_ids = [
            product["id"]
            for product in context["candidate_products"][:max_products]
        ]
        return product_ids, "Lựa chọn kiểm thử từ danh sách ứng viên hợp lệ."

    monkeypatch.setattr("mai_agent.vision.service.select_skin_products", select)


def test_medical_text_is_blocked_before_image_is_sent_to_vlm() -> None:
    provider = FakeProvider()
    service = SkinConsultationService(settings=_settings(), provider=provider)

    result = service.consult(
        image_bytes=b"image",
        mime_type="image/jpeg",
        message="Chẩn đoán giúp tôi đây là bệnh gì",
        customer={"skin_type": "da dầu"},
    )

    assert result.status == "medical_out_of_scope"
    assert provider.analyze_calls == 0
    assert provider.research_calls == 0


def test_red_flag_never_calls_web_search() -> None:
    provider = FakeProvider(_observation(open_wound=True))
    service = SkinConsultationService(settings=_settings(), provider=provider)

    result = service.consult(
        image_bytes=b"image",
        mime_type="image/jpeg",
        message="Da này phù hợp sản phẩm nào?",
        customer={"skin_type": "da dầu"},
    )

    assert result.status == "medical_red_flag"
    assert provider.analyze_calls == 1
    assert provider.research_calls == 0


def test_medical_review_combines_supportive_products_with_doctor_advice() -> None:
    provider = FakeProvider(
        _observation(
            visible_shine="none",
            visible_dryness="noticeable",
            visible_redness="noticeable",
            visible_spots="none",
            widespread_rash=True,
            widespread_rash_confidence=0.92,
            widespread_rash_reason="Nhiều dấu hiệu riêng biệt ở trán và hai má.",
        )
    )
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch(
        "mai_agent.vision.service.generate_skin_reply",
        return_value=(
            "Chăm sóc hỗ trợ với OA003 và OA006; bạn nên sắp xếp khám bác sĩ da liễu."
        ),
    ) as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={},
        )

    assert result.status == "recommended_with_medical_review"
    assert result.product_ids == ["OA001", "OA003"]
    context = generate.call_args.args[0]
    assert context["care_mode"] == "medical_review"
    assert context["medical_review_context"]["doctor_review_recommended"] is True
    assert context["routine_order_product_ids"] == ["OA001", "OA003"]
    assert all(
        "medical_review" in item["allowed_care_modes"]
        for item in context["retrieved_product_knowledge"]
    )


def test_hard_red_flag_still_stops_before_product_reply_model() -> None:
    provider = FakeProvider(_observation(blistering=True))
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch("mai_agent.vision.service.generate_skin_reply") as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={},
        )

    assert result.status == "medical_red_flag"
    assert result.product_ids == []
    generate.assert_not_called()


def test_unrelated_image_stops_before_product_matching_and_reply_model() -> None:
    provider = FakeProvider(
        _observation(
            image_subject="unrelated",
            face_count=0,
            facial_skin_visible=False,
            relevance_confidence=0.99,
            relevance_reason="Ảnh đồ vật, không có khuôn mặt.",
            visible_shine="unclear",
            visible_dryness="unclear",
            visible_redness="unclear",
            visible_spots="unclear",
            confidence=0,
        )
    )
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch("mai_agent.vision.service.generate_skin_reply") as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={},
        )

    assert result.status == "irrelevant_image"
    assert "không cho thấy rõ vùng da mặt" in result.reply
    assert result.product_ids == []
    generate.assert_not_called()


def test_multiple_faces_stop_before_product_reply_model() -> None:
    provider = FakeProvider(_observation(face_count=2))
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch("mai_agent.vision.service.generate_skin_reply") as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={},
        )

    assert result.status == "need_single_face"
    assert "nhiều hơn một khuôn mặt" in result.reply
    generate.assert_not_called()


def test_product_image_redirects_to_product_question_without_skin_advice() -> None:
    provider = FakeProvider(
        _observation(
            image_subject="cosmetic_product",
            face_count=0,
            facial_skin_visible=False,
            relevance_reason="Ảnh chai mỹ phẩm.",
            confidence=0,
        )
    )
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch("mai_agent.vision.service.generate_skin_reply") as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Sản phẩm này dùng như thế nào?",
            customer={},
        )

    assert result.status == "product_image"
    assert "tên hoặc mã SKU" in result.reply
    assert result.product_ids == []
    generate.assert_not_called()


def test_image_without_skin_type_uses_local_knowledge_and_visible_features() -> None:
    provider = FakeProvider()
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch(
        "mai_agent.vision.service.generate_skin_reply",
        return_value="Gợi ý theo ảnh.",
    ):
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={"skin_type": "da khô", "allergies": ["niacinamide"]},
        )

    assert result.status == "recommended"
    assert result.reply == "Gợi ý theo ảnh."
    assert provider.research_calls == 0
    assert result.product_ids


def test_stale_profile_is_not_treated_as_current_confirmation() -> None:
    assert (
        extract_confirmed_skin_type(
            "Da này phù hợp sản phẩm nào?",
            {
                "skin_type": "da dầu",
                "last_updated": "2025-01-01T00:00:00+07:00",
            },
        )
        is None
    )


def test_no_visible_signal_is_decided_by_product_selection_model() -> None:
    provider = FakeProvider(
        _observation(
            visible_shine="none",
            visible_dryness="none",
            visible_redness="none",
            visible_spots="none",
        )
    )
    service = SkinConsultationService(settings=_settings(), provider=provider)

    with patch(
        "mai_agent.vision.service.generate_skin_reply",
        return_value="Routine cơ bản từ kho tri thức.",
    ) as generate:
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da này phù hợp sản phẩm nào?",
            customer={},
        )

    assert result.status == "recommended"
    assert result.product_ids
    generate.assert_called_once()


def test_safe_result_uses_local_knowledge_without_exposing_external_sources() -> None:
    provider = FakeProvider()
    service = SkinConsultationService(settings=_settings(), provider=provider)
    customer = {
        "skin_type": "da dầu",
        "allergies": [],
        "sensitivities": [],
        "excluded_ingredients": [],
        "disliked_product_ids": [],
    }

    with patch("mai_agent.vision.service.generate_skin_reply") as generate:
        generate.return_value = "Gợi ý đã được kiểm tra.\nNguồn: https://aad.org/example"
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da dầu này phù hợp sản phẩm nào?",
            customer=customer,
        )

    assert result.status == "recommended"
    assert result.reply == "Gợi ý đã được kiểm tra."
    assert result.product_ids == ["OA002", "OA003", "OA005", "OA007"]
    assert provider.research_calls == 0
    business_context = generate.call_args.args[0]
    assert "match_decision" not in business_context
    assert "verified_general_care_guidance" not in business_context
    assert business_context["routine_order_product_ids"] == ["OA002", "OA003", "OA005", "OA007"]
    assert business_context["retrieved_product_knowledge"][0]["product_id"] == "OA002"
    assert all(
        "match_evidence" not in item
        for item in business_context["retrieved_product_knowledge"]
    )


def test_safe_result_creates_detailed_langfuse_observations() -> None:
    provider = FakeProvider()
    langfuse = FakeLangfuse()
    service = SkinConsultationService(
        settings=_settings(),
        provider=provider,
        langfuse_client=langfuse,
    )
    customer = {
        "skin_type": "da dầu",
        "allergies": [],
        "sensitivities": [],
        "excluded_ingredients": [],
        "disliked_product_ids": [],
    }

    with patch(
        "mai_agent.vision.service.generate_skin_reply",
        return_value="Gợi ý an toàn.",
    ):
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da dầu này phù hợp sản phẩm nào?",
            customer=customer,
        )

    assert result.status == "recommended"
    assert [item.attributes["name"] for item in langfuse.observations] == [
        "medical-safety",
        "google-vlm",
        "image-safety-classification",
        "local-product-knowledge",
        "misa-product-selection",
        "misa-final-response",
    ]
    assert langfuse.observations[1].attributes["input"] == {
        "mime_type": "image/jpeg",
        "size_bytes": 5,
    }
