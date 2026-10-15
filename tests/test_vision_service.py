from unittest.mock import patch

from mai_agent.config import Settings
from mai_agent.vision.models import SkinObservation, SkinResearch
from mai_agent.vision.recommendation import extract_confirmed_skin_type
from mai_agent.vision.service import SkinConsultationService


def _observation(**updates) -> SkinObservation:
    values = {
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

    def research_skin_care(
        self,
        observation: SkinObservation,
        skin_type: str,
    ) -> SkinResearch:
        self.research_calls += 1
        return SkinResearch(
            summary="Ưu tiên làm sạch dịu nhẹ và chống nắng hằng ngày.",
            source_domains=["aad.org"],
        )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        google_api_key="test-key",
        skin_vision_min_confidence=0.6,
    )


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


def test_missing_confirmed_skin_type_asks_questions_before_search() -> None:
    provider = FakeProvider()
    service = SkinConsultationService(settings=_settings(), provider=provider)

    result = service.consult(
        image_bytes=b"image",
        mime_type="image/jpeg",
        message="Da này phù hợp sản phẩm nào?",
        customer={},
    )

    assert result.status == "need_confirmation"
    assert provider.research_calls == 0


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


def test_safe_result_searches_then_uses_catalog_without_exposing_sources() -> None:
    provider = FakeProvider()
    service = SkinConsultationService(settings=_settings(), provider=provider)
    customer = {
        "skin_type": "da dầu",
        "allergies": [],
        "sensitivities": [],
        "excluded_ingredients": [],
        "disliked_product_ids": [],
    }

    with patch("mai_agent.vision.service.generate_reply") as generate:
        generate.return_value = "Gợi ý đã được kiểm tra.\nNguồn: https://aad.org/example"
        result = service.consult(
            image_bytes=b"image",
            mime_type="image/jpeg",
            message="Da dầu này phù hợp sản phẩm nào?",
            customer=customer,
        )

    assert result.status == "recommended"
    assert result.reply == "Gợi ý đã được kiểm tra."
    assert result.product_ids == ["OA002", "OA007"]
    assert provider.research_calls == 1
    business_context = generate.call_args.args[0]
    assert "aad.org" not in business_context
    assert "http" not in business_context
    assert "OA002" in business_context
