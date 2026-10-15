from mai_agent.vision.models import SkinObservation
from mai_agent.vision.safety import classify_skin_scope


def _observation(**updates) -> SkinObservation:
    values = {
        "image_quality": "good",
        "lighting_quality": "good",
        "visible_shine": "t_zone",
        "visible_dryness": "none",
        "visible_redness": "mild",
        "visible_spots": "few",
        "confidence": 0.85,
    }
    values.update(updates)
    return SkinObservation.model_validate(values)


def test_diagnosis_request_is_out_of_scope() -> None:
    assert (
        classify_skin_scope(
            "Nhìn ảnh giúp tôi chẩn đoán đây là bệnh gì",
            _observation(),
            minimum_confidence=0.6,
        )
        == "medical_out_of_scope"
    )

def test_visible_red_flag_blocks_cosmetic_recommendation() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(eye_or_lip_swelling=True),
            minimum_confidence=0.6,
        )
        == "medical_red_flag"
    )


def test_filtered_or_low_confidence_image_needs_more_information() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(filter_suspected=True, confidence=0.55),
            minimum_confidence=0.6,
        )
        == "need_more_information"
    )


def test_safe_cosmetic_question_can_continue() -> None:
    assert (
        classify_skin_scope(
            "Da hỗn hợp này hợp sản phẩm nào?",
            _observation(),
            minimum_confidence=0.6,
        )
        == "cosmetic_care"
    )
