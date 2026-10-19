from mai_agent.vision.models import SkinObservation
from mai_agent.vision.safety import classify_skin_scope


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


def test_unrelated_image_is_rejected_before_skin_analysis() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(
                image_subject="unrelated",
                face_count=0,
                facial_skin_visible=False,
                relevance_confidence=0.99,
                relevance_reason="Ảnh phong cảnh.",
            ),
            minimum_confidence=0.6,
            minimum_relevance_confidence=0.85,
        )
        == "irrelevant_image"
    )


def test_multiple_faces_require_a_single_face_image() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(face_count=2),
            minimum_confidence=0.6,
            minimum_relevance_confidence=0.85,
        )
        == "need_single_face"
    )


def test_product_image_is_routed_out_of_skin_analysis() -> None:
    assert (
        classify_skin_scope(
            "Sản phẩm này dùng như nào?",
            _observation(
                image_subject="cosmetic_product",
                face_count=0,
                facial_skin_visible=False,
            ),
            minimum_confidence=0.6,
        )
        == "product_image"
    )


def test_other_skin_region_is_outside_current_image_scope() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(
                image_subject="other_skin",
                face_count=0,
                facial_skin_visible=False,
            ),
            minimum_confidence=0.6,
        )
        == "unsupported_skin_region"
    )


def test_verified_widespread_rash_uses_medical_review_care_mode() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(
                widespread_rash=True,
                widespread_rash_confidence=0.92,
                widespread_rash_reason="Nhiều tổn thương riêng biệt ở trán và hai má.",
            ),
            minimum_confidence=0.6,
            minimum_red_flag_confidence=0.85,
        )
        == "medical_review"
    )


def test_unverified_widespread_rash_requests_better_evidence() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(
                widespread_rash=True,
                widespread_rash_confidence=0.72,
                widespread_rash_reason="Không đủ rõ để xác nhận.",
            ),
            minimum_confidence=0.6,
            minimum_red_flag_confidence=0.85,
        )
        == "need_more_information"
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


def test_dark_image_needs_more_information() -> None:
    assert (
        classify_skin_scope(
            "Da này hợp sản phẩm nào?",
            _observation(lighting_quality="dark"),
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
