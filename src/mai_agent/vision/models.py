from typing import Literal

from pydantic import BaseModel, Field


class SkinObservation(BaseModel):
    """Quan sát bề mặt từ ảnh; cố ý không có trường chẩn đoán bệnh."""

    image_subject: Literal[
        "facial_skin",
        "cosmetic_product",
        "other_skin",
        "unrelated",
        "unclear",
    ]
    face_count: int = Field(ge=0)
    facial_skin_visible: bool
    relevance_confidence: float = Field(ge=0, le=1)
    relevance_reason: str
    image_quality: Literal["good", "poor", "unusable"]
    lighting_quality: Literal["good", "dark", "overexposed"]
    filter_suspected: bool = False
    visible_shine: Literal["none", "t_zone", "widespread", "unclear"] = "unclear"
    visible_dryness: Literal["none", "mild", "noticeable", "unclear"] = "unclear"
    visible_redness: Literal["none", "mild", "noticeable", "unclear"] = "unclear"
    visible_spots: Literal["none", "few", "many", "unclear"] = "unclear"
    eye_or_lip_swelling: bool = False
    open_wound: bool = False
    blistering: bool = False
    pus_or_crusting: bool = False
    widespread_rash: bool = False
    widespread_rash_confidence: float = Field(default=0, ge=0, le=1)
    widespread_rash_reason: str = ""
    confidence: float = Field(ge=0, le=1)


class WidespreadRashVerification(BaseModel):
    """Kết quả xác minh lần hai cho cờ dễ nhầm với đỏ hoặc khô da."""

    confirmed: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class SkinConsultationResult(BaseModel):
    status: Literal[
        "recommended",
        "recommended_with_medical_review",
        "need_confirmation",
        "need_better_image",
        "medical_out_of_scope",
        "medical_red_flag",
        "product_image",
        "unsupported_skin_region",
        "irrelevant_image",
        "need_single_face",
        "insufficient_knowledge",
        "no_matching_product",
    ]
    reply: str
    observation: SkinObservation | None = None
    product_ids: list[str] = Field(default_factory=list)
