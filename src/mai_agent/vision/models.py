from typing import Literal

from pydantic import BaseModel, Field


class SkinObservation(BaseModel):
    """Quan sát bề mặt từ ảnh; cố ý không có trường chẩn đoán bệnh."""

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
    confidence: float = Field(ge=0, le=1)


class SkinResearch(BaseModel):
    """Kết quả nghiên cứu nội bộ; nguồn không được chuyển sang câu trả lời khách."""

    summary: str
    source_domains: list[str] = Field(default_factory=list)


class SkinConsultationResult(BaseModel):
    status: Literal[
        "recommended",
        "need_confirmation",
        "need_better_image",
        "medical_out_of_scope",
        "medical_red_flag",
        "research_unavailable",
        "no_matching_product",
    ]
    reply: str
    observation: SkinObservation | None = None
    product_ids: list[str] = Field(default_factory=list)
