import re
import unicodedata
from typing import Literal

from mai_agent.vision.models import SkinObservation

SkinScope = Literal[
    "cosmetic_care",
    "need_more_information",
    "need_single_face",
    "product_image",
    "unsupported_skin_region",
    "irrelevant_image",
    "medical_out_of_scope",
    "medical_review",
    "medical_red_flag",
]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )


def asks_for_medical_advice(message: str) -> bool:
    text = _normalize(message)
    return bool(
        re.search(
            r"\b(chan doan|benh gi|co phai.*benh|viem da gi|ke.*thuoc|thuoc boi|"
            r"dieu tri|chua khoi|diagnos|what disease|is this (eczema|rosacea|psoriasis)|"
            r"prescription|medicine|medical treatment|cure)\b",
            text,
        )
    )


def has_medical_red_flag(
    observation: SkinObservation,
    *,
    minimum_red_flag_confidence: float = 0.85,
) -> bool:
    hard_red_flag = any(
        (
            observation.eye_or_lip_swelling,
            observation.open_wound,
            observation.blistering,
            observation.pus_or_crusting,
        )
    )
    return hard_red_flag


def needs_medical_review(
    observation: SkinObservation,
    *,
    minimum_red_flag_confidence: float = 0.85,
) -> bool:
    return (
        observation.widespread_rash
        and observation.widespread_rash_confidence >= minimum_red_flag_confidence
        and bool(observation.widespread_rash_reason.strip())
    )


def classify_skin_scope(
    message: str,
    observation: SkinObservation,
    *,
    minimum_confidence: float,
    minimum_relevance_confidence: float = 0.85,
    minimum_red_flag_confidence: float = 0.85,
) -> SkinScope:
    if asks_for_medical_advice(message):
        return "medical_out_of_scope"
    if observation.image_subject == "cosmetic_product":
        return "product_image"
    if observation.image_subject == "other_skin":
        return "unsupported_skin_region"
    if observation.image_subject == "unrelated":
        return "irrelevant_image"
    if observation.face_count > 1:
        return "need_single_face"
    if (
        observation.image_subject != "facial_skin"
        or observation.face_count != 1
        or not observation.facial_skin_visible
        or observation.relevance_confidence < minimum_relevance_confidence
    ):
        return "need_more_information"
    if has_medical_red_flag(
        observation,
        minimum_red_flag_confidence=minimum_red_flag_confidence,
    ):
        return "medical_red_flag"
    if needs_medical_review(
        observation,
        minimum_red_flag_confidence=minimum_red_flag_confidence,
    ):
        return "medical_review"
    if observation.widespread_rash:
        # Cờ chưa được xác minh đủ mạnh: không tư vấn sản phẩm nhưng cũng không đưa
        # cảnh báo khẩn cấp như một kết luận chắc chắn.
        return "need_more_information"
    if (
        observation.image_quality != "good"
        or observation.lighting_quality != "good"
        or observation.filter_suspected
        or observation.confidence < minimum_confidence
    ):
        return "need_more_information"
    return "cosmetic_care"
