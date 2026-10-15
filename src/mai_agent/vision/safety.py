import re
import unicodedata
from typing import Literal

from mai_agent.vision.models import SkinObservation

SkinScope = Literal[
    "cosmetic_care",
    "need_more_information",
    "medical_out_of_scope",
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


def has_medical_red_flag(observation: SkinObservation) -> bool:
    return any(
        (
            observation.eye_or_lip_swelling,
            observation.open_wound,
            observation.blistering,
            observation.pus_or_crusting,
            observation.widespread_rash,
        )
    )


def classify_skin_scope(
    message: str,
    observation: SkinObservation,
    *,
    minimum_confidence: float,
) -> SkinScope:
    if asks_for_medical_advice(message):
        return "medical_out_of_scope"
    if has_medical_red_flag(observation):
        return "medical_red_flag"
    if (
        observation.image_quality != "good"
        or observation.filter_suspected
        or observation.confidence < minimum_confidence
    ):
        return "need_more_information"
    return "cosmetic_care"
