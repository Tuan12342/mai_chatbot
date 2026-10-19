import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

from mai_agent.catalog import load_products
from mai_agent.vision.knowledge_base import product_supports_care_mode


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )


def extract_skin_type_from_message(message: str) -> str | None:
    text = _normalize(message)
    aliases = {
        "da dầu": ("da dau", "oily skin"),
        "da khô": ("da kho", "dry skin"),
        "da hỗn hợp": ("da hon hop", "combination skin"),
        "da nhạy cảm": ("da nhay cam", "sensitive skin"),
        "da thường": ("da thuong", "normal skin"),
    }
    for skin_type, candidates in aliases.items():
        if any(re.search(rf"\b{re.escape(candidate)}\b", text) for candidate in candidates):
            return skin_type
    return None


def extract_confirmed_skin_type(message: str, customer: dict[str, Any]) -> str | None:
    aliases = {
        "da dầu",
        "da khô",
        "da hỗn hợp",
        "da nhạy cảm",
        "da thường",
    }
    if skin_type := extract_skin_type_from_message(message):
        return skin_type
    stored = customer.get("skin_type")
    if stored not in aliases:
        return None
    try:
        last_updated = datetime.fromisoformat(str(customer["last_updated"]))
        now = datetime.now(last_updated.tzinfo) if last_updated.tzinfo else datetime.now()
        if now - last_updated > timedelta(days=90):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return stored


def eligible_catalog_products(
    customer: dict[str, Any],
    skin_type: str | None,
    *,
    care_mode: str = "standard",
) -> list[dict[str, Any]]:
    """Lọc điều kiện nghiệp vụ bắt buộc; không chấm điểm hay tự chọn sản phẩm."""
    excluded = {
        value.casefold()
        for field in ("allergies", "sensitivities", "excluded_ingredients")
        for value in customer.get(field, [])
        if isinstance(value, str) and value.strip()
    }
    disliked = {value.upper() for value in customer.get("disliked_product_ids", [])}
    eligible: list[dict[str, Any]] = []

    for product in load_products():
        if product.get("stock", 0) <= 0 or product["id"].upper() in disliked:
            continue
        if not product_supports_care_mode(product["id"], care_mode):
            continue
        supported = {value.casefold() for value in product.get("skin_types", [])}
        normalized_skin_type = skin_type.casefold() if skin_type else ""
        if (
            normalized_skin_type
            and normalized_skin_type not in supported
            and "mọi loại da" not in supported
        ):
            continue
        ingredients = [value.casefold() for value in product.get("ingredients", [])]
        if any(blocked in ingredient for blocked in excluded for ingredient in ingredients):
            continue
        eligible.append(product)

    return eligible
