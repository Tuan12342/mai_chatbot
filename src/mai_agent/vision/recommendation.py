import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

from mai_agent.catalog import load_products
from mai_agent.vision.models import SkinObservation


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )


def extract_confirmed_skin_type(message: str, customer: dict[str, Any]) -> str | None:
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


def _category_priority(observation: SkinObservation) -> list[str]:
    if observation.visible_dryness == "noticeable" or observation.visible_redness == "noticeable":
        return ["Kem dưỡng", "Sữa rửa mặt", "Kem chống nắng", "Serum"]
    if observation.visible_shine in {"t_zone", "widespread"} or observation.visible_spots in {
        "few",
        "many",
    }:
        return ["Sữa rửa mặt", "Kem dưỡng", "Kem chống nắng", "Serum"]
    return ["Sữa rửa mặt", "Kem dưỡng", "Kem chống nắng", "Toner"]


def match_catalog_products(
    observation: SkinObservation,
    customer: dict[str, Any],
    skin_type: str,
    *,
    limit: int = 2,
) -> list[dict[str, Any]]:
    excluded = {
        value.casefold()
        for field in ("allergies", "sensitivities", "excluded_ingredients")
        for value in customer.get(field, [])
        if isinstance(value, str) and value.strip()
    }
    disliked = {value.upper() for value in customer.get("disliked_product_ids", [])}
    priorities = _category_priority(observation)
    ranked: list[tuple[int, dict[str, Any]]] = []

    for product in load_products():
        if product.get("stock", 0) <= 0 or product["id"].upper() in disliked:
            continue
        supported = {value.casefold() for value in product.get("skin_types", [])}
        if skin_type.casefold() not in supported and "mọi loại da" not in supported:
            continue
        ingredients = [value.casefold() for value in product.get("ingredients", [])]
        if any(blocked in ingredient for blocked in excluded for ingredient in ingredients):
            continue
        try:
            category_score = priorities.index(product["category"])
        except ValueError:
            category_score = len(priorities) + 1
        exact_skin_bonus = 0 if skin_type.casefold() in supported else 1
        ranked.append((category_score * 10 + exact_skin_bonus, product))

    ranked.sort(key=lambda item: (item[0], -item[1]["stock"], item[1]["price_vnd"]))
    return [product for _, product in ranked[:limit]]
