import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCTS_FILE = Path(__file__).resolve().parents[2] / "data" / "products.json"


def normalize_product_reference(text: str) -> str:
    """Chuẩn hóa tên/mã sản phẩm để dùng chung cho đặt hàng và tìm tài liệu."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


@lru_cache
def load_products() -> list[dict[str, Any]]:
    """Đọc catalog một lần và cache trong suốt tiến trình."""
    with PRODUCTS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def find_product(product_id: str) -> dict[str, Any] | None:
    """Tìm chính xác một sản phẩm theo mã."""
    normalized_id = product_id.strip().lower()
    return next(
        (product for product in load_products() if product["id"].lower() == normalized_id),
        None,
    )


def find_alternative_products(
    product_id: str,
    *,
    skin_type: str | None = None,
    excluded_ingredients: list[str] | None = None,
    limit=5,
) -> list[dict[str, Any]]:
    """Tìm sản phẩm cùng danh mục, còn hàng và phù hợp hồ sơ da."""
    unavailable_product = find_product(product_id)
    if unavailable_product is None or limit <= 0:
        return []

    normalized_skin_type = (skin_type or "").strip().lower()
    excluded = {
        ingredient.strip().lower()
        for ingredient in excluded_ingredients or []
        if ingredient.strip()
    }
    alternatives: list[dict[str, Any]] = []

    for product in load_products():
        if product["id"] == unavailable_product["id"] or product["stock"] <= 0:
            continue
        if product["category"] != unavailable_product["category"]:
            continue

        supported_skin_types = {item.lower() for item in product["skin_types"]}
        if (
            normalized_skin_type
            and "mọi loại da" not in supported_skin_types
            and normalized_skin_type not in supported_skin_types
        ):
            continue

        ingredients = {ingredient.lower() for ingredient in product["ingredients"]}
        if any(blocked in ingredient for blocked in excluded for ingredient in ingredients):
            continue

        alternatives.append(
            {
                "id": product["id"],
                "name": product["name"],
                "price_vnd": product["price_vnd"],
                "stock": product["stock"],
                "skin_types": product["skin_types"],
            }
        )

    alternatives.sort(
        key=lambda product: (
            abs(product["price_vnd"] - unavailable_product["price_vnd"]),
            -product["stock"],
        )
    )
    return alternatives[:limit]
