import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCTS_FILE = Path(__file__).resolve().parents[2] / "data" / "products.json"

# Alias thuộc dữ liệu nghiệp vụ, không để model tự suy đoán SKU. Một alias có thể
# trỏ tới nhiều SKU; lớp order phải yêu cầu khách chọn khi kết quả còn mơ hồ.
PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "sua rua mat diu nhe": ("OA001",),
    "gentle cleanser": ("OA001",),
    "mild cleanser": ("OA001",),
    "gel rua mat": ("OA002",),
    "purifying gel": ("OA002",),
    "gel cleanser": ("OA002",),
    "toner": ("OA003",),
    "toner cap am": ("OA003",),
    "hydrating toner": ("OA003",),
    "serum phuc hoi": ("OA004",),
    "barrier repair serum": ("OA004",),
    "barrier repair": ("OA004",),
    "serum lam sang": ("OA005",),
    "brightening serum": ("OA005",),
    "brightening": ("OA005",),
    "kem ceramide": ("OA006",),
    "ceramide cream": ("OA006",),
    "kem oil free": ("OA007",),
    "oil free gel cream": ("OA007",),
    "oil free cream": ("OA007",),
    "kem chong nang": ("OA008",),
    "kcn": ("OA008",),
    "sunscreen": ("OA008",),
    "daily uv fluid": ("OA008",),
    "protector solar": ("OA008",),
    "防晒霜": ("OA008",),
    "선크림": ("OA008",),
    "mat na": ("OA009",),
    "soothing mask": ("OA009",),
    "face mask": ("OA009",),
    "nuoc tay trang": ("OA010",),
    "micellar water": ("OA010",),
    "makeup remover": ("OA010",),
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "sua rua mat": ("OA001", "OA002"),
    "cleanser": ("OA001", "OA002"),
    "limpiador": ("OA001", "OA002"),
    "洁面": ("OA001", "OA002"),
    "클렌저": ("OA001", "OA002"),
    "serum": ("OA004", "OA005"),
    "suero": ("OA004", "OA005"),
    "精华": ("OA004", "OA005"),
    "세럼": ("OA004", "OA005"),
    "kem duong": ("OA006", "OA007"),
    "moisturizer": ("OA006", "OA007"),
    "crema hidratante": ("OA006", "OA007"),
    "面霜": ("OA006", "OA007"),
    "크림": ("OA006", "OA007"),
}


def normalize_product_reference(text: str) -> str:
    """Chuẩn hóa tên/mã sản phẩm để dùng chung cho đặt hàng và tìm tài liệu."""
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(
        "".join(char if char.isalnum() else " " for char in without_accents).split()
    )


def product_alias_candidates(reference: str) -> list[str]:
    """Trả SKU theo alias xác định; alias cụ thể dài hơn luôn được ưu tiên."""
    normalized = normalize_product_reference(reference)
    if not normalized:
        return []

    aliases = {**CATEGORY_ALIASES, **PRODUCT_ALIASES}
    matches = [
        (len(normalized_alias), product_ids)
        for alias, product_ids in aliases.items()
        if _contains_alias(normalized, normalized_alias := normalize_product_reference(alias))
    ]
    if not matches:
        return []
    longest = max(length for length, _ in matches)
    product_ids: list[str] = []
    for length, candidate_ids in matches:
        if length != longest:
            continue
        for product_id in candidate_ids:
            if product_id not in product_ids:
                product_ids.append(product_id)
    return product_ids


def extract_product_mentions(text: str) -> list[str]:
    """Lấy các SKU được nhắc rõ trong câu mà không cần gọi embedding hay LLM."""
    normalized = normalize_product_reference(text)
    mentions: list[tuple[int, str]] = []
    for alias, product_ids in PRODUCT_ALIASES.items():
        normalized_alias = normalize_product_reference(alias)
        position = (
            normalized.find(normalized_alias)
            if any(ord(character) > 127 for character in normalized_alias)
            else f" {normalized} ".find(f" {normalized_alias} ")
        )
        if position < 0 or len(product_ids) != 1:
            continue
        mentions.append((position, product_ids[0]))
    mentions.sort()
    return list(dict.fromkeys(product_id for _, product_id in mentions))


def _contains_alias(normalized: str, alias: str) -> bool:
    if any(ord(character) > 127 for character in alias):
        return alias in normalized
    return normalized == alias or f" {alias} " in f" {normalized} "


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
    disliked_product_ids: list[str] | None = None,
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
    disliked = {product_id.strip().upper() for product_id in disliked_product_ids or []}
    alternatives: list[dict[str, Any]] = []

    for product in load_products():
        if product["id"] == unavailable_product["id"] or product["stock"] <= 0:
            continue
        if product["id"].upper() in disliked:
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
