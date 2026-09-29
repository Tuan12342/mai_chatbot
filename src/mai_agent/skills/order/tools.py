import re
import unicodedata
from typing import TypedDict

from mai_agent.catalog import load_products
from mai_agent.skills.product.vector_store import resolve_product_references
from mai_agent.state import ProductCandidate


class ParsedOrderRequest(TypedDict):
    product_reference: str | None
    quantity: int | None


class ProductResolution(TypedDict):
    status: str
    product: ProductCandidate | None
    candidates: list[ProductCandidate]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


def extract_order_request(text: str) -> ParsedOrderRequest:
    """Tách số lượng và phần tên/mã sản phẩm từ một câu đặt hàng."""
    cleaned = text.strip()
    quantity: int | None = None

    quantity_patterns = [
        r"\b(?:mua|lấy|đặt)\s+(\d+)\b",
        r"\b(\d+)\s*(?:sản phẩm|sp|cái|chai|lọ|tuýp)\b",
        r"^\s*(\d+)\s*$",
    ]
    quantity_match = next(
        (match for pattern in quantity_patterns if (match := re.search(pattern, cleaned, re.I))),
        None,
    )
    if quantity_match:
        quantity = int(quantity_match.group(1))
        number_start, number_end = quantity_match.span(1)
        cleaned = f"{cleaned[:number_start]} {cleaned[number_end:]}"

    cleaned = re.sub(
        r"\b(?:cho tôi|cho mình|tôi muốn|mình muốn|muốn|mua|đặt|lấy|giúp tôi|giúp mình)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\b(?:sản phẩm|sp|cái|chai|lọ|tuýp)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    product_reference = " ".join(cleaned.strip(" ,.-").split()) or None
    return {"product_reference": product_reference, "quantity": quantity}


def resolve_order_product(reference: str) -> ProductResolution:
    """Resolve tên/mã và trả danh sách lựa chọn nếu sản phẩm còn mơ hồ."""
    products = load_products()
    normalized_reference = _normalize(reference)

    sku_match = re.search(r"\boa\d+\b", normalized_reference)
    if sku_match:
        product = next(
            (item for item in products if item["id"].lower() == sku_match.group(0)),
            None,
        )
        if product:
            candidate = {"product_id": product["id"], "product_name": product["name"]}
            return {"status": "resolved", "product": candidate, "candidates": [candidate]}

    exact_name = next(
        (item for item in products if _normalize(item["name"]) == normalized_reference),
        None,
    )
    if exact_name:
        candidate = {"product_id": exact_name["id"], "product_name": exact_name["name"]}
        return {"status": "resolved", "product": candidate, "candidates": [candidate]}

    matches = [
        {"product_id": item["id"], "product_name": item["name"]}
        for item in products
        if normalized_reference in _normalize(item["name"])
    ]
    if len(matches) == 1:
        return {"status": "resolved", "product": matches[0], "candidates": matches}
    if len(matches) > 1:
        return {"status": "ambiguous", "product": None, "candidates": matches}

    fuzzy = resolve_product_references([reference])
    if fuzzy["resolved"]:
        product = fuzzy["resolved"][0]
        candidate = {"product_id": product["id"], "product_name": product["name"]}
        return {"status": "resolved", "product": candidate, "candidates": [candidate]}

    return {"status": "not_found", "product": None, "candidates": []}
