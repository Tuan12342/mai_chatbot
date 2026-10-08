import re
from typing import TypedDict

from mai_agent.catalog import load_products, normalize_product_reference
from mai_agent.skills.product.vector_store import resolve_product_references
from mai_agent.state import ProductCandidate


class ProductResolution(TypedDict):
    status: str
    product: ProductCandidate | None
    candidates: list[ProductCandidate]


def resolve_order_product(reference: str) -> ProductResolution:
    """Resolve tên/mã và trả danh sách lựa chọn nếu sản phẩm còn mơ hồ."""
    products = load_products()
    normalized_reference = normalize_product_reference(reference)
    if not normalized_reference:
        return {"status": "not_found", "product": None, "candidates": []}

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
        (
            item
            for item in products
            if normalize_product_reference(item["name"]) == normalized_reference
        ),
        None,
    )
    if exact_name:
        candidate = {"product_id": exact_name["id"], "product_name": exact_name["name"]}
        return {"status": "resolved", "product": candidate, "candidates": [candidate]}

    # Tìm theo tên hoặc danh mục
    matches = [
        {"product_id": item["id"], "product_name": item["name"]}
        for item in products
        if normalized_reference in normalize_product_reference(item["name"])
        or normalized_reference == normalize_product_reference(item["category"])
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
