import re
from typing import TypedDict

from mai_agent.catalog import (
    extract_product_mentions,
    load_products,
    normalize_product_reference,
    product_alias_candidates,
)
from mai_agent.skills.product.vector_store import resolve_product_references
from mai_agent.state import ProductCandidate


class ProductResolution(TypedDict):
    status: str
    product: ProductCandidate | None
    candidates: list[ProductCandidate]


def normalize_order_items(
    text: str,
    extracted_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Chuẩn hóa entity model trích xuất trước khi chạy nghiệp vụ giỏ hàng.

    SKU trong catalog luôn thắng cách gọi tự do của model. Khi câu đặt hàng nhắc
    rõ sản phẩm nhưng không nói số lượng, giữ ``quantity=None`` để hỏi lại khách.
    """
    items = [dict(item) for item in extracted_items]
    mentions = extract_product_mentions(text)
    referenced_ids: set[str] = set()

    # Ambiguous business aliases (for example "serum phục hồi" mapping to
    # OA004 and OA011) are intentionally not returned by extract_product_mentions.
    # Preserve the original reference when the model omitted it so the order
    # state machine can present every valid SKU instead of asking a generic question.
    alias_candidates = product_alias_candidates(text)
    if not items and len(alias_candidates) > 1:
        items.append({"product_reference": text, "quantity": None})

    for item in items:
        reference = str(item.get("product_reference") or "").strip()
        if not reference:
            continue
        resolution = resolve_order_product(reference)
        if resolution["status"] == "resolved" and resolution["product"]:
            product_id = resolution["product"]["product_id"]
            item["product_reference"] = product_id
            referenced_ids.add(product_id)

    missing_reference_items = [item for item in items if not item.get("product_reference")]
    remaining_mentions = [product_id for product_id in mentions if product_id not in referenced_ids]
    for item, product_id in zip(missing_reference_items, remaining_mentions, strict=False):
        item["product_reference"] = product_id
        referenced_ids.add(product_id)

    for product_id in remaining_mentions[len(missing_reference_items) :]:
        items.append({"product_reference": product_id, "quantity": None})

    return items


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

    alias_ids = product_alias_candidates(reference)
    if alias_ids:
        candidates = [
            {"product_id": item["id"], "product_name": item["name"]}
            for item in products
            if item["id"] in alias_ids
        ]
        if len(candidates) == 1:
            return {"status": "resolved", "product": candidates[0], "candidates": candidates}
        return {"status": "ambiguous", "product": None, "candidates": candidates}

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
