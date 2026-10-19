import re
from typing import Any

from langchain_core.tools import tool

from mai_agent.catalog import (
    extract_product_mentions,
    find_alternative_products,
    find_product,
    load_products,
    normalize_product_reference,
    product_alias_candidates,
)
from mai_agent.skills.product.vector_store import (
    get_product_vector_store,
    resolve_product_references,
)

INGREDIENT_ALIASES = {
    "hương liệu": ("hương liệu", "fragrance", "parfum"),
    "fragrance": ("hương liệu", "fragrance", "parfum"),
    "cồn": ("alcohol", "ethanol", "cồn"),
    "alcohol": ("alcohol", "ethanol", "cồn"),
}


def _avoid_ingredient_terms(*groups: list[str] | None) -> set[str]:
    terms: set[str] = set()
    for value in (item for group in groups for item in group or []):
        normalized = value.strip().casefold()
        if normalized:
            terms.update(INGREDIENT_ALIASES.get(normalized, (normalized,)))
    return terms


@tool
def search_products(keyword: str) -> list[dict[str, Any]]:
    """Tìm sản phẩm theo mã, tên hoặc danh mục.

    Dùng tool này khi khách hỏi về thành phần, công dụng,
    cách dùng, loại da phù hợp, giá hoặc thông tin sản phẩm."""
    normalized_keyword = normalize_product_reference(keyword)
    if not normalized_keyword:
        return []

    alias_ids = set(product_alias_candidates(keyword))
    matches = []
    for product in load_products():
        searchable_text = normalize_product_reference(
            " ".join([product["id"], product["name"], product["category"]])
        )
        if product["id"] in alias_ids or normalized_keyword in searchable_text:
            matches.append(
                {
                    "id": product["id"],
                    "name": product["name"],
                    "category": product["category"],
                    "skin_types": product["skin_types"],
                    "ingredients": product["ingredients"],
                    "benefits": product["benefits"],
                    "usage": product["usage"],
                    "price_vnd": product["price_vnd"],
                    "stock": product["stock"],
                }
            )
    return matches


@tool
def check_product_stock(
    product_id: str,
    quantity: int,
    skin_type: str | None = None,
    excluded_ingredients: list[str] | None = None,
    sensitivities: list[str] | None = None,
    allergies: list[str] | None = None,
    disliked_product_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Kiểm tra sản phẩm có tồn tại và còn đủ số lượng để bán hay không."""

    if quantity <= 0:
        return {"available": False, "reason": "Số lượng phải lớn hơn 0."}

    product = find_product(product_id)
    if product is None:
        return {"available": False, "reason": "Không tìm thấy sản phẩm."}
    if product["stock"] >= quantity:
        return {
            "status": "available",
            "available": True,
            "product_id": product["id"],
            "product_name": product["name"],
            "quantity": quantity,
            "unit_price": product["price_vnd"],
            "subtotal": product["price_vnd"] * quantity,
        }

    if product["stock"] > 0:
        return {
            "status": "partial_stock",
            "available": False,
            "product_id": product["id"],
            "product_name": product["name"],
            "reason": "Không đủ số lượng yêu cầu.",
            "requested_quantity": quantity,
            "current_stock": product["stock"],
            "alternatives": [],
        }

    return {
        "status": "out_of_stock",
        "available": False,
        "product_id": product["id"],
        "product_name": product["name"],
        "reason": "Sản phẩm đã hết hàng.",
        "requested_quantity": quantity,
        "current_stock": 0,
        "alternatives": find_alternative_products(
            product_id,
            skin_type=skin_type,
            excluded_ingredients=list(
                _avoid_ingredient_terms(excluded_ingredients, sensitivities, allergies)
            ),
            disliked_product_ids=disliked_product_ids,
        ),
    }


@tool
def search_product_knowledge(
    query: str,
    product_references: list[str] | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    """Semantic search có kiểm soát trên tài liệu sản phẩm của shop.

    product_references chứa mọi mã/tên cần tra. Khi so sánh, phải truyền cả hai
    sản phẩm để tool lấy evidence riêng cho từng SKU.
    """
    references = list(product_references or [])
    if not references:
        references = re.findall(r"\bOA\d+\b", query, flags=re.IGNORECASE)
    if not references:
        references = extract_product_mentions(query)

    resolution = resolve_product_references(references)
    resolved_products = resolution["resolved"]
    unresolved = resolution["unresolved"]
    if references and not resolved_products:
        return {
            "status": "product_not_resolved",
            "resolved_products": [],
            "unresolved_references": unresolved,
            "evidence": [],
            "instruction": "Không suy đoán sản phẩm; hãy hỏi khách xác nhận tên hoặc mã SKU.",
        }

    product_ids = [product["id"] for product in resolved_products]
    evidence = get_product_vector_store().search(
        query,
        product_ids=product_ids or None,
        top_k=min(max(top_k, 1), 12),
    )
    return {
        "status": "ok" if evidence else "no_evidence",
        "resolved_products": resolved_products,
        "unresolved_references": unresolved,
        "evidence": evidence,
        "instruction": (
            "Chỉ trả lời từ evidence. Nếu evidence không chứa thông tin được hỏi, "
            "hãy nói tài liệu shop chưa có thông tin."
        ),
    }


PRODUCT_TOOLS = [
    search_products,
    search_product_knowledge,
    check_product_stock,
]
