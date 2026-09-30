import re
from typing import Any

from langchain_core.tools import tool

from mai_agent.catalog import find_alternative_products, find_product, load_products
from mai_agent.skills.product.vector_store import (
    get_product_vector_store,
    resolve_product_references,
)


@tool
def search_products(keyword: str) -> list[dict[str, Any]]:
    """Tìm sản phẩm theo mã, tên hoặc danh mục.

    Dùng tool này khi khách hỏi về thành phần, công dụng,
    cách dùng, loại da phù hợp, giá hoặc thông tin sản phẩm."""
    normalized_keyword = keyword.strip().lower()
    if not normalized_keyword:
        return []

    matches = []
    for product in load_products():
        searchable_text = " ".join([product["id"], product["name"], product["category"]]).lower()
        if normalized_keyword in searchable_text:
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
            "reason": "Không đủ số lượng yêu cầu.",
            "requested_quantity": quantity,
            "current_stock": product["stock"],
            "alternatives": [],
        }

    return {
        "status": "out_of_stock",
        "available": False,
        "reason": "Sản phẩm đã hết hàng.",
        "requested_quantity": quantity,
        "current_stock": 0,
        "alternatives": find_alternative_products(
            product_id,
            skin_type=skin_type,
            excluded_ingredients=excluded_ingredients,
        ),
    }


@tool
def resolve_product_names(references: list[str]) -> dict[str, list[Any]]:
    """Chuẩn hóa mã/tên sản phẩm, kể cả khi khách gõ thiếu hoặc sai nhẹ.

    Luôn dùng trước khi truy xuất nếu khách nêu tên sản phẩm không phải mã SKU chính xác.
    """
    return resolve_product_references(references)


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
    resolve_product_names,
    search_product_knowledge,
    check_product_stock,
]
