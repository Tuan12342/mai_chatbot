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

    Không dùng tool này để tìm sản phẩm theo thành phần."""
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
def search_products_by_ingredient(ingredient: str) -> dict[str, Any]:
    """Tra chính xác sản phẩm có liệt kê một thành phần trong catalog.

    Dùng khi khách hỏi shop có thành phần X không, sản phẩm nào chứa X, hoặc
    hỏi vai trò của thành phần X trong sản phẩm shop. ``ingredient`` phải là tên
    thành phần, không phải tính từ mô tả, loại da, công dụng hay loại sản phẩm.
    Không dùng khi khách hỏi "thành phần của sản phẩm X"; trường hợp đó dùng
    ``search_product_knowledge`` cho sản phẩm X. Không suy ra thành phần từ công dụng.
    """
    ingredient_query = ingredient.strip()
    normalized_query = normalize_product_reference(ingredient_query)
    if not normalized_query:
        return {
            "status": "invalid_ingredient",
            "ingredient_query": ingredient_query,
            "products": [],
            "instruction": "Hãy yêu cầu khách nêu rõ tên thành phần cần tìm.",
        }

    products: list[dict[str, Any]] = []
    for product in load_products():
        matching_ingredients = [
            listed_ingredient
            for listed_ingredient in product["ingredients"]
            if normalized_query
            in normalize_product_reference(listed_ingredient)
        ]
        if not matching_ingredients:
            continue
        products.append(
            {
                "id": product["id"],
                "name": product["name"],
                "matching_ingredients": matching_ingredients,
                "ingredients": product["ingredients"],
                "benefits": product["benefits"],
                "usage": product["usage"],
                "price_vnd": product["price_vnd"],
                "stock": product["stock"],
            }
        )

    return {
        "status": "found" if products else "ingredient_not_found",
        "ingredient_query": ingredient_query,
        "products": products,
        "instruction": (
            "Chỉ khẳng định sản phẩm chứa thành phần khi sản phẩm nằm trong "
            "danh sách này. Catalog chỉ liệt kê thành phần nổi bật, không phải bảng "
            "INCI đầy đủ. Khi không tìm thấy, chỉ được nói catalog chưa liệt kê thành "
            "phần đó; tuyệt đối không suy ra sản phẩm không chứa hoặc an toàn với "
            "người cần tránh thành phần. Không thay thế bằng sản phẩm có công dụng "
            "tương tự. Nếu đầu vào thực ra là tính chất hoặc loại sản phẩm chứ không "
            "phải thành phần, quay lại tra sản phẩm/danh mục được nhắc trong câu hỏi. "
            "Khi khách hỏi công dụng riêng của thành phần, chỉ mô tả công dụng tổng thể "
            "mà catalog gắn với sản phẩm chứa nó; không tự quy quan hệ nhân quả cho một "
            "thành phần nếu catalog không nêu."
        ),
    }


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

    product_references chứa mọi mã/tên sản phẩm hoặc mô tả đủ phân biệt cần tra.
    Dùng tool này khi khách hỏi thành phần CỦA sản phẩm, công dụng, cách dùng hoặc
    độ phù hợp. Không đưa tên một ingredient vào product_references. Khi so sánh,
    phải truyền cả hai sản phẩm để tool lấy evidence riêng cho từng SKU.
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
            "catalog_match": False,
            "resolved_products": [],
            "unresolved_references": unresolved,
            "evidence": [],
            "instruction": (
                "Tra cứu đã hoàn tất và catalog hiện không có sản phẩm khớp tham chiếu "
                "khách nêu. Hãy nói rõ không tìm thấy trong catalog hiện tại. Không hỏi "
                "SKU như thể sản phẩm chắc chắn tồn tại, không khẳng định shop có nhiều "
                "phiên bản và không tự thay bằng sản phẩm gần giống."
            ),
        }

    if not references:
        return {
            "status": "product_reference_required",
            "resolved_products": [],
            "unresolved_references": [],
            "evidence": [],
            "instruction": (
                "Không dùng semantic similarity để tự chọn sản phẩm. Nếu khách "
                "tìm theo thành phần, hãy gọi search_products_by_ingredient; nếu "
                "không, hãy yêu cầu tên, mã hoặc danh mục sản phẩm rõ ràng."
            ),
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
    search_products_by_ingredient,
    search_product_knowledge,
    check_product_stock,
]
