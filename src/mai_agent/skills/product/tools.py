from typing import Any

from langchain_core.tools import tool

from mai_agent.catalog import find_product, load_products


@tool
def search_products(keyword: str) -> list[dict[str, Any]]:
    """Tìm sản phẩm theo tên, mã hoặc danh mục trong catalog."""
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
                    "price_vnd": product["price_vnd"],
                    "stock": product["stock"],
                }
            )
    return matches


@tool
def check_product_stock(product_id: str, quantity: int) -> dict[str, Any]:
    """Kiểm tra sản phẩm có tồn tại và còn đủ số lượng để bán hay không."""

    product = find_product(product_id)
    if product is None:
        return {"available": False, "reason": "Không tìm thấy sản phẩm."}
    if product["stock"] < quantity:
        return {
            "available": False,
            "reason": "Không đủ trong kho.",
            "current_stock": product["stock"],
        }
    return {
        "available": True,
        "product_id": product["id"],
        "product_name": product["name"],
        "quantity": quantity,
        "unit_price": product["price_vnd"],
        "subtotal": product["price_vnd"] * quantity,
    }


PRODUCT_TOOLS = [search_products, check_product_stock]
