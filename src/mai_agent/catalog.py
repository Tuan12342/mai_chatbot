import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PRODUCTS_FILE = Path(__file__).resolve().parents[2] / "data" / "products.json"


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
