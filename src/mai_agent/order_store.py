import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ORDERS_FILE = Path(__file__).resolve().parents[2] / "data" / "orders.json"


@lru_cache
def load_orders() -> list[dict[str, Any]]:
    """Đọc dữ liệu đơn mẫu; có thể thay bằng repository/API thật sau này."""
    with ORDERS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def find_orders_by_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return [order for order in load_orders() if order["customer_id"] == customer_id]
