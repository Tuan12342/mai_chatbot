import json
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

ORDERS_FILE = Path(__file__).resolve().parents[2] / "data" / "orders.json"


@lru_cache
def load_orders() -> list[dict[str, Any]]:
    """Đọc dữ liệu đơn mẫu; có thể thay bằng repository/API thật sau này."""
    with ORDERS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def find_orders_by_customer_id(customer_id: str) -> list[dict[str, Any]]:
    return [order for order in load_orders() if order["customer_id"] == customer_id]


def find_order_by_id(order_id: str) -> dict[str, Any] | None:
    normalized_id = order_id.strip().upper()
    return next(
        (
            order
            for order in load_orders()
            if str(order.get("order_id", "")).upper() == normalized_id
        ),
        None,
    )


def save_order(order: dict[str, Any]) -> None:
    """Lưu đơn đã xác nhận bằng thao tác thay file để tránh ghi JSON dang dở."""
    orders = [dict(item) for item in load_orders()]
    order_id = str(order.get("order_id", "")).strip().upper()
    if not order_id or any(
        str(item.get("order_id", "")).strip().upper() == order_id for item in orders
    ):
        return
    orders.append(order)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=ORDERS_FILE.parent,
        delete=False,
    ) as file:
        json.dump(orders, file, ensure_ascii=False, indent=2)
        temporary_path = Path(file.name)
    temporary_path.replace(ORDERS_FILE)
    load_orders.cache_clear()
