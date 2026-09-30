import re
import unicodedata
from typing import Any

from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.order_store import find_orders_by_customer_id


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    without_accents = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def verify_customer_identity(
    *,
    zalo_user_id: str,
    provided_name: str,
    phone_last_four: str,
) -> bool:
    """Xác minh trong phạm vi hồ sơ đã map với chính Zalo user ID hiện tại."""
    customer = find_customer_by_zalo_id(zalo_user_id)
    if customer is None:
        return False

    supplied_digits = _digits(phone_last_four)
    return (
        _normalize_name(provided_name) == _normalize_name(customer["name"])
        and len(supplied_digits) == 4
        and _digits(customer["phone"]).endswith(supplied_digits)
    )


def lookup_verified_customer_orders(zalo_user_id: str) -> list[dict[str, Any]]:
    """Trả dữ liệu đơn tối thiểu; hàm gọi phải xác minh danh tính trước."""
    customer = find_customer_by_zalo_id(zalo_user_id)
    if customer is None:
        return []

    return [
        {
            "order_id": order["order_id"],
            "status": order["status"],
            "payment_status": order["payment_status"],
            "total_amount": order["total_amount"],
            "items": order["items"],
        }
        for order in find_orders_by_customer_id(customer["zalo_user_id"])
    ]
