import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CUSTOMERS_FILE = Path(__file__).resolve().parents[2] / "data" / "customers.json"


@lru_cache
def load_customers() -> list[dict[str, Any]]:
    """Đọc hồ sơ khách đã được shop map với Zalo user ID."""
    with CUSTOMERS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def find_customer_by_zalo_id(zalo_user_id: str) -> dict[str, Any] | None:
    return next(
        (customer for customer in load_customers() if customer["zalo_user_id"] == zalo_user_id),
        None,
    )
