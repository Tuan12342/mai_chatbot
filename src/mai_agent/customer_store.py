import json
import os
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

CUSTOMERS_FILE = Path(__file__).resolve().parents[2] / "data" / "customers.json"
CUSTOMERS_LOCK = Lock()


@lru_cache
def load_customers() -> list[dict[str, Any]]:
    """Đọc hồ sơ khách đã được shop map với Zalo user ID."""
    with CUSTOMERS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def find_customer_by_zalo_id(zalo_user_id: str) -> dict[str, Any] | None:
    customer = next(
        (customer for customer in load_customers() if customer["zalo_user_id"] == zalo_user_id),
        None,
    )
    return deepcopy(customer) if customer is not None else None


def update_customer_profile(zalo_user_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Cập nhật hồ sơ dài hạn bằng thao tác thay file nguyên tử."""
    allowed_fields = {
        "skin_type",
        "is_sensitive",
        "sensitivities",
        "allergies",
        "concerns",
        "excluded_ingredients",
        "disliked_product_ids",
        "profile_metadata",
        "last_updated",
    }
    invalid_fields = set(updates) - allowed_fields
    if invalid_fields:
        raise ValueError(f"Trường hồ sơ không được hỗ trợ: {sorted(invalid_fields)}")

    with CUSTOMERS_LOCK:
        customers = deepcopy(load_customers())
        customer = next(
            (item for item in customers if item["zalo_user_id"] == zalo_user_id),
            None,
        )
        if customer is None:
            raise ValueError(f"Không tìm thấy khách hàng {zalo_user_id}.")

        customer.update(updates)
        customer["last_updated"] = updates.get(
            "last_updated",
            datetime.now().astimezone().isoformat(),
        )
        CUSTOMERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=CUSTOMERS_FILE.parent,
                prefix="customers-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                json.dump(customers, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_path = temporary_file.name
            os.replace(temporary_path, CUSTOMERS_FILE)
        finally:
            if temporary_path and Path(temporary_path).exists():
                Path(temporary_path).unlink()

        load_customers.cache_clear()
        return deepcopy(customer)
