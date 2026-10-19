import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from mai_agent.catalog import load_products

KNOWLEDGE_FILE = Path(__file__).resolve().parents[3] / "data" / "product_knowledge.json"


@lru_cache
def load_product_knowledge() -> dict[str, dict[str, Any]]:
    with KNOWLEDGE_FILE.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Kho tri thức sản phẩm phải là một danh sách JSON.")

    catalog_ids = {product["id"] for product in load_products()}
    knowledge: dict[str, dict[str, Any]] = {}
    for record in records:
        product_id = str(record.get("product_id", "")).strip().upper()
        if not product_id or product_id in knowledge:
            raise ValueError("Kho tri thức có SKU trống hoặc trùng lặp.")
        if product_id not in catalog_ids:
            raise ValueError(f"SKU {product_id} trong kho tri thức không có trong catalog.")
        care_modes = record.get("allowed_care_modes")
        if not isinstance(care_modes, list) or not care_modes:
            raise ValueError(f"SKU {product_id} thiếu allowed_care_modes.")
        if not set(care_modes) <= {"standard", "medical_review"}:
            raise ValueError(f"SKU {product_id} có allowed_care_modes không hợp lệ.")
        knowledge[product_id] = record

    missing = catalog_ids - knowledge.keys()
    if missing:
        raise ValueError(f"Kho tri thức thiếu SKU: {', '.join(sorted(missing))}.")
    return knowledge


def product_supports_care_mode(product_id: str, care_mode: str) -> bool:
    record = load_product_knowledge()[product_id.upper()]
    return care_mode in record["allowed_care_modes"]


def knowledge_for_products(
    product_ids: list[str],
) -> list[dict[str, Any]]:
    knowledge = load_product_knowledge()
    records: list[dict[str, Any]] = []
    for product_id in product_ids:
        normalized_id = product_id.strip().upper()
        if normalized_id not in knowledge:
            continue
        record = {
            "product_id": normalized_id,
            "routine_role": knowledge[normalized_id]["routine_role"],
            "routine_role_en": knowledge[normalized_id].get("routine_role_en"),
            "allowed_care_modes": knowledge[normalized_id]["allowed_care_modes"],
            "why": knowledge[normalized_id]["why"],
            "why_en": knowledge[normalized_id].get("why_en", []),
            "cautions": knowledge[normalized_id]["cautions"],
            "cautions_en": knowledge[normalized_id].get("cautions_en", []),
        }
        records.append(record)
    return records
