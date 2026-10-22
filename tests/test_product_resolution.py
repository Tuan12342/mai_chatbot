from mai_agent.catalog import extract_product_mentions
from mai_agent.skills.product.vector_store import resolve_product_references


def test_alias_with_unknown_qualifier_is_not_substituted() -> None:
    result = resolve_product_references(["toner AHA"])

    assert result["resolved"] == []
    assert result["unresolved"] == ["toner AHA"]


def test_specific_catalog_alias_still_resolves() -> None:
    result = resolve_product_references(["sữa rửa mặt dịu nhẹ"])

    assert result["resolved"] == [
        {"id": "OA001", "name": "Sữa rửa mặt dịu nhẹ OA Gentle Cleanser"}
    ]
    assert result["unresolved"] == []


def test_unique_catalog_categories_are_resolved_from_data() -> None:
    mentions = extract_product_mentions("Cho tôi thông tin toner, mặt nạ và tẩy trang")

    assert mentions == ["OA003", "OA009", "OA010"]
