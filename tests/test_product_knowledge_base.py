from mai_agent.vision.knowledge_base import (
    knowledge_for_products,
    load_product_knowledge,
    product_supports_care_mode,
)


def test_knowledge_base_covers_every_catalog_product() -> None:
    knowledge = load_product_knowledge()
    assert set(knowledge) == {f"OA{number:03d}" for number in range(1, 11)}
    assert all(
        record.get("routine_role_en")
        and record.get("why_en")
        and record.get("cautions_en")
        for record in knowledge.values()
    )


def test_retrieved_knowledge_is_limited_to_selected_products() -> None:
    evidence = knowledge_for_products(["OA003", "OA008"])

    assert [item["product_id"] for item in evidence] == ["OA003", "OA008"]
    assert all(item["why"] and item["cautions"] for item in evidence)
    assert all(
        item["routine_role_en"] and item["why_en"] and item["cautions_en"]
        for item in evidence
    )
    assert all("match_evidence" not in item for item in evidence)


def test_retrieved_knowledge_normalizes_product_ids() -> None:
    assert knowledge_for_products(["oa003"])[0]["product_id"] == "OA003"


def test_medical_review_only_allows_curated_supportive_products() -> None:
    assert product_supports_care_mode("OA001", "medical_review")
    assert product_supports_care_mode("OA003", "medical_review")
    assert product_supports_care_mode("OA006", "medical_review")
    assert not product_supports_care_mode("OA002", "medical_review")
    assert not product_supports_care_mode("OA005", "medical_review")
