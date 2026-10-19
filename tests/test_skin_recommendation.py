from mai_agent.vision.recommendation import eligible_catalog_products


def test_candidate_filter_does_not_rank_from_visible_features() -> None:
    candidate_ids = [
        product["id"]
        for product in eligible_catalog_products({}, "da hỗn hợp")
    ]

    assert candidate_ids == ["OA002", "OA003", "OA005", "OA007", "OA008", "OA010"]


def test_candidate_filter_keeps_only_medical_review_products() -> None:
    candidate_ids = [
        product["id"]
        for product in eligible_catalog_products({}, None, care_mode="medical_review")
    ]

    assert candidate_ids == ["OA001", "OA003", "OA006"]


def test_candidate_filter_applies_exclusions_and_dislikes() -> None:
    candidates = eligible_catalog_products(
        {
            "excluded_ingredients": ["niacinamide"],
            "disliked_product_ids": ["OA003"],
        },
        "da dầu",
    )

    assert [product["id"] for product in candidates] == ["OA008", "OA010"]
