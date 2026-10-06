from mai_agent.skills.order.node import _validated_quantity


def test_inferred_default_quantity_is_rejected() -> None:
    item = {
        "quantity": 1,
        "quantity_explicitly_provided": False,
        "quantity_evidence": "",
    }

    assert _validated_quantity("I want to buy OA Purifying Gel", item) is None


def test_sku_digits_are_not_quantity_evidence() -> None:
    item = {
        "quantity": 2,
        "quantity_explicitly_provided": True,
        "quantity_evidence": "2",
    }

    assert _validated_quantity("I want to buy OA002", item) is None


def test_explicit_word_quantity_is_accepted() -> None:
    item = {
        "quantity": 2,
        "quantity_explicitly_provided": True,
        "quantity_evidence": "two",
    }

    assert _validated_quantity("I want to buy two OA Purifying Gel", item) == 2


def test_explicit_numeric_quantity_is_accepted() -> None:
    item = {
        "quantity": 3,
        "quantity_explicitly_provided": True,
        "quantity_evidence": "3",
    }

    assert _validated_quantity("Cho tôi 3 sản phẩm OA002", item) == 3
