import json

from langchain_core.messages import HumanMessage, ToolMessage

from mai_agent.skills.order import node as order_node_module
from mai_agent.skills.order.node import _contextualize_requested_items
from mai_agent.skills.order.tools import normalize_order_items
from mai_agent.skills.product import tools as product_tools_module
from mai_agent.skills.product.node import (
    _has_product_tool_result_for_latest_user,
    _recent_product_candidates,
)
from mai_agent.skills.product.tools import (
    search_product_knowledge,
    search_products_by_ingredient,
)


def _tool_message(name: str, payload: object) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        name=name,
        tool_call_id="tool-1",
    )


def test_omitted_product_uses_single_previous_candidate() -> None:
    assert _contextualize_requested_items(
        [{"product_reference": None, "quantity": 2}],
        [{"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}],
    ) == [{"product_reference": "OA002", "quantity": 2}]


def test_named_product_without_quantity_does_not_default_to_one() -> None:
    assert normalize_order_items(
        "cho tôi gel rửa mặt",
        [{"product_reference": "OA002", "quantity": None}],
    ) == [{"product_reference": "OA002", "quantity": None}]


def test_quantity_is_reinterpreted_semantically_when_initial_evidence_is_incomplete(
    monkeypatch,
) -> None:
    decisions = iter(
        [
            {
                "name": "provide_complete_order_request",
                "args": {
                    "items": [
                        {
                            "product_reference": None,
                            "quantity": 1,
                            "quantity_explicitly_provided": False,
                            "quantity_evidence": "",
                        }
                    ],
                    "address": "",
                    "phone": "",
                },
            },
            {
                "name": "use_explicit_order_quantity",
                "args": {"quantity": 1, "evidence": "1 sản phẩm"},
            },
        ]
    )
    monkeypatch.setattr(
        order_node_module,
        "invoke_tool",
        lambda *_args, **_kwargs: next(decisions),
    )

    result = order_node_module._interpret_order_request(
        "cho mình 1 sản phẩm",
        [{"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}],
    )

    assert result["items"] == [{"product_reference": None, "quantity": 1}]


def test_quantity_model_does_not_treat_sku_or_phone_as_quantity(monkeypatch) -> None:
    decisions = iter(
        [
            {
                "name": "provide_complete_order_request",
                "args": {
                    "items": [
                        {
                            "product_reference": "OA002",
                            "quantity": None,
                            "quantity_explicitly_provided": False,
                            "quantity_evidence": "",
                        }
                    ],
                    "address": "",
                    "phone": "0901234567",
                },
            },
            {"name": "no_explicit_order_quantity", "args": {}},
        ]
    )
    monkeypatch.setattr(
        order_node_module,
        "invoke_tool",
        lambda *_args, **_kwargs: next(decisions),
    )

    result = order_node_module._interpret_order_request(
        "mua OA002, số điện thoại 0901234567",
        [{"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}],
    )

    assert result["items"] == [{"product_reference": "OA002", "quantity": None}]


def test_one_product_from_single_recent_candidate_moves_to_shipping(monkeypatch) -> None:
    decisions = iter(
        [
            {
                "name": "provide_complete_order_request",
                "args": {
                    "items": [
                        {
                            "product_reference": None,
                            "quantity": 1,
                            "quantity_explicitly_provided": False,
                            "quantity_evidence": "",
                        }
                    ],
                    "address": "",
                    "phone": "",
                },
            },
            {
                "name": "use_explicit_order_quantity",
                "args": {"quantity": 1, "evidence": "1 sản phẩm"},
            },
        ]
    )
    monkeypatch.setattr(
        order_node_module,
        "invoke_tool",
        lambda *_args, **_kwargs: next(decisions),
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda *_args: {
            "status": "available",
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 1,
            "unit_price": 219_000,
        },
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="cho mình 1 sản phẩm")],
            "session": {
                "current_step": "idle",
                "cart": [],
                "last_product_candidates": [
                    {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
                ],
            },
        }
    )

    assert result["session"]["current_step"] == "collecting_address"
    assert result["session"]["cart"] == [
        {
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 1,
            "unit_price": 219_000,
        }
    ]


def test_previous_candidates_are_collected_from_product_tool_result() -> None:
    messages = [
        HumanMessage(content="Tư vấn sản phẩm"),
        _tool_message(
            "search_product_knowledge",
            {
                "resolved_products": [
                    {"id": "OA002", "name": "Gel rửa mặt OA Purifying Gel"},
                    {"id": "OA008", "name": "Kem chống nắng OA Daily UV Fluid"},
                ]
            },
        ),
    ]

    assert _recent_product_candidates(messages, "Dùng OA002 và OA008.") == [
        {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"},
        {"product_id": "OA008", "product_name": "Kem chống nắng OA Daily UV Fluid"},
    ]


def test_product_question_requires_retrieval_before_final_answer() -> None:
    initial_messages = [
        HumanMessage(content="Sữa rửa mặt dịu nhẹ có thành phần gì?"),
    ]
    retrieved_messages = [
        *initial_messages,
        _tool_message(
            "search_product_knowledge",
            {
                "resolved_products": [
                    {"id": "OA001", "name": "Sữa rửa mặt dịu nhẹ OA Gentle Cleanser"}
                ],
                "evidence": [],
            },
        ),
    ]

    assert _has_product_tool_result_for_latest_user(initial_messages) is False
    assert _has_product_tool_result_for_latest_user(retrieved_messages) is True


def test_vitamin_c_is_not_in_catalog() -> None:
    result = search_products_by_ingredient.invoke({"ingredient": "Vitamin C"})

    assert result["status"] == "ingredient_not_found"
    assert result["products"] == []


def test_ingredient_search_only_returns_exact_catalog_matches() -> None:
    niacinamide = search_products_by_ingredient.invoke({"ingredient": "Niacinamide"})
    ceramide = search_products_by_ingredient.invoke({"ingredient": "Ceramide"})
    hyaluronic_acid = search_products_by_ingredient.invoke(
        {"ingredient": "Hyaluronic Acid"}
    )

    assert [product["id"] for product in niacinamide["products"]] == [
        "OA002",
        "OA005",
        "OA007",
    ]
    assert [product["id"] for product in ceramide["products"]] == ["OA004", "OA006"]
    assert [product["id"] for product in hyaluronic_acid["products"]] == [
        "OA003",
        "OA007",
    ]


def test_unscoped_knowledge_search_does_not_use_nearest_product(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("Vector search must not run without an exact product")

    monkeypatch.setattr(product_tools_module, "get_product_vector_store", fail_if_called)
    result = search_product_knowledge.invoke(
        {"query": "Có serum Vitamin C không?", "product_references": []}
    )

    assert result["status"] == "product_reference_required"
    assert result["evidence"] == []


def test_found_ingredient_results_preserve_exact_product_candidates() -> None:
    messages = [
        HumanMessage(content="Sản phẩm nào có Ceramide?"),
        _tool_message(
            "search_products_by_ingredient",
            {
                "status": "found",
                "ingredient_query": "Ceramide",
                "products": [
                    {
                        "id": "OA004",
                        "name": "Serum phục hồi OA Barrier Repair Serum",
                        "matching_ingredients": ["Ceramide NP"],
                    },
                    {
                        "id": "OA006",
                        "name": "Kem dưỡng OA Ceramide Cream",
                        "matching_ingredients": ["Ceramide Complex"],
                    },
                ],
            },
        ),
    ]

    assert _recent_product_candidates(messages) == [
        {
            "product_id": "OA004",
            "product_name": "Serum phục hồi OA Barrier Repair Serum",
        },
        {"product_id": "OA006", "product_name": "Kem dưỡng OA Ceramide Cream"},
    ]


def test_selecting_product_can_understand_two_bottles_of_gel(monkeypatch) -> None:
    monkeypatch.setattr(order_node_module, "_interpret_order_control", lambda *_args: "continue")
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_request",
        lambda _text, *_args: {
            # Kết quả tool-calling sau khi model đối chiếu "gel" với danh sách
            # sản phẩm vừa tư vấn; nghiệp vụ phía sau vẫn xác thực SKU và tồn kho.
            "items": [{"product_reference": "OA002", "quantity": 2}],
            "address": "",
            "phone": "",
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda *_args: {
            "status": "available",
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 2,
            "unit_price": 219_000,
        },
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="cho tôi 2 lọ gel đi")],
            "session": {
                "current_step": "selecting_product",
                "cart": [],
                "last_product_candidates": [
                    {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"},
                    {"product_id": "OA007", "product_name": "Kem dưỡng OA Oil-Free Gel Cream"},
                    {"product_id": "OA008", "product_name": "Kem chống nắng OA Daily UV Fluid"},
                ],
            },
        }
    )

    assert result["session"]["current_step"] == "collecting_address"
    assert result["session"]["cart"] == [
        {
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 2,
            "unit_price": 219_000,
        }
    ]


def test_selected_product_without_quantity_moves_to_quantity_step(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_request",
        lambda _text, *_args: {
            "items": [{"product_reference": "OA002", "quantity": None}],
            "address": "",
            "phone": "",
        },
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="cho tôi gel rửa mặt")],
            "session": {
                "current_step": "idle",
                "cart": [],
                "last_product_candidates": [
                    {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
                ],
            },
        }
    )

    assert result["session"]["current_step"] == "collecting_quantity"
    assert result["session"]["pending_product_reference"] == "OA002"
    assert result["session"]["cart"] == []


def test_order_flow_can_return_to_product_selection_from_any_step(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_control",
        lambda *_args: "selecting_product",
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="quay lại bước chọn sản phẩm")],
            "session": {
                "current_step": "confirming_alternative",
                "cart": [],
                "pending_product_id": "OA001",
                "last_product_candidates": [
                    {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
                ],
            },
        }
    )

    assert result["session"]["current_step"] == "selecting_product"
    assert result["session"]["pending_product_id"] is None
    assert result["session"]["pending_product_candidates"] == [
        {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
    ]


def test_order_navigation_uses_model_tool_decision(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "invoke_tool",
        lambda *_args, **_kwargs: {"name": "return_to_shipping_step", "args": {}},
    )

    assert order_node_module._interpret_order_control(
        "Tôi muốn sửa lại nơi nhận hàng",
        "confirming_order",
        [],
    ) == "collecting_address"


def test_order_control_returns_structured_cart_change(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "invoke_tool",
        lambda *_args, **_kwargs: {
            "name": "modify_cart",
            "args": {
                "changes": [
                    {
                        "action": "update_quantity",
                        "product_id": "OA002",
                        "quantity": 3,
                        "replacement_product_id": None,
                    }
                ]
            },
        },
    )

    assert order_node_module._interpret_order_control(
        "cho tôi 3 lọ đi",
        "collecting_address",
        [
            {
                "product_id": "OA002",
                "product_name": "Gel rửa mặt OA Purifying Gel",
                "quantity": 1,
                "unit_price": 219_000,
            }
        ],
    ) == {
        "action": "modify",
        "changes": [
            {
                "action": "update_quantity",
                "product_id": "OA002",
                "quantity": 3,
                "replacement_product_id": None,
            }
        ],
    }


def test_quantity_update_from_shipping_step_keeps_recent_product_context(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_control",
        lambda *_args: {
            "action": "modify",
            "changes": [
                {
                    "action": "update_quantity",
                    "product_id": "OA002",
                    "quantity": 3,
                }
            ],
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda _state, product_id, quantity: {
            "status": "available",
            "product_id": product_id,
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": quantity,
            "unit_price": 219_000,
        },
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="cho tôi 3 lọ đi")],
            "session": {
                "current_step": "collecting_address",
                "cart": [
                    {
                        "product_id": "OA002",
                        "product_name": "Gel rửa mặt OA Purifying Gel",
                        "quantity": 1,
                        "unit_price": 219_000,
                    }
                ],
                "last_product_candidates": [
                    {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"},
                    {"product_id": "OA007", "product_name": "Kem dưỡng OA Oil-Free Gel Cream"},
                    {"product_id": "OA008", "product_name": "Kem chống nắng OA Daily UV Fluid"},
                ],
            },
        }
    )

    assert result["session"]["current_step"] == "confirming_cart_revision"
    assert result["session"]["cart"][0]["quantity"] == 1
    assert result["session"]["pending_cart_revision"][0]["product_id"] == "OA002"
    assert result["session"]["pending_cart_revision"][0]["quantity"] == 3


def test_empty_cart_revision_is_kept_as_draft_for_the_next_change(monkeypatch) -> None:
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)
    original_cart = [
        {
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 1,
            "unit_price": 219_000,
        }
    ]

    result = order_node_module._prepare_cart_revision(
        {"session": {"language_code": "vi"}},
        original_cart=original_cart,
        base_cart=original_cart,
        changes=[{"action": "remove", "product_id": "OA002"}],
    )

    assert result["session"]["current_step"] == "confirming_cart_revision"
    assert result["session"]["cart"] == original_cart
    assert result["session"]["pending_cart_revision"] == []


def test_add_after_empty_revision_uses_draft_instead_of_old_cart(monkeypatch) -> None:
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_control",
        lambda *_args: {
            "action": "modify",
            "changes": [
                {
                    "action": "add",
                    "product_id": "OA002",
                    "quantity": 1,
                }
            ],
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda _state, product_id, quantity: {
            "status": "available",
            "product_id": product_id,
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": quantity,
            "unit_price": 219_000,
        },
    )
    monkeypatch.setattr(order_node_module, "generate_reply", lambda text, **_kwargs: text)

    result = order_node_module.order_node(
        {
            "messages": [HumanMessage(content="thêm sữa rửa mặt cho tôi")],
            "session": {
                "current_step": "confirming_cart_revision",
                "cart": [
                    {
                        "product_id": "OA002",
                        "product_name": "Gel rửa mặt OA Purifying Gel",
                        "quantity": 1,
                        "unit_price": 219_000,
                    }
                ],
                "pending_cart_revision": [],
            },
        }
    )

    assert result["session"]["current_step"] == "confirming_cart_revision"
    assert result["session"]["pending_cart_revision"] == [
        {
            "product_id": "OA002",
            "product_name": "Gel rửa mặt OA Purifying Gel",
            "quantity": 1,
            "unit_price": 219_000,
        }
    ]
