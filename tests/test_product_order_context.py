import json

from langchain_core.messages import HumanMessage, ToolMessage

from mai_agent.skills.order import node as order_node_module
from mai_agent.skills.order.node import _contextualize_requested_items
from mai_agent.skills.product import tools as product_tools
from mai_agent.skills.product.node import _recent_product_candidates


def _tool_message(name: str, payload: object, call_id: str = "tool-1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        name=name,
        tool_call_id=call_id,
    )


def test_mixed_order_resolves_omitted_product_before_multi_item_processing() -> None:
    items = [
        {"product_reference": None, "quantity": 2},
        {"product_reference": "OA003", "quantity": 1},
    ]

    assert _contextualize_requested_items(
        items,
        [{"product_id": "OA004", "product_name": "Barrier Repair Serum"}],
    ) == [
        {"product_reference": "OA004", "quantity": 2},
        {"product_reference": "OA003", "quantity": 1},
    ]


def test_mixed_order_does_not_guess_when_prior_context_is_ambiguous() -> None:
    items = [{"product_reference": None, "quantity": 2}]

    assert _contextualize_requested_items(
        items,
        [
            {"product_id": "OA004", "product_name": "Barrier Repair Serum"},
            {"product_id": "OA005", "product_name": "Brightening Serum"},
        ],
    ) == items


def test_explicit_new_product_replaces_product_waiting_for_quantity(monkeypatch) -> None:
    state = {
        "messages": [HumanMessage(content="Đổi sang OA003, lấy 2 chai")],
        "session": {
            "current_step": "collecting_quantity",
            "pending_product_reference": "OA002",
            "cart": [],
        },
    }
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_control",
        lambda *_args: "continue",
    )
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_request",
        lambda *_args: {
            "items": [{"product_reference": "OA003", "quantity": 2}],
            "address": "",
            "phone": "",
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "resolve_order_product",
        lambda reference: {
            "status": "resolved",
            "product": {"product_id": reference, "product_name": "Sản phẩm mới"},
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda _state, product_id, quantity: {
            "status": "available",
            "product_id": product_id,
            "product_name": "Sản phẩm mới",
            "quantity": quantity,
            "unit_price": 100_000,
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_reply",
        lambda _text, session_update, **updates: {
            "session": session_update,
            **updates,
        },
    )

    result = order_node_module.order_node(state)

    assert result["session"]["cart"] == [
        {
            "product_id": "OA003",
            "product_name": "Sản phẩm mới",
            "quantity": 2,
            "unit_price": 100_000,
        }
    ]


def test_multi_item_order_keeps_shipping_details_from_same_message(monkeypatch) -> None:
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Mua OA002 và OA003, giao 123 Nguyễn Trãi, "
                    "số điện thoại 0901234567"
                )
            )
        ],
        "session": {"current_step": "idle", "cart": []},
    }
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_request",
        lambda *_args: {
            "items": [
                {"product_reference": "OA002", "quantity": 1},
                {"product_reference": "OA003", "quantity": 2},
            ],
            "address": "123 Nguyễn Trãi",
            "phone": "0901234567",
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "resolve_order_product",
        lambda reference: {
            "status": "resolved",
            "product": {"product_id": reference, "product_name": reference},
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda _state, product_id, quantity: {
            "status": "available",
            "product_id": product_id,
            "product_name": product_id,
            "quantity": quantity,
            "unit_price": 100_000,
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_reply",
        lambda _text, session_update, **updates: {
            "session": session_update,
            **updates,
        },
    )

    result = order_node_module.order_node(state)

    assert result["session"]["current_step"] == "confirming_order"
    assert result["session"]["pending_shipping_address"] == "123 Nguyễn Trãi"
    assert result["session"]["pending_phone"] == "0901234567"
    assert [item["product_id"] for item in result["session"]["cart"]] == [
        "OA002",
        "OA003",
    ]


def test_multi_item_order_remembers_products_while_waiting_for_quantities(monkeypatch) -> None:
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Mua OA002 và OA003, giao 123 Nguyễn Trãi, "
                    "số điện thoại 0901234567"
                )
            )
        ],
        "session": {"current_step": "idle", "cart": []},
    }
    requested_items = [
        {"product_reference": "OA002", "quantity": None},
        {"product_reference": "OA003", "quantity": None},
    ]
    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_request",
        lambda *_args: {
            "items": requested_items,
            "address": "123 Nguyễn Trãi",
            "phone": "0901234567",
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_reply",
        lambda _text, session_update, **updates: {
            "session": session_update,
            **updates,
        },
    )

    result = order_node_module.order_node(state)

    assert result["session"]["current_step"] == "collecting_multi_quantities"
    assert result["session"]["pending_order_items"] == requested_items
    assert result["session"]["pending_shipping_address"] == "123 Nguyễn Trãi"
    assert result["session"]["pending_phone"] == "0901234567"


def test_multi_item_quantities_resume_from_pending_products(monkeypatch) -> None:
    pending_items = [
        {"product_reference": "OA002", "quantity": None},
        {"product_reference": "OA003", "quantity": None},
    ]
    state = {
        "messages": [HumanMessage(content="Mỗi loại 2")],
        "session": {
            "current_step": "collecting_multi_quantities",
            "cart": [],
            "pending_order_items": pending_items,
            "pending_shipping_address": "123 Nguyễn Trãi",
            "pending_phone": "0901234567",
        },
    }
    received_pending_items = []

    def interpret(_text, _candidates, supplied_pending_items):
        received_pending_items.extend(supplied_pending_items)
        return {
            "items": [
                {"product_reference": "OA002", "quantity": 2},
                {"product_reference": "OA003", "quantity": 2},
            ],
            "address": "",
            "phone": "",
        }

    monkeypatch.setattr(
        order_node_module,
        "_interpret_order_control",
        lambda *_args: "continue",
    )
    monkeypatch.setattr(order_node_module, "_interpret_order_request", interpret)
    monkeypatch.setattr(
        order_node_module,
        "resolve_order_product",
        lambda reference: {
            "status": "resolved",
            "product": {"product_id": reference, "product_name": reference},
            "candidates": [],
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_check_stock",
        lambda _state, product_id, quantity: {
            "status": "available",
            "product_id": product_id,
            "product_name": product_id,
            "quantity": quantity,
            "unit_price": 100_000,
        },
    )
    monkeypatch.setattr(
        order_node_module,
        "_reply",
        lambda _text, session_update, **updates: {
            "session": session_update,
            **updates,
        },
    )

    result = order_node_module.order_node(state)

    assert received_pending_items == pending_items
    assert result["session"]["current_step"] == "confirming_order"
    assert result["session"]["pending_order_items"] == []
    assert [item["quantity"] for item in result["session"]["cart"]] == [2, 2]


def test_partial_stock_result_keeps_product_identity(monkeypatch) -> None:
    product = {"id": "SKU-PARTIAL", "name": "Limited Product", "stock": 2}
    monkeypatch.setattr(product_tools, "find_product", lambda _product_id: product)

    result = product_tools.check_product_stock.invoke(
        {"product_id": product["id"], "quantity": 3},
    )

    assert result["status"] == "partial_stock"
    assert result["product_id"] == product["id"]
    assert result["product_name"] == product["name"]


def test_out_of_stock_result_keeps_product_identity(monkeypatch) -> None:
    product = {"id": "SKU-EMPTY", "name": "Unavailable Product", "stock": 0}
    monkeypatch.setattr(product_tools, "find_product", lambda _product_id: product)
    monkeypatch.setattr(
        product_tools,
        "find_alternative_products",
        lambda *_args, **_kwargs: [],
    )

    result = product_tools.check_product_stock.invoke(
        {"product_id": product["id"], "quantity": 2},
    )

    assert result["status"] == "out_of_stock"
    assert result["product_id"] == product["id"]
    assert result["product_name"] == product["name"]


def test_broad_search_remembers_only_product_surfaced_in_final_answer() -> None:
    messages = [
        HumanMessage(content="Tư vấn serum phù hợp cho tôi"),
        _tool_message(
            "search_products",
            [
                {"id": "OA004", "name": "Barrier Repair Serum"},
                {"id": "OA005", "name": "Brightening Serum"},
            ],
        ),
    ]

    assert _recent_product_candidates(
        messages,
        "Với nhu cầu hiện tại, em đề xuất OA004 - Barrier Repair Serum.",
    ) == [{"product_id": "OA004", "product_name": "Barrier Repair Serum"}]


def test_broad_search_keeps_multiple_products_when_answer_surfaces_multiple() -> None:
    messages = [
        HumanMessage(content="So sánh hai serum"),
        _tool_message(
            "search_products",
            [
                {"id": "OA004", "name": "Barrier Repair Serum"},
                {"id": "OA005", "name": "Brightening Serum"},
            ],
        ),
    ]

    assert _recent_product_candidates(
        messages,
        "OA004 phục hồi hàng rào da, còn OA005 hỗ trợ làm sáng.",
    ) == [
        {"product_id": "OA004", "product_name": "Barrier Repair Serum"},
        {"product_id": "OA005", "product_name": "Brightening Serum"},
    ]
