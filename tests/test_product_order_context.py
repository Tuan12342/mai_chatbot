import json

from langchain_core.messages import HumanMessage, ToolMessage

from mai_agent.skills.order.node import _contextualize_requested_items
from mai_agent.skills.product.node import _recent_product_candidates
from mai_agent.skills.product import tools as product_tools


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
