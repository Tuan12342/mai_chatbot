import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph
from mai_agent.skills.order.tools import extract_order_request


@pytest.fixture(autouse=True)
def deterministic_order_replies(monkeypatch):
    monkeypatch.setattr(
        "mai_agent.skills.order.node._generate_order_reply",
        lambda required_content: required_content,
    )


def _conversation():
    graph = create_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "order-test"}}
    return graph, config


def _invoke(graph, config, text: str):
    return graph.invoke(
        {
            "user_id": "customer-1",
            "messages": [HumanMessage(content=text)],
        },
        config=config,
    )


def test_extract_order_request_gets_quantity_and_product() -> None:
    parsed = extract_order_request("Cho tôi mua 2 sản phẩm Serum phục hồi OA004")

    assert parsed == {
        "product_reference": "Serum phục hồi OA004",
        "quantity": 2,
    }


def test_order_with_product_and_quantity_asks_for_address() -> None:
    graph, config = _conversation()

    result = _invoke(graph, config, "Cho tôi mua 2 OA004")

    assert result["session"]["current_step"] == "collecting_address"
    assert result["session"]["cart"][0]["product_id"] == "OA004"
    assert result["session"]["cart"][0]["quantity"] == 2
    assert "địa chỉ" in result["reply"]


def test_full_product_name_and_quantity_are_not_requested_again() -> None:
    graph, config = _conversation()

    result = _invoke(
        graph,
        config,
        "Cho tôi mua 2 sản phẩm Serum phục hồi OA Barrier Repair Serum abc",
    )

    assert result["session"]["current_step"] == "collecting_address"
    assert result["session"]["cart"][0]["product_id"] == "OA011"
    assert result["session"]["cart"][0]["quantity"] == 2
    assert "địa chỉ" in result["reply"]


def test_ambiguous_product_keeps_quantity_after_customer_selects_sku() -> None:
    graph, config = _conversation()

    first = _invoke(graph, config, "Cho tôi mua 2 serum")
    second = _invoke(graph, config, "OA004")

    assert first["session"]["current_step"] == "selecting_product"
    assert first["session"]["pending_quantity"] == 2
    assert len(first["session"]["pending_product_candidates"]) >= 2
    assert second["session"]["current_step"] == "collecting_address"
    assert second["session"]["cart"][0]["quantity"] == 2


def test_order_flow_collects_address_and_confirms_order() -> None:
    graph, config = _conversation()

    _invoke(graph, config, "Mua 2 OA004")
    address_result = _invoke(
        graph,
        config,
        "123 Nguyễn Trãi, Quận 5, TP.HCM, 0901234567",
    )
    confirmed = _invoke(graph, config, "xác nhận")

    assert address_result["session"]["current_step"] == "confirming_order"
    assert "OA004" in address_result["reply"]
    assert confirmed["session"]["current_step"] == "idle"
    assert confirmed["session"]["cart"] == []
    assert confirmed["active_order"]["status"] == "confirmed"
    assert confirmed["active_order"]["items"][0]["quantity"] == 2


def test_order_rejects_quantity_above_stock() -> None:
    graph, config = _conversation()

    result = _invoke(graph, config, "Mua 999 OA004")

    assert result["session"]["current_step"] == "confirming_partial_quantity"
    assert result["session"]["pending_product_id"] == "OA004"
    assert "28" in result["reply"]


def test_customer_sees_alternative_details_before_quantity() -> None:
    graph, config = _conversation()

    offered = _invoke(graph, config, "Mua 1 OA001")
    details = _invoke(graph, config, "có")
    quantity = _invoke(graph, config, "5")

    assert offered["session"]["current_step"] == "confirming_alternative"
    assert details["session"]["current_step"] == "collecting_quantity"
    assert "OA002" in details["reply"]
    assert "219,000" in details["reply"]
    assert quantity["session"]["current_step"] == "collecting_address"
    assert quantity["session"]["cart"][0]["product_id"] == "OA002"
    assert quantity["session"]["cart"][0]["quantity"] == 5
