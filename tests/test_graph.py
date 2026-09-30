import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph
from mai_agent.routing import classify_intent_node


@pytest.fixture(autouse=True)
def deterministic_order_replies(monkeypatch):
    monkeypatch.setattr(
        "mai_agent.skills.order.node._generate_order_reply",
        lambda required_content: required_content,
    )


def invoke(message: str):
    graph = create_agent_graph()
    return graph.invoke({"user_id": "test-user", "messages": [HumanMessage(content=message)]})


def test_greeting_route() -> None:
    result = invoke("Xin chào Mai")
    assert result["session"]["current_intent"] == "greeting"
    assert "mình là Mai" in result["reply"]


def test_recommendation_route() -> None:
    result = classify_intent_node(
        {
            "user_id": "test-user",
            "messages": [HumanMessage(content="Tư vấn sản phẩm cho da dầu")],
        }
    )

    assert result["session"]["current_intent"] == "recommendation"


def test_order_route() -> None:
    result = invoke("Tôi muốn mua serum")
    assert result["session"]["current_intent"] == "order"
    assert result["session"]["current_step"] == "collecting_quantity"
    assert result["session"]["cart"] == []
    assert "số lượng" in result["reply"]


def test_unknown_route() -> None:
    result = invoke("abc xyz")
    assert result["session"]["current_intent"] == "unknown"
    assert "chưa hiểu rõ" in result["reply"]


def test_checkpointer_preserves_messages_between_turns() -> None:
    graph = create_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "conversation-1"}}

    graph.invoke(
        {"user_id": "zalo-user-1", "messages": [HumanMessage(content="Xin chào Mai")]},
        config=config,
    )
    result = graph.invoke(
        {"user_id": "zalo-user-1", "messages": [HumanMessage(content="abc xyz")]},
        config=config,
    )

    assert len(result["messages"]) == 4
    assert result["session"]["session_id"] == "zalo-user-1"


def test_partial_session_update_does_not_clear_cart() -> None:
    graph = create_agent_graph()
    cart = [
        {
            "product_id": "SP001",
            "product_name": "Serum",
            "quantity": 1,
            "unit_price": 200_000,
        }
    ]

    result = graph.invoke(
        {
            "user_id": "zalo-user-1",
            "session": {"session_id": "session-1", "cart": cart},
            "messages": [HumanMessage(content="Xin chào Mai")],
        }
    )

    assert result["session"]["cart"] == cart
