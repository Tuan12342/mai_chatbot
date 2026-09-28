from langchain_core.messages import HumanMessage

from mai_agent.graph import create_agent_graph


def invoke(message: str):
    graph = create_agent_graph()
    return graph.invoke({"user_id": "test-user", "messages": [HumanMessage(content=message)]})


def test_greeting_route() -> None:
    result = invoke("Xin chào Mai")
    assert result["intent"] == "greeting"
    assert "mình là Mai" in result["reply"]


def test_recommendation_route() -> None:
    result = invoke("Tư vấn sản phẩm cho da dầu")
    assert result["intent"] == "recommendation"
    assert "loại da" in result["reply"]


def test_order_route() -> None:
    result = invoke("Tôi muốn mua serum")
    assert result["intent"] == "order"
    assert "số lượng" in result["reply"]


def test_unknown_route() -> None:
    result = invoke("abc xyz")
    assert result["intent"] == "unknown"
    assert "chưa hiểu rõ" in result["reply"]
