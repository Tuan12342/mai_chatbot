from langchain_core.messages import HumanMessage

from mai_agent.graph import create_agent_graph
from mai_agent.routing import classify_intent_node
from mai_agent.skills.product.tools import (
    check_product_stock,
    search_product_knowledge,
    search_products,
)


def test_product_code_routes_to_assistant() -> None:
    result = classify_intent_node(
        {
            "user_id": "test-user",
            "messages": [HumanMessage(content="OA004")],
        }
    )

    assert result["session"]["current_intent"] == "product_question"


def test_search_products_returns_catalog_details() -> None:
    results = search_products.invoke({"keyword": "OA004"})

    assert len(results) == 1
    assert results[0]["id"] == "OA004"
    assert results[0]["benefits"]
    assert results[0]["ingredients"]
    assert results[0]["usage"]


def test_check_product_stock_rejects_non_positive_quantity() -> None:
    result = check_product_stock.invoke({"product_id": "OA004", "quantity": 0})

    assert result == {
        "available": False,
        "reason": "Số lượng phải lớn hơn 0.",
    }


def test_partial_stock_does_not_suggest_alternatives() -> None:
    result = check_product_stock.invoke(
        {
            "product_id": "OA004",
            "quantity": 999,
            "skin_type": "da dầu",
            "excluded_ingredients": [],
        }
    )

    assert result["available"] is False
    assert result["status"] == "partial_stock"
    assert result["current_stock"] == 28
    assert result["alternatives"] == []


def test_graph_contains_product_tool_node() -> None:
    graph = create_agent_graph()

    assert "product_tools" in graph.nodes


def test_knowledge_search_refuses_unresolved_product() -> None:
    result = search_product_knowledge.invoke(
        {
            "query": "Công dụng của serum không tồn tại là gì?",
            "product_references": ["serum không tồn tại"],
        }
    )

    assert result["status"] == "product_not_resolved"
    assert result["evidence"] == []
