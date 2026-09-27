from langgraph.graph import END, START, StateGraph

from mai_agent.nodes import (
    assistant_node,
    classify_intent_node,
    fallback_node,
    greeting_node,
    order_node,
)
from mai_agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "greeting":
        return "greeting"
    if intent == "order":
        return "order"
    if intent in {"product_question", "recommendation"}:
        return "assistant"
    return "fallback"


def create_agent_graph():
    """Tạo và compile LangGraph cho agent Mai."""
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("greeting", greeting_node)
    builder.add_node("assistant", assistant_node)
    builder.add_node("order", order_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "greeting": "greeting",
            "assistant": "assistant",
            "order": "order",
            "fallback": "fallback",
        },
    )
    builder.add_edge("greeting", END)
    builder.add_edge("assistant", END)
    builder.add_edge("order", END)
    builder.add_edge("fallback", END)

    return builder.compile()
