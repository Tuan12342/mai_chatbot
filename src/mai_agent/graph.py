from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mai_agent.routing import classify_intent_node, detect_language_node, fallback_node
from mai_agent.skills.handoff import (
    create_handoff_node,
    handoff_guard_node,
    human_wait_node,
    route_handoff,
)
from mai_agent.skills.order import order_node
from mai_agent.skills.product import PRODUCT_TOOLS, assistant_node
from mai_agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("session", {}).get("current_intent", "unknown")
    if intent == "order":
        return "order"
    if intent in {"product_question", "recommendation"}:
        return "assistant"
    return "fallback"


def create_agent_graph(*, checkpointer=None):
    """Tạo và compile LangGraph cho agent Mai."""
    builder = StateGraph(AgentState)

    builder.add_node("detect_language", detect_language_node)
    builder.add_node("handoff_guard", handoff_guard_node)
    builder.add_node("create_handoff", create_handoff_node)
    builder.add_node("human_wait", human_wait_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("assistant", assistant_node)
    builder.add_node("product_tools", ToolNode(PRODUCT_TOOLS))
    builder.add_node("order", order_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "handoff_guard")
    builder.add_conditional_edges(
        "handoff_guard",
        route_handoff,
        {
            "create": "create_handoff",
            "wait": "human_wait",
            "continue": "classify_intent",
        },
    )
    builder.add_edge("create_handoff", END)
    builder.add_edge("human_wait", END)
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "assistant": "assistant",
            "order": "order",
            "fallback": "fallback",
        },
    )
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "product_tools",
            "__end__": END,
        },
    )
    builder.add_edge("product_tools", "assistant")
    builder.add_edge("order", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
