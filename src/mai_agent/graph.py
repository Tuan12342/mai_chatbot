from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mai_agent.handoff import handoff_guard_node, handoff_node
from mai_agent.routing import classify_intent_node, fallback_node
from mai_agent.skills.order import order_node
from mai_agent.skills.order_lookup import order_lookup_node
from mai_agent.skills.product import PRODUCT_TOOLS, assistant_node
from mai_agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("session", {}).get("current_intent", "unknown")
    if intent == "order":
        return "order"
    if intent == "order_lookup":
        return "order_lookup"
    if intent in {"product_question", "recommendation"}:
        return "assistant"
    return "fallback"


def route_after_handoff_guard(state: AgentState) -> str:
    status = state.get("session", {}).get("handoff_status", "none")
    if status == "requested":
        return "handoff"
    if status == "active":
        return "human_active"
    return "classify_intent"


def create_agent_graph(*, checkpointer=None):
    """Tạo và compile LangGraph cho agent Mai."""
    builder = StateGraph(AgentState)

    builder.add_node("handoff_guard", handoff_guard_node)
    builder.add_node("handoff", handoff_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("assistant", assistant_node)
    builder.add_node("product_tools", ToolNode(PRODUCT_TOOLS))
    builder.add_node("order", order_node)
    builder.add_node("order_lookup", order_lookup_node)
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "handoff_guard")
    builder.add_conditional_edges(
        "handoff_guard",
        route_after_handoff_guard,
        {
            "handoff": "handoff",
            "human_active": END,
            "classify_intent": "classify_intent",
        },
    )
    builder.add_edge("handoff", END)
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "assistant": "assistant",
            "order": "order",
            "order_lookup": "order_lookup",
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
    builder.add_edge("order_lookup", END)
    builder.add_edge("fallback", END)

    return builder.compile(checkpointer=checkpointer)
