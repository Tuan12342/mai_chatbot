from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mai_agent.handoff import (
    handoff_guard_node as session_handoff_guard_node,
)
from mai_agent.handoff import (
    handoff_node as session_handoff_node,
)
from mai_agent.policy import policy_guard_node, route_after_policy_guard
from mai_agent.profile_memory import (
    profile_confirmation_node,
    profile_memory_node,
    route_after_profile_confirmation,
    route_after_profile_memory,
)
from mai_agent.routing import classify_intent_node, detect_language_node, fallback_node
from mai_agent.skills.handoff import (
    create_handoff_node,
    handoff_guard_node,
    human_wait_node,
    route_handoff,
)
from mai_agent.skills.order import order_node
from mai_agent.skills.order_lookup import order_lookup_node
from mai_agent.skills.product import PRODUCT_TOOLS, assistant_node, recommendation_node
from mai_agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    intent = state.get("session", {}).get("current_intent", "unknown")
    if intent == "order":
        return "order"
    if intent == "order_lookup":
        return "order_lookup"
    if intent in {"product_question", "recommendation"}:
        return "profile_memory" if intent == "recommendation" else "assistant"
    return "fallback"


def route_after_handoff_guard(state: AgentState) -> str:
    """Route cho giao diện handoff thử nghiệm dùng state ngay trong session."""
    status = state.get("session", {}).get("handoff_status", "none")
    if status == "requested":
        return "handoff"
    if status == "active":
        return "human_active"
    return "classify_intent"


def create_agent_graph(*, checkpointer=None, handoff_backend: str = "persistent"):
    """Tạo và compile LangGraph cho agent Mai.

    ``persistent`` dùng kho handoff nghiệp vụ hiện tại. ``session`` dành cho giao
    diện handoff lab, nơi nhân viên cập nhật trực tiếp checkpoint của phiên.
    """
    builder = StateGraph(AgentState)

    builder.add_node("detect_language", detect_language_node)
    builder.add_node("policy_guard", policy_guard_node)
    builder.add_node("profile_confirmation", profile_confirmation_node)
    builder.add_node("profile_memory", profile_memory_node)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("assistant", assistant_node)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("product_tools", ToolNode(PRODUCT_TOOLS))
    builder.add_node("order", order_node)
    builder.add_node("order_lookup", order_lookup_node)
    builder.add_node("fallback", fallback_node)

    if handoff_backend == "session":
        builder.add_node("handoff_guard", session_handoff_guard_node)
        builder.add_node("handoff", session_handoff_node)
    else:
        builder.add_node("handoff_guard", handoff_guard_node)
        builder.add_node("create_handoff", create_handoff_node)
        builder.add_node("human_wait", human_wait_node)

    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "policy_guard")
    builder.add_conditional_edges(
        "policy_guard",
        route_after_policy_guard,
        {
            "end": END,
            "handoff_guard": "handoff_guard",
            "profile_confirmation": "profile_confirmation",
        },
    )

    if handoff_backend == "session":
        builder.add_conditional_edges(
            "handoff_guard",
            route_after_handoff_guard,
            {
                "handoff": "handoff",
                "human_active": END,
                "classify_intent": "profile_confirmation",
            },
        )
        builder.add_edge("handoff", END)
    else:
        builder.add_conditional_edges(
            "handoff_guard",
            route_handoff,
            {
                "create": "create_handoff",
                "wait": "human_wait",
                "continue": "profile_confirmation",
            },
        )
        builder.add_edge("create_handoff", END)
        builder.add_edge("human_wait", END)

    builder.add_conditional_edges(
        "profile_confirmation",
        route_after_profile_confirmation,
        {
            "end": END,
            "classify_intent": "classify_intent",
            "recommendation": "recommendation",
        },
    )
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "assistant": "assistant",
            "profile_memory": "profile_memory",
            "order": "order",
            "order_lookup": "order_lookup",
            "fallback": "fallback",
        },
    )
    builder.add_conditional_edges(
        "profile_memory",
        route_after_profile_memory,
        {"end": END, "assistant": "assistant", "recommendation": "recommendation"},
    )
    builder.add_edge("recommendation", END)
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
