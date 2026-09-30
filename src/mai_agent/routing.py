import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.state import AgentState, Intent, SessionState


def latest_user_text(state: AgentState) -> str:
    """Lấy nội dung tin nhắn gần nhất của người dùng."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    text = latest_user_text(state).lower()
    session = state.get("session", {})
    if session.get("current_step") == "verifying_order_lookup":
        intent: Intent = "order_lookup"
    elif session.get("current_step", "idle") != "idle":
        intent: Intent = "order"
    elif any(word in text for word in ["xin chào", "chào", "hello", "hi"]):
        intent: Intent = "greeting"
    elif any(
        phrase in text
        for phrase in [
            "tra đơn",
            "đơn cũ",
            "đơn của tôi",
            "đơn của mình",
            "kiểm tra đơn hàng",
            "đơn đang giao",
        ]
    ):
        intent = "order_lookup"
    elif any(word in text for word in ["mua", "đặt", "chốt đơn", "order"]):
        intent = "order"
    elif any(word in text for word in ["tư vấn", "gợi ý", "phù hợp", "recommend"]):
        intent = "recommendation"
    elif (
        any(
            word in text
            for word in [
                "sản phẩm",
                "thành phần",
                "công dụng",
                "cách dùng",
                "giá",
                "tồn kho",
                "còn hàng",
                "serum",
                "kem",
            ]
        )
        or re.search(r"\boa\d+\b", text) is not None
    ):
        intent = "product_question"
    else:
        intent = "unknown"
    session_update: SessionState = {
        "session_id": session.get("session_id", state.get("user_id", "anonymous")),
        "current_intent": intent,
        "current_step": session.get("current_step", "idle"),
        "cart": session.get("cart", []),
    }
    update: dict[str, Any] = {"session": session_update}
    customer = find_customer_by_zalo_id(state.get("user_id", ""))
    if customer is not None:
        update["customer"] = customer
    return update


def fallback_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Em chưa hiểu rõ ý mình. Mình có thể hỏi về sản phẩm, "
        "xin tư vấn theo loại da hoặc nói sản phẩm muốn mua nhé."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
