from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from mai_agent.state import AgentState, Intent


def latest_user_text(state: AgentState) -> str:
    """Lấy nội dung tin nhắn gần nhất của người dùng."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def classify_intent_node(state: AgentState) -> dict[str, Intent]:
    """Phân loại intent ban đầu bằng luật đơn giản, dễ kiểm thử."""
    text = latest_user_text(state).lower()
    if any(word in text for word in ["xin chào", "chào", "hello", "hi"]):
        intent: Intent = "greeting"
    elif any(word in text for word in ["mua", "đặt", "chốt đơn", "order"]):
        intent = "order"
    elif any(word in text for word in ["tư vấn", "gợi ý", "phù hợp", "recommend"]):
        intent = "recommendation"
    elif any(word in text for word in ["thành phần", "công dụng", "cách dùng", "serum", "kem"]):
        intent = "product_question"
    else:
        intent = "unknown"
    return {"intent": intent}


def fallback_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Em chưa hiểu rõ ý mình. Mình có thể hỏi về sản phẩm, "
        "xin tư vấn theo loại da hoặc nói sản phẩm muốn mua nhé."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
