from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from mai_agent.config import get_settings
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.state import AgentState, Intent


def latest_user_text(state: AgentState) -> str:
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


def greeting_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Chào mình, em là Mai của OA Cosmetics. "
        "Em có thể giúp mình tìm hiểu sản phẩm hoặc tư vấn theo loại da nhé."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


def order_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Mình cho em tên sản phẩm và số lượng muốn đặt nhé. "
        "Ở bước tiếp theo em sẽ bổ sung giỏ hàng và state machine cho đơn hàng."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


def fallback_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Em chưa hiểu rõ ý mình. Mình có thể hỏi về sản phẩm, "
        "xin tư vấn theo loại da hoặc nói sản phẩm muốn mua nhé."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


def assistant_node(state: AgentState) -> dict[str, Any]:
    """Dùng LangChain ChatOpenAI nếu có key, nếu không trả lời ở chế độ demo."""
    settings = get_settings()
    user_text = latest_user_text(state)

    if settings.demo_mode or not settings.openai_api_key:
        reply = _demo_reply(state.get("intent", "unknown"), user_text)
    else:
        model = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0.2,
        )
        response = model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Intent đã nhận diện: {state.get('intent', 'unknown')}\n"
                        f"Khách hỏi: {user_text}"
                    )
                ),
            ]
        )
        reply = str(response.content)

    return {"reply": reply, "messages": [AIMessage(content=reply)]}


def _demo_reply(intent: Intent, user_text: str) -> str:
    if intent == "recommendation":
        return (
            "Để tư vấn phù hợp, mình cho em biết loại da, "
            "vấn đề đang quan tâm và thành phần từng bị kích ứng nhé."
        )
    if intent == "product_question":
        return (
            f"Em đã ghi nhận câu hỏi “{user_text}”. "
            "Hiện agent lõi chưa nối catalog/RAG nên em chưa dùng dữ liệu sản phẩm để trả lời."
        )
    return "Em đã nhận được câu hỏi của mình nhưng chưa đủ thông tin để xử lý chính xác."
