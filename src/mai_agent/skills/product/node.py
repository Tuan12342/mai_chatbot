from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.routing import latest_user_text
from mai_agent.state import AgentState, Intent


def assistant_node(state: AgentState) -> dict[str, Any]:
    """Trả lời câu hỏi sản phẩm bằng Gemma hoặc phản hồi demo."""
    settings = get_settings()
    user_text = latest_user_text(state)

    if settings.demo_mode or not settings.google_api_key:
        reply = _demo_reply(state.get("intent", "unknown"), user_text)
    else:
        model = ChatGoogleGenerativeAI(
            model=settings.google_model,
            api_key=settings.google_api_key,
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
            f"Mình đã ghi nhận câu hỏi “{user_text}”. "
            "Hiện agent lõi chưa nối catalog/RAG nên mình chưa dùng dữ liệu sản phẩm để trả lời."
        )
    return "Em đã nhận được câu hỏi của mình nhưng chưa đủ thông tin để xử lý chính xác."
