from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.reply_generator import generate_reply
from mai_agent.state import AgentState, Intent, SessionState


def latest_user_text(state: AgentState) -> str:
    """Lấy nội dung tin nhắn gần nhất của người dùng."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


@tool
def route_order() -> str:
    """Khách muốn mua, đặt, sửa hoặc xác nhận một đơn hàng mới."""
    return "order"


@tool
def route_order_lookup() -> str:
    """Khách muốn tra cứu đơn hàng cũ hoặc trạng thái giao hàng."""
    return "order_lookup"


@tool
def route_recommendation() -> str:
    """Khách cần tư vấn hoặc gợi ý sản phẩm theo nhu cầu và tình trạng da."""
    return "recommendation"


@tool
def route_product_question() -> str:
    """Khách hỏi thông tin, giá, thành phần, công dụng hoặc tồn kho sản phẩm."""
    return "product_question"


@tool
def route_unknown() -> str:
    """Yêu cầu không thuộc các năng lực mua hàng, tra đơn hay tư vấn sản phẩm."""
    return "unknown"


ROUTING_TOOLS = [
    route_order,
    route_order_lookup,
    route_recommendation,
    route_product_question,
    route_unknown,
]


def _interpret_intent(text: str) -> Intent:
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(ROUTING_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Phân loại ý định chính và bắt buộc gọi đúng một routing tool; "
                    "không tự trả lời khách. Ưu tiên route_order nếu khách yêu cầu "
                    "lấy, mua hoặc chốt số lượng cụ thể, kể cả khi dùng 'loại này', "
                    "'sản phẩm này', 'loại 1', 'loại 2' hoặc kèm mã SKU. Ví dụ, "
                    "'lấy 5 chai loại này và 10 OA003' bắt buộc là route_order. "
                    "Nếu khách chỉ hỏi thông tin thì chọn route_product_question. "
                    "Nếu khách cần lựa chọn sản phẩm theo da hoặc cung cấp thêm đặc "
                    "điểm da để tiếp tục tư vấn thì chọn route_recommendation."
                )
            ),
            HumanMessage(content=text),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn intent cho tin nhắn.")
    tool_to_intent: dict[str, Intent] = {
        "route_order": "order",
        "route_order_lookup": "order_lookup",
        "route_recommendation": "recommendation",
        "route_product_question": "product_question",
        "route_unknown": "unknown",
    }
    tool_name = response.tool_calls[0]["name"]
    if tool_name not in tool_to_intent:
        raise RuntimeError(f"Gemini gọi routing tool không được hỗ trợ: {tool_name}")
    return tool_to_intent[tool_name]


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    text = latest_user_text(state)
    session = state.get("session", {})
    if session.get("current_step") == "verifying_order_lookup":
        intent: Intent = "order_lookup"
    elif session.get("current_step", "idle") != "idle":
        intent: Intent = "order"
    else:
        intent = _interpret_intent(text)
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
    reply = generate_reply(
        "Chưa hiểu rõ yêu cầu của khách. Hỏi lại một cách thân thiện và gợi ý "
        "khách có thể hỏi về sản phẩm, xin tư vấn theo loại da hoặc nói sản phẩm muốn mua."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
