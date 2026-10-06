from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.reply_generator import generate_reply
from mai_agent.state import AgentState, Intent, LanguageCode, SessionState


def latest_user_text(state: AgentState) -> str:
    """Lấy nội dung tin nhắn gần nhất của người dùng."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


@tool
def use_vietnamese() -> str:
    """Tin nhắn chủ yếu bằng tiếng Việt."""
    return "vi"


@tool
def use_english() -> str:
    """Tin nhắn chủ yếu bằng tiếng Anh."""
    return "en"


@tool
def keep_current_language() -> str:
    """Tin nhắn không đủ ngôn ngữ để xác định, giữ ngôn ngữ hiện tại."""
    return "keep"


LANGUAGE_TOOLS = [use_vietnamese, use_english, keep_current_language]


def detect_language_node(state: AgentState) -> dict[str, Any]:
    """Phát hiện ngôn ngữ lượt hiện tại mà không làm mất state nghiệp vụ."""
    session = state.get("session", {})
    customer = state.get("customer", {})
    current_language = session.get("language_code")
    if current_language not in {"vi", "en"}:
        preferred_language = customer.get("preferred_language")
        current_language = preferred_language if preferred_language in {"vi", "en"} else "vi"

    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(LANGUAGE_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Xác định ngôn ngữ mà trợ lý phải dùng cho lượt trả lời tiếp theo "
                    "và bắt buộc gọi đúng một tool. Gọi use_vietnamese khi tin nhắn "
                    "thể hiện rõ tiếng Việt; gọi use_english khi thể hiện rõ tiếng Anh. "
                    "Gọi keep_current_language với nội dung trung tính hoặc không đủ "
                    "bằng chứng như SKU, tên sản phẩm, tên INCI, con số, số lượng, số "
                    "điện thoại, địa chỉ ngắn, emoji, hoặc câu xác nhận rất ngắn. Việc "
                    "đổi ngôn ngữ không được hiểu là bắt đầu phiên mới. Không trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Ngôn ngữ hiện tại: {current_language}\n"
                    f"Tin nhắn mới: {latest_user_text(state)}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không trả kết quả nhận diện ngôn ngữ.")

    tool_name = response.tool_calls[0]["name"]
    language_by_tool: dict[str, LanguageCode] = {
        "use_vietnamese": "vi",
        "use_english": "en",
        "keep_current_language": current_language,
    }
    if tool_name not in language_by_tool:
        raise RuntimeError(f"Gemini gọi language tool không được hỗ trợ: {tool_name}")
    return {"session": {"language_code": language_by_tool[tool_name]}}


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


@tool
def route_in_scope() -> str:
    """Tin nhắn thuộc phạm vi mỹ phẩm, chăm sóc da, mua hàng hoặc đơn hàng."""
    return "in_scope"


@tool
def route_out_of_scope() -> str:
    """Tin nhắn không thuộc phạm vi hỗ trợ của OA Cosmetics."""
    return "out_of_scope"


ROUTING_TOOLS = [
    route_order,
    route_order_lookup,
    route_recommendation,
    route_product_question,
    route_unknown,
]

SCOPE_TOOLS = [route_in_scope, route_out_of_scope]


def _is_out_of_scope(text: str, current_step: str) -> bool:
    """Dùng Gemini chặn yêu cầu ngoài phạm vi trước khi phân luồng."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(SCOPE_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Xác định tin nhắn có thuộc phạm vi hỗ trợ của OA Cosmetics "
                    "hay không và bắt buộc gọi đúng một tool. route_in_scope cho "
                    "các nội dung: mỹ phẩm, chăm sóc da, thành phần, công dụng, "
                    "giá, tồn kho, tư vấn sản phẩm, mua hàng, giỏ hàng, "
                    "thông tin giao hàng, thanh toán, tra đơn, giao hàng, đổi trả "
                    "hoặc khiếu nại về shop. Lời chào và câu trả lời ngắn như "
                    "'có', 'không', 'đúng', số lượng, địa chỉ hay số điện "
                    "thoại là route_in_scope khi phù hợp bước hội thoại hiện tại. "
                    "route_out_of_scope cho tin tức, thời sự, chính trị, thể thao, "
                    "thời tiết, chứng khoán, tiền mã hóa, lập trình, toán học, "
                    "lịch sử, kiến thức phổ thông hoặc yêu cầu khác không liên "
                    "quan đến hoạt động của shop. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Bước hội thoại hiện tại: {current_step}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không trả kết quả kiểm tra phạm vi.")
    tool_name = response.tool_calls[0]["name"]
    if tool_name == "route_out_of_scope":
        return True
    if tool_name == "route_in_scope":
        return False
    raise RuntimeError(f"Gemini gọi scope tool không được hỗ trợ: {tool_name}")


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
    current_step = session.get("current_step", "idle")
    if _is_out_of_scope(text, current_step):
        intent: Intent = "unknown"
    elif current_step == "verifying_order_lookup":
        intent = "order_lookup"
    elif current_step != "idle":
        intent = "order"
    else:
        intent = _interpret_intent(text)
    session_update: SessionState = {
        "session_id": session.get("session_id", state.get("user_id", "anonymous")),
        "language_code": session.get("language_code", "vi"),
        "current_intent": intent,
        "current_step": current_step,
        "cart": session.get("cart", []),
    }
    update: dict[str, Any] = {"session": session_update}
    customer = find_customer_by_zalo_id(state.get("user_id", ""))
    if customer is not None:
        update["customer"] = customer
    return update


def fallback_node(state: AgentState) -> dict[str, Any]:
    reply = generate_reply(
        "Yêu cầu của khách nằm ngoài phạm vi hỗ trợ hoặc chưa đủ "
        "rõ. Thông báo ngắn gọn rằng Mai chỉ hỗ trợ thông tin mỹ phẩm, "
        "tư vấn chăm sóc da, mua hàng và tra cứu đơn hàng. Không trả "
        "lời nội dung ngoài phạm vi. Mời khách đặt câu hỏi liên quan "
        "đến sản phẩm hoặc nhu cầu chăm sóc da.",
        response_language=state.get("session", {}).get("language_code", "vi"),
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
