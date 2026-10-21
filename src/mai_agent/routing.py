from typing import Any, Literal, get_args

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.llm import invoke_tool
from mai_agent.reply_generator import generate_reply
from mai_agent.state import (
    AgentState,
    ConversationStep,
    Intent,
    LanguageCode,
    LanguageSignal,
    SessionState,
)


def latest_user_text(state: AgentState) -> str:
    """Lấy nội dung tin nhắn gần nhất của người dùng."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


@tool
def explicit_language_request(language: Literal["vi", "en"]) -> str:
    """Người dùng trực tiếp yêu cầu bot trả lời bằng tiếng Việt hoặc tiếng Anh."""
    return language


@tool
def clear_message_language(language: Literal["vi", "en"]) -> str:
    """Không có yêu cầu đổi tiếng, nhưng nội dung đủ rõ là tiếng Việt hoặc tiếng Anh."""
    return language


@tool
def ambiguous_language_signal() -> str:
    """Tin nhắn quá ngắn, dùng chung nhiều ngôn ngữ, là mã/số hoặc không phải Việt/Anh."""
    return "ambiguous"


LANGUAGE_SIGNAL_TOOLS = [
    explicit_language_request,
    clear_message_language,
    ambiguous_language_signal,
]


def _interpret_language_signal(
    text: str,
    current_language: LanguageCode,
) -> tuple[LanguageSignal, LanguageCode | None]:
    """Phân loại lý do đổi/giữ ngôn ngữ bằng tool-calling của model."""
    call = invoke_tool(
        LANGUAGE_SIGNAL_TOOLS,
        [
            SystemMessage(
                content=(
                    "Phân loại tín hiệu ngôn ngữ của tin nhắn và bắt buộc gọi đúng một "
                    "tool. Chỉ hỗ trợ vi và en. Ưu tiên cao nhất là "
                    "explicit_language_request khi người dùng trực tiếp yêu cầu bot trả "
                    "lời, chuyển hoặc tiếp tục bằng một ngôn ngữ; tham số language là "
                    "ngôn ngữ họ yêu cầu, không phải ngôn ngữ câu yêu cầu. Nếu không có "
                    "yêu cầu trực tiếp và câu đủ rõ, gọi clear_message_language theo ngôn "
                    "ngữ nội dung. Gọi ambiguous_language_signal cho lời xác nhận hoặc "
                    "cảm thán ngắn dùng chung như ok/okay, mã SKU, con số, ký hiệu, nội "
                    "dung quá ngắn để chắc chắn, hoặc ngôn ngữ ngoài vi/en. Với tín hiệu "
                    "mập mờ phải giữ ngôn ngữ hiện tại. Không tự trả lời người dùng."
                )
            ),
            HumanMessage(
                content=(
                    f"Ngôn ngữ hiện tại: {current_language}\n"
                    f"Tin nhắn mới nhất: {text}"
                )
            ),
        ],
        error_message="Model không phân loại được tín hiệu ngôn ngữ.",
    )
    if call["name"] == "ambiguous_language_signal":
        return "ambiguous", None
    if call["name"] == "explicit_language_request":
        signal: LanguageSignal = "explicit_request"
    elif call["name"] == "clear_message_language":
        signal = "clear_language"
    else:
        raise RuntimeError(f"Model gọi tool ngôn ngữ không được hỗ trợ: {call['name']}")

    language = call["args"].get("language")
    if language not in {"vi", "en"}:
        raise RuntimeError("Model trả về ngôn ngữ không được hỗ trợ.")
    return signal, language


def detect_language_node(state: AgentState) -> dict[str, Any]:
    session = state.get("session", {})
    customer = state.get("customer", {})
    current_language = session.get("language_code")
    if current_language not in {"vi", "en"}:
        preferred_language = customer.get("preferred_language")
        current_language = preferred_language if preferred_language in {"vi", "en"} else "vi"

    signal, signaled_language = _interpret_language_signal(
        latest_user_text(state).strip(),
        current_language,
    )
    language: LanguageCode = signaled_language or current_language
    return {
        "session": {
            "language_code": language,
            "language_signal": signal,
        }
    }


@tool
def route_order() -> str:
    """Khách muốn lấy/mua sản phẩm, kể cả đã chọn sản phẩm nhưng chưa nói số lượng."""
    return "order"


@tool
def route_order_lookup() -> str:
    """Khách muốn tra cứu trạng thái hoặc thông tin của một đơn hàng đã tạo trước đó."""
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
def route_greeting() -> str:
    """Khách chỉ đang chào hỏi hoặc bắt đầu cuộc trò chuyện."""
    return "greeting"


@tool
def route_unknown() -> str:
    """Yêu cầu chưa đủ rõ hoặc nằm ngoài các nghiệp vụ được hỗ trợ."""
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
    route_greeting,
    route_order,
    route_order_lookup,
    route_recommendation,
    route_product_question,
    route_unknown,
]

SCOPE_TOOLS = [route_in_scope, route_out_of_scope]


def _is_out_of_scope(text: str, current_step: str) -> bool:
    """Dùng Gemini chặn yêu cầu ngoài phạm vi trước khi phân luồng."""
    call = invoke_tool(
        SCOPE_TOOLS,
        [
            SystemMessage(
                content=(
                    "Xác định tin nhắn có thuộc phạm vi hỗ trợ của OA Cosmetics "
                    "hay không và bắt buộc gọi đúng một tool. route_in_scope cho "
                    "các nội dung: mỹ phẩm, chăm sóc da, thành phần, công dụng, "
                    "giá, tồn kho, tư vấn sản phẩm, mua hàng, giỏ hàng, "
                    "thông tin giao hàng, thanh toán, tra đơn, giao hàng, đổi trả "
                    "hoặc khiếu nại về shop. Lời chào, kể cả cách viết thân mật "
                    "hoặc rút gọn như 'chào', 'hi', 'hello', 'alo', 'a lô', 'lô', "
                    "và câu trả lời ngắn như "
                    "'có', 'không', 'đúng', số lượng, địa chỉ hay số điện "
                    "thoại là route_in_scope khi phù hợp bước hội thoại hiện tại. "
                    "Đặc biệt, khi bước hiện tại là "
                    "awaiting_recommendation_clarification, phải hiểu câu trả lời "
                    "ngắn theo yêu cầu tư vấn đang chờ và chọn route_in_scope, trừ "
                    "khi khách chuyển rõ sang chủ đề không liên quan. "
                    "route_out_of_scope cho tin tức, thời sự, chính trị, thể thao, "
                    "thời tiết, chứng khoán, tiền mã hóa, lập trình, toán học, "
                    "lịch sử, kiến thức phổ thông hoặc yêu cầu khác không liên "
                    "quan đến hoạt động của shop. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(f"Bước hội thoại hiện tại: {current_step}\nTin nhắn khách: {text}")
            ),
        ],
        error_message="Gemini không trả kết quả kiểm tra phạm vi.",
    )
    tool_name = call["name"]
    if tool_name == "route_out_of_scope":
        return True
    if tool_name == "route_in_scope":
        return False
    raise RuntimeError(f"Gemini gọi scope tool không được hỗ trợ: {tool_name}")


def _interpret_intent(
    text: str,
    current_step: str,
    pending_context: dict[str, Any],
) -> Intent:
    call = invoke_tool(
        ROUTING_TOOLS,
        [
            SystemMessage(
                content=(
                    "Phân loại ý định chính và bắt buộc gọi đúng một routing tool; "
                    "không tự trả lời khách. Chọn route_greeting khi khách chỉ chào "
                    "hỏi hoặc bắt đầu trò chuyện, kể cả cách viết thân mật hay rút "
                    "gọn như 'chào', 'hi', 'hello', 'alo', 'a lô', 'lô'. Nếu lời "
                    "chào đi kèm một nhu cầu cụ thể thì chọn intent của nhu cầu đó, "
                    "không chọn route_greeting. Ưu tiên route_order nếu khách yêu cầu "
                    "lấy, mua hoặc chốt số lượng cụ thể, kể cả khi dùng 'loại này', "
                    "'sản phẩm này', 'loại 1', 'loại 2' hoặc kèm mã SKU. "
                    "Khi ngữ cảnh có recently_recommended_products và khách chọn một "
                    "sản phẩm cụ thể để nhận/lấy, phải chọn route_order dù khách chưa "
                    "nói số lượng; order flow sẽ hỏi số lượng ở bước sau. Không phân "
                    "loại trường hợp đã chọn mua này thành route_product_question. "
                    "Chọn route_order_lookup khi khách muốn xem danh sách, trạng thái, "
                    "thanh toán, giao hàng hoặc thông tin của đơn đã tạo trước đó. Hệ "
                    "thống tự lấy tất cả đơn theo tài khoản của phiên hiện tại, không "
                    "cần khách cung cấp mã đơn hoặc số điện thoại. "
                    "Nếu khách chỉ hỏi thông tin thì chọn route_product_question. "
                    "Nếu khách cần lựa chọn sản phẩm theo da hoặc cung cấp thêm đặc "
                    "điểm da để tiếp tục tư vấn thì chọn route_recommendation. "
                    "Nếu bước hiện tại là awaiting_recommendation_clarification, hãy "
                    "dùng dữ liệu đang chờ để hiểu câu trả lời ngắn theo ngữ cảnh và "
                    "chọn route_recommendation, trừ khi khách chuyển rõ sang ý định khác. "
                    "Nếu đơn đang nhập dở, hãy hiểu tin nhắn dựa trên bước hiện tại "
                    "và dữ liệu đang chờ. Chọn route_order khi khách đang cung cấp, "
                    "xác nhận, sửa, từ chối hoặc hủy thông tin của đơn. Nếu khách đổi "
                    "sang hỏi sản phẩm hoặc cần tư vấn thì chọn đúng intent tương ứng. "
                    "Nếu chưa xác định được thì chọn route_unknown; không mặc định "
                    "route_order chỉ vì đơn đang nhập dở."
                )
            ),
            HumanMessage(
                content=(
                    f"Bước hội thoại hiện tại: {current_step}\n"
                    f"Ngữ cảnh hội thoại và đơn đang chờ: {pending_context}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ],
        error_message="Gemini không chọn intent cho tin nhắn.",
    )
    tool_to_intent: dict[str, Intent] = {
        "route_greeting": "greeting",
        "route_order": "order",
        "route_order_lookup": "order_lookup",
        "route_recommendation": "recommendation",
        "route_product_question": "product_question",
        "route_unknown": "unknown",
    }
    tool_name = call["name"]
    if tool_name not in tool_to_intent:
        raise RuntimeError(f"Gemini gọi routing tool không được hỗ trợ: {tool_name}")
    return tool_to_intent[tool_name]


def _pending_context(session: dict[str, Any]) -> dict[str, Any]:
    context = {
        key: value
        for key, value in session.items()
        if key.startswith("pending_") and value not in (None, "", [])
    }
    if session.get("last_product_candidates"):
        context["recently_recommended_products"] = session["last_product_candidates"]
    if session.get("cart"):
        context["current_cart"] = session["cart"]
    return context


def classify_intent_node(state: AgentState) -> dict[str, Any]:
    text = latest_user_text(state)
    session = state.get("session", {})
    current_step = session.get("current_step", "idle")
    if current_step not in get_args(ConversationStep):
        current_step = "idle"
    if _is_out_of_scope(text, current_step):
        intent: Intent = "unknown"
    else:
        intent = _interpret_intent(text, current_step, _pending_context(session))
    session_update: SessionState = {
        "session_id": session.get("session_id", state.get("user_id", "anonymous")),
        "language_code": session.get("language_code", "vi"),
        "current_intent": intent,
        "current_step": current_step,
        "cart": session.get("cart", []),
        "handoff_status": session.get("handoff_status", "none"),
    }
    update: dict[str, Any] = {"session": session_update}
    customer = find_customer_by_zalo_id(state.get("user_id", ""))
    if customer is not None:
        update["customer"] = customer
    return update


def greeting_node(state: AgentState) -> dict[str, Any]:
    """Tạo lời chào tự nhiên mà không cố định nguyên văn câu trả lời."""
    customer_name = str(state.get("customer", {}).get("name") or "").strip()
    name_context = (
        f"Tên khách đã biết là {customer_name}; có thể gọi tên một cách tự nhiên. "
        if customer_name
        else "Không biết tên khách nên không tự đặt hoặc đoán tên. "
    )
    reply = generate_reply(
        "Khách vừa chào hoặc bắt đầu cuộc trò chuyện. "
        f"{name_context}"
        "Chào lại ngắn gọn, thân thiện với vai trò Mai của OA Cosmetics và mở lời "
        "để khách chia sẻ nhu cầu về sản phẩm, chăm sóc da hoặc đơn hàng. Không liệt "
        "kê dài dòng và không nói rằng yêu cầu của khách nằm ngoài phạm vi.",
        response_language=state.get("session", {}).get("language_code", "vi"),
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}


def fallback_node(state: AgentState) -> dict[str, Any]:
    session = state.get("session", {})
    current_step = session.get("current_step", "idle")
    if current_step != "idle":
        pending_context = _pending_context(session)
        reply = generate_reply(
            "Chưa xác định được khách muốn tiếp tục đơn hay chuyển sang yêu cầu khác. "
            "Không tự suy diễn, không thay đổi giỏ và không xóa dữ liệu đã thu thập. "
            f"Bước hiện tại là {current_step}; dữ liệu đang chờ là {pending_context}. "
            "Nói rằng tiến trình vẫn được giữ và hỏi một câu ngắn để khách làm rõ ý định.",
            response_language=session.get("language_code", "vi"),
        )
        return {"reply": reply, "messages": [AIMessage(content=reply)]}

    reply = generate_reply(
        "Yêu cầu của khách nằm ngoài phạm vi hỗ trợ hoặc chưa đủ "
        "rõ. Thông báo ngắn gọn rằng Mai chỉ hỗ trợ thông tin mỹ phẩm, "
        "tư vấn chăm sóc da, đặt đơn mới và xem đơn của tài khoản hiện tại. Không trả "
        "lời nội dung ngoài phạm vi. Mời khách đặt câu hỏi liên quan "
        "đến sản phẩm hoặc nhu cầu chăm sóc da.",
        response_language=state.get("session", {}).get("language_code", "vi"),
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
