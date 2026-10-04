import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.reply_generator import generate_reply
from mai_agent.state import AgentState

HandoffTrigger = Literal[
    "direct_request",
    "strong_negative_unresolved",
    "repeated_unresolved_complaint",
]


@tool
def request_human_handoff(reason: str, trigger: HandoffTrigger) -> str:
    """Chuyển cho nhân viên khi hội thoại thỏa một trigger handoff rõ ràng."""
    return f"{trigger}: {reason}"


@tool
def continue_with_ai(reason: str) -> str:
    """Tiếp tục để AI xử lý khi chưa đủ căn cứ cần nhân viên thật."""
    return reason


HANDOFF_TOOLS = [request_human_handoff, continue_with_ai]


def _decision_context(state: AgentState) -> str:
    session = state.get("session", {})
    safe_order = _safe_order_context(state)
    return (
        f"Bước hội thoại hiện tại: {session.get('current_step', 'idle')}\n"
        f"Giỏ hàng/đơn hiện tại: {safe_order}\n"
        "Lịch sử hội thoại nằm trong các message tiếp theo."
    )


def _interpret_handoff(state: AgentState) -> tuple[bool, str | None, str | None]:
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(HANDOFF_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Bạn là bộ điều phối handoff, không trả lời khách. Đọc toàn bộ ngữ cảnh "
                    "và bắt buộc gọi đúng một tool. Chỉ gọi request_human_handoff khi có ít "
                    "nhất một căn cứ rõ ràng: (1) khách yêu cầu trực tiếp gặp/gọi nhân viên "
                    "hoặc chủ shop; (2) khách thể hiện tiêu cực mạnh và vấn đề cụ thể vẫn chưa "
                    "được giải quyết; (3) cùng một khiếu nại/vấn đề đã được khách nhắc lại ít "
                    "nhất hai lần mà câu trả lời trước chưa giải quyết. Không handoff chỉ vì một "
                    "từ tiêu cực, một câu khó chịu đơn lẻ, câu hỏi bình thường, hoặc chỉ vì bot "
                    "chưa chắc câu trả lời. Hãy hiểu ngữ nghĩa và tham chiếu trong hội thoại, "
                    "không dò theo danh sách từ khóa. reason phải ngắn gọn, cụ thể, hữu ích cho "
                    "nhân viên và không chép số điện thoại, địa chỉ hay dữ liệu riêng tư."
                )
            ),
            SystemMessage(content=_decision_context(state)),
            *state.get("messages", [])[-16:],
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không đưa ra quyết định handoff.")
    call = response.tool_calls[0]
    if call["name"] == "continue_with_ai":
        return False, None, None
    if call["name"] != "request_human_handoff":
        raise RuntimeError(f"Gemini gọi handoff tool không được hỗ trợ: {call['name']}")
    arguments = call.get("args", {})
    return True, str(arguments.get("reason", "Cần nhân viên hỗ trợ.")), str(
        arguments.get("trigger", "direct_request")
    )


def _redact_private_data(text: str) -> str:
    text = re.sub(
        r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)",
        "[đã ẩn số điện thoại]",
        text,
    )
    return re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[đã ẩn email]",
        text,
        flags=re.IGNORECASE,
    )


def _recent_conversation(state: AgentState, limit: int = 12) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            speaker = "customer"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            speaker = (
                "human_agent"
                if message.additional_kwargs.get("sender") == "human_agent"
                else "assistant"
            )
        else:
            continue
        content = _redact_private_data(str(message.content).strip())
        if content:
            history.append({"speaker": speaker, "content": content})
    return history[-limit:]


def _safe_order_context(state: AgentState) -> dict[str, Any]:
    session = state.get("session", {})
    active_order = state.get("active_order", {})
    order = {
        key: active_order[key]
        for key in ("order_id", "items", "status", "payment_status", "total_amount")
        if key in active_order
    }
    return {
        "current_step": session.get("current_step", "idle"),
        "cart": session.get("cart", []),
        "active_order": order or None,
    }


def build_handoff_package(
    state: AgentState,
    *,
    reason: str | None = None,
    trigger: str | None = None,
) -> dict[str, Any]:
    """Tạo context đủ dùng cho nhân viên nhưng loại dữ liệu định danh không cần thiết."""
    session = state.get("session", {})
    return {
        "reason": reason or session.get("handoff_reason"),
        "trigger": trigger or session.get("handoff_trigger"),
        "conversation_history": _recent_conversation(state),
        "order_context": _safe_order_context(state),
    }


def handoff_guard_node(state: AgentState) -> dict[str, Any]:
    session = state.get("session", {})
    status = session.get("handoff_status", "none")

    if status == "active":
        # Vẫn nhận và cập nhật tin mới cho nhân viên, nhưng tuyệt đối không sinh trả lời AI.
        package = build_handoff_package(state)
        return {
            "reply": "",
            "response_owner": "human",
            "session": {"handoff_status": "active", "handoff_package": package},
        }

    if status == "resolved":
        return {
            "response_owner": "ai",
            "session": {
                "handoff_status": "none",
                "handoff_reason": None,
                "handoff_trigger": None,
                "handoff_package": {},
            },
        }

    should_handoff, reason, trigger = _interpret_handoff(state)
    if not should_handoff:
        return {"response_owner": "ai"}

    package = build_handoff_package(state, reason=reason, trigger=trigger)
    return {
        "response_owner": "human",
        "session": {
            "handoff_status": "requested",
            "handoff_reason": reason,
            "handoff_trigger": trigger,
            "handoff_package": package,
        },
    }


def handoff_node(state: AgentState) -> dict[str, Any]:
    reply = generate_reply(
        "Thông báo ngắn gọn rằng yêu cầu đã được chuyển cho nhân viên của shop và nhân viên "
        "đã nhận được nội dung trao đổi trước đó, nên khách không cần trình bày lại. Không hứa "
        "thời gian phản hồi cụ thể."
    )
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "response_owner": "human",
        "session": {"handoff_status": "active"},
    }


def human_agent_update(reply: str, *, resume_ai: bool = False) -> dict[str, Any]:
    """Payload để adapter inbox ghi nhận phản hồi nhân viên và mở lại bot khi đã xử lý xong."""
    return {
        "messages": [AIMessage(content=reply, additional_kwargs={"sender": "human_agent"})],
        "reply": reply,
        "response_owner": "ai" if resume_ai else "human",
        "session": {
            "handoff_status": "resolved" if resume_ai else "active",
            "handoff_resolution_summary": reply if resume_ai else None,
        },
    }
