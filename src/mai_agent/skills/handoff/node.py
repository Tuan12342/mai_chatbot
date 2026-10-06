from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.order_store import find_orders_by_customer_id
from mai_agent.reply_generator import generate_reply
from mai_agent.routing import latest_user_text
from mai_agent.skills.handoff.store import (
    append_customer_message,
    get_handoff,
    save_handoff,
)
from mai_agent.skills.handoff.tools import HANDOFF_TOOLS
from mai_agent.state import AgentState, HandoffAction

ACTIVE_HANDOFF_STATUSES = {"waiting_for_human", "human_active"}


def _conversation_text(state: AgentState, *, limit: int = 20) -> str:
    lines: list[str] = []
    for message in state.get("messages", [])[-limit:]:
        if isinstance(message, HumanMessage):
            role = "Khách"
        elif isinstance(message, AIMessage):
            role = "Mai"
        else:
            continue
        content = str(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _assess_latest_message(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(HANDOFF_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Đánh giá tin nhắn MỚI NHẤT của khách trong ngữ cảnh hội thoại và "
                    "bắt buộc gọi assess_handoff. Hiểu theo ngữ nghĩa tiếng Việt, không "
                    "dò hay phụ thuộc vào danh sách từ khóa. Phân biệt việc khách từ chối "
                    "một sản phẩm với phàn nàn thực sự. requests_human chỉ dựa trên ý muốn "
                    "được người thật tiếp nhận. Giữ issue_type nhất quán khi khách nhắc lại "
                    "cùng một vấn đề. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Hội thoại gần đây:\n{_conversation_text(state)}\n\n"
                    f"Tin nhắn cần đánh giá: {latest_user_text(state)}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không trả kết quả đánh giá handoff.")
    return dict(response.tool_calls[0]["args"])


def _serialize_messages(state: AgentState, *, limit: int = 30) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in state.get("messages", [])[-limit:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = str(message.content).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def handoff_guard_node(state: AgentState) -> dict[str, Any]:
    handoff = state.get("handoff", {})
    handoff_id = handoff.get("handoff_id")
    if handoff.get("status") in ACTIVE_HANDOFF_STATUSES and handoff_id:
        stored = get_handoff(handoff_id)
        stored_status = stored.get("status") if stored else handoff.get("status")
        if stored_status in ACTIVE_HANDOFF_STATUSES:
            append_customer_message(handoff_id, latest_user_text(state))
            return {
                "handoff_action": "wait",
                "handoff": {"status": stored_status},
            }

        if stored_status == "resolved":
            issue_type = handoff.get("issue_type", "other")
            counts = dict(handoff.get("complaint_counts", {}))
            counts[issue_type] = 0
            return {
                "handoff_action": "continue",
                "handoff": {
                    "status": "resolved",
                    "complaint_counts": counts,
                    "resolution_summary": str(stored.get("resolution_summary", "")),
                },
            }

    assessment = _assess_latest_message(state)
    issue_type = str(assessment.get("issue_type", "none"))
    counts = dict(handoff.get("complaint_counts", {}))
    if assessment.get("issue_resolved") and issue_type != "none":
        counts[issue_type] = 0
    elif assessment.get("is_complaint") and issue_type != "none":
        counts[issue_type] = counts.get(issue_type, 0) + 1

    complaint_count = counts.get(issue_type, 0)
    should_handoff = bool(
        assessment.get("requests_human")
        or assessment.get("negative_intensity") == "strong"
        or (
            assessment.get("is_complaint")
            and complaint_count >= 2
            and not assessment.get("issue_resolved")
        )
    )
    action: HandoffAction = "create" if should_handoff else "continue"
    return {
        "handoff_action": action,
        "handoff": {
            "status": handoff.get("status", "inactive"),
            "issue_type": issue_type,
            "reason": str(assessment.get("reason", "")),
            "severity": str(assessment.get("negative_intensity", "none")),
            "complaint_count": complaint_count,
            "complaint_counts": counts,
        },
    }


def route_handoff(state: AgentState) -> HandoffAction:
    return state.get("handoff_action", "continue")


def create_handoff_node(state: AgentState) -> dict[str, Any]:
    now = datetime.now().astimezone().isoformat()
    handoff = state.get("handoff", {})
    handoff_id = f"HO-{datetime.now():%Y%m%d}-{uuid4().hex[:8].upper()}"
    user_id = state.get("user_id", "anonymous")
    customer = state.get("customer") or find_customer_by_zalo_id(user_id) or {}
    customer_id = str(customer.get("zalo_user_id", user_id))
    package = {
        "handoff_id": handoff_id,
        "user_id": user_id,
        "status": "waiting_for_human",
        "reason": handoff.get("reason", "Khách cần người thật hỗ trợ."),
        "issue_type": handoff.get("issue_type", "other"),
        "severity": handoff.get("severity", "none"),
        "complaint_count": handoff.get("complaint_count", 0),
        "customer": customer,
        "conversation_history": _serialize_messages(state),
        "current_cart": state.get("session", {}).get("cart", []),
        "current_step": state.get("session", {}).get("current_step", "idle"),
        "active_order": state.get("active_order", {}),
        "related_orders": find_orders_by_customer_id(customer_id),
        "messages_after_handoff": [],
        "created_at": now,
        "updated_at": now,
    }
    save_handoff(package)
    reply = generate_reply(
        "Thông báo ngắn gọn rằng yêu cầu cùng lịch sử hội thoại và thông tin đơn liên "
        "quan đã được chuyển đầy đủ cho chủ shop; người thật sẽ tiếp nhận hỗ trợ. Không "
        "hỏi khách cung cấp lại thông tin và không tự hứa thời gian phản hồi cụ thể.",
        response_language=state.get("session", {}).get("language_code", "vi"),
    )
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "handoff": {
            "handoff_id": handoff_id,
            "status": "waiting_for_human",
            "created_at": now,
        },
    }


def human_wait_node(_: AgentState) -> dict[str, Any]:
    """Giữ im lặng trong lúc người thật đang tiếp nhận nhưng vẫn lưu tin nhắn vào state."""
    return {"reply": ""}
