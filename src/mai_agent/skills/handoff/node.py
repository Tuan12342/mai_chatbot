from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.llm import invoke_tool
from mai_agent.order_store import find_orders_by_customer_id
from mai_agent.policy import RESPONSE_GUIDANCE, triage_requires_handoff
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
    roles = {"user": "Khách", "assistant": "Mai"}
    return "\n".join(
        f"{roles[message['role']]}: {message['content']}"
        for message in _serialize_messages(state, limit=limit)
    )


def _assess_latest_message(state: AgentState) -> dict[str, Any]:
    call = invoke_tool(
        HANDOFF_TOOLS,
        [
            SystemMessage(
                content=(
                    "Đánh giá tin nhắn MỚI NHẤT bằng schema ngữ nghĩa tổng quát và bắt "
                    "buộc gọi assess_message. Không dò từ khóa hoặc đối chiếu câu mẫu. "
                    "Phân biệt mục tiêu của khách, thao tác họ yêu cầu và thẩm quyền xử "
                    "lý: allowed nếu bot tự xử lý được, needs_human nếu cần phê duyệt hoặc "
                    "xác minh ngoài dữ liệu, prohibited nếu phải từ chối. is_complaint chỉ "
                    "đúng với phản ánh sự cố hoặc không hài lòng; severity=strong chỉ khi "
                    "có cường độ hoặc hậu quả đáng kể. explicit_human_preference chỉ đúng "
                    "khi khách muốn người thật tiếp nhận. Giữ issue_type nhất quán cho "
                    "cùng một vấn đề và không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Hội thoại gần đây:\n{_conversation_text(state)}\n\n"
                    f"Tin nhắn cần đánh giá: {latest_user_text(state)}"
                )
            ),
        ],
        error_message="Gemini không trả kết quả đánh giá handoff.",
    )
    return dict(call["args"])


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

    assessment = dict(state.get("triage") or _assess_latest_message(state))
    issue_type = str(assessment.get("issue_type", "none"))
    counts = dict(handoff.get("complaint_counts", {}))
    if assessment.get("issue_resolved") and issue_type != "none":
        counts[issue_type] = 0
    elif assessment.get("is_complaint") and issue_type != "none":
        counts[issue_type] = counts.get(issue_type, 0) + 1

    complaint_count = counts.get(issue_type, 0)
    should_handoff = triage_requires_handoff(
        assessment,
        complaint_count=complaint_count,
    )
    action: HandoffAction = "create" if should_handoff else "continue"
    return {
        "handoff_action": action,
        "handoff": {
            "status": handoff.get("status", "inactive"),
            "issue_type": issue_type,
            "reason": str(assessment.get("reason", "")),
            "severity": str(assessment.get("severity", "none")),
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
    issue_type = str(handoff.get("issue_type", "other"))
    followup_by_issue = {
        "allergy_safety": (
            "Khuyên khách ngưng sản phẩm ngay; nếu phản ứng nặng, lan rộng hoặc đau rát, "
            "khuyên đi khám bác sĩ. Không chẩn đoán và không nhận lỗi thay shop."
        ),
        "payment": (
            "Đề nghị khách gửi ảnh giao dịch đã che thông tin bảo mật để chủ shop đối "
            "chiếu; không yêu cầu mật khẩu, OTP, số thẻ hoặc CVV."
        ),
        "delivery": (
            "Ghi nhận đúng sự cố giao nhận mà khách đã mô tả; nói rằng thông tin đơn và "
            "giao nhận liên quan đã được đính kèm để người thật kiểm tra. Không hỏi khách "
            "lặp lại vấn đề vừa nêu và không tự hứa thời gian giao hoặc bồi thường."
        ),
        "product_quality": (
            "Nếu phản ánh liên quan bao bì, hạn dùng hoặc chất lượng hàng nhận được, "
            "đề nghị khách gửi ảnh sản phẩm/bao bì làm bằng chứng; không tự kết luận."
        ),
        "refund": (
            "Đề nghị khách gửi ảnh sản phẩm hoặc chứng từ liên quan nếu có để người thật "
            "xử lý; không tự hứa hình thức hay thời điểm hoàn tiền."
        ),
        "service": "Mời khách mô tả ngắn gọn sự việc để chuyển kèm cho chủ shop.",
        "wrong_item": (
            "Ghi nhận rõ khách đặt một sản phẩm nhưng nhận sản phẩm khác; có thể đề nghị "
            "gửi ảnh hàng nhận được để người thật đối chiếu, nhưng không trì hoãn handoff."
        ),
        "other": "Nếu khách chưa nói lý do khiếu nại, hỏi một câu ngắn để bổ sung lý do.",
    }
    followup = followup_by_issue.get(issue_type, "")
    response_constraints = " ".join(
        RESPONSE_GUIDANCE[requirement]
        for requirement in state.get("triage", {}).get("response_requirements", [])
        if requirement in RESPONSE_GUIDANCE
    )
    reply = generate_reply(
        "Thông báo ngắn gọn rằng yêu cầu handoff đã được tạo và hiện đang chờ chủ shop "
        "hoặc người thật tiếp nhận. Lịch sử hội thoại và thông tin đơn liên quan đã được "
        "đính kèm. Không nói rằng chủ shop đã tiếp nhận, đang xử lý hoặc chắc chắn sẽ "
        "liên hệ trong một thời gian cụ thể. "
        "Hướng dẫn bổ sung theo loại vấn đề và chính sách: "
        f"{followup} {response_constraints}",
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
