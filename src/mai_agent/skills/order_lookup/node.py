import re
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.reply_generator import generate_reply
from mai_agent.routing import latest_user_text
from mai_agent.skills.order_lookup.tools import (
    lookup_verified_customer_orders,
    verify_customer_identity,
)
from mai_agent.state import AgentState

MAX_VERIFICATION_ATTEMPTS = 3
_LOOKUP_RESPONSE_LANGUAGE: ContextVar[str] = ContextVar(
    "lookup_response_language",
    default="vi",
)


def _reply(text: str, session: dict[str, Any]) -> dict[str, Any]:
    rendered_text = generate_reply(
        text,
        response_language=_LOOKUP_RESPONSE_LANGUAGE.get(),
    )
    return {
        "reply": rendered_text,
        "messages": [AIMessage(content=rendered_text)],
        "session": session,
    }


def _parse_verification_answer(text: str) -> tuple[str, str] | None:
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
    if match is None:
        return None
    name = " ".join((text[: match.start()] + text[match.end() :]).strip(" ,-:").split())
    if not name:
        return None
    return name, match.group(1)


def _format_orders(orders: list[dict[str, Any]]) -> str:
    if not orders:
        return "Em đã xác minh thông tin nhưng hiện chưa thấy đơn hàng nào của mình."

    lines = ["Em đã xác minh. Các đơn hàng của mình:"]
    for order in orders:
        item_summary = ", ".join(
            f"{item['product_name']} x{item['quantity']}" for item in order["items"]
        )
        lines.append(
            f"- {order['order_id']}: {item_summary}; "
            f"trạng thái {order['status']}; tổng {order['total_amount']:,}đ."
        )
    return "\n".join(lines)


def order_lookup_node(state: AgentState) -> dict[str, Any]:
    """Xác minh nhẹ rồi mới trả đơn của Zalo user ID đang trò chuyện."""
    _LOOKUP_RESPONSE_LANGUAGE.set(
        state.get("session", {}).get("language_code", "vi")
    )
    user_id = state.get("user_id", "")
    session = state.get("session", {})
    attempts = session.get("verification_attempts", 0)

    if session.get("identity_verified"):
        return _reply(
            _format_orders(lookup_verified_customer_orders(user_id)),
            {"current_step": "idle"},
        )

    if session.get("current_step") != "verifying_order_lookup":
        return _reply(
            "Để bảo vệ thông tin đơn hàng, mình cho em xin tên và 4 số cuối "
            "số điện thoại đã đăng ký (ví dụ: Lan, 4567).",
            {
                "current_step": "verifying_order_lookup",
                "identity_verified": False,
                "verification_attempts": attempts,
            },
        )

    parsed = _parse_verification_answer(latest_user_text(state))
    verified = parsed is not None and verify_customer_identity(
        zalo_user_id=user_id,
        provided_name=parsed[0],
        phone_last_four=parsed[1],
    )
    if verified:
        return _reply(
            _format_orders(lookup_verified_customer_orders(user_id)),
            {
                "current_step": "idle",
                "identity_verified": True,
                "verification_attempts": 0,
            },
        )

    attempts += 1
    if attempts >= MAX_VERIFICATION_ATTEMPTS:
        return _reply(
            "Em chưa thể xác minh thông tin. Yêu cầu tra đơn đã được tạm dừng; "
            "mình vui lòng liên hệ shop để được hỗ trợ.",
            {
                "current_step": "idle",
                "identity_verified": False,
                "verification_attempts": attempts,
            },
        )

    return _reply(
        "Thông tin chưa khớp. Em chưa thể cung cấp dữ liệu đơn hàng. "
        "Mình vui lòng nhập lại tên và đúng 4 số cuối số điện thoại đã đăng ký.",
        {
            "current_step": "verifying_order_lookup",
            "identity_verified": False,
            "verification_attempts": attempts,
        },
    )
