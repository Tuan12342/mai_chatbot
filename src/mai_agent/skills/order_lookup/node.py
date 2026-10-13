import re
from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.customer_store import find_customer_by_zalo_id
from mai_agent.order_store import find_order_by_id
from mai_agent.reply_generator import generate_reply
from mai_agent.routing import latest_user_text
from mai_agent.state import AgentState

ORDER_ID_PATTERN = re.compile(r"\bORDER-[A-Z0-9]+\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])(?:\+?84|0)[\d .()-]{7,14}\d(?!\d)"
)


def _normalize_phone(value: str) -> str | None:
    normalized = re.sub(r"[\s.()-]+", "", value.strip())
    if normalized.startswith("+84"):
        normalized = f"0{normalized[3:]}"
    elif normalized.startswith("84"):
        normalized = f"0{normalized[2:]}"
    if re.fullmatch(r"0\d{9}", normalized):
        return normalized
    return None


def _extract_phone(text: str) -> str | None:
    for match in PHONE_PATTERN.finditer(text):
        if phone := _normalize_phone(match.group(0)):
            return phone
    return None


def _order_phone(order: dict[str, Any]) -> str | None:
    shipping_address = order.get("shipping_address", {})
    if isinstance(shipping_address, dict):
        phone = _normalize_phone(str(shipping_address.get("phone", "")))
        if phone:
            return phone

    customer = find_customer_by_zalo_id(str(order.get("customer_id", "")))
    if customer is None:
        return None
    return _normalize_phone(str(customer.get("phone", "")))


def _reply(instruction: str, session: dict[str, Any]) -> dict[str, Any]:
    reply = generate_reply(
        instruction,
        response_language=session.get("language_code", "vi"),
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)], "session": session}


def order_lookup_node(state: AgentState) -> dict[str, Any]:
    """Chỉ trả dữ liệu đơn cũ sau khi mã đơn và số điện thoại cùng khớp."""
    text = latest_user_text(state)
    session = state.get("session", {})
    order_match = ORDER_ID_PATTERN.search(text)
    order_id = (
        order_match.group(0).upper()
        if order_match
        else session.get("pending_order_lookup_id")
    )
    phone = _extract_phone(text)

    if not order_id or not phone:
        missing = []
        if not order_id:
            missing.append("mã đơn")
        if not phone:
            missing.append("số điện thoại đã dùng khi đặt hàng")
        return _reply(
            "Để bảo vệ thông tin cá nhân, yêu cầu khách cung cấp "
            f"{', '.join(missing)} trước khi tra cứu. Không xác nhận đơn có tồn tại "
            "hay không và không tiết lộ bất kỳ thông tin đơn hàng nào.",
            {
                "current_step": "verifying_order_lookup",
                "pending_order_lookup_id": order_id,
                "pending_order_lookup_attempts": session.get(
                    "pending_order_lookup_attempts", 0
                ),
            },
        )

    order = find_order_by_id(order_id)
    verified = (
        order is not None
        and str(order.get("customer_id", "")) == state.get("user_id", "")
        and _order_phone(order) == phone
    )
    if not verified:
        attempts = session.get("pending_order_lookup_attempts", 0) + 1
        return _reply(
            "Không xác minh được thông tin tra cứu. Trả lời chung rằng mã đơn hoặc "
            "số điện thoại chưa khớp và đề nghị khách kiểm tra lại; không nói trường "
            "nào sai, không xác nhận đơn có tồn tại và không tiết lộ dữ liệu đơn.",
            {
                "current_step": "verifying_order_lookup",
                "pending_order_lookup_id": None,
                "pending_order_lookup_attempts": attempts,
            },
        )

    items = ", ".join(
        f"{item['product_id']} - {item['product_name']} x{item['quantity']}"
        for item in order.get("items", [])
    )
    return _reply(
        f"Đã xác thực đúng mã đơn và số điện thoại. Thông báo thông tin đơn {order_id}: "
        f"sản phẩm {items or 'không có dữ liệu'}; trạng thái {order.get('status', 'unknown')}; "
        f"trạng thái thanh toán {order.get('payment_status', 'unknown')}; tổng tiền "
        f"{order.get('total_amount', 0):,} đồng. Không đọc lại số điện thoại, địa chỉ "
        "hoặc dữ liệu cá nhân khác.",
        {
            "current_step": "idle",
            "pending_order_lookup_id": None,
            "pending_order_lookup_attempts": 0,
        },
    )
