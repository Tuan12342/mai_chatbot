import json
from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.order_store import find_orders_by_customer_id
from mai_agent.reply_generator import generate_reply
from mai_agent.state import AgentState


def _reply(instruction: str, session: dict[str, Any]) -> dict[str, Any]:
    reply = generate_reply(
        instruction,
        response_language=session.get("language_code", "vi"),
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)], "session": session}


def _safe_order_view(order: dict[str, Any]) -> dict[str, Any]:
    """Chỉ giữ dữ liệu cần hiển thị, loại bỏ địa chỉ và thông tin liên hệ."""
    items = []
    for item in order.get("items", []):
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "product_id": str(item.get("product_id", "")),
                "product_name": str(item.get("product_name", "")),
                "quantity": item.get("quantity", 0),
                "unit_price": item.get("unit_price", 0),
            }
        )
    return {
        "order_id": str(order.get("order_id", "")),
        "status": str(order.get("status", "unknown")),
        "payment_status": str(order.get("payment_status", "unknown")),
        "total_amount": order.get("total_amount", 0),
        "items": items,
    }


def order_lookup_node(state: AgentState) -> dict[str, Any]:
    """Hiển thị mọi đơn thuộc đúng tài khoản của phiên hiện tại."""
    session = state.get("session", {})
    customer_id = str(state.get("user_id") or "").strip()
    next_session = {
        "language_code": session.get("language_code", "vi"),
        "current_step": "idle",
        "pending_order_lookup_id": None,
        "pending_order_lookup_attempts": 0,
    }

    if not customer_id:
        return _reply(
            "Không xác định được tài khoản đang đăng nhập nên không được truy vấn hay "
            "hiển thị dữ liệu đơn hàng. Thông báo ngắn gọn rằng chưa thể tải lịch sử "
            "đơn của tài khoản hiện tại và đề nghị khách thử đăng nhập lại. Không hỏi "
            "mã đơn hoặc số điện thoại.",
            next_session,
        )

    # Repository đã lọc theo customer_id. Kiểm tra lại quyền sở hữu tại node để
    # tránh lộ dữ liệu nếu implementation của repository thay đổi hoặc trả dư bản ghi.
    owned_orders = [
        order
        for order in find_orders_by_customer_id(customer_id)
        if str(order.get("customer_id", "")) == customer_id
    ]
    safe_orders = [_safe_order_view(order) for order in owned_orders]

    if not safe_orders:
        return _reply(
            "Tài khoản hiện tại không có đơn hàng nào. Thông báo điều này một cách "
            "tự nhiên; không hỏi mã đơn hoặc số điện thoại và không suy đoán dữ liệu "
            "của tài khoản khác.",
            next_session,
        )

    order_data = json.dumps(safe_orders, ensure_ascii=False)
    return _reply(
        "Hiển thị đầy đủ tất cả đơn hàng dưới đây của tài khoản hiện tại. Mỗi đơn phải "
        "xuất hiện đúng một lần và gồm mã đơn, danh sách sản phẩm cùng số lượng, trạng "
        "thái đơn, trạng thái thanh toán và tổng tiền. Diễn đạt rõ ràng, tự nhiên, có "
        "thể dùng danh sách để dễ đọc. Giữ nguyên chính xác mã đơn, mã và tên sản phẩm; "
        "không thêm dữ liệu không có, không hỏi mã đơn hay số điện thoại, không nhắc "
        "địa chỉ hoặc thông tin liên hệ. Dữ liệu đơn đã được lọc theo tài khoản:\n"
        f"{order_data}",
        next_session,
    )
