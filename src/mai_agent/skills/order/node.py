import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.catalog import find_product
from mai_agent.config import get_settings
from mai_agent.routing import latest_user_text
from mai_agent.skills.order.tools import (
    extract_order_request,
    resolve_order_product,
)
from mai_agent.skills.product.tools import check_product_stock
from mai_agent.state import AgentState


def _reply(
    text: str,
    session_update: dict[str, Any],
    **state_updates: Any,
) -> dict[str, Any]:
    rendered_text = _generate_order_reply(text)
    return {
        "reply": rendered_text,
        "messages": [AIMessage(content=rendered_text)],
        "session": session_update,
        **state_updates,
    }


def _generate_order_reply(required_content: str) -> str:
    """Cho Gemini diễn đạt sau khi code đã quyết định kết quả nghiệp vụ."""
    try:
        settings = get_settings()
        model = ChatGoogleGenerativeAI(
            model=settings.google_model,
            api_key=settings.google_api_key,
        )
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        "Bạn là Mai, tư vấn viên OA Cosmetics. Viết 1-2 câu ngắn, "
                        "thân thiện và cùng ngôn ngữ với khách. Chỉ diễn đạt lại nội "
                        "dung bắt buộc do hệ thống đã kiểm tra. Không thay đổi hoặc tự "
                        "thêm sản phẩm, mã SKU, số lượng, giá, tồn kho, trạng thái, "
                        "quyết định xác thực hay hành động tiếp theo. Không nói rằng "
                        "bạn đang nhận chỉ dẫn từ hệ thống."
                    )
                ),
                HumanMessage(
                    content=(
                        "Nội dung và hành động bắt buộc phải được giữ nguyên:\n"
                        f"{json.dumps(required_content, ensure_ascii=False)}"
                    )
                ),
            ]
        )
        reply = response.text.strip()
        return reply or required_content
    except Exception:
        return required_content


def order_node(state: AgentState) -> dict[str, Any]:
    text = latest_user_text(state)
    session = state.get("session", {})
    current_step = session.get("current_step", "idle")
    cart = session.get("cart", [])

    if current_step == "confirming_alternative":
        normalized_text = text.lower().strip()
        if normalized_text in {"có", "đồng ý", "ok", "xem", "xem nhé"}:
            product_id = session.get("pending_product_id")
            product = find_product(product_id) if product_id else None
            if product is None:
                return _reply(
                    "Sản phẩm thay thế không còn khả dụng. Hỏi khách chọn sản phẩm khác.",
                    {"current_step": "selecting_product", "cart": cart},
                )

            return _reply(
                f"Giới thiệu sản phẩm thay thế {product['id']} - {product['name']}. "
                f"Danh mục: {product['category']}; phù hợp: "
                f"{', '.join(product['skin_types'])}; thành phần chính: "
                f"{', '.join(product['ingredients'])}; công dụng: "
                f"{', '.join(product['benefits'])}; cách dùng: {product['usage']}; "
                f"giá {product['price_vnd']:,} đồng; hiện còn {product['stock']} sản phẩm. "
                "Sau khi giới thiệu đầy đủ, hỏi khách muốn mua số lượng bao nhiêu.",
                {
                    "current_step": "collecting_quantity",
                    "cart": cart,
                    "pending_product_id": product["id"],
                    "pending_product_reference": product["id"],
                },
            )

        return _reply(
            "Khách không muốn xem sản phẩm thay thế. Hỏi khách muốn chọn sản phẩm khác không.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_product_id": None,
                "pending_product_reference": None,
            },
        )

    if current_step == "collecting_address":
        phone_match = re.search(r"(?<!\d)(?:\+?84|0)\d{8,10}(?!\d)", text)
        phone = phone_match.group(0) if phone_match else None
        address = (
            " ".join(
                (text[: phone_match.start()] + text[phone_match.end() :])
                .strip(" ,-:")
                .split()
            )
            if phone_match
            else ""
        )
        if not address or not phone:
            return _reply(
                "Thông tin giao hàng chưa đầy đủ. Hỏi khách cung cấp cả địa chỉ "
                "giao hàng và số điện thoại người nhận.",
                {"current_step": "collecting_address", "cart": cart},
            )

        total_amount = sum(item["quantity"] * item["unit_price"] for item in cart)
        item_summary = ", ".join(
            f"{item['product_id']} - {item['product_name']} x{item['quantity']}"
            for item in cart
        )
        return _reply(
            f"Đọc lại đơn để khách xác nhận: {item_summary}; tổng tiền "
            f"{total_amount:,} đồng; địa chỉ {address}; số điện thoại {phone}. "
            "Hỏi khách có xác nhận đặt đơn không.",
            {
                "current_step": "confirming_order",
                "cart": cart,
                "pending_shipping_address": address,
                "pending_phone": phone,
            },
        )

    if current_step == "confirming_order":
        if text.lower().strip() in {"có", "đồng ý", "ok", "xác nhận", "đặt đơn"}:
            address = session.get("pending_shipping_address", "")
            phone = session.get("pending_phone", "")
            total_amount = sum(item["quantity"] * item["unit_price"] for item in cart)
            order_id = f"ORDER-{uuid.uuid4().hex[:8].upper()}"
            active_order = {
                "order_id": order_id,
                "customer_id": state.get("user_id", "anonymous"),
                "items": cart,
                "shipping_address": {
                    "address_id": "pending",
                    "address_line": address,
                    "phone": phone,
                },
                "status": "confirmed",
                "payment_status": "pending",
                "total_amount": total_amount,
            }
            return _reply(
                f"Đơn {order_id} đã được xác nhận, tổng tiền {total_amount:,} đồng. "
                "Thông báo đặt đơn thành công.",
                {"current_step": "idle", "cart": []},
                active_order=active_order,
            )
        return _reply(
            "Đơn chưa được xác nhận. Hỏi khách muốn xác nhận hay sửa thông tin đơn.",
            {"current_step": "confirming_order", "cart": cart},
        )

    # Khách đang trả lời câu hỏi:
    # "Có muốn lấy toàn bộ số lượng còn lại không?"
    if current_step == "confirming_partial_quantity":
        normalized_text = text.lower().strip()

        if normalized_text in {
            "có",
            "đồng ý",
            "ok",
            "lấy",
            "lấy nhé",
            "lấy hết",
        }:
            product_id = session.get("pending_product_id")
            available_quantity = session.get("pending_available_quantity")

            if not product_id or not available_quantity:
                return _reply(
                    "Em chưa tìm thấy thông tin sản phẩm đang chờ xác nhận.",
                    {
                        "current_step": "selecting_product",
                        "cart": cart,
                    },
                )

            product = find_product(product_id)
            if product is None:
                return _reply(
                    "Sản phẩm này hiện không còn trong danh mục.",
                    {
                        "current_step": "selecting_product",
                        "cart": cart,
                    },
                )

            updated_cart = [
                *cart,
                {
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "quantity": available_quantity,
                    "unit_price": product["price_vnd"],
                },
            ]

            return _reply(
                f"Em đã thêm {available_quantity} {product['name']} vào giỏ. "
                "Hỏi khách cung cấp địa chỉ giao hàng và số điện thoại người nhận.",
                {
                    "current_step": "collecting_address",
                    "cart": updated_cart,
                    "pending_product_id": None,
                    "pending_quantity": None,
                    "pending_available_quantity": None,
                },
            )

        return _reply(
            "Dạ, em chưa thêm sản phẩm vào giỏ. "
            "Mình muốn chọn sản phẩm hoặc số lượng khác không ạ?",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_product_id": None,
                "pending_quantity": None,
                "pending_available_quantity": None,
            },
        )

    parsed = extract_order_request(text)
    product_reference = parsed["product_reference"]
    quantity = parsed["quantity"]

    if current_step == "collecting_quantity" and session.get("pending_product_reference"):
        product_reference = session["pending_product_reference"]
    if session.get("pending_product_candidates") and session.get("pending_quantity"):
        product_reference = text
        quantity = session["pending_quantity"]

    if not product_reference:
        return _reply(
            "Bạn cho mình tên hoặc mã sản phẩm muốn đặt nhé.",
            {
                "current_step": "selecting_product",
                "cart": cart,
            },
        )

    if quantity is None:
        return _reply(
            "Mình muốn mua số lượng bao nhiêu ạ?",
            {
                "current_step": "collecting_quantity",
                "cart": cart,
                "pending_product_reference": product_reference,
            },
        )

    product_resolution = resolve_order_product(product_reference)

    if product_resolution["status"] == "not_found":
        return _reply(
            "Em chưa tìm thấy sản phẩm này. "
            "Mình kiểm tra lại tên hoặc mã sản phẩm giúp em nhé.",
            {
                "current_step": "selecting_product",
                "cart": cart,
            },
        )

    if product_resolution["status"] == "ambiguous":
        candidates = ", ".join(
            f"{candidate['product_id']} - {candidate['product_name']}"
            for candidate in product_resolution["candidates"]
        )

        return _reply(
            f"Em tìm thấy nhiều sản phẩm phù hợp: {candidates}. "
            "Mình chọn giúp em mã sản phẩm nhé.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_quantity": quantity,
                "pending_product_candidates": product_resolution["candidates"],
            },
        )

    selected_product = product_resolution["product"]
    if selected_product is None:
        return _reply(
            "Em chưa xác định được sản phẩm.",
            {
                "current_step": "selecting_product",
                "cart": cart,
            },
        )

    customer = state.get("customer", {})

    stock_result = check_product_stock.invoke(
        {
            "product_id": selected_product["product_id"],
            "quantity": quantity,
            "skin_type": customer.get("skin_type"),
            "excluded_ingredients": customer.get(
                "excluded_ingredients",
                [],
            ),
        }
    )

    if stock_result["status"] == "partial_stock":
        current_stock = stock_result["current_stock"]

        return _reply(
            f"{selected_product['product_name']} hiện chỉ còn "
            f"{current_stock} sản phẩm. "
            f"Mình có muốn lấy {current_stock} sản phẩm không ạ?",
            {
                "current_step": "confirming_partial_quantity",
                "cart": cart,
                "pending_product_id": selected_product["product_id"],
                "pending_quantity": quantity,
                "pending_available_quantity": current_stock,
            },
        )

    if stock_result["status"] == "out_of_stock":
        alternatives = stock_result.get("alternatives", [])
        if alternatives:
            recommended_product = alternatives[0]
            alternative_text = ", ".join(
                f"{item['id']} - {item['name']}" for item in alternatives
            )
            reply = (
                f"{selected_product['product_name']} hiện đã hết hàng. "
                f"Em gợi ý: {alternative_text}. Hỏi khách có muốn xem sản phẩm "
                "thay thế không."
            )
        else:
            reply = (
                f"{selected_product['product_name']} hiện đã hết hàng và chưa có "
                "sản phẩm thay thế phù hợp."
            )

        return _reply(
            reply,
            {
                "current_step": (
                    "confirming_alternative" if alternatives else "selecting_product"
                ),
                "cart": cart,
                "pending_product_id": (
                    recommended_product["id"] if alternatives else None
                ),
            },
        )

    updated_cart = [
        *cart,
        {
            "product_id": stock_result["product_id"],
            "product_name": stock_result["product_name"],
            "quantity": stock_result["quantity"],
            "unit_price": stock_result["unit_price"],
        },
    ]

    return _reply(
        f"Em đã thêm {quantity} {stock_result['product_name']} vào giỏ. "
        "Hỏi khách cung cấp địa chỉ giao hàng và số điện thoại người nhận.",
        {
            "current_step": "collecting_address",
            "cart": updated_cart,
        },
    )
