import re
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from mai_agent.catalog import find_product
from mai_agent.config import get_settings
from mai_agent.reply_generator import generate_reply
from mai_agent.routing import latest_user_text
from mai_agent.skills.order.tools import resolve_order_product
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


def _generate_order_reply(business_context: str) -> str:
    """Yêu cầu Gemini viết phản hồi từ kết quả nghiệp vụ đã được kiểm tra."""
    return generate_reply(business_context)


@tool
def choose_offered_product(product_id: str) -> str:
    """Chọn một mã sản phẩm trong danh sách shop vừa giới thiệu."""
    return product_id


@tool
def ask_for_other_products() -> str:
    """Khách hỏi còn lựa chọn/sản phẩm/mẫu nào khác trong nhóm vừa tư vấn."""
    return "other_products"


@tool
def search_different_product(query: str) -> str:
    """Khách đổi sang tìm một tên, mã hoặc danh mục sản phẩm khác."""
    return query


@tool
def decline_offered_products() -> str:
    """Khách không muốn chọn các sản phẩm vừa được giới thiệu."""
    return "decline"


SELECTION_TOOLS = [
    choose_offered_product,
    ask_for_other_products,
    search_different_product,
    decline_offered_products,
]


@tool
def confirm_order() -> str:
    """Khách đồng ý và xác nhận đặt đúng đơn hàng vừa được đọc lại."""
    return "confirm"


@tool
def request_order_changes(change_request: str = "") -> str:
    """Khách chưa xác nhận và muốn sửa hoặc chưa nói rõ cần sửa gì trong đơn."""
    return change_request


@tool
def cancel_order() -> str:
    """Khách nói rõ muốn hủy đơn hoặc không mua nữa."""
    return "cancel"


ORDER_CONFIRMATION_TOOLS = [
    confirm_order,
    request_order_changes,
    cancel_order,
]


@tool
def accept_offer() -> str:
    """Khách đồng ý với đề nghị đang chờ trong bước hiện tại."""
    return "accept"


@tool
def decline_offer() -> str:
    """Khách từ chối đề nghị đang chờ trong bước hiện tại."""
    return "decline"


class OrderRequestItem(BaseModel):
    product_reference: str = Field(
        description="Tên hoặc mã SKU của đúng một sản phẩm",
    )
    quantity: int | None = Field(
        default=None,
        description="Số lượng khách yêu cầu; null nếu khách chưa nêu",
    )


@tool
def provide_order_request(items: list[OrderRequestItem]) -> list[dict[str, Any]]:
    """Trích tất cả sản phẩm và số lượng tương ứng trong yêu cầu mua hàng."""
    return [item.model_dump() for item in items]


BINARY_CHOICE_TOOLS = [accept_offer, decline_offer]
ORDER_REQUEST_TOOLS = [provide_order_request]


@tool
def accept_available_stock() -> str:
    """Khách đồng ý lấy toàn bộ số lượng hiện còn của sản phẩm đang chờ."""
    return "accept"


@tool
def decline_available_stock() -> str:
    """Khách không muốn lấy số lượng hiện còn của sản phẩm đang chờ."""
    return "decline"


@tool
def modify_partial_order(
    accepted_quantity: int,
    additional_product_id: str = "",
    additional_quantity: int | None = None,
) -> dict[str, Any]:
    """Lấy một số lượng của sản phẩm hiện tại và có thể thêm một SKU khác."""
    return {
        "accepted_quantity": accepted_quantity,
        "additional_product_id": additional_product_id or None,
        "additional_quantity": additional_quantity,
    }


PARTIAL_STOCK_TOOLS = [
    accept_available_stock,
    decline_available_stock,
    modify_partial_order,
]


class CartChange(BaseModel):
    action: Literal["add", "update_quantity", "remove", "replace"]
    product_id: str = Field(description="SKU hiện tại hoặc SKU cần thêm")
    quantity: int | None = Field(default=None, description="Số lượng mới")
    replacement_product_id: str | None = Field(
        default=None,
        description="SKU thay thế, chỉ dùng cho action replace",
    )


@tool
def revise_order_before_confirmation(
    changes: list[CartChange] | None = None,
    address: str = "",
    phone: str = "",
) -> dict[str, Any]:
    """Sửa nhiều slot của đơn: giỏ hàng, địa chỉ và/hoặc số điện thoại."""
    return {
        "changes": [change.model_dump() for change in changes or []],
        "address": address,
        "phone": phone,
    }


ORDER_CONFIRMATION_TOOLS.append(revise_order_before_confirmation)


@tool
def provide_shipping_details(address: str = "", phone: str = "") -> dict[str, str]:
    """Trích địa chỉ và/hoặc số điện thoại; để trống trường khách chưa cung cấp."""
    return {"address": address, "phone": phone}


@tool
def modify_cart(changes: list[CartChange]) -> list[dict[str, Any]]:
    """Khách muốn thêm, xóa, thay sản phẩm hoặc đổi số lượng trong giỏ."""
    return [change.model_dump() for change in changes]


@tool
def cancel_pending_order() -> str:
    """Khách muốn hủy toàn bộ đơn đang đặt."""
    return "cancel"


ADDRESS_STEP_TOOLS = [provide_shipping_details, modify_cart, cancel_pending_order]


@tool
def confirm_cart_revision() -> str:
    """Khách đồng ý áp dụng toàn bộ giỏ hàng đã sửa."""
    return "confirm"


@tool
def revise_cart_again(changes: list[CartChange]) -> list[dict[str, Any]]:
    """Khách chưa đồng ý và cung cấp thêm thay đổi cho giỏ hàng nháp."""
    return [change.model_dump() for change in changes]


@tool
def discard_cart_revision() -> str:
    """Khách bỏ bản sửa và giữ nguyên giỏ hàng cũ."""
    return "discard"


CART_REVISION_TOOLS = [
    confirm_cart_revision,
    revise_cart_again,
    discard_cart_revision,
]


@tool
def continue_order_flow() -> str:
    """Tin nhắn tiếp tục xử lý bước đặt hàng hiện tại, không hủy toàn bộ đơn."""
    return "continue"


@tool
def cancel_entire_order_flow() -> str:
    """Khách nói rõ muốn thôi mua, hủy hoặc dừng toàn bộ đơn đang làm."""
    return "cancel"


ORDER_CONTROL_TOOLS = [continue_order_flow, cancel_entire_order_flow]


def _interpret_pending_selection(
    text: str,
    candidates: list[dict[str, str]],
) -> tuple[str, str | None]:
    """Để Gemini chọn hành động tiếp theo bằng tool calling."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(SELECTION_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Đọc câu mới nhất trong ngữ cảnh khách đang chọn sản phẩm. "
                    "Bắt buộc gọi đúng một tool: choose_offered_product nếu khách "
                    "chọn mã trong danh sách; ask_for_other_products nếu khách hỏi "
                    "còn lựa chọn khác; search_different_product nếu khách muốn tìm "
                    "một sản phẩm hoặc danh mục mới; decline_offered_products nếu "
                    "khách từ chối hoặc không muốn chọn các sản phẩm vừa được giới "
                    "thiệu. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Danh sách vừa giới thiệu: {candidates}\n"
                    f"Tin nhắn mới nhất: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động cho bước chọn sản phẩm.")
    call = response.tool_calls[0]
    if call["name"] == "choose_offered_product":
        return "choose", call["args"]["product_id"]
    if call["name"] == "ask_for_other_products":
        return "other_products", None
    if call["name"] == "decline_offered_products":
        return "decline", None
    if call["name"] == "search_different_product":
        query = call["args"].get("query")
        if query:
            return "search", query
        raise RuntimeError("Tool search_different_product thiếu tham số query.")
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")


def _interpret_order_confirmation(
    text: str,
    order_summary: dict[str, Any],
) -> tuple[str, Any]:
    """Để Gemini hiểu câu xác nhận, sửa hoặc hủy đơn bằng tool calling."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(ORDER_CONFIRMATION_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Khách đang trả lời sau khi đơn hàng đã được đọc lại. Bắt buộc "
                    "gọi đúng một tool: confirm_order khi khách đồng ý/chốt đơn; "
                    "revise_order_before_confirmation khi khách đã nói rõ thay đổi "
                    "sản phẩm, số lượng, địa chỉ hoặc số điện thoại, kể cả nhiều thay "
                    "đổi trong một câu; request_order_changes khi khách chỉ từ chối "
                    "hoặc muốn sửa nhưng chưa nói rõ; cancel_order chỉ khi khách nói "
                    "rõ muốn hủy đơn hoặc không mua nữa. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Đơn đang chờ xác nhận: {order_summary}\n"
                    f"Tin nhắn mới nhất: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động xác nhận đơn hàng.")
    call = response.tool_calls[0]
    if call["name"] == "confirm_order":
        return "confirm", None
    if call["name"] == "cancel_order":
        return "cancel", None
    if call["name"] == "request_order_changes":
        return "edit", call["args"].get("change_request") or None
    if call["name"] == "revise_order_before_confirmation":
        return "revise_slots", call["args"]
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")


def _interpret_binary_choice(text: str, pending_offer: str) -> bool:
    """Để Gemini hiểu khách đồng ý hay từ chối đề nghị đang chờ."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(BINARY_CHOICE_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Khách đang trả lời một đề nghị cụ thể. Bắt buộc gọi accept_offer "
                    "nếu khách đồng ý hoặc decline_offer nếu khách từ chối. Không tự "
                    "trả lời khách."
                )
            ),
            HumanMessage(
                content=f"Đề nghị đang chờ: {pending_offer}\nTin nhắn khách: {text}"
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động đồng ý hoặc từ chối.")
    tool_name = response.tool_calls[0]["name"]
    if tool_name == "accept_offer":
        return True
    if tool_name == "decline_offer":
        return False
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {tool_name}")


def _interpret_order_request(
    text: str,
    offered_candidates: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Để Gemini trích sản phẩm và số lượng thay cho luật từ khóa."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(ORDER_REQUEST_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Trích yêu cầu mua hàng từ tin nhắn và bắt buộc gọi "
                    "provide_order_request. Mỗi sản phẩm phải là một item riêng với "
                    "đúng số lượng đi kèm; tuyệt đối không cộng số lượng của nhiều "
                    "sản phẩm. Chuyển số lượng viết bằng chữ thành số. Nếu khách nói "
                    "'loại 1', 'loại 2'..., ánh xạ theo đúng thứ tự danh sách đã giới "
                    "thiệu và trả mã SKU tương ứng. Không tự trả lời khách."
                    "- “loại này”, “sản phẩm này” là sản phẩm vừa được tư vấn gần nhất. "
                    "- “loại 1”, “loại 2” ánh xạ theo thứ tự offered_candidates. "
                    "- Mỗi sản phẩm phải tạo một item riêng. "
                    "- Không cộng số lượng của nhiều sản phẩm. "
                    "Chuyển số lượng viết bằng chữ thành số."
                )
            ),
            HumanMessage(
                content=(
                    f"Danh sách đã giới thiệu theo thứ tự: {offered_candidates or []}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không trích được yêu cầu đặt hàng.")
    call = response.tool_calls[0]
    if call["name"] != "provide_order_request":
        raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")
    items = call["args"].get("items", [])
    return {
        "items": [
            {
                "product_reference": item.get("product_reference") or None,
                "quantity": item.get("quantity"),
            }
            for item in items
        ]
    }


def _interpret_partial_stock_response(
    text: str,
    pending_product: dict[str, Any],
    offered_candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """Để Gemini xử lý đồng ý, từ chối hoặc sửa yêu cầu khi thiếu hàng."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(PARTIAL_STOCK_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Khách đang trả lời đề nghị lấy số lượng còn trong kho. Bắt buộc "
                    "gọi đúng một tool. Dùng accept_available_stock khi khách chỉ đồng "
                    "ý lấy toàn bộ số còn lại; decline_available_stock khi từ chối; "
                    "modify_partial_order khi khách nêu số lượng cụ thể hoặc muốn lấy "
                    "thêm SKU khác. Nếu khách nói 'loại 1', 'loại 2', hãy ánh xạ theo "
                    "thứ tự danh sách sản phẩm đã giới thiệu và truyền mã SKU thật vào "
                    "additional_product_id. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Sản phẩm đang thiếu hàng: {pending_product}\n"
                    f"Danh sách đã giới thiệu theo thứ tự: {offered_candidates}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động xử lý số lượng còn lại.")
    call = response.tool_calls[0]
    if call["name"] == "accept_available_stock":
        return {"action": "accept"}
    if call["name"] == "decline_available_stock":
        return {"action": "decline"}
    if call["name"] == "modify_partial_order":
        return {"action": "modify", **call["args"]}
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")


def _process_multi_item_order(
    state: AgentState,
    requested_items: list[dict[str, Any]],
    cart: list[dict[str, Any]],
) -> dict[str, Any]:
    """Phân giải và kiểm tra độc lập mọi sản phẩm trong một yêu cầu nhiều dòng."""
    resolved_quantities: dict[str, int] = {}
    resolution_errors: list[str] = []

    for item in requested_items:
        reference = item.get("product_reference")
        quantity = item.get("quantity")
        if not reference or not isinstance(quantity, int) or quantity <= 0:
            resolution_errors.append(
                f"Yêu cầu chưa đủ tên/mã và số lượng hợp lệ: {item}."
            )
            continue

        resolution = resolve_order_product(reference)
        if resolution["status"] == "not_found":
            resolution_errors.append(f"Không tìm thấy sản phẩm: {reference}.")
            continue
        if resolution["status"] == "ambiguous":
            choices = ", ".join(
                candidate["product_id"] for candidate in resolution["candidates"]
            )
            resolution_errors.append(
                f"Sản phẩm {reference} còn mơ hồ; các mã phù hợp: {choices}."
            )
            continue

        product = resolution["product"]
        if product is None:
            resolution_errors.append(f"Chưa xác định được sản phẩm: {reference}.")
            continue
        product_id = product["product_id"]
        resolved_quantities[product_id] = (
            resolved_quantities.get(product_id, 0) + quantity
        )

    if resolution_errors:
        return _reply(
            "Không thể xử lý đơn nhiều sản phẩm vì các lỗi sau:\n"
            + "\n".join(f"- {error}" for error in resolution_errors)
            + "\nHỏi khách cung cấp lại mã SKU và số lượng cho từng sản phẩm.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_order_items": requested_items,
            },
        )

    customer = state.get("customer", {})
    checked_items: list[dict[str, Any]] = []
    stock_errors: list[str] = []
    for product_id, quantity in resolved_quantities.items():
        stock = check_product_stock.invoke(
            {
                "product_id": product_id,
                "quantity": quantity,
                "skin_type": customer.get("skin_type"),
                "excluded_ingredients": customer.get("excluded_ingredients", []),
            }
        )
        if stock["status"] == "available":
            checked_items.append(
                {
                    "product_id": stock["product_id"],
                    "product_name": stock["product_name"],
                    "quantity": stock["quantity"],
                    "unit_price": stock["unit_price"],
                }
            )
        elif stock["status"] == "partial_stock":
            stock_errors.append(
                f"{product_id}: yêu cầu {quantity}, hiện chỉ còn "
                f"{stock['current_stock']}."
            )
        else:
            stock_errors.append(f"{product_id}: hiện đã hết hàng.")

    if stock_errors:
        return _reply(
            "Chưa thêm sản phẩm nào của yêu cầu này vào giỏ vì tồn kho chưa đáp ứng:\n"
            + "\n".join(f"- {error}" for error in stock_errors)
            + "\nHỏi khách xác nhận lại số lượng cho từng mã; không tự thay thế sản phẩm.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_order_items": requested_items,
            },
        )

    updated_cart = [*cart, *checked_items]
    return _reply(
        f"Đã kiểm tra và thêm riêng từng sản phẩm vào giỏ: {checked_items}. "
        "Hỏi khách cung cấp địa chỉ giao hàng và số điện thoại người nhận.",
        {
            "current_step": "collecting_address",
            "cart": updated_cart,
            "pending_product_candidates": [],
            "pending_order_items": [],
        },
    )


def _interpret_address_step(
    text: str,
    cart: list[dict[str, Any]],
    offered_candidates: list[dict[str, str]],
) -> dict[str, Any]:
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(ADDRESS_STEP_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Khách đang ở bước cung cấp thông tin giao hàng. Bắt buộc gọi "
                    "provide_shipping_details nếu khách đưa địa chỉ và số điện thoại; "
                    "modify_cart nếu khách đổi ý muốn thêm, xóa, thay sản phẩm hoặc "
                    "đổi số lượng; cancel_pending_order chỉ khi khách muốn hủy toàn "
                    "bộ đơn. Với provide_shipping_details, không được sao chép số "
                    "điện thoại sang address hoặc tự bịa trường còn thiếu; hãy để "
                    "chuỗi rỗng. Nếu khách chưa ghi địa chỉ hay số điện thoại thì "
                    "hỏi lại. Khi sửa giỏ, phải truyền SKU thật và không tự đặt giá."
                )
            ),
            HumanMessage(
                content=(
                    f"Giỏ hiện tại: {cart}\n"
                    f"Sản phẩm đã giới thiệu: {offered_candidates}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động tại bước địa chỉ.")
    call = response.tool_calls[0]
    if call["name"] == "provide_shipping_details":
        return {"action": "shipping", **call["args"]}
    if call["name"] == "modify_cart":
        return {"action": "modify", "changes": call["args"].get("changes", [])}
    if call["name"] == "cancel_pending_order":
        return {"action": "cancel"}
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")


def _interpret_order_control(
    text: str,
    current_step: str,
    cart: list[dict[str, Any]],
) -> str:
    """Nhận diện yêu cầu hủy toàn bộ ở mọi bước của đơn đang làm."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(ORDER_CONTROL_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Xác định khách có muốn hủy toàn bộ đơn đang làm hay không. "
                    "Chỉ gọi cancel_entire_order_flow khi khách nói rõ muốn thôi mua, "
                    "khỏi mua nữa, hủy đơn hoặc dừng toàn bộ việc đặt hàng. Một câu "
                    "'không' đơn lẻ hoặc từ chối một đề nghị cục bộ phải gọi "
                    "continue_order_flow. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Bước hiện tại: {current_step}\nGiỏ hiện tại: {cart}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động tiếp tục hoặc hủy đơn.")
    tool_name = response.tool_calls[0]["name"]
    if tool_name == "cancel_entire_order_flow":
        return "cancel"
    if tool_name == "continue_order_flow":
        return "continue"
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {tool_name}")


def _clean_cancelled_order_session() -> dict[str, Any]:
    """Xóa toàn bộ dữ liệu tạm để đơn bị hủy không còn treo trong session."""
    return {
        "current_step": "idle",
        "cart": [],
        "pending_product_id": None,
        "pending_product_reference": None,
        "pending_product_candidates": [],
        "last_product_candidates": [],
        "pending_quantity": None,
        "pending_order_items": [],
        "pending_shipping_address": None,
        "pending_phone": None,
        "pending_available_quantity": None,
        "pending_cart_revision": [],
        "cart_revision_reason": None,
        "pending_revised_shipping_address": None,
        "pending_revised_phone": None,
    }


def _is_valid_shipping_address(address: str) -> bool:
    """Địa chỉ phải có nội dung chữ, không được chỉ là số điện thoại/con số."""
    letters = [character for character in address if character.isalpha()]
    return len(letters) >= 3


def _interpret_cart_revision_confirmation(
    text: str,
    original_cart: list[dict[str, Any]],
    revised_cart: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    ).bind_tools(CART_REVISION_TOOLS, tool_choice="any")
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Khách đang xác nhận giỏ hàng sau khi sửa. Bắt buộc gọi "
                    "confirm_cart_revision nếu đồng ý; revise_cart_again nếu muốn "
                    "sửa thêm và phải truyền đầy đủ changes; discard_cart_revision "
                    "nếu muốn bỏ bản sửa để giữ giỏ cũ. Không tự trả lời khách."
                )
            ),
            HumanMessage(
                content=(
                    f"Giỏ cũ: {original_cart}\nGiỏ sửa: {revised_cart}\n"
                    f"Tin nhắn khách: {text}"
                )
            ),
        ]
    )
    if not response.tool_calls:
        raise RuntimeError("Gemini không chọn hành động xác nhận giỏ sửa.")
    call = response.tool_calls[0]
    if call["name"] == "confirm_cart_revision":
        return {"action": "confirm"}
    if call["name"] == "discard_cart_revision":
        return {"action": "discard"}
    if call["name"] == "revise_cart_again":
        return {"action": "revise", "changes": call["args"].get("changes", [])}
    raise RuntimeError(f"Gemini gọi tool không được hỗ trợ: {call['name']}")


def _prepare_cart_revision(
    state: AgentState,
    original_cart: list[dict[str, Any]],
    base_cart: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    reason: str,
    revised_address: str | None = None,
    revised_phone: str | None = None,
) -> dict[str, Any]:
    revised_cart = [dict(item) for item in base_cart]
    errors: list[str] = []

    for change in changes:
        action = change.get("action")
        product_id = str(change.get("product_id", "")).upper()
        if action not in {"add", "update_quantity", "remove", "replace"}:
            errors.append(f"Hành động sửa giỏ không hợp lệ: {action}.")
            continue
        index = next(
            (
                item_index
                for item_index, item in enumerate(revised_cart)
                if item["product_id"] == product_id
            ),
            None,
        )

        if action == "remove":
            if index is None:
                errors.append(f"{product_id} không có trong giỏ.")
            else:
                revised_cart.pop(index)
            continue

        target_id = product_id
        quantity = change.get("quantity")
        if action == "replace":
            if index is None:
                errors.append(f"{product_id} không có trong giỏ để thay thế.")
                continue
            target_id = str(change.get("replacement_product_id", "")).upper()
            if quantity is None:
                quantity = revised_cart[index]["quantity"]
            revised_cart.pop(index)
        elif action == "update_quantity":
            if index is None:
                errors.append(f"{product_id} không có trong giỏ để đổi số lượng.")
                continue
            revised_cart.pop(index)
        elif action == "add" and index is not None:
            quantity = (quantity or 0) + revised_cart[index]["quantity"]
            revised_cart.pop(index)

        product = find_product(target_id)
        if product is None:
            errors.append(f"Không tìm thấy sản phẩm {target_id}.")
            continue
        if not isinstance(quantity, int) or quantity <= 0:
            errors.append(f"Số lượng của {target_id} phải lớn hơn 0.")
            continue
        revised_cart.append(
            {
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": quantity,
                "unit_price": product["price_vnd"],
            }
        )

    consolidated: dict[str, dict[str, Any]] = {}
    for item in revised_cart:
        product = find_product(item["product_id"])
        if product is None:
            errors.append(f"Không tìm thấy sản phẩm {item['product_id']}.")
            continue
        if product["id"] in consolidated:
            consolidated[product["id"]]["quantity"] += item["quantity"]
        else:
            consolidated[product["id"]] = {
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": item["quantity"],
                "unit_price": product["price_vnd"],
            }
    revised_cart = list(consolidated.values())

    if not revised_cart:
        errors.append("Giỏ hàng sau khi sửa không được để trống.")

    customer = state.get("customer", {})
    for item in revised_cart:
        stock = check_product_stock.invoke(
            {
                "product_id": item["product_id"],
                "quantity": item["quantity"],
                "skin_type": customer.get("skin_type"),
                "excluded_ingredients": customer.get("excluded_ingredients", []),
            }
        )
        if stock["status"] == "partial_stock":
            errors.append(
                f"{item['product_id']} yêu cầu {item['quantity']}, chỉ còn "
                f"{stock['current_stock']}."
            )
        elif stock["status"] == "out_of_stock":
            errors.append(f"{item['product_id']} đã hết hàng.")

    if errors:
        return _reply(
            "Chưa thể tạo bản sửa vì:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\nHỏi khách cung cấp lại thay đổi hợp lệ.",
            {
                "current_step": "collecting_address",
                "cart": original_cart,
                "pending_cart_revision": [],
                "pending_revised_shipping_address": None,
                "pending_revised_phone": None,
            },
        )

    summary = ", ".join(
        f"{item['product_id']} - {item['product_name']} x{item['quantity']} "
        f"({item['unit_price']:,} đồng/sản phẩm)"
        for item in revised_cart
    )
    total = sum(item["quantity"] * item["unit_price"] for item in revised_cart)
    shipping_summary = ""
    if revised_address or revised_phone:
        shipping_summary = (
            f" Địa chỉ sau sửa: {revised_address or 'chưa có'}; "
            f"số điện thoại: {revised_phone or 'chưa có'}."
        )
    return _reply(
        f"Đọc lại toàn bộ giỏ sau khi sửa: {summary}; tổng tiền {total:,} đồng. "
        f"{shipping_summary} "
        "Hỏi khách xác nhận áp dụng toàn bộ thay đổi trước khi đi tiếp.",
        {
            "current_step": "confirming_cart_revision",
            "cart": original_cart,
            "pending_cart_revision": revised_cart,
            "cart_revision_reason": reason,
            "pending_revised_shipping_address": revised_address,
            "pending_revised_phone": revised_phone,
        },
    )


def order_node(state: AgentState) -> dict[str, Any]:
    text = latest_user_text(state)
    session = state.get("session", {})
    current_step = session.get("current_step", "idle")
    cart = session.get("cart", [])

    if current_step != "idle":
        control_action = _interpret_order_control(text, current_step, cart)
        if control_action == "cancel":
            return _reply(
                "Khách đã yêu cầu dừng và hủy toàn bộ đơn đang làm. Xác nhận chưa "
                "đặt đơn, chưa giữ sản phẩm nào và đã xóa toàn bộ thông tin tạm của "
                "đơn này. Không hỏi tiếp địa chỉ, số lượng hay xác nhận đơn.",
                _clean_cancelled_order_session(),
            )

    if current_step == "confirming_alternative":
        accepted = _interpret_binary_choice(
            text,
            "Xem thông tin chi tiết sản phẩm thay thế được đề xuất.",
        )
        if accepted:
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

    if current_step == "confirming_cart_revision":
        revised_cart = session.get("pending_cart_revision", [])
        revision_action = _interpret_cart_revision_confirmation(
            text,
            cart,
            revised_cart,
        )
        if revision_action["action"] == "confirm":
            revised_address = session.get("pending_revised_shipping_address")
            revised_phone = session.get("pending_revised_phone")
            if revised_address and revised_phone:
                total = sum(
                    item["quantity"] * item["unit_price"] for item in revised_cart
                )
                order_id = f"ORDER-{uuid.uuid4().hex[:8].upper()}"
                active_order = {
                    "order_id": order_id,
                    "customer_id": state.get("user_id", "anonymous"),
                    "items": revised_cart,
                    "shipping_address": {
                        "address_id": "pending",
                        "address_line": revised_address,
                        "phone": revised_phone,
                    },
                    "status": "confirmed",
                    "payment_status": "pending",
                    "total_amount": total,
                }
                return _reply(
                    f"Bản sửa đã được áp dụng và đơn {order_id} đã được xác nhận; "
                    f"tổng tiền {total:,} đồng; địa chỉ {revised_address}; số điện "
                    f"thoại {revised_phone}. Thông báo cập nhật và đặt đơn thành công, "
                    "không hỏi xác nhận thêm lần nữa.",
                    {
                        "current_step": "idle",
                        "cart": [],
                        "pending_cart_revision": [],
                        "cart_revision_reason": None,
                        "pending_shipping_address": None,
                        "pending_phone": None,
                        "pending_revised_shipping_address": None,
                        "pending_revised_phone": None,
                    },
                    active_order=active_order,
                )
            return _reply(
                "Khách đã xác nhận toàn bộ sản phẩm, số lượng và giá sau khi sửa. "
                "Thông báo đã áp dụng bản sửa và hỏi lại địa chỉ giao hàng cùng số "
                "điện thoại người nhận.",
                {
                    "current_step": "collecting_address",
                    "cart": revised_cart,
                    "pending_cart_revision": [],
                    "cart_revision_reason": None,
                    "pending_shipping_address": None,
                    "pending_phone": None,
                    "pending_revised_shipping_address": None,
                    "pending_revised_phone": None,
                },
            )
        if revision_action["action"] == "discard":
            return _reply(
                "Khách bỏ bản sửa. Thông báo vẫn giữ nguyên giỏ cũ và hỏi địa chỉ "
                "giao hàng cùng số điện thoại người nhận.",
                {
                    "current_step": "collecting_address",
                    "cart": cart,
                    "pending_cart_revision": [],
                    "cart_revision_reason": None,
                    "pending_revised_shipping_address": None,
                    "pending_revised_phone": None,
                },
            )
        return _prepare_cart_revision(
            state,
            original_cart=cart,
            base_cart=revised_cart,
            changes=revision_action["changes"],
            reason=text,
            revised_address=session.get("pending_revised_shipping_address"),
            revised_phone=session.get("pending_revised_phone"),
        )

    if current_step == "collecting_address":
        address_action = _interpret_address_step(
            text,
            cart,
            session.get("last_product_candidates", []),
        )
        if address_action["action"] == "modify":
            return _prepare_cart_revision(
                state,
                original_cart=cart,
                base_cart=cart,
                changes=address_action["changes"],
                reason=text,
            )
        if address_action["action"] == "cancel":
            return _reply(
                "Khách yêu cầu hủy toàn bộ đơn đang đặt. Xác nhận đã hủy và xóa "
                "giỏ hàng đang chờ.",
                {
                    "current_step": "idle",
                    "cart": [],
                    "pending_cart_revision": [],
                    "pending_shipping_address": None,
                    "pending_phone": None,
                },
            )

        supplied_address = " ".join(
            str(address_action.get("address", "")).split()
        )
        phone_text = str(address_action.get("phone", ""))
        phone_match = re.fullmatch(r"(?:\+?84|0)\d{9,10}", phone_text)
        supplied_phone = phone_match.group(0) if phone_match else None

        saved_address = str(session.get("pending_shipping_address") or "")
        address = (
            supplied_address
            if _is_valid_shipping_address(supplied_address)
            else saved_address if _is_valid_shipping_address(saved_address) else None
        )
        phone = supplied_phone or session.get("pending_phone")

        missing_fields = []
        if not address:
            missing_fields.append("địa chỉ giao hàng có tên đường/khu vực")
        if not phone:
            missing_fields.append("số điện thoại hợp lệ")
        if missing_fields:
            return _reply(
                "Đã lưu các thông tin giao hàng hợp lệ khách vừa cung cấp. Còn thiếu: "
                f"{', '.join(missing_fields)}. Hỏi khách chỉ cung cấp phần còn thiếu. "
                "Không được dùng số điện thoại làm địa chỉ.",
                {
                    "current_step": "collecting_address",
                    "cart": cart,
                    "pending_shipping_address": address,
                    "pending_phone": phone,
                },
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
        address = session.get("pending_shipping_address", "")
        phone = session.get("pending_phone", "")
        total_amount = sum(item["quantity"] * item["unit_price"] for item in cart)
        confirmation_action, change_request = _interpret_order_confirmation(
            text,
            {
                "items": cart,
                "total_amount": total_amount,
                "shipping_address": address,
                "phone": phone,
            },
        )

        if confirmation_action == "revise_slots":
            revision = change_request or {}
            supplied_address = " ".join(str(revision.get("address", "")).split())
            supplied_phone = str(revision.get("phone", ""))
            if supplied_address and not _is_valid_shipping_address(supplied_address):
                return _reply(
                    "Địa chỉ mới chưa hợp lệ vì không có đủ nội dung chữ. Hỏi khách "
                    "cung cấp lại địa chỉ; chưa thay đổi đơn.",
                    {"current_step": "confirming_order", "cart": cart},
                )
            if supplied_phone and not re.fullmatch(
                r"(?:\+?84|0)\d{9,10}",
                supplied_phone,
            ):
                return _reply(
                    "Số điện thoại mới sai định dạng. Hỏi khách cung cấp lại; chưa "
                    "thay đổi đơn.",
                    {"current_step": "confirming_order", "cart": cart},
                )
            return _prepare_cart_revision(
                state,
                original_cart=cart,
                base_cart=cart,
                changes=revision.get("changes", []),
                reason=text,
                revised_address=supplied_address or address,
                revised_phone=supplied_phone or phone,
            )

        if confirmation_action == "confirm":
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

        if confirmation_action == "cancel":
            return _reply(
                "Khách đã yêu cầu hủy đơn và không mua nữa. Xác nhận đơn chưa được "
                "đặt và đã xóa giỏ hàng đang chờ.",
                {
                    "current_step": "idle",
                    "cart": [],
                    "pending_shipping_address": None,
                    "pending_phone": None,
                },
            )

        return _reply(
            "Khách chưa xác nhận đơn và muốn chỉnh sửa. "
            f"Nội dung khách muốn sửa: {change_request or 'chưa nói rõ'}. "
            "Hỏi khách cung cấp lại địa chỉ giao hàng và số điện thoại người nhận; "
            "không xác nhận đơn lúc này.",
            {
                "current_step": "collecting_address",
                "cart": cart,
                "pending_shipping_address": None,
                "pending_phone": None,
            },
        )

    # Khách đang trả lời câu hỏi:
    # "Có muốn lấy toàn bộ số lượng còn lại không?"
    if current_step == "confirming_partial_quantity":
        product_id = session.get("pending_product_id")
        available_quantity = session.get("pending_available_quantity")
        product = find_product(product_id) if product_id else None
        partial_action = _interpret_partial_stock_response(
            text,
            {
                "product_id": product_id,
                "product_name": product["name"] if product else None,
                "available_quantity": available_quantity,
                "requested_quantity": session.get("pending_quantity"),
            },
            session.get("last_product_candidates", []),
        )
        if partial_action["action"] in {"accept", "modify"}:
            if not product_id or not available_quantity:
                return _reply(
                    "Em chưa tìm thấy thông tin sản phẩm đang chờ xác nhận.",
                    {
                        "current_step": "selecting_product",
                        "cart": cart,
                    },
                )

            if product is None:
                return _reply(
                    "Sản phẩm này hiện không còn trong danh mục.",
                    {
                        "current_step": "selecting_product",
                        "cart": cart,
                    },
                )

            accepted_quantity = (
                available_quantity
                if partial_action["action"] == "accept"
                else partial_action.get("accepted_quantity")
            )
            if (
                not isinstance(accepted_quantity, int)
                or accepted_quantity <= 0
                or accepted_quantity > available_quantity
            ):
                return _reply(
                    f"Số lượng chấp nhận phải từ 1 đến {available_quantity} cho "
                    f"{product['id']} - {product['name']}. Hỏi khách nhập lại số lượng.",
                    {
                        "current_step": "confirming_partial_quantity",
                        "cart": cart,
                        "pending_product_id": product_id,
                        "pending_quantity": session.get("pending_quantity"),
                        "pending_available_quantity": available_quantity,
                    },
                )

            updated_cart = [
                *cart,
                {
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "quantity": accepted_quantity,
                    "unit_price": product["price_vnd"],
                },
            ]

            additional_product_id = partial_action.get("additional_product_id")
            additional_quantity = partial_action.get("additional_quantity")
            if additional_product_id and additional_quantity:
                allowed_ids = {
                    candidate["product_id"]
                    for candidate in session.get("last_product_candidates", [])
                }
                if additional_product_id not in allowed_ids:
                    return _reply(
                        f"Đã giữ {accepted_quantity} {product['name']} trong giỏ tạm. "
                        f"Mã bổ sung {additional_product_id} không nằm trong danh sách "
                        "đã giới thiệu. Hỏi khách chọn lại đúng mã SKU.",
                        {
                            "current_step": "selecting_product",
                            "cart": updated_cart,
                            "pending_product_candidates": session.get(
                                "last_product_candidates",
                                [],
                            ),
                            "pending_quantity": additional_quantity,
                        },
                    )

                additional_product = find_product(additional_product_id)
                if additional_product is None:
                    return _reply(
                        f"Đã giữ {accepted_quantity} {product['name']} trong giỏ tạm. "
                        f"Không tìm thấy sản phẩm bổ sung {additional_product_id}. "
                        "Hỏi khách chọn lại mã SKU.",
                        {
                            "current_step": "selecting_product",
                            "cart": updated_cart,
                            "pending_product_candidates": session.get(
                                "last_product_candidates",
                                [],
                            ),
                            "pending_quantity": additional_quantity,
                        },
                    )
                additional_stock = check_product_stock.invoke(
                    {
                        "product_id": additional_product_id,
                        "quantity": additional_quantity,
                        "skin_type": state.get("customer", {}).get("skin_type"),
                        "excluded_ingredients": state.get("customer", {}).get(
                            "excluded_ingredients",
                            [],
                        ),
                    }
                )
                if additional_stock["status"] == "partial_stock":
                    current_stock = additional_stock["current_stock"]
                    return _reply(
                        f"Đã giữ {accepted_quantity} {product['name']} trong giỏ tạm. "
                        f"{additional_product['name']} chỉ còn {current_stock} sản phẩm. "
                        f"Hỏi khách có muốn lấy {current_stock} sản phẩm không.",
                        {
                            "current_step": "confirming_partial_quantity",
                            "cart": updated_cart,
                            "pending_product_id": additional_product_id,
                            "pending_quantity": additional_quantity,
                            "pending_available_quantity": current_stock,
                        },
                    )
                if additional_stock["status"] == "out_of_stock":
                    return _reply(
                        f"Đã giữ {accepted_quantity} {product['name']} trong giỏ tạm. "
                        f"Sản phẩm {additional_product_id} đã hết hàng. Hỏi khách chọn "
                        "một sản phẩm khác trong danh sách.",
                        {
                            "current_step": "selecting_product",
                            "cart": updated_cart,
                            "pending_product_candidates": session.get(
                                "last_product_candidates",
                                [],
                            ),
                            "pending_quantity": additional_quantity,
                        },
                    )
                updated_cart.append(
                    {
                        "product_id": additional_product["id"],
                        "product_name": additional_product["name"],
                        "quantity": additional_quantity,
                        "unit_price": additional_product["price_vnd"],
                    }
                )

            return _reply(
                f"Đã thêm các sản phẩm đã kiểm tra vào giỏ: {updated_cart}. "
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

    pending_candidates = session.get("pending_product_candidates", [])
    offered_candidates = pending_candidates or session.get("last_product_candidates", [])
    parsed = _interpret_order_request(text, offered_candidates)
    requested_items = parsed["items"]
    if len(requested_items) > 1:
        return _process_multi_item_order(state, requested_items, cart)

    first_item = requested_items[0] if requested_items else {}
    product_reference = first_item.get("product_reference")
    quantity = first_item.get("quantity")

    pending_action: str | None = None
    if pending_candidates:
        pending_action, interpreted_reference = _interpret_pending_selection(
            text,
            pending_candidates,
        )

    if pending_action == "other_products":
        available_products = [
            product
            for candidate in pending_candidates
            if (product := find_product(candidate["product_id"])) is not None
            and product["stock"] > 0
        ]
        available_summary = "\n".join(
            f"- {product['id']} - {product['name']}; giá "
            f"{product['price_vnd']:,} đồng; còn {product['stock']} sản phẩm."
            for product in available_products
        )
        return _reply(
            "Khách đang hỏi tiếp xem có lựa chọn nào khác trong nhóm sản phẩm "
            "vừa được tư vấn hay không. Đây là toàn bộ lựa chọn hiện còn hàng:\n"
            f"{available_summary}\n"
            "Trả lời rõ rằng ngoài danh sách này hiện không còn lựa chọn nào khác "
            "đang có hàng. Không giới thiệu sản phẩm hết hàng. Hỏi khách có muốn "
            "chọn một mã trong danh sách không.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_quantity": session.get("pending_quantity"),
                "pending_product_candidates": pending_candidates,
            },
        )

    if pending_action == "decline":
        return _reply(
            "Khách không muốn chọn các sản phẩm vừa được giới thiệu. Xác nhận rằng "
            "chưa thêm sản phẩm nào vào giỏ và hỏi khách muốn tìm danh mục khác hay "
            "cần hỗ trợ nội dung gì tiếp theo.",
            {
                "current_step": "idle",
                "cart": cart,
                "pending_product_id": None,
                "pending_product_reference": None,
                "pending_product_candidates": [],
                "pending_quantity": None,
            },
        )

    if current_step == "collecting_quantity" and session.get("pending_product_reference"):
        product_reference = session["pending_product_reference"]
    if pending_candidates:
        product_reference = interpreted_reference
        if quantity is None:
            quantity = session.get("pending_quantity")
        if pending_action == "search":
            pending_candidates = []

    if not product_reference:
        return _reply(
            "Bạn cho mình tên hoặc mã sản phẩm muốn đặt nhé.",
            {
                "current_step": "selecting_product",
                "cart": cart,
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
                "pending_product_candidates": (
                    [] if pending_action == "search" else pending_candidates
                ),
            },
        )

    if product_resolution["status"] == "ambiguous":
        candidate_products = [
            product
            for candidate in product_resolution["candidates"]
            if (product := find_product(candidate["product_id"])) is not None
            and product["stock"] > 0
        ]
        available_candidates = [
            {"product_id": product["id"], "product_name": product["name"]}
            for product in candidate_products
        ]
        if not candidate_products:
            return _reply(
                "Các sản phẩm thuộc loại khách yêu cầu hiện đều đã hết hàng. "
                "Thông báo chưa có sản phẩm còn hàng và hỏi khách muốn xem loại "
                "sản phẩm khác không; không giới thiệu sản phẩm hết hàng.",
                {
                    "current_step": "selecting_product",
                    "cart": cart,
                    "pending_quantity": quantity,
                    "pending_product_candidates": [],
                },
            )
        candidates = "\n".join(
            f"- {product['id']} - {product['name']}; phù hợp: "
            f"{', '.join(product['skin_types'])}; công dụng: "
            f"{', '.join(product['benefits'])}; giá {product['price_vnd']:,} đồng; "
            f"tồn kho: {product['stock']}."
            for product in candidate_products
        )
        skin_type = state.get("customer", {}).get("skin_type")
        recommended_ids = [
            product["id"]
            for product in candidate_products
            if skin_type
            and (
                skin_type in product["skin_types"]
                or "mọi loại da" in product["skin_types"]
            )
        ]
        recommendation = (
            f" Theo hồ sơ da {skin_type} của khách, ưu tiên gợi ý "
            f"{', '.join(recommended_ids)}."
            if recommended_ids
            else ""
        )

        return _reply(
            f"Khách mới nêu loại sản phẩm nên chưa thể hỏi số lượng. "
            f"Giới thiệu ngắn gọn các lựa chọn sau:\n{candidates}"
            f"{recommendation} Hỏi khách chọn một mã sản phẩm; chưa hỏi số lượng.",
            {
                "current_step": "selecting_product",
                "cart": cart,
                "pending_quantity": quantity,
                "pending_product_candidates": available_candidates,
                "last_product_candidates": available_candidates,
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

    if quantity is None:
        return _reply(
            f"Khách đã chọn {selected_product['product_id']} - "
            f"{selected_product['product_name']}. Hỏi khách muốn mua số lượng bao nhiêu.",
            {
                "current_step": "collecting_quantity",
                "cart": cart,
                "pending_product_reference": selected_product["product_id"],
                "pending_product_candidates": [],
                "pending_quantity": None,
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
