from typing import Annotated, Any, Literal, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["greeting", "product_question", "recommendation", "order", "unknown"]

ConversationStep = Literal[
    "idle",
    "selecting_product",
    "collecting_quantity",
    "confirming_alternative",
    "confirming_partial_quantity",
    "collecting_address",
    "confirming_order",
    "verifying_order_lookup",
]

OrderStatus = Literal[
    "draft",
    "awaiting_confirmation",
    "confirmed",
    "processing",
    "completed",
    "cancelled",
]

PaymentStatus = Literal[
    "pending",
    "paid",
    "failed",
    "refunded",
]


class CartItem(TypedDict):
    product_id: str
    product_name: str
    quantity: int
    unit_price: int


class ProductCandidate(TypedDict):
    product_id: str
    product_name: str


class Address(TypedDict, total=False):
    address_id: Required[str]
    address_line: Required[str]
    recipient_name: str
    phone: str
    ward: str
    district: str
    province: str
    is_default: bool


class SessionState(TypedDict, total=False):
    session_id: Required[str]
    current_intent: Intent
    current_step: ConversationStep
    cart: list[CartItem]
    active_order_id: str
    pending_product_id: str | None
    pending_product_reference: str | None
    pending_product_candidates: list[ProductCandidate]
    pending_quantity: int | None
    pending_shipping_address: str | None
    pending_phone: str | None
    pending_available_quantity: int | None


class CustomerProfile(TypedDict, total=False):
    zalo_user_id: Required[str]
    name: str
    skin_type: str
    is_sensitive: bool
    sensitivities: list[str]
    allergies: list[str]
    purchase_history: list[str]
    addresses: list[Address]


class OrderState(TypedDict, total=False):
    order_id: Required[str]
    customer_id: Required[str]
    items: Required[list[CartItem]]
    shipping_address: Address
    status: OrderStatus
    payment_status: PaymentStatus
    total_amount: int
    cancel_reason: str


def merge_mapping(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Gộp cập nhật từng phần mà không làm mất các trường state cùng tầng."""
    return {**left, **right}


class AgentState(TypedDict, total=False):
    """Trạng thái làm việc của graph trong một phiên hội thoại."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    reply: str
    session: Annotated[SessionState, merge_mapping]

    # Các bản chụp được nạp từ kho dữ liệu lâu dài khi cần.
    customer: Annotated[CustomerProfile, merge_mapping]
    active_order: Annotated[OrderState, merge_mapping]
