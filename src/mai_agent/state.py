from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["greeting", "product_question", "recommendation", "order", "unknown"]

OrderStep = Literal[
    "selecting_product",
    "collecting_quantity",
    "collecting_address",
    "comfirming",
    "complete",
    "cancelled",
]

ConversationStep = Literal[
    "idle",
    "selecting_product",
    "collecting_quantity",
    "collecting_address",
    "confirming_order",
]

PaymentStatus = Literal[
    "pending",
    "failed",
    "success",
]


class CartItem(TypedDict):
    product_id: str
    product_name: str
    quantity: int
    unit_price: int


class SessionState(TypedDict, total=False):
    current_intent: Intent
    current_step: ConversationStep
    cart: list[CartItem]


class CustomerProfile(TypedDict, total=False):
    name: str
    skin_type: str
    allergies: list[str]
    purchase_history: list[str]
    addresses: list[str]


class OrderState(TypedDict, total=False):
    order_id: str
    items: list[CartItem]
    shipping_address: str
    payment_status: PaymentStatus
    step: OrderStep


class AgentState(TypedDict, total=False):
    """Trạng thái được truyền qua tất cả node của LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    intent: Intent
    reply: str
    current_step: OrderStep
    cart: list[CartItem]

    # Customer profile
    customer: CustomerProfile

    # Order state
    order: OrderState
