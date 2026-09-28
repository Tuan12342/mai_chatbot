from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.state import AgentState


def order_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Bạn cho mình tên sản phẩm và số lượng muốn đặt nhé. "
        "Ở bước tiếp theo mình sẽ bổ sung giỏ hàng và state machine cho đơn hàng."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
