from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.state import AgentState


def order_node(state: AgentState) -> dict[str, Any]:
    reply = "Bạn cho mình tên sản phẩm và số lượng muốn đặt nhé."
    session = state.get("session", {})
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "session": {
            "session_id": session.get("session_id", state.get("user_id", "anonymous")),
            "current_step": "selecting_product",
            "cart": session.get("cart", []),
        },
    }
