from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.state import AgentState


def greeting_node(_: AgentState) -> dict[str, Any]:
    reply = (
        "Chào bạn, mình là Mai của OA Cosmetics. "
        "Mình có thể giúp bạn tìm hiểu sản phẩm hoặc tư vấn theo loại da."
    )
    return {"reply": reply, "messages": [AIMessage(content=reply)]}
