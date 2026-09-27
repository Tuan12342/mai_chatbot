from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["greeting", "product_question", "recommendation", "order", "unknown"]


class AgentState(TypedDict, total=False):
    """Trạng thái được truyền qua tất cả node của LangGraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    intent: Intent
    reply: str
