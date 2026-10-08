from langchain_core.messages import BaseMessage, ToolCall
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings


def create_chat_model() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    )


def invoke_tool(
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    error_message: str,
) -> ToolCall:
    """Yêu cầu Gemini chọn tool và trả về tên cùng tham số đã trích xuất."""
    response = create_chat_model().bind_tools(tools, tool_choice="any").invoke(messages)
    if not response.tool_calls:
        raise RuntimeError(error_message)
    return response.tool_calls[0]
