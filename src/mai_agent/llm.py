from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolCall
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from mai_agent.config import get_settings


def create_chat_model() -> BaseChatModel:
    settings = get_settings()
    provider = settings.llm_provider.strip().casefold()
    if provider == "misa":
        if not settings.misa_api_key:
            raise ValueError("MISA_API_KEY chưa được cấu hình.")
        return ChatOpenAI(
            model=settings.misa_model,
            api_key=settings.misa_api_key,
            base_url=settings.misa_base_url,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if provider == "google":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY chưa được cấu hình.")
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            api_key=settings.google_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"LLM_PROVIDER không được hỗ trợ: {settings.llm_provider}")


def invoke_tool(
    tools: list[BaseTool],
    messages: list[BaseMessage],
    *,
    error_message: str,
) -> ToolCall:
    """Yêu cầu model chọn tool và trả về tên cùng tham số đã trích xuất."""
    model = create_chat_model().bind_tools(tools, tool_choice="any")
    current_messages = messages
    for _attempt in range(3):
        response = model.invoke(current_messages)
        if response.tool_calls:
            return response.tool_calls[0]
        current_messages = [
            SystemMessage(
                content=(
                    "Phản hồi trước không gọi tool như yêu cầu. Hãy thử lại và bắt buộc "
                    "gọi đúng một tool phù hợp; không trả lời bằng văn bản thường."
                )
            ),
            *messages,
        ]
    raise RuntimeError(error_message)
