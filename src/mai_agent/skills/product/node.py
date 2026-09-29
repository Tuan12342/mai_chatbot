from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.skills.product.tools import PRODUCT_TOOLS
from mai_agent.state import AgentState


def assistant_node(state: AgentState) -> dict[str, Any]:
    """Gọi Gemini và giữ nguyên tool calls để LangGraph xử lý."""
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("Thiếu GOOGLE_API_KEY để gọi Gemini.")

    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    )
    model_with_tools = model.bind_tools(PRODUCT_TOOLS)
    system_message = SystemMessage(
        content=(
            f"{SYSTEM_PROMPT}\n\n"
            "Quy tắc sử dụng công cụ:\n"
            "- Dùng resolve_product_names khi tên sản phẩm bị viết tắt, sai nhẹ hoặc mơ hồ.\n"
            "- Dùng search_product_knowledge khi hỏi thành phần, công dụng, cách dùng, "
            "độ phù hợp hoặc so sánh sản phẩm.\n"
            "- Dùng search_products khi cần tìm theo mã, tên hoặc danh mục.\n"
            "- Dùng check_product_stock khi hỏi tồn kho hoặc số lượng có thể mua.\n"
            "- Chỉ trả lời thông tin sản phẩm từ kết quả công cụ; không tự suy đoán."
        )
    )
    response = model_with_tools.invoke([system_message, *state.get("messages", [])])
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))

    # Lượt yêu cầu tool thường chưa có nội dung trả lời. Giữ nguyên AIMessage
    # để tools_condition đọc được response.tool_calls.
    reply = response.text
    return {"reply": reply, "messages": [response]}
