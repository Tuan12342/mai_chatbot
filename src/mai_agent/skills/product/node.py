from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.skills.product.tools import PRODUCT_TOOLS
from mai_agent.state import AgentState


def assistant_node(state: AgentState) -> dict[str, Any]:
    settings = get_settings()
    customer = state.get("customer", {})
    profile_context = (
        "Hồ sơ được hệ thống nạp cho đúng Zalo user hiện tại:\n"
        f"- Loại da: {customer.get('skin_type', 'chưa biết')}\n"
        f"- Thành phần cần loại trừ: {customer.get('excluded_ingredients', [])}\n"
        "Không hỏi hoặc tự suy đoán Zalo user ID."
    )
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
            "  Khi thiếu hàng, truyền loại da và thành phần cần loại trừ từ hồ sơ trên "
            "để nhận gợi ý thay thế.\n"
            f"{profile_context}\n"
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
