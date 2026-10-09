from typing import Any

from langchain_core.messages import SystemMessage

from mai_agent.llm import create_chat_model
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.skills.product.tools import PRODUCT_TOOLS
from mai_agent.state import AgentState


def assistant_node(state: AgentState) -> dict[str, Any]:
    customer = state.get("customer", {})
    session = state.get("session", {})
    response_language = session.get("language_code", "vi")
    language_name = "English" if response_language == "en" else "Vietnamese"
    pending_product_reference = session.get("pending_product_reference")
    pending_product_context = (
        f"Sản phẩm đang được chọn trong luồng đặt hàng: {pending_product_reference}. "
        "Khi khách nói 'sản phẩm này', 'loại này' hoặc cách gọi tương tự, dùng mã này "
        "để tra cứu. Chỉ tư vấn ở lượt hiện tại và giữ nguyên tiến trình đặt hàng."
        if pending_product_reference
        else "Không có sản phẩm nào đang chờ trong luồng đặt hàng."
    )
    profile_context = (
        "Hồ sơ được hệ thống nạp cho đúng Zalo user hiện tại:\n"
        f"- Loại da: {customer.get('skin_type', 'chưa biết')}\n"
        f"- Thành phần cần loại trừ: {customer.get('excluded_ingredients', [])}\n"
        "Không hỏi hoặc tự suy đoán Zalo user ID."
    )
    model_with_tools = create_chat_model().bind_tools(PRODUCT_TOOLS)
    system_message = SystemMessage(
        content=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Ngôn ngữ bắt buộc của câu trả lời hiện tại: {language_name}.\n"
            "Dù ngôn ngữ trả lời thay đổi, phải tiếp tục dùng hồ sơ, lịch sử và "
            "trạng thái phiên đã có. Không dịch SKU, tên riêng sản phẩm, thương hiệu "
            "hoặc tên thành phần INCI trong kết quả công cụ.\n"
            f"{pending_product_context}\n"
            "Quy tắc sử dụng công cụ:\n"
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

    # Lượt yêu cầu tool thường chưa có nội dung trả lời. Giữ nguyên AIMessage
    # để tools_condition đọc được response.tool_calls.
    reply = response.text
    return {"reply": reply, "messages": [response]}
