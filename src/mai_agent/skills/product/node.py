import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from mai_agent.catalog import find_product
from mai_agent.llm import create_chat_model
from mai_agent.prompts import SYSTEM_PROMPT
from mai_agent.skills.product.tools import PRODUCT_TOOLS
from mai_agent.state import AgentState


def _tool_payload(message: ToolMessage) -> Any:
    """Decode the value serialized by LangGraph's ToolNode."""
    content = message.content
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None


def _recent_product_candidates(
    messages: list[Any],
    response_text: str | None = None,
) -> list[dict[str, str]]:
    """Collect products resolved and actually surfaced in this consultation turn."""
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index

    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    def add_candidate(product_id: Any, product_name: Any) -> None:
        if not isinstance(product_id, str) or not isinstance(product_name, str):
            return
        normalized_id = product_id.strip().upper()
        normalized_name = product_name.strip()
        if not normalized_id or not normalized_name or normalized_id in seen_ids:
            return
        seen_ids.add(normalized_id)
        candidates.append(
            {"product_id": normalized_id, "product_name": normalized_name},
        )

    for message in messages[latest_human_index + 1 :]:
        if not isinstance(message, ToolMessage):
            continue
        payload = _tool_payload(message)
        if message.name == "search_products" and isinstance(payload, list):
            for product in payload:
                if isinstance(product, dict):
                    add_candidate(product.get("id"), product.get("name"))
        elif message.name == "search_product_knowledge" and isinstance(payload, dict):
            for product in payload.get("resolved_products", []):
                if isinstance(product, dict):
                    add_candidate(product.get("id"), product.get("name"))
        elif message.name == "check_product_stock" and isinstance(payload, dict):
            add_candidate(payload.get("product_id"), payload.get("product_name"))

            for alternative in payload.get("alternatives", []):
                if isinstance(alternative, dict):
                    add_candidate(alternative.get("id"), alternative.get("name"))

    if response_text is None or len(candidates) <= 1:
        return candidates


    mentioned_by_id = [
        candidate
        for candidate in candidates
        if re.search(
            rf"(?<!\w){re.escape(candidate['product_id'])}(?!\w)",
            response_text,
            flags=re.IGNORECASE,
        )
    ]
    if mentioned_by_id:
        return mentioned_by_id

    normalized_response = response_text.casefold()
    return [
        candidate
        for candidate in candidates
        if candidate["product_name"].casefold() in normalized_response
    ]


def _product_candidates_from_ids(product_ids: list[str]) -> list[dict[str, str]]:
    """Khôi phục ngữ cảnh sản phẩm ảnh bằng dữ liệu catalog, không để model đoán."""
    candidates: list[dict[str, str]] = []
    for product_id in product_ids:
        product = find_product(product_id)
        if product is not None:
            candidates.append(
                {"product_id": product["id"], "product_name": product["name"]}
            )
    return candidates


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
    skin_product_ids = session.get("pending_skin_product_ids", [])
    skin_followup_context = (
        "Khách đang trả lời lời mời hướng dẫn chi tiết sau khi phân tích ảnh da. "
        f"Các SKU vừa được gợi ý là: {', '.join(skin_product_ids)}. "
        "Nếu khách đồng ý, dùng công cụ tra các SKU này rồi hướng dẫn một quy trình "
        "ngắn, rõ thứ tự và cách dùng. Nếu khách từ chối, xác nhận ngắn gọn và không "
        "gợi ý thêm. Không chẩn đoán tình trạng da từ ảnh."
        if session.get("current_step") == "awaiting_skin_guidance_confirmation"
        else "Không có lượt tư vấn ảnh nào đang chờ phản hồi."
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
            f"{skin_followup_context}\n"
            "Quy tắc sử dụng công cụ:\n"
            "- Dùng search_product_knowledge khi hỏi thành phần, công dụng, cách dùng, "
            "độ phù hợp hoặc so sánh sản phẩm.\n"
            "- Dùng search_products khi cần tìm theo mã, tên hoặc danh mục.\n"
            "- Dùng check_product_stock khi hỏi tồn kho hoặc số lượng có thể mua.\n"
            "  Khi thiếu hàng, truyền loại da và thành phần cần loại trừ từ hồ sơ trên "
            "để nhận gợi ý thay thế.\n"
            f"{profile_context}\n"
            "- Khi giới thiệu, mô tả hoặc đề xuất sản phẩm trong câu trả lời cuối, luôn "
            "ghi kèm mã SKU lấy từ kết quả công cụ để duy trì đúng ngữ cảnh hội thoại.\n"
            "- Chỉ trả lời thông tin sản phẩm từ kết quả công cụ; không tự suy đoán."
        )
    )
    messages = state.get("messages", [])
    response = model_with_tools.invoke([system_message, *messages])

    # Lượt yêu cầu tool thường chưa có nội dung trả lời. Giữ nguyên AIMessage
    # để tools_condition đọc được response.tool_calls.
    reply = response.text
    update: dict[str, Any] = {"reply": reply, "messages": [response]}
    # Persist context after tool use so the next turn can resolve a phrase such as
    # "mua 2 lo" to the single product that was just discussed.
    if not response.tool_calls:
        candidates = _recent_product_candidates(messages, reply)
        if not candidates and session.get("current_step") == "awaiting_skin_guidance_confirmation":
            candidates = _product_candidates_from_ids(skin_product_ids)
        session_update: dict[str, Any] = {}
        # Không xóa ngữ cảnh cũ chỉ vì lượt hiện tại không gọi tool sản phẩm.
        # Điều này cho phép khách chuyển thẳng sang đặt hàng ở lượt kế tiếp.
        if candidates:
            session_update["last_product_candidates"] = candidates
        if session.get("current_step") == "awaiting_skin_guidance_confirmation":
            session_update.update(
                {
                    "current_step": "idle",
                    "pending_skin_product_ids": [],
                }
            )
        update["session"] = session_update
    return update
