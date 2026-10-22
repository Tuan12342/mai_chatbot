import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from mai_agent.catalog import extract_product_mentions, find_product
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
        elif message.name == "search_products_by_ingredient" and isinstance(payload, dict):
            for product in payload.get("products", []):
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


def _has_product_tool_result_for_latest_user(messages: list[Any]) -> bool:
    """Chỉ cho phép trả lời factual sau khi lượt hiện tại đã tra dữ liệu shop."""
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index
    return any(
        isinstance(message, ToolMessage) and message.name in {tool.name for tool in PRODUCT_TOOLS}
        for message in messages[latest_human_index + 1 :]
    )


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
    messages = state.get("messages", [])
    latest_user_message = next(
        (
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    resolved_mentions = extract_product_mentions(latest_user_message)
    resolved_mention_context = (
        "Catalog resolver đã xác định được các SKU được gọi đích danh trong tin nhắn: "
        f"{', '.join(resolved_mentions)}. Phải dùng các SKU này khi gọi công cụ và không "
        "yêu cầu khách nêu lại tên."
        if resolved_mentions
        else "Catalog resolver chưa xác định được một SKU duy nhất từ cách gọi hiện tại."
    )
    must_retrieve_product_data = (
        session.get("current_step") != "awaiting_skin_guidance_confirmation"
        and not _has_product_tool_result_for_latest_user(messages)
    )
    bind_options = {"tool_choice": "any"} if must_retrieve_product_data else {}
    model_with_tools = create_chat_model().bind_tools(PRODUCT_TOOLS, **bind_options)
    system_message = SystemMessage(
        content=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Ngôn ngữ bắt buộc của câu trả lời hiện tại: {language_name}.\n"
            "Dù ngôn ngữ trả lời thay đổi, phải tiếp tục dùng hồ sơ, lịch sử và "
            "trạng thái phiên đã có. Không dịch SKU, tên riêng sản phẩm, thương hiệu "
            "hoặc tên thành phần INCI trong kết quả công cụ.\n"
            f"{pending_product_context}\n"
            f"{skin_followup_context}\n"
            f"{resolved_mention_context}\n"
            "Quy tắc sử dụng công cụ:\n"
            "- Khi khách hỏi shop có/chứa thành phần X hoặc sản phẩm nào có X, bắt "
            "buộc dùng search_products_by_ingredient. Thành phần phải khớp dữ liệu "
            "ingredients trong catalog; tuyệt đối không lấy công dụng gần giống để "
            "khẳng định sản phẩm chứa X.\n"
            "- Khi khách hỏi thành phần X có vai trò gì trong sản phẩm, dùng "
            "search_products_by_ingredient cho X. Chỉ liên hệ với lợi ích tổng thể "
            "được catalog ghi cho sản phẩm; không tự gán một công dụng chưa có evidence "
            "cho riêng thành phần.\n"
            "- Phân biệt câu hỏi 'sản phẩm nào có thành phần X' với câu hỏi 'thành "
            "phần của sản phẩm X là gì'. Với câu hỏi thành phần của sản phẩm/danh mục "
            "đã nêu, phải tra sản phẩm đó; không được coi tên hoặc tính chất sản phẩm "
            "như 'dịu nhẹ', 'toner', 'mặt nạ', 'tẩy trang' là tên thành phần.\n"
            "  Một từ đứng trong tên/mô tả sản phẩm không trở thành ingredient chỉ vì "
            "câu hỏi có chữ 'thành phần'. Hãy ưu tiên resolve toàn bộ cụm sản phẩm bằng "
            "search_product_knowledge hoặc search_products trước.\n"
            "- Danh sách ingredients là các thành phần nổi bật, không phải bảng INCI "
            "đầy đủ. Nếu không thấy thành phần X, chỉ nói tài liệu shop chưa liệt kê X; "
            "không được suy ra 'không chứa X', 'X-free' hoặc an toàn với người cần "
            "tránh X. Nếu khách cần tránh thành phần đó, đề nghị xác nhận với shop.\n"
            "- Khi công cụ không tìm thấy sản phẩm hoặc thành phần, hãy tự diễn đạt "
            "tự nhiên rằng catalog hiện không có dữ liệu khớp. Không nói shop có nhiều "
            "phiên bản, không tự đưa sản phẩm gần giống và không dùng một câu trả lời "
            "mẫu cố định. Chỉ hỏi lại khi tham chiếu thật sự mơ hồ hoặc có vẻ gõ sai.\n"
            "- Dùng search_product_knowledge khi hỏi thành phần, công dụng, cách dùng, "
            "độ phù hợp hoặc so sánh một sản phẩm đã xác định rõ.\n"
            "- Khi khách so sánh nhiều sản phẩm đã nêu rõ, truyền đầy đủ tất cả tên/mã "
            "vào product_references, trình bày điểm liên quan của từng sản phẩm rồi "
            "kết luận trực tiếp sản phẩm nào phù hợp hơn theo đúng tiêu chí khách hỏi. "
            "Không bỏ bớt một sản phẩm, không biến câu hỏi thành danh sách gợi ý chung "
            "và không hỏi khách có muốn sắp xếp routine hay không.\n"
            "- Dùng search_products khi cần tìm theo mã, tên hoặc danh mục.\n"
            "- Dùng check_product_stock khi hỏi tồn kho hoặc số lượng có thể mua.\n"
            "  Khi thiếu hàng, truyền loại da và thành phần cần loại trừ từ hồ sơ trên "
            "để nhận gợi ý thay thế.\n"
            f"{profile_context}\n"
            "- Khi khách hỏi thông tin về một sản phẩm đã nêu rõ, không dùng loại da "
            "hay thành phần loại trừ trong hồ sơ để đổi câu hỏi thành tư vấn sản phẩm.\n"
            "- Tự hiểu theo ngữ nghĩa xem khách đã gọi đích danh sản phẩm nào. Nếu tên "
            "tự nhiên đủ phân biệt với catalog, dùng chính tên đó để tra cứu và không "
            "hỏi lại SKU. Chỉ hỏi làm rõ khi thực sự còn từ hai lựa chọn hợp lệ trở lên.\n"
            "- Nếu kết quả công cụ chỉ resolve đúng một sản phẩm, chỉ trả lời về sản "
            "phẩm đó; không nêu sản phẩm khác trừ khi kết quả công cụ có alternatives.\n"
            "- Khi giới thiệu, mô tả hoặc đề xuất sản phẩm trong câu trả lời cuối, luôn "
            "ghi kèm mã SKU lấy từ kết quả công cụ để duy trì đúng ngữ cảnh hội thoại.\n"
            "- Chỉ trả lời thông tin sản phẩm từ kết quả công cụ; không tự suy đoán."
            " Không bổ sung kiến thức mỹ phẩm phổ thông nếu evidence không có. Nếu "
            "evidence không trả lời được đúng điều khách hỏi, nói rõ tài liệu shop "
            "chưa đề cập thay vì đưa lời khuyên hoặc khẳng định bên ngoài dữ liệu. Mọi "
            "con số, thời lượng, tần suất và điều kiện sử dụng trong câu trả lời cuối "
            "phải xuất hiện trong kết quả công cụ; không thêm ví dụ định lượng từ kiến "
            "thức chung."
        )
    )
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
