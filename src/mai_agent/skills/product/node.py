import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mai_agent.catalog import find_product, product_alias_candidates
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
        if message.name in {"search_products", "recommend_products"} and isinstance(payload, list):
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

    # Tool searches may return a broad internal result set. Prefer SKUs explicitly
    # surfaced in the final answer so follow-up purchases do not become falsely
    # ambiguous. Product names are a fallback for responses that omit the SKU.
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


def _format_price(price_vnd: int, language: str) -> str:
    separator = "." if language in {"vi", "es"} else ","
    return f"{price_vnd:,}".replace(",", separator)


PRODUCT_DISPLAY_NAMES = {
    "OA001": "OA Gentle Cleanser",
    "OA002": "OA Purifying Gel",
    "OA003": "OA Hydrating Toner",
    "OA004": "OA Barrier Repair Serum",
    "OA005": "OA Brightening Serum",
    "OA006": "OA Ceramide Cream",
    "OA007": "OA Oil-Free Gel Cream",
    "OA008": "OA Daily UV Fluid SPF 50+ PA++++",
    "OA009": "OA Soothing Mask",
    "OA010": "OA Micellar Water",
}

SKIN_TYPE_TRANSLATIONS = {
    "en": {
        "da dầu": "oily skin",
        "da hỗn hợp": "combination skin",
        "da khô": "dry skin",
        "da nhạy cảm": "sensitive skin",
        "da thường": "normal skin",
        "mọi loại da": "all skin types",
    },
    "zh": {
        "da dầu": "油性皮肤",
        "da hỗn hợp": "混合性皮肤",
        "da khô": "干性皮肤",
        "da nhạy cảm": "敏感性皮肤",
        "da thường": "中性皮肤",
        "mọi loại da": "所有肤质",
    },
    "ko": {
        "da dầu": "지성 피부",
        "da hỗn hợp": "복합성 피부",
        "da khô": "건성 피부",
        "da nhạy cảm": "민감성 피부",
        "da thường": "중성 피부",
        "mọi loại da": "모든 피부 타입",
    },
    "es": {
        "da dầu": "piel grasa",
        "da hỗn hợp": "piel mixta",
        "da khô": "piel seca",
        "da nhạy cảm": "piel sensible",
        "da thường": "piel normal",
        "mọi loại da": "todo tipo de piel",
    },
}


def _direct_catalog_products(text: str) -> list[dict[str, Any]]:
    """Resolve explicit SKU first, then deterministic aliases."""
    explicit_ids = re.findall(r"\bOA\d{3}\b", text, flags=re.IGNORECASE)
    product_ids = [product_id.upper() for product_id in explicit_ids]
    if not product_ids:
        product_ids = product_alias_candidates(text)
    return [
        product
        for product_id in dict.fromkeys(product_ids)
        if (product := find_product(product_id)) is not None
    ]


def _localized_product_fact(product: dict[str, Any], fact: str, language: str) -> str:
    sku = product["id"]
    name = product["name"] if language == "vi" else PRODUCT_DISPLAY_NAMES.get(sku, product["name"])
    price = _format_price(product["price_vnd"], language)
    translations = SKIN_TYPE_TRANSLATIONS.get(language, {})
    skin_types = ", ".join(translations.get(value, value) for value in product["skin_types"])
    ingredients = ", ".join(product["ingredients"])
    usage = product["usage"]
    if language == "en":
        templates = {
            "price": f"{name} ({sku}) costs {price} VND.",
            "skin": f"{name} ({sku}) is suitable for: {skin_types}.",
            "ingredients": f"The main ingredients of {name} ({sku}) are {ingredients}.",
            "usage": f"How to use {name} ({sku}): {usage}",
            "sunscreen": f"Yes. {name} ({sku}) has SPF 50+ PA++++ and costs {price} VND.",
        }
    elif language == "zh":
        templates = {
            "price": f"{name}（{sku}）的价格是 {price} 越南盾。",
            "skin": f"{name}（{sku}）适合以下肤质：{skin_types}。",
            "ingredients": f"{name}（{sku}）的主要成分是：{ingredients}。",
            "usage": f"{name}（{sku}）的使用方法：{usage}",
            "sunscreen": f"有。{name}（{sku}）是 SPF 50+ PA++++，价格为 {price} 越南盾。",
        }
    elif language == "ko":
        templates = {
            "price": f"{name}({sku})의 가격은 {price}동입니다.",
            "skin": f"{name}({sku})는 다음 피부 타입에 적합합니다: {skin_types}.",
            "ingredients": f"{name}({sku})의 주요 성분은 {ingredients}입니다.",
            "usage": f"{name}({sku}) 사용법: {usage}",
            "sunscreen": f"네. {name}({sku})는 SPF 50+ PA++++이며 가격은 {price}동입니다.",
        }
    elif language == "es":
        templates = {
            "price": f"{name} ({sku}) cuesta {price} VND.",
            "skin": f"{name} ({sku}) es adecuado para: {skin_types}.",
            "ingredients": f"Los ingredientes principales de {name} ({sku}) son {ingredients}.",
            "usage": f"Modo de uso de {name} ({sku}): {usage}",
            "sunscreen": f"Sí. {name} ({sku}) tiene SPF 50+ PA++++ y cuesta {price} VND.",
        }
    else:
        templates = {
            "price": f"{name} ({sku}) có giá {price}đ.",
            "skin": f"{name} ({sku}) phù hợp với: {skin_types}.",
            "ingredients": f"Thành phần chính của {name} ({sku}) là {ingredients}.",
            "usage": f"Cách dùng {name} ({sku}): {usage}",
            "sunscreen": f"Có ạ. {name} ({sku}) có SPF 50+ PA++++, giá {price}đ.",
        }
    return templates[fact]


def _direct_catalog_answer(
    text: str,
    language: str,
    intent: str,
    customer: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer simple catalog facts without an embedding or model round-trip."""
    products = _direct_catalog_products(text)
    normalized = text.casefold()

    if intent == "recommendation" and len(products) > 1:
        skin_type = customer.get("skin_type")
        if "da dầu" in normalized:
            skin_type = "da dầu"
        matches = [
            product
            for product in products
            if skin_type in product["skin_types"] or "mọi loại da" in product["skin_types"]
        ]
        if len(matches) == 1:
            product = matches[0]
            return _localized_product_fact(product, "skin", language), matches

    if len(products) != 1:
        return None
    product = products[0]
    if re.search(
        r"giá|bao nhiêu|how much|price|cost|多少钱|价格|얼마|가격|cuánto|precio",
        normalized,
    ):
        fact = "price"
    elif re.search(r"spf|防晒霜|선크림|protector solar", normalized):
        fact = "sunscreen"
    elif re.search(r"thành phần|hoạt chất|ingredient|成分|성분|ingrediente", normalized):
        fact = "ingredients"
    elif re.search(r"dùng|bước|đắp|use|apply|使用|사용|cómo se usa", normalized):
        fact = "usage"
    elif re.search(r"hợp|loại da|suitable|skin type|适合|피부|adecuado", normalized):
        fact = "skin"
    else:
        return None
    return _localized_product_fact(product, fact, language), products


def assistant_node(state: AgentState) -> dict[str, Any]:
    customer = state.get("customer", {})
    session = state.get("session", {})
    response_language = session.get("language_code", "vi")
    language_name = {
        "vi": "Vietnamese",
        "en": "English",
        "zh": "Simplified Chinese",
        "ko": "Korean",
        "es": "Spanish",
    }.get(response_language, "Vietnamese")
    pending_product_reference = session.get("pending_product_reference")
    latest_text = next(
        (
            str(message.content)
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage)
        ),
        "",
    )
    direct_answer = _direct_catalog_answer(
        latest_text,
        response_language,
        str(session.get("current_intent", "")),
        customer,
    )
    if direct_answer is not None:
        reply, matched_products = direct_answer
        candidates = [
            {"product_id": product["id"], "product_name": product["name"]}
            for product in matched_products
        ]
        return {
            "reply": reply,
            "messages": [AIMessage(content=reply)],
            "session": {"last_product_candidates": candidates},
        }

    catalog_matches = _direct_catalog_products(latest_text)
    catalog_context = (
        "Catalog resolver đã ánh xạ cách gọi của khách thành: "
        + ", ".join(f"{product['id']} - {product['name']}" for product in catalog_matches)
        + ". Phải truyền các SKU này vào công cụ; nếu có nhiều SKU thì không tự chọn."
        if catalog_matches
        else "Catalog resolver chưa có ánh xạ chắc chắn cho cách gọi trong câu hiện tại."
    )
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
        f"- Vấn đề quan tâm: {customer.get('concerns', [])}\n"
        f"- Dị ứng/nhạy cảm: {customer.get('allergies', []) + customer.get('sensitivities', [])}\n"
        f"- Không gợi ý các SKU: {customer.get('disliked_product_ids', [])}\n"
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
            f"{catalog_context}\n"
            "Quy tắc sử dụng công cụ:\n"
            "- Dùng search_product_knowledge khi hỏi thành phần, công dụng, cách dùng, "
            "độ phù hợp hoặc so sánh sản phẩm.\n"
            "- Dùng search_products khi cần tìm theo mã, tên hoặc danh mục.\n"
            "- Khi khách cần gợi ý/lựa chọn sản phẩm, bắt buộc dùng recommend_products "
            "và truyền đầy đủ loại da, vấn đề quan tâm, dị ứng/nhạy cảm, thành phần "
            "loại trừ, SKU không thích từ hồ sơ. "
            "Không đề xuất SKU nằm trong danh sách không thích.\n"
            "- Dùng check_product_stock khi hỏi tồn kho hoặc số lượng có thể mua.\n"
            "  Khi thiếu hàng, truyền loại da, dị ứng/nhạy cảm, thành phần cần loại trừ "
            "từ hồ sơ trên cùng danh sách SKU không thích để nhận gợi ý thay thế.\n"
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
        update["session"] = {"last_product_candidates": candidates}
    return update
