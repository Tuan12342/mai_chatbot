import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from mai_agent.config import get_settings
from mai_agent.llm import invoke_tool
from mai_agent.routing import latest_user_text
from mai_agent.state import AgentState
from mai_agent.vision.knowledge_base import knowledge_for_products
from mai_agent.vision.recommendation import eligible_catalog_products

ConsultationSubject = Literal["self", "other", "unknown"]
ConsultationSkinType = Literal[
    "da dầu",
    "da khô",
    "da hỗn hợp",
    "da nhạy cảm",
    "da thường",
]
ProductCategory = Literal[
    "cleanser",
    "toner",
    "serum",
    "moisturizer",
    "sunscreen",
    "mask",
    "makeup_remover",
    "routine",
    "unspecified",
]

CATEGORY_TO_CATALOG = {
    "cleanser": "Sữa rửa mặt",
    "toner": "Toner",
    "serum": "Serum",
    "moisturizer": "Kem dưỡng",
    "sunscreen": "Kem chống nắng",
    "mask": "Mặt nạ",
    "makeup_remover": "Tẩy trang",
}


class ProductRecommendationChoice(BaseModel):
    product_id: str
    evidence_keys: list[str] = Field(min_length=1, max_length=3)


@tool
def recommend_now(
    request: str,
    subject: ConsultationSubject,
    skin_type: ConsultationSkinType | None = None,
    concerns: list[str] | None = None,
    product_category: ProductCategory = "unspecified",
    excluded_ingredients: list[str] | None = None,
) -> dict[str, Any]:
    """Yêu cầu đã đủ rõ để chọn và tư vấn sản phẩm ngay."""
    return {
        "request": request,
        "subject": subject,
        "skin_type": skin_type,
        "concerns": concerns or [],
        "product_category": product_category,
        "excluded_ingredients": excluded_ingredients or [],
    }


@tool
def ask_recommendation_clarification(
    question: str,
    request_so_far: str = "",
    missing_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Thiếu thông tin thiết yếu nên cần hỏi đúng một câu trước khi tư vấn."""
    return {
        "question": question,
        "request_so_far": request_so_far,
        "missing_fields": missing_fields or [],
    }


@tool
def choose_recommended_products(
    recommendations: list[ProductRecommendationChoice],
) -> dict[str, Any]:
    """Chọn SKU và dẫn đúng khóa bằng chứng từ danh sách ứng viên đã lọc."""
    return {
        "recommendations": [recommendation.model_dump() for recommendation in recommendations]
    }


RECOMMENDATION_ACTION_TOOLS = [recommend_now, ask_recommendation_clarification]
RECOMMENDATION_SELECTION_TOOLS = [choose_recommended_products]


def _interpret_recommendation_action(
    text: str,
    pending: dict[str, Any],
    customer: dict[str, Any],
    response_language: str,
) -> dict[str, Any]:
    language_name = "English" if response_language == "en" else "Vietnamese"
    call = invoke_tool(
        RECOMMENDATION_ACTION_TOOLS,
        [
            SystemMessage(
                content=(
                    "Hiểu yêu cầu tư vấn mỹ phẩm và bắt buộc gọi đúng một tool. "
                    "subject=self khi khách đang hỏi cho chính họ, kể cả khi chủ ngữ "
                    "được lược bỏ theo hội thoại tự nhiên; subject=other khi khách nói "
                    "rõ đang hỏi cho mẹ, bạn bè hoặc người khác; chỉ dùng unknown khi "
                    "thực sự không xác định được. Trích xuất skin_type, concerns, nhóm "
                    "sản phẩm và thành phần cần tránh CHỈ từ yêu cầu hiện tại hoặc yêu "
                    "cầu đang chờ. Không lấy loại da trong trusted_profile để điền vào "
                    "skin_type của request. Gọi recommend_now khi khách đã nêu loại da, "
                    "vấn đề, mục tiêu, loại sản phẩm hoặc đang bổ sung đủ cho yêu cầu còn "
                    "chờ. Một nhu cầu như sản phẩm cho da có mụn, khô, bóng dầu hoặc đỏ "
                    "đã đủ để tư vấn; không bắt khách chọn trước danh mục. Chỉ gọi "
                    "ask_recommendation_clarification khi cả yêu cầu và hồ sơ phù hợp đều "
                    "không cho biết nhu cầu. Khi có pending_request, ghép ý nghĩa câu mới "
                    "với yêu cầu đó nhưng không biến dữ liệu của người khác thành hồ sơ "
                    f"khách. Câu hỏi làm rõ phải viết bằng {language_name}."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "current_message": text,
                        "pending_request": pending,
                        "trusted_profile_for_self_only": {
                            "skin_type": customer.get("skin_type"),
                            "concerns": customer.get("concerns", []),
                            "excluded_ingredients": customer.get(
                                "excluded_ingredients", []
                            ),
                        },
                    },
                    ensure_ascii=False,
                )
            ),
        ],
        error_message="Model không chọn được hành động tư vấn sản phẩm.",
    )
    args = dict(call.get("args", {}))
    if call["name"] == "recommend_now":
        request = str(args.get("request", "")).strip()
        if not request:
            request = str(pending.get("request", "")).strip() or text
        return {
            "action": "recommend",
            "request": request,
            "subject": args.get("subject", "unknown"),
            "skin_type": args.get("skin_type"),
            "concerns": list(args.get("concerns") or []),
            "product_category": args.get("product_category", "unspecified"),
            "excluded_ingredients": list(args.get("excluded_ingredients") or []),
        }
    if call["name"] == "ask_recommendation_clarification":
        question = str(args.get("question", "")).strip()
        if not question:
            raise RuntimeError("Model không cung cấp câu hỏi làm rõ.")
        request_so_far = str(args.get("request_so_far", "")).strip()
        return {
            "action": "clarify",
            "question": question,
            "request": request_so_far or str(pending.get("request", "")).strip() or text,
            "missing_fields": list(args.get("missing_fields") or []),
        }
    raise RuntimeError(f"Model gọi tool tư vấn không được hỗ trợ: {call['name']}")


def _grounded_knowledge(
    product_ids: list[str],
    response_language: str,
) -> list[dict[str, Any]]:
    grounded: list[dict[str, Any]] = []
    for record in knowledge_for_products(product_ids):
        if response_language == "en":
            reasons = record.get("why_en") or record.get("why", [])
            cautions = record.get("cautions_en") or record.get("cautions", [])
            routine_role = record.get("routine_role_en") or record["routine_role"]
        else:
            reasons = record.get("why", [])
            cautions = record.get("cautions", [])
            routine_role = record["routine_role"]
        evidence = {
            f"why_{index}": text
            for index, text in enumerate(reasons, start=1)
        }
        grounded.append(
            {
                "product_id": record["product_id"],
                "routine_role": routine_role,
                "evidence": evidence,
                "cautions": cautions,
            }
        )
    return grounded


def _select_recommendation_products(
    context: dict[str, Any],
    *,
    max_products: int,
) -> list[dict[str, Any]]:
    candidates = {
        str(product.get("id", "")).strip().upper(): product
        for product in context.get("candidate_products", [])
        if isinstance(product, dict) and product.get("id")
    }
    knowledge = {
        str(record.get("product_id", "")).strip().upper(): record
        for record in context.get("product_knowledge", [])
        if isinstance(record, dict) and record.get("product_id")
    }
    call = invoke_tool(
        RECOMMENDATION_SELECTION_TOOLS,
        [
            SystemMessage(
                content=(
                    "Chọn sản phẩm mỹ phẩm bằng suy luận ngữ nghĩa và bắt buộc gọi "
                    "choose_recommended_products. Mỗi lựa chọn phải gồm product_id và "
                    "ít nhất một evidence_key có thật trong evidence của đúng SKU. Chỉ "
                    "chọn SKU trong candidate_products; không tạo SKU hoặc lý do mới. "
                    "Tôn trọng request_context, loại da, vấn đề, danh mục và thành phần "
                    "cần tránh. Với routine hoặc yêu cầu chung, không chọn hai sản phẩm "
                    "cùng danh mục. Không vượt quá max_products. Nếu không đủ căn cứ, "
                    "trả recommendations rỗng. Không mô tả mỹ phẩm như thuốc."
                )
            ),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
        ],
        error_message="Model không chọn được sản phẩm tư vấn.",
    )
    if call["name"] != "choose_recommended_products":
        return []

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_categories: set[str] = set()
    requested_category = context.get("request_context", {}).get(
        "product_category", "unspecified"
    )
    raw_recommendations = call.get("args", {}).get("recommendations", [])
    if not isinstance(raw_recommendations, list):
        return []

    for raw_choice in raw_recommendations:
        if not isinstance(raw_choice, dict):
            continue
        product_id = str(raw_choice.get("product_id", "")).strip().upper()
        if product_id in seen_ids or product_id not in candidates or product_id not in knowledge:
            continue
        allowed_evidence = knowledge[product_id].get("evidence", {})
        raw_evidence_keys = raw_choice.get("evidence_keys")
        if not isinstance(raw_evidence_keys, list):
            continue
        evidence_keys = list(
            dict.fromkeys(
                key.strip()
                for key in raw_evidence_keys
                if isinstance(key, str) and key.strip()
            )
        )
        if not evidence_keys or any(key not in allowed_evidence for key in evidence_keys):
            continue
        category = str(candidates[product_id].get("category", ""))
        if requested_category in {"routine", "unspecified"} and category in seen_categories:
            continue
        validated.append(
            {
                "product_id": product_id,
                "evidence_keys": evidence_keys[:3],
            }
        )
        seen_ids.add(product_id)
        seen_categories.add(category)
        if len(validated) >= max_products:
            break
    return validated


def _candidate_payload(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "skin_types": product["skin_types"],
        "ingredients": product["ingredients"],
        "benefits": product["benefits"],
        "usage": product["usage"],
        "price_vnd": product["price_vnd"],
        "stock": product["stock"],
    }


def _effective_profile(
    action: dict[str, Any],
    customer: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    subject = action.get("subject", "unknown")
    if subject == "self":
        profile = dict(customer)
        skin_type = action.get("skin_type") or customer.get("skin_type")
        excluded = list(customer.get("excluded_ingredients", []))
        excluded.extend(action.get("excluded_ingredients", []))
        profile["excluded_ingredients"] = list(dict.fromkeys(excluded))
        return profile, skin_type

    # Không áp hồ sơ của người gửi cho người khác hoặc một chủ thể chưa xác định.
    return {
        "excluded_ingredients": list(action.get("excluded_ingredients", [])),
        "allergies": [],
        "sensitivities": [],
        "disliked_product_ids": [],
    }, action.get("skin_type")


def _filter_requested_category(
    products: list[dict[str, Any]],
    product_category: str,
) -> list[dict[str, Any]]:
    catalog_category = CATEGORY_TO_CATALOG.get(product_category)
    if catalog_category is None:
        return products
    return [product for product in products if product.get("category") == catalog_category]


def _format_price(price_vnd: int) -> str:
    return f"{price_vnd:,}".replace(",", ".") + " VNĐ"


def _render_recommendations(
    selected: list[dict[str, Any]],
    *,
    response_language: str,
) -> str:
    if not selected:
        if response_language == "en":
            return (
                "I couldn't find an in-stock product with enough verified information "
                "for this request. Could you share the main skin concern or product type?"
            )
        return (
            "Mình chưa tìm thấy sản phẩm còn hàng có đủ căn cứ cho nhu cầu này. "
            "Bạn cho mình biết vấn đề da chính hoặc loại sản phẩm mong muốn nhé?"
        )

    if response_language == "en":
        intro = "You can consider the following products:"
        role_label = "Role"
        reason_label = "Why it fits"
        usage_label = "How to use"
        caution_label = "Note"
        outro = "Would you like me to arrange these into a morning or evening routine?"
    else:
        intro = "Bạn có thể tham khảo các sản phẩm sau:"
        role_label = "Vai trò"
        reason_label = "Lý do phù hợp"
        usage_label = "Cách dùng"
        caution_label = "Lưu ý"
        outro = "Bạn muốn mình sắp xếp các sản phẩm này thành routine sáng hay tối?"

    blocks = [intro]
    for index, item in enumerate(selected, start=1):
        product = item["product"]
        evidence = "; ".join(item["evidence"])
        block = [
            f"{index}. **{product['name']} ({product['id']})** — "
            f"{_format_price(product['price_vnd'])}",
            f"- {role_label}: {item['routine_role']}",
            f"- {reason_label}: {evidence}",
            f"- {usage_label}: {product['usage']}",
        ]
        if item.get("caution"):
            block.append(f"- {caution_label}: {item['caution']}")
        blocks.append("\n".join(block))
    blocks.append(outro)
    return "\n\n".join(blocks)


def recommendation_node(state: AgentState) -> dict[str, Any]:
    session = state.get("session", {})
    customer = state.get("customer", {})
    response_language = session.get("language_code", "vi")
    pending = session.get("pending_recommendation") or {}
    request_text = (
        str(pending.get("request", "")).strip()
        if state.get("profile_memory_resume_intent") == "recommendation"
        else latest_user_text(state)
    )
    action = _interpret_recommendation_action(
        request_text or latest_user_text(state),
        pending,
        customer,
        response_language,
    )

    if action["action"] == "clarify":
        question = action["question"]
        return {
            "reply": question,
            "messages": [AIMessage(content=question)],
            "profile_memory_resume_intent": None,
            "session": {
                "current_step": "awaiting_recommendation_clarification",
                "pending_recommendation": {
                    "request": action["request"],
                    "missing_fields": action.get("missing_fields", []),
                    "last_question": question,
                },
            },
        }

    effective_profile, effective_skin_type = _effective_profile(action, customer)
    products = eligible_catalog_products(
        effective_profile,
        effective_skin_type,
        care_mode="standard",
    )
    products = _filter_requested_category(
        products,
        action.get("product_category", "unspecified"),
    )
    candidate_products = [_candidate_payload(product) for product in products]
    candidate_knowledge = _grounded_knowledge(
        [product["id"] for product in candidate_products],
        response_language,
    )
    max_products = max(1, min(get_settings().skin_recommendation_max_products, 5))
    selection_context = {
        "request_context": {
            "request": action["request"],
            "subject": action.get("subject", "unknown"),
            "skin_type": effective_skin_type or "not_provided",
            "concerns": action.get("concerns", []),
            "product_category": action.get("product_category", "unspecified"),
            "excluded_ingredients": effective_profile.get("excluded_ingredients", []),
        },
        "max_products": max_products,
        "candidate_products": candidate_products,
        "product_knowledge": candidate_knowledge,
    }
    choices = (
        _select_recommendation_products(selection_context, max_products=max_products)
        if candidate_products
        else []
    )
    products_by_id = {product["id"]: product for product in candidate_products}
    knowledge_by_id = {
        record["product_id"]: record for record in candidate_knowledge
    }
    selected: list[dict[str, Any]] = []
    for choice in choices:
        product_id = choice["product_id"]
        record = knowledge_by_id[product_id]
        selected.append(
            {
                "product": products_by_id[product_id],
                "routine_role": record["routine_role"],
                "evidence": [record["evidence"][key] for key in choice["evidence_keys"]],
                "caution": next(iter(record.get("cautions", [])), ""),
            }
        )

    reply = _render_recommendations(selected, response_language=response_language)
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "profile_memory_resume_intent": None,
        "session": {
            "current_step": "idle",
            "pending_recommendation": None,
            "pending_resume_intent": None,
            "pending_resume_request": None,
            "last_product_candidates": [
                {
                    "product_id": item["product"]["id"],
                    "product_name": item["product"]["name"],
                }
                for item in selected
            ],
        },
    }
