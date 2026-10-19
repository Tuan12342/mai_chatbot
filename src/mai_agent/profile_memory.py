import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from mai_agent.customer_store import find_customer_by_zalo_id, update_customer_profile
from mai_agent.llm import invoke_tool
from mai_agent.routing import latest_user_text
from mai_agent.skills.product.vector_store import resolve_product_references
from mai_agent.state import AgentState

PROFILE_MAX_AGE = timedelta(days=90)
LIST_PROFILE_FIELDS = (
    "sensitivities",
    "allergies",
    "concerns",
    "excluded_ingredients",
)


@tool
def remember_profile_information(
    skin_type: Literal["da dầu", "da khô", "da hỗn hợp", "da nhạy cảm", "da thường"] | None = None,
    sensitivities: list[str] | None = None,
    allergies: list[str] | None = None,
    concerns: list[str] | None = None,
    excluded_ingredients: list[str] | None = None,
    disliked_products: list[str] | None = None,
    cleared_fields: list[Literal["sensitivities", "allergies", "concerns", "excluded_ingredients"]]
    | None = None,
) -> str:
    """Ghi nhận CHỈ thông tin khách nói rõ về da, dị ứng hoặc sản phẩm không thích."""
    return "profile_information"


@tool
def no_profile_information() -> str:
    """Tin nhắn không trực tiếp cung cấp hay sửa thông tin hồ sơ da/sở thích."""
    return "no_profile_information"


@tool
def third_party_profile_context() -> str:
    """Thông tin da hoặc nhu cầu trong tin nhắn thuộc về một người khác, không phải khách."""
    return "third_party_profile_context"


@tool
def consultation_context_only() -> str:
    """Loại da/vấn đề chỉ là điều kiện tìm sản phẩm, không phải cập nhật hồ sơ cá nhân."""
    return "consultation_context_only"


PROFILE_EXTRACTION_TOOLS = [
    remember_profile_information,
    no_profile_information,
    third_party_profile_context,
    consultation_context_only,
]


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_accents = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return without_accents.replace("đ", "d")


def _is_affirmative(text: str) -> bool:
    normalized = _normalize_text(text).strip()
    return bool(
        re.search(
            r"\b(dung|co|vang|u|uh|ok|okay|van vay|chinh xac|yes|yup|correct)\b",
            normalized,
        )
    )


def _is_negative(text: str) -> bool:
    normalized = _normalize_text(text).strip()
    return bool(re.search(r"\b(khong|khong con|sai|no|not anymore)\b", normalized))


def _reply(text: str, *, session_update: dict[str, Any]) -> dict[str, Any]:
    return {
        "reply": text,
        "messages": [AIMessage(content=text)],
        "session": session_update,
        "profile_memory_handled": True,
    }


def _language(state: AgentState) -> str:
    return state.get("session", {}).get("language_code", "vi")


def _confirmation_text(updates: dict[str, Any], language: str) -> str:
    details = ", ".join(f"{key}: {value}" for key, value in updates.items())
    if language == "en":
        return f"I found a change in your skin profile ({details}). Should I save this update?"
    return f"Mình thấy hồ sơ da có thay đổi ({details}). Bạn xác nhận lưu cập nhật này nhé?"


def _is_explicit_memory_request(text: str) -> bool:
    return bool(re.search(r"\b(nhớ|lưu|ghi nhớ|remember|save)\b", text.casefold()))


def _saved_profile_text(updates: dict[str, Any], language: str) -> str:
    details = ", ".join(f"{key}: {value}" for key, value in updates.items())
    if language == "en":
        return f"I've saved only the information you just provided ({details})."
    return f"Mình đã lưu đúng thông tin bạn vừa cung cấp ({details})."


def _new_information_for_reply(extracted: dict[str, Any]) -> dict[str, Any]:
    """Keep confirmations scoped to this turn instead of echoing merged history."""
    information: dict[str, Any] = {}
    if extracted.get("skin_type"):
        information["skin_type"] = extracted["skin_type"]
    for field in LIST_PROFILE_FIELDS:
        if extracted.get(field):
            information[field] = extracted[field]
    if extracted.get("disliked_products"):
        information["disliked_products"] = extracted["disliked_products"]
    for field in extracted.get("cleared_fields", []):
        if field in LIST_PROFILE_FIELDS:
            information[field] = []
    return information


def _stale_text(customer: dict[str, Any], language: str) -> str:
    skin_type = customer.get("skin_type", "unknown")
    if language == "en":
        return (
            f"Your saved skin type is {skin_type}, but this profile is over 90 days old. "
            "Is it still correct?"
        )
    return (
        f"Lần trước bạn cho biết da của mình là {skin_type}, nhưng hồ sơ đã quá 90 ngày. "
        "Hiện vẫn vậy chứ?"
    )


def _profile_is_stale(customer: dict[str, Any]) -> bool:
    raw_timestamp = customer.get("last_updated")
    if not raw_timestamp:
        return True
    try:
        updated_at = datetime.fromisoformat(str(raw_timestamp))
    except ValueError:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.astimezone()
    return datetime.now().astimezone() - updated_at > PROFILE_MAX_AGE


def _with_profile_metadata(
    customer: dict[str, Any],
    updates: dict[str, Any],
    *,
    source: Literal["user_explicit", "user_reconfirmed"],
) -> dict[str, Any]:
    """Gắn nguồn và thời điểm xác nhận cho từng thuộc tính được lưu."""
    timestamp = datetime.now().astimezone().isoformat()
    metadata = dict(customer.get("profile_metadata", {}))
    for field in updates:
        if field in {"profile_metadata", "last_updated"}:
            continue
        metadata[field] = {
            "source": source,
            "updated_at": timestamp,
            "confirmed_at": timestamp,
        }
    return {**updates, "profile_metadata": metadata}


def _recommendation_resume_update(state: AgentState) -> dict[str, Any]:
    """Giữ yêu cầu gốc khi profile confirmation tạm ngắt luồng tư vấn."""
    if state.get("session", {}).get("current_intent") != "recommendation":
        return {}
    request = latest_user_text(state).strip()
    if not request:
        return {}
    return {
        "pending_resume_intent": "recommendation",
        "pending_resume_request": request,
    }


def _resume_recommendation_after_confirmation(
    state: AgentState,
    customer: dict[str, Any],
    cleared_pending: dict[str, Any],
) -> dict[str, Any] | None:
    session = state.get("session", {})
    request = str(session.get("pending_resume_request", "")).strip()
    if session.get("pending_resume_intent") != "recommendation" or not request:
        return None
    return {
        "customer": customer,
        "profile_memory_handled": True,
        "profile_memory_resume_intent": "recommendation",
        "session": {
            **cleared_pending,
            "current_intent": "recommendation",
            "pending_resume_intent": None,
            "pending_resume_request": None,
            "pending_recommendation": {"request": request},
        },
    }


def profile_confirmation_node(state: AgentState) -> dict[str, Any]:
    """Xử lý câu trả lời cho cập nhật đang chờ trước bước phân loại intent."""
    session = state.get("session", {})
    pending = session.get("pending_profile_update")
    reason = session.get("pending_profile_reason")
    if not pending:
        return {"profile_memory_handled": False}

    text = latest_user_text(state)
    language = _language(state)
    cleared_pending = {
        "pending_profile_update": None,
        "pending_profile_reason": None,
    }
    user_id = state.get("user_id", "")
    if _is_affirmative(text):
        source = "user_reconfirmed" if reason == "stale" else "user_explicit"
        confirmed_updates = _with_profile_metadata(
            state.get("customer", {}),
            pending,
            source=source,
        )
        customer = update_customer_profile(user_id, confirmed_updates)
        if resumed := _resume_recommendation_after_confirmation(
            state,
            customer,
            cleared_pending,
        ):
            return resumed
        reply = (
            "I've updated your skin profile and will use it for future consultations."
            if language == "en"
            else "Mình đã cập nhật hồ sơ da và sẽ dùng thông tin này cho các lần tư vấn sau."
        )
        result = _reply(reply, session_update=cleared_pending)
        result["customer"] = customer
        return result

    if _is_negative(text):
        if reason == "stale":
            customer = update_customer_profile(
                user_id,
                {
                    "skin_type": "",
                    "sensitivities": [],
                    "allergies": [],
                    "concerns": [],
                    "excluded_ingredients": [],
                },
            )
            if resumed := _resume_recommendation_after_confirmation(
                state,
                customer,
                cleared_pending,
            ):
                return resumed
            reply = (
                "I won't use the old profile. Please tell me your current skin type and concerns."
                if language == "en"
                else (
                    "Mình sẽ không dùng hồ sơ cũ. Bạn cho mình biết loại da và "
                    "vấn đề da hiện tại nhé."
                )
            )
            result = _reply(reply, session_update=cleared_pending)
            result["customer"] = customer
            return result
        if resumed := _resume_recommendation_after_confirmation(
            state,
            state.get("customer", {}),
            cleared_pending,
        ):
            return resumed
        reply = (
            "Okay, I haven't changed your saved profile."
            if language == "en"
            else "Được nhé, mình chưa thay đổi hồ sơ đã lưu."
        )
        return _reply(reply, session_update=cleared_pending)

    reply = (
        "Please confirm yes or no so I can update your skin profile safely."
        if language == "en"
        else "Bạn xác nhận có hoặc không để mình cập nhật hồ sơ da cho chính xác nhé."
    )
    return _reply(reply, session_update={})


def _extract_profile_information(text: str) -> dict[str, Any]:
    call = invoke_tool(
        PROFILE_EXTRACTION_TOOLS,
        [
            SystemMessage(
                content=(
                    "Trích xuất memory hồ sơ chỉ từ điều khách trực tiếp khẳng định "
                    "về CHÍNH HỌ trong tin nhắn hiện tại. Không lưu loại da, dị ứng, "
                    "vấn đề hoặc sở thích của mẹ, bạn bè, khách hàng hay người khác "
                    "vào hồ sơ người gửi; khi các thông tin đó thuộc về người khác, "
                    "gọi third_party_profile_context. Gọi consultation_context_only "
                    "khi loại da/vấn đề chỉ được dùng làm điều kiện hỏi hoặc tìm sản "
                    "phẩm, nhưng khách không khẳng định đó là hồ sơ của chính họ. Ví "
                    "dụ một câu hỏi chung về sản phẩm cho da khô không phải là yêu cầu "
                    "đổi hồ sơ; còn lời khẳng định rõ da của tôi/mình là da khô mới là "
                    "dữ liệu hồ sơ. Không suy luận loại da từ nhu cầu hoặc sản phẩm. "
                    "concerns là vấn đề như mụn, nám, lão hóa. "
                    "disliked_products chỉ gồm mã/tên sản phẩm khách nói không thích "
                    "hoặc không muốn được gợi ý. Nếu khách nói rõ không còn một nhóm "
                    "thông tin nào, đưa tên nhóm đó vào cleared_fields. Nếu không có "
                    "dữ liệu hồ sơ rõ ràng, gọi no_profile_information. Không tự điền "
                    "giá trị thiếu."
                )
            ),
            HumanMessage(content=text),
        ],
        error_message="Gemini không trích xuất được thông tin hồ sơ.",
    )
    if call["name"] == "third_party_profile_context":
        return {"_profile_subject": "other"}
    if call["name"] == "consultation_context_only":
        return {"_consultation_context_only": True}
    if call["name"] == "no_profile_information":
        return {}
    if call["name"] != "remember_profile_information":
        raise RuntimeError(f"Gemini gọi profile tool không được hỗ trợ: {call['name']}")
    return {
        key: value for key, value in call.get("args", {}).items() if value not in (None, "", [])
    }


def _merge_new_information(customer: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if skin_type := extracted.get("skin_type"):
        updates["skin_type"] = str(skin_type).strip().casefold()

    for field in LIST_PROFILE_FIELDS:
        new_values = extracted.get(field, [])
        if not new_values:
            continue
        existing = list(customer.get(field, []))
        seen = {str(value).casefold() for value in existing}
        updates[field] = existing + [
            str(value).strip()
            for value in new_values
            if str(value).strip() and str(value).casefold() not in seen
        ]

    for field in extracted.get("cleared_fields", []):
        if field in LIST_PROFILE_FIELDS:
            updates[field] = []

    references = extracted.get("disliked_products", [])
    if references:
        resolution = resolve_product_references(references)
        existing_ids = list(customer.get("disliked_product_ids", []))
        disliked_ids = existing_ids + [
            product["id"] for product in resolution["resolved"] if product["id"] not in existing_ids
        ]
        if disliked_ids != existing_ids:
            updates["disliked_product_ids"] = disliked_ids
    return updates


def profile_memory_node(state: AgentState) -> dict[str, Any]:
    """Học dữ liệu được nói rõ và chặn việc dùng hồ sơ cũ/chưa xác nhận."""
    customer = state.get("customer") or find_customer_by_zalo_id(state.get("user_id", "")) or {}
    latest_text = latest_user_text(state)
    extracted = _extract_profile_information(latest_text)
    is_third_party_request = extracted.pop("_profile_subject", None) == "other"
    is_consultation_context_only = bool(extracted.pop("_consultation_context_only", False))
    updates = _merge_new_information(customer, extracted)
    language = _language(state)

    old_skin_type = str(customer.get("skin_type", "")).casefold()
    new_skin_type = str(updates.get("skin_type", "")).casefold()
    if old_skin_type and new_skin_type and old_skin_type != new_skin_type:
        reply = _confirmation_text(updates, language)
        return _reply(
            reply,
            session_update={
                "pending_profile_update": updates,
                "pending_profile_reason": "changed",
                **_recommendation_resume_update(state),
            },
        )

    cleared_existing_fields = [
        field
        for field in LIST_PROFILE_FIELDS
        if field in updates and not updates[field] and customer.get(field)
    ]
    if cleared_existing_fields:
        reply = _confirmation_text(updates, language)
        return _reply(
            reply,
            session_update={
                "pending_profile_update": updates,
                "pending_profile_reason": "changed",
                **_recommendation_resume_update(state),
            },
        )

    if updates:
        saved_updates = _new_information_for_reply(extracted) or dict(updates)
        updates = _with_profile_metadata(customer, updates, source="user_explicit")
        customer = update_customer_profile(state.get("user_id", ""), updates)
        if _is_explicit_memory_request(latest_text):
            result = _reply(
                _saved_profile_text(saved_updates, language),
                session_update={},
            )
            result["customer"] = customer
            return result

    if (
        customer.get("skin_type")
        and not is_third_party_request
        and not is_consultation_context_only
        and _profile_is_stale(customer)
    ):
        snapshot = {
            "skin_type": customer.get("skin_type", ""),
            **{field: customer.get(field, []) for field in LIST_PROFILE_FIELDS},
        }
        return _reply(
            _stale_text(customer, language),
            session_update={
                "pending_profile_update": snapshot,
                "pending_profile_reason": "stale",
                **_recommendation_resume_update(state),
            },
        )

    return {"customer": customer, "profile_memory_handled": False}


def route_after_profile_confirmation(state: AgentState) -> str:
    if state.get("profile_memory_resume_intent") == "recommendation":
        return "recommendation"
    return "end" if state.get("profile_memory_handled") else "classify_intent"


def route_after_profile_memory(state: AgentState) -> str:
    if state.get("profile_memory_handled"):
        return "end"
    if state.get("session", {}).get("current_intent") == "recommendation":
        return "recommendation"
    return "assistant"
