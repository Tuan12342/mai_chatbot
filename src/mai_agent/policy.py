from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from mai_agent.llm import ModelRefusalError, invoke_tool
from mai_agent.reply_generator import generate_reply
from mai_agent.routing import latest_user_text
from mai_agent.state import AgentState

UserGoal = Literal[
    "get_information", "get_recommendation", "buy_product", "lookup_own_orders",
    "change_existing_order", "report_problem", "request_human", "translate",
    "social_response", "other",
]
RequestedOperation = Literal[
    "none", "read", "create", "update", "cancel", "refund",
    "delete", "disclose_sensitive_data", "medical_assessment", "translate",
]
AssistantAuthority = Literal["allowed", "needs_human", "prohibited"]
IssueType = Literal[
    "product_quality", "wrong_item", "delivery", "payment", "refund",
    "allergy_safety", "service", "order_change", "other", "none",
]
Severity = Literal["none", "mild", "strong"]
RiskDomain = Literal[
    "medical", "product_reaction", "payment_secrets", "privacy", "fraud",
]
ResponseRequirement = Literal[
    "acknowledge", "apologize", "explain_limitation", "refuse_unsafe_request",
    "protect_privacy", "protect_payment_secrets", "stop_product",
    "seek_medical_help", "consult_professional", "translate_only",
    "identify_assistant",
]
VerificationLevel = Literal["none", "catalog", "official"]
ProtectedResourceScope = Literal["none", "self", "other", "ambiguous"]
AccessDecision = Literal["not_applicable", "allow", "clarify", "deny"]


@tool
def assess_message(
    user_goals: Annotated[list[UserGoal], "Các mục tiêu đồng thời của người dùng."],
    requested_operation: RequestedOperation,
    assistant_authority: Annotated[
        AssistantAuthority,
        "allowed nếu bot được tự xử lý; needs_human nếu cần quyền hoặc xác minh của "
        "người thật; prohibited nếu bot phải từ chối.",
    ],
    is_complaint: bool,
    issue_type: IssueType,
    severity: Severity,
    explicit_human_preference: Annotated[
        bool,
        "True khi người dùng muốn người thật tiếp nhận; không suy ra chỉ vì họ cần hỗ trợ.",
    ],
    verification_level: Annotated[
        VerificationLevel,
        "Mức bằng chứng cần để trả lời: none nếu không cần tra cứu; catalog nếu dữ liệu "
        "sản phẩm/nội bộ là đủ; official nếu phải xác minh bằng hồ sơ, chứng từ hoặc "
        "nguồn có thẩm quyền.",
    ],
    protected_resource_scope: Annotated[
        ProtectedResourceScope,
        "Chủ thể của tài nguyên riêng tư đang được yêu cầu: self cho chính tài khoản "
        "hiện tại; other cho người/tài khoản khác; ambiguous khi chưa xác định; none "
        "khi yêu cầu không truy cập dữ liệu riêng tư.",
    ],
    safety_domains: list[RiskDomain],
    response_requirements: list[ResponseRequirement],
    issue_resolved: bool,
    reason: str,
    target_language: Literal["vi", "en", "unknown"] = "unknown",
) -> dict[str, Any]:
    """Phân tích theo các chiều ngữ nghĩa độc lập, không theo từng test case."""
    return {
        "user_goals": user_goals,
        "requested_operation": requested_operation,
        "assistant_authority": assistant_authority,
        "is_complaint": is_complaint,
        "issue_type": issue_type,
        "severity": severity,
        "explicit_human_preference": explicit_human_preference,
        "verification_level": verification_level,
        "protected_resource_scope": protected_resource_scope,
        "safety_domains": safety_domains,
        "response_requirements": response_requirements,
        "issue_resolved": issue_resolved,
        "reason": reason,
        "target_language": target_language,
    }


POLICY_TOOLS = [assess_message]


@tool
def assess_protected_resource_access(
    protected_resource_scope: ProtectedResourceScope,
    requested_operation: RequestedOperation,
    decision: AccessDecision,
    reason: str,
) -> dict[str, Any]:
    """Independently assess access to private customer-owned resources."""
    return {
        "protected_resource_scope": protected_resource_scope,
        "requested_operation": requested_operation,
        "decision": decision,
        "reason": reason,
    }


PROTECTED_ACCESS_TOOLS = [assess_protected_resource_access]

RESPONSE_GUIDANCE: dict[ResponseRequirement, str] = {
    "acknowledge": (
        "Ghi nhận nội dung của khách một cách lịch sự và hỏi ngắn gọn họ cần hỗ trợ gì "
        "tiếp theo nếu chưa rõ."
    ),
    "apologize": (
        "Xin lỗi chân thành, không biện minh, không tự nhận trách nhiệm pháp lý và tập "
        "trung vào nhu cầu hiện tại của khách."
    ),
    "explain_limitation": (
        "Nói rõ giới hạn thông tin hoặc thẩm quyền; không phỏng đoán hay khẳng định điều "
        "chưa được dữ liệu xác minh."
    ),
    "refuse_unsafe_request": (
        "Từ chối ngắn gọn phần yêu cầu không an toàn hoặc không được phép, không cung "
        "cấp chỉ dẫn giúp vượt qua giới hạn đó."
    ),
    "protect_privacy": (
        "Không tiết lộ dữ liệu cá nhân, dữ liệu của khách khác hoặc thông tin nội bộ; "
        "chỉ hướng khách tới kênh và dữ liệu họ có quyền truy cập."
    ),
    "protect_payment_secrets": (
        "Nhắc khách không gửi mật khẩu, OTP, PIN, số thẻ hoặc CVV; chứng từ nếu cần "
        "phải che thông tin bảo mật."
    ),
    "stop_product": "Khuyên khách ngưng dùng sản phẩm đang gây phản ứng.",
    "seek_medical_help": (
        "Nếu triệu chứng nặng, lan rộng, đau rát hoặc khó chịu nhiều, khuyên khách đi "
        "khám; không chẩn đoán."
    ),
    "consult_professional": (
        "Khuyên khách hỏi bác sĩ hoặc chuyên gia phù hợp trước quyết định có rủi ro sức khỏe."
    ),
    "translate_only": (
        "Chỉ dịch đúng nội dung được yêu cầu, giữ nguyên ý nghĩa, tên riêng, SKU và tên "
        "thành phần; không biến yêu cầu dịch thành tư vấn."
    ),
    "identify_assistant": (
        "Giới thiệu Mai là trợ lý ảo của OA Cosmetics; không giả làm người thật và không "
        "bịa thông tin liên hệ cá nhân."
    ),
}


def triage_requires_handoff(
    assessment: dict[str, Any], *, complaint_count: int = 0
) -> bool:
    """Áp dụng quy tắc tổng quát lên kết quả ngữ nghĩa của model."""
    if assessment.get("provider_refused") or assessment.get("access_denied") or (
        _is_unauthorized_cross_subject_operation(assessment)
    ):
        return False
    is_complaint = bool(assessment.get("is_complaint"))
    return bool(
        assessment.get("explicit_human_preference")
        or assessment.get("assistant_authority") == "needs_human"
        or (is_complaint and assessment.get("severity") == "strong")
        or (is_complaint and complaint_count >= 2 and not assessment.get("issue_resolved"))
    )


def _is_unauthorized_cross_subject_operation(assessment: dict[str, Any]) -> bool:
    return bool(
        assessment.get("protected_resource_scope") == "other"
        and assessment.get("requested_operation")
        in {
            "read",
            "update",
            "cancel",
            "refund",
            "delete",
            "disclose_sensitive_data",
        }
    )


def _extend_unique(assessment: dict[str, Any], field: str, *values: str) -> None:
    """Append constraints without duplicates or mutating the original lists."""
    assessment[field] = list(dict.fromkeys([*(assessment.get(field) or []), *values]))


def _provider_refusal(assessment: dict[str, Any]) -> dict[str, Any]:
    refused = dict(assessment)
    refused["provider_refused"] = True
    return _normalize_assessment(refused)


def _normalize_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    """Enforce semantic invariants shared by policy and handoff routing.

    A referral to a qualified professional is not a request for a shop employee.
    For a medical assessment without an active complaint or an explicit request for
    a person, the assistant must limit its answer and recommend a clinician instead
    of creating a business handoff.
    """
    normalized = dict(assessment)
    if (
        normalized.get("requested_operation") == "medical_assessment"
        and not normalized.get("is_complaint")
        and not normalized.get("explicit_human_preference")
    ):
        normalized["assistant_authority"] = "prohibited"

        _extend_unique(normalized, "safety_domains", "medical")
        _extend_unique(
            normalized, "response_requirements", "explain_limitation", "consult_professional"
        )

    if (
        normalized.get("verification_level") == "official"
        and normalized.get("assistant_authority") != "prohibited"
    ):
        normalized["assistant_authority"] = "needs_human"
        _extend_unique(normalized, "response_requirements", "explain_limitation")

    if normalized.get("access_denied") or _is_unauthorized_cross_subject_operation(normalized):
        normalized["assistant_authority"] = "prohibited"
        _extend_unique(normalized, "safety_domains", "privacy")
        _extend_unique(
            normalized,
            "response_requirements",
            "protect_privacy",
            "refuse_unsafe_request",
            "explain_limitation",
        )

    if normalized.get("provider_refused"):
        normalized["assistant_authority"] = "prohibited"
        _extend_unique(
            normalized, "response_requirements", "refuse_unsafe_request", "explain_limitation"
        )

    return normalized


def _merge_protected_access(
    assessment: dict[str, Any], access: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(assessment)
    scope = access.get("protected_resource_scope", "none")
    operation = access.get("requested_operation", "none")
    if scope != "none":
        merged["protected_resource_scope"] = scope
    if operation != "none":
        merged["requested_operation"] = operation
    if access.get("decision") == "deny":
        merged["access_denied"] = True
    return _normalize_assessment(merged)


def _handled(reply: str, assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "policy_handled": True,
        "triage": assessment,
    }


def _assess_policy(state: AgentState) -> dict[str, Any]:
    messages = [
        SystemMessage(
            content=(
                "Phân tích tin nhắn mới nhất theo các chiều độc lập và bắt buộc gọi "
                "assess_message. Dựa vào ý nghĩa toàn bộ hội thoại, không dò từ khóa, "
                "không đối chiếu câu mẫu và không tự trả lời khách. user_goals mô tả điều "
                "khách muốn đạt được và có thể có nhiều mục tiêu; requested_operation mô "
                "tả loại thao tác, không mô tả chủ đề. assistant_authority=allowed khi "
                "trợ lý có thể xử lý bằng dữ liệu và công cụ hiện có; needs_human khi cần "
                "phê duyệt, thay đổi dữ liệu đã chốt, xác minh chính thức hoặc điều tra "
                "ngoài dữ liệu; prohibited khi yêu cầu xâm phạm riêng tư, gian lận, tiết "
                "lộ bí mật hoặc đưa ra chẩn đoán/đánh giá y khoa mà trợ lý không được "
                "thực hiện. Không dùng needs_human chỉ vì dữ liệu sản phẩm có thể thiếu: "
                "luồng sản phẩm có thể tra catalog rồi nói rõ giới hạn. "
                "needs_human chỉ nói đến nhân viên của shop có thẩm quyền giải quyết; "
                "bác sĩ, dược sĩ hoặc chuyên gia độc lập không phải là handoff của shop. "
                "Khi khách yêu cầu đánh giá tương tác giữa mỹ phẩm và thuốc, chẩn đoán "
                "hoặc quyết định y khoa, dùng requested_operation=medical_assessment, "
                "assistant_authority=prohibited và consult_professional; không dùng "
                "needs_human nếu khách không yêu cầu nhân viên và không khiếu nại. "
                "Sửa giỏ hàng đang tạo khác với sửa một đơn đã được tạo. is_complaint chỉ "
                "đúng khi khách phản ánh sự cố hoặc không hài lòng, không phải chỉ từ chối "
                "một gợi ý. severity=strong cần dấu hiệu tức giận, đe dọa, mất niềm tin "
                "hoặc rủi ro đáng kể; lời nhắc lịch sự thường là mild. issue_type chỉ dùng "
                "để nhóm cùng một vấn đề qua nhiều lượt. issue_resolved chỉ đúng khi chính "
                "khách xác nhận vấn đề đã ổn. explicit_human_preference chỉ phản ánh mong "
                "muốn người thật tiếp nhận, không đồng nghĩa nhu cầu hỗ trợ nói chung. "
                "verification_level=none khi không cần bằng chứng, catalog khi dữ liệu "
                "catalog hoặc dữ liệu nội bộ hiện có đủ để trả lời, và official khi câu "
                "trả lời phải được chứng minh bằng hồ sơ, chứng từ, giấy phép, chứng nhận "
                "hoặc nguồn có thẩm quyền. Với verification_level=official, không suy "
                "đoán, không hứa rằng trợ lý có thể tự kiểm tra và dùng needs_human để "
                "nhân viên shop xác minh, trừ khi bản thân yêu cầu là prohibited. "
                "protected_resource_scope chỉ mô tả tài nguyên riêng tư như đơn hàng, "
                "tài khoản, hồ sơ khách hàng, địa chỉ, số điện thoại hoặc thanh toán; "
                "không dùng cho thông tin sản phẩm công khai. Dùng self khi dữ liệu thuộc "
                "tài khoản hiện tại, other khi thuộc bất kỳ người hoặc tài khoản khác, kể "
                "cả khi khách nói đã được nhờ hoặc cho phép, và ambiguous khi chưa rõ chủ "
                "thể. Hệ thống không có cơ chế ủy quyền chéo tài khoản: mọi thao tác đọc, "
                "sửa, hủy, hoàn tiền hoặc tiết lộ trên tài nguyên của other đều prohibited, "
                "cần protect_privacy và refuse_unsafe_request; không handoff để vượt qua "
                "kiểm soát quyền và không thay bằng việc hiển thị dữ liệu của tài khoản "
                "hiện tại. "
                "safety_domains mô tả miền rủi ro; response_requirements chỉ chứa ràng "
                "buộc thực sự cần có trong câu trả lời. product_reaction và "
                "issue_type=allergy_safety chỉ dùng khi khách nói họ đang có triệu chứng "
                "hoặc phản ứng thực tế sau khi dùng; không dùng cho câu hỏi giả định, hỏi "
                "độ phù hợp, nguy cơ có thể xảy ra, tin đồn về tác dụng phụ hay việc khách "
                "nêu một dị ứng có sẵn để kiểm tra thành phần. Chỉ phản ứng đang xảy ra mới "
                "cần stop_product và seek_medical_help. Câu hỏi chẩn đoán hoặc thuốc cần "
                "giới hạn y khoa và consult_professional. Câu hỏi an toàn trong thai kỳ "
                "không phải phản ứng sản phẩm và không cần handoff: từ chối khẳng định an "
                "toàn, nói rõ giới hạn và khuyên hỏi bác sĩ. Nhu cầu mỹ phẩm thông thường "
                "như chọn sản phẩm theo loại da, mụn hoặc vết thâm không phải chẩn đoán y "
                "khoa; để luồng recommendation xử lý từ catalog. Câu hỏi phối hợp mỹ "
                "phẩm khi chưa có phản ứng cũng để luồng sản phẩm kiểm tra catalog; không "
                "gắn stop_product, product_reaction hoặc needs_human. Yêu cầu dữ liệu riêng "
                "tư hoặc bí mật thanh toán cần bảo vệ dữ liệu và từ chối phần không được "
                "phép. Yêu cầu dịch chỉ cần translate_only. Câu hỏi bot là ai cần "
                "identify_assistant. Nếu khách yêu cầu trợ lý xin lỗi thì dùng apologize; "
                "nếu chính khách xin lỗi thì dùng acknowledge. Với câu hỏi sản phẩm, tồn "
                "kho, đơn hàng hoặc mua hàng thông thường, để response_requirements=[]; "
                "không dùng acknowledge chỉ vì trợ lý nên trả lời lịch sự. Không tạo nhãn "
                "mới cho một tình huống cụ thể; biểu diễn bằng tổ hợp các trường tổng quát."
            )
        ),
        HumanMessage(
            content=(
                f"Hội thoại gần đây:\n{_conversation_text(state)}\n\n"
                f"Tin nhắn mới nhất: {latest_user_text(state)}"
            )
        ),
    ]
    try:
        call = invoke_tool(
            POLICY_TOOLS,
            messages,
            error_message="Model không phân tích được policy và handoff.",
        )
    except ModelRefusalError:
        # A provider refusal is already a terminal safety decision. Do not retry it
        # through a second classifier or invent a privacy-related explanation.
        return _provider_refusal({})
    except RuntimeError:
        assessment = {
            "user_goals": ["other"],
            "requested_operation": "none",
            "assistant_authority": "allowed",
            "is_complaint": False,
            "issue_type": "other",
            "severity": "none",
            "explicit_human_preference": False,
            "verification_level": "none",
            "protected_resource_scope": "none",
            "safety_domains": [],
            "response_requirements": [],
            "issue_resolved": False,
            "reason": (
                "Structured triage không khả dụng; tiếp tục tới luồng nghiệp vụ có "
                "ràng buộc an toàn riêng."
            ),
            "target_language": "unknown",
        }
    else:
        if call["name"] != "assess_message":
            raise RuntimeError(f"Model gọi triage tool không được hỗ trợ: {call['name']}")
        assessment = dict(call["args"])

    assessment = _normalize_assessment(assessment)
    if _is_unauthorized_cross_subject_operation(assessment):
        return assessment
    try:
        access_call = invoke_tool(
            PROTECTED_ACCESS_TOOLS,
            [
                SystemMessage(
                    content=(
                        "Bạn là lớp kiểm soát quyền truy cập độc lập. Chỉ phân tích xem "
                        "tin nhắn mới nhất có yêu cầu đọc, sửa, hủy, xóa, hoàn tiền hoặc "
                        "tiết lộ tài nguyên riêng tư hay không và bắt buộc gọi tool. Tài "
                        "nguyên riêng tư gồm đơn hàng, tài khoản, hồ sơ khách hàng, địa "
                        "chỉ, số điện thoại và dữ liệu thanh toán; thông tin sản phẩm công "
                        "khai không thuộc nhóm này. scope=self khi tài nguyên thuộc chính "
                        "tài khoản hiện tại; other khi thuộc người hoặc tài khoản khác; "
                        "ambiguous khi chưa xác định được chủ thể. Hệ thống không hỗ trợ "
                        "ủy quyền chéo tài khoản, vì vậy lời khách nói rằng người khác đã "
                        "nhờ hoặc đồng ý không phải bằng chứng xác minh: chọn decision=deny "
                        "cho thao tác trên tài nguyên của other. Không trả lời khách."
                    )
                ),
                HumanMessage(content=latest_user_text(state)),
            ],
            error_message="Model không phân tích được quyền truy cập tài nguyên.",
        )
    except ModelRefusalError:
        return _provider_refusal(assessment)
    except RuntimeError:
        return assessment
    if access_call["name"] != "assess_protected_resource_access":
        return assessment
    return _merge_protected_access(assessment, dict(access_call["args"]))


def _conversation_text(state: AgentState, *, limit: int = 12) -> str:
    lines: list[str] = []
    for message in state.get("messages", [])[-limit:]:
        if isinstance(message, HumanMessage):
            role = "Khách"
        elif isinstance(message, AIMessage):
            role = "Mai"
        else:
            continue
        content = str(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _complaint_count(state: AgentState, assessment: dict[str, Any]) -> int:
    if not assessment.get("is_complaint"):
        return 0
    issue_type = str(assessment.get("issue_type", "none"))
    previous = state.get("handoff", {}).get("complaint_counts", {}).get(issue_type, 0)
    return previous + (1 if issue_type != "none" else 0)


def _reply_guidance(state: AgentState, assessment: dict[str, Any]) -> str:
    requirements = [
        str(requirement) for requirement in assessment.get("response_requirements", [])
    ]
    parts = [
        RESPONSE_GUIDANCE[requirement]
        for requirement in requirements
        if requirement in RESPONSE_GUIDANCE
    ]
    if "translate_only" in requirements:
        parts.append(
            f"Ngôn ngữ đích: {assessment.get('target_language', 'unknown')}. "
            f"Tin nhắn nguyên văn: {latest_user_text(state)}"
        )
    if assessment.get("assistant_authority") == "prohibited" and not parts:
        parts.extend(
            [
                RESPONSE_GUIDANCE["refuse_unsafe_request"],
                RESPONSE_GUIDANCE["explain_limitation"],
            ]
        )
    return " ".join(parts)


def _requires_policy_response(assessment: dict[str, Any]) -> bool:
    """Không để các chỉ dẫn hội thoại phụ chiếm luồng nghiệp vụ chính."""
    goals = set(assessment.get("user_goals", []))
    requirements = set(assessment.get("response_requirements", []))
    if assessment.get("assistant_authority") == "prohibited":
        return True
    if goals.intersection({"translate", "social_response"}):
        return True
    return bool(
        requirements
        & {
            "refuse_unsafe_request",
            "protect_privacy",
            "protect_payment_secrets",
            "stop_product",
            "seek_medical_help",
            "translate_only",
            "identify_assistant",
        }
    )


def policy_guard_node(state: AgentState) -> dict[str, Any]:
    """Phân tích đa chiều; handoff được ưu tiên trước phản hồi policy."""
    assessment = _assess_policy(state)
    complaint_count = _complaint_count(state, assessment)
    if triage_requires_handoff(assessment, complaint_count=complaint_count):
        return {"policy_handled": False, "triage": assessment}

    if not _requires_policy_response(assessment):
        return {"policy_handled": False, "triage": assessment}

    guidance = _reply_guidance(state, assessment)
    if not guidance:
        return {"policy_handled": False, "triage": assessment}

    reply = generate_reply(
        guidance,
        response_language=state.get("session", {}).get("language_code", "vi"),
    )
    return _handled(reply, assessment)


def route_after_policy_guard(state: AgentState) -> str:
    return "end" if state.get("policy_handled") else "handoff_guard"
