from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from mai_agent.llm import invoke_tool
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
    "disclose_sensitive_data", "medical_assessment", "translate",
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
        "safety_domains": safety_domains,
        "response_requirements": response_requirements,
        "issue_resolved": issue_resolved,
        "reason": reason,
        "target_language": target_language,
    }


POLICY_TOOLS = [assess_message]

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
    is_complaint = bool(assessment.get("is_complaint"))
    return bool(
        assessment.get("explicit_human_preference")
        or assessment.get("assistant_authority") == "needs_human"
        or (is_complaint and assessment.get("severity") == "strong")
        or (is_complaint and complaint_count >= 2 and not assessment.get("issue_resolved"))
    )


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
                "Sửa giỏ hàng đang tạo khác với sửa một đơn đã được tạo. is_complaint chỉ "
                "đúng khi khách phản ánh sự cố hoặc không hài lòng, không phải chỉ từ chối "
                "một gợi ý. severity=strong cần dấu hiệu tức giận, đe dọa, mất niềm tin "
                "hoặc rủi ro đáng kể; lời nhắc lịch sự thường là mild. issue_type chỉ dùng "
                "để nhóm cùng một vấn đề qua nhiều lượt. issue_resolved chỉ đúng khi chính "
                "khách xác nhận vấn đề đã ổn. explicit_human_preference chỉ phản ánh mong "
                "muốn người thật tiếp nhận, không đồng nghĩa nhu cầu hỗ trợ nói chung. "
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
    except RuntimeError:
        return {
            "user_goals": ["other"],
            "requested_operation": "none",
            "assistant_authority": "allowed",
            "is_complaint": False,
            "issue_type": "other",
            "severity": "none",
            "explicit_human_preference": False,
            "safety_domains": [],
            "response_requirements": [],
            "issue_resolved": False,
            "reason": (
                "Structured triage không khả dụng; tiếp tục tới luồng nghiệp vụ có "
                "ràng buộc an toàn riêng."
            ),
            "target_language": "unknown",
        }
    if call["name"] != "assess_message":
        raise RuntimeError(f"Model gọi triage tool không được hỗ trợ: {call['name']}")
    return dict(call["args"])


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
