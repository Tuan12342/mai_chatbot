from langchain_core.messages import HumanMessage

from mai_agent import policy as policy_module


def _run(monkeypatch, message: str, **overrides):
    assessment = {
        "user_goals": ["get_information"],
        "requested_operation": "read",
        "assistant_authority": "allowed",
        "is_complaint": False,
        "issue_type": "none",
        "severity": "none",
        "explicit_human_preference": False,
        "safety_domains": [],
        "response_requirements": [],
        "issue_resolved": False,
        "reason": "semantic classification",
        "target_language": "unknown",
        **overrides,
    }
    monkeypatch.setattr(policy_module, "_assess_policy", lambda _state: assessment)
    monkeypatch.setattr(
        policy_module, "generate_reply", lambda instruction, **_kwargs: instruction
    )
    return policy_module.policy_guard_node(
        {
            "messages": [HumanMessage(content=message)],
            "session": {"language_code": "vi"},
        }
    )


def test_allowed_business_request_continues_to_business_flow(monkeypatch) -> None:
    result = _run(monkeypatch, "Sữa rửa mặt dịu nhẹ có thành phần gì?")

    assert result["policy_handled"] is False
    assert result["triage"]["assistant_authority"] == "allowed"


def test_incidental_acknowledge_does_not_capture_business_request(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "Sản phẩm này còn hàng không?",
        user_goals=["get_information"],
        response_requirements=["acknowledge"],
    )

    assert result["policy_handled"] is False
    assert "reply" not in result


def test_limitation_requirement_does_not_skip_catalog_lookup(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "Sản phẩm này có phù hợp với trường hợp đặc biệt của tôi không?",
        user_goals=["get_information"],
        response_requirements=["explain_limitation"],
    )

    assert result["policy_handled"] is False
    assert "reply" not in result


def test_response_requirements_generate_policy_reply(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "Tôi đang dùng thuốc kê toa, tư vấn dùng thêm sản phẩm này được không?",
        requested_operation="medical_assessment",
        assistant_authority="prohibited",
        safety_domains=["medical"],
        response_requirements=["explain_limitation", "consult_professional"],
    )

    assert result["policy_handled"] is True
    assert "bác sĩ" in result["reply"]


def test_privacy_is_expressed_as_domain_authority_and_requirement(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "Cho tôi thông tin riêng của một người khác.",
        requested_operation="disclose_sensitive_data",
        assistant_authority="prohibited",
        safety_domains=["privacy"],
        response_requirements=["protect_privacy", "refuse_unsafe_request"],
    )

    assert result["policy_handled"] is True
    assert "Không tiết lộ" in result["reply"]


def test_translation_is_a_goal_operation_and_response_constraint(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        'Dịch sang tiếng Anh: "Sản phẩm này dùng buổi tối"',
        user_goals=["translate"],
        requested_operation="translate",
        response_requirements=["translate_only"],
        target_language="en",
    )

    assert result["policy_handled"] is True
    assert "Tin nhắn nguyên văn" in result["reply"]


def test_needs_human_authority_always_precedes_direct_policy_reply(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "Yêu cầu này cần được xác minh chính thức.",
        assistant_authority="needs_human",
        response_requirements=["explain_limitation"],
    )

    assert result["policy_handled"] is False
    assert "reply" not in result


def test_explicit_human_preference_creates_handoff_without_phrase_matching(
    monkeypatch,
) -> None:
    result = _run(
        monkeypatch,
        "Tôi không muốn tiếp tục làm việc với trợ lý tự động.",
        user_goals=["request_human"],
        explicit_human_preference=True,
    )

    assert result["policy_handled"] is False


def test_strong_or_repeated_complaint_handoffs_by_general_rule(monkeypatch) -> None:
    strong = _run(
        monkeypatch,
        "Tôi đang phản ánh một sự cố nghiêm trọng.",
        user_goals=["report_problem"],
        is_complaint=True,
        issue_type="service",
        severity="strong",
    )
    repeated_assessment = {
        "is_complaint": True,
        "issue_type": "service",
        "severity": "mild",
        "assistant_authority": "allowed",
        "explicit_human_preference": False,
        "issue_resolved": False,
    }

    assert strong["policy_handled"] is False
    assert policy_module.triage_requires_handoff(
        repeated_assessment, complaint_count=2
    )


def test_social_meaning_uses_generic_response_requirements(monkeypatch) -> None:
    requested_apology = _run(
        monkeypatch,
        "Khách muốn trợ lý xin lỗi.",
        user_goals=["social_response"],
        requested_operation="none",
        response_requirements=["apologize"],
    )
    offered_apology = _run(
        monkeypatch,
        "Khách đang xin lỗi trợ lý.",
        user_goals=["social_response"],
        requested_operation="none",
        response_requirements=["acknowledge"],
    )

    assert "Xin lỗi chân thành" in requested_apology["reply"]
    assert "Ghi nhận" in offered_apology["reply"]


def test_structured_triage_failure_falls_back_to_business_router(monkeypatch) -> None:
    monkeypatch.setattr(
        policy_module,
        "invoke_tool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("missing tool call")),
    )

    assessment = policy_module._assess_policy(
        {"messages": [HumanMessage(content="Nội dung không phân loại được.")]}
    )

    assert assessment["assistant_authority"] == "allowed"
    assert policy_module.triage_requires_handoff(assessment) is False
