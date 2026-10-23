import pytest
from langchain_core.messages import HumanMessage

from mai_agent import policy as policy_module


def test_provider_refusal_stops_without_second_classification(monkeypatch) -> None:
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise policy_module.ModelRefusalError("Provider declined the request")

    monkeypatch.setattr(policy_module, "invoke_tool", refuse)
    result = policy_module._assess_policy(
        {"messages": [HumanMessage(content="Unclassified request")]}
    )

    assert len(calls) == 1
    assert result["assistant_authority"] == "prohibited"
    assert result.get("protected_resource_scope") != "other"
    assert "privacy" not in result.get("safety_domains", [])
    assert not policy_module.triage_requires_handoff(result)
    assert policy_module._requires_policy_response(result)


@pytest.mark.parametrize("operation", ["read", "update", "cancel", "delete", "refund"])
def test_confirmed_cross_subject_denial_skips_redundant_check(monkeypatch, operation):
    calls = []

    def classify(*args, **kwargs):
        calls.append(args)
        assert len(calls) == 1
        return {
            "name": "assess_message",
            "args": {
                "requested_operation": operation,
                "protected_resource_scope": "other",
                "assistant_authority": "allowed",
                "explicit_human_preference": True,
            },
        }

    monkeypatch.setattr(policy_module, "invoke_tool", classify)
    result = policy_module._assess_policy(
        {"messages": [HumanMessage(content="Private resource request")]}
    )

    assert len(calls) == 1
    assert result["assistant_authority"] == "prohibited"
    assert not policy_module.triage_requires_handoff(result)


def test_access_denial_without_operation_cannot_be_overridden_by_handoff() -> None:
    assessment = policy_module._merge_protected_access(
        {"explicit_human_preference": True, "requested_operation": "none"},
        {"decision": "deny", "protected_resource_scope": "ambiguous"},
    )

    assert assessment["assistant_authority"] == "prohibited"
    assert not policy_module.triage_requires_handoff(assessment)


def test_normalization_is_idempotent_and_does_not_mutate_input() -> None:
    source = {
        "requested_operation": "update",
        "protected_resource_scope": "other",
        "safety_domains": [],
        "response_requirements": [],
    }
    first = policy_module._normalize_assessment(source)

    assert policy_module._normalize_assessment(first) == first
    assert source["safety_domains"] == []
    assert source["response_requirements"] == []


def test_allowed_triage_still_runs_independent_access_check(monkeypatch) -> None:
    calls = []

    def classify(tools, messages, **kwargs):
        calls.append(tools[0].name)
        if len(calls) == 1:
            return {
                "name": "assess_message",
                "args": {"assistant_authority": "allowed", "requested_operation": "none"},
            }
        return {
            "name": "assess_protected_resource_access",
            "args": {
                "requested_operation": "read",
                "protected_resource_scope": "other",
                "decision": "deny",
            },
        }

    monkeypatch.setattr(policy_module, "invoke_tool", classify)
    result = policy_module._assess_policy(
        {"messages": [HumanMessage(content="Private resource request")]}
    )

    assert calls == ["assess_message", "assess_protected_resource_access"]
    assert result["assistant_authority"] == "prohibited"
    assert "privacy" in result["safety_domains"]


def _run(monkeypatch, message: str, **overrides):
    assessment = {
        "user_goals": ["get_information"],
        "requested_operation": "read",
        "assistant_authority": "allowed",
        "is_complaint": False,
        "issue_type": "none",
        "severity": "none",
        "explicit_human_preference": False,
        "verification_level": "none",
        "protected_resource_scope": "none",
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


def test_medical_professional_referral_is_not_shop_handoff() -> None:
    assessment = policy_module._normalize_assessment(
        {
            "user_goals": ["get_information"],
            "requested_operation": "medical_assessment",
            "assistant_authority": "needs_human",
            "is_complaint": False,
            "issue_type": "none",
            "severity": "none",
            "explicit_human_preference": False,
            "safety_domains": [],
            "response_requirements": [],
            "issue_resolved": False,
            "reason": "Cần chuyên gia có chuyên môn y tế đánh giá.",
            "target_language": "unknown",
        }
    )

    assert assessment["assistant_authority"] == "prohibited"
    assert assessment["safety_domains"] == ["medical"]
    assert "explain_limitation" in assessment["response_requirements"]
    assert "consult_professional" in assessment["response_requirements"]
    assert policy_module.triage_requires_handoff(assessment) is False


def test_medical_complaint_can_still_require_business_handoff() -> None:
    assessment = policy_module._normalize_assessment(
        {
            "requested_operation": "medical_assessment",
            "assistant_authority": "needs_human",
            "is_complaint": True,
            "issue_type": "allergy_safety",
            "severity": "strong",
            "explicit_human_preference": False,
            "safety_domains": ["medical", "product_reaction"],
            "response_requirements": ["stop_product", "seek_medical_help"],
            "issue_resolved": False,
        }
    )

    assert assessment["assistant_authority"] == "needs_human"
    assert policy_module.triage_requires_handoff(assessment) is True


def test_official_verification_requires_business_handoff() -> None:
    assessment = policy_module._normalize_assessment(
        {
            "requested_operation": "read",
            "assistant_authority": "allowed",
            "is_complaint": False,
            "issue_type": "none",
            "severity": "none",
            "explicit_human_preference": False,
            "verification_level": "official",
            "safety_domains": [],
            "response_requirements": [],
            "issue_resolved": False,
        }
    )

    assert assessment["assistant_authority"] == "needs_human"
    assert assessment["response_requirements"] == ["explain_limitation"]
    assert policy_module.triage_requires_handoff(assessment) is True


def test_cross_subject_private_operation_is_refused_without_handoff() -> None:
    assessment = policy_module._normalize_assessment(
        {
            "user_goals": ["change_existing_order", "request_human"],
            "requested_operation": "update",
            "assistant_authority": "needs_human",
            "is_complaint": False,
            "issue_type": "order_change",
            "severity": "none",
            "explicit_human_preference": True,
            "verification_level": "none",
            "protected_resource_scope": "other",
            "safety_domains": [],
            "response_requirements": [],
            "issue_resolved": False,
        }
    )

    assert assessment["assistant_authority"] == "prohibited"
    assert assessment["safety_domains"] == ["privacy"]
    assert assessment["response_requirements"] == [
        "protect_privacy",
        "refuse_unsafe_request",
        "explain_limitation",
    ]
    assert policy_module.triage_requires_handoff(assessment) is False


def test_independent_access_assessment_overrides_missed_main_triage() -> None:
    assessment = policy_module._merge_protected_access(
        {
            "requested_operation": "none",
            "assistant_authority": "allowed",
            "is_complaint": False,
            "explicit_human_preference": False,
            "protected_resource_scope": "none",
            "safety_domains": [],
            "response_requirements": [],
        },
        {
            "protected_resource_scope": "other",
            "requested_operation": "delete",
            "decision": "deny",
            "reason": "Tài nguyên thuộc một tài khoản khác.",
        },
    )

    assert assessment["requested_operation"] == "delete"
    assert assessment["protected_resource_scope"] == "other"
    assert assessment["assistant_authority"] == "prohibited"
    assert policy_module.triage_requires_handoff(assessment) is False


def test_public_product_information_is_not_treated_as_private_data() -> None:
    assessment = policy_module._normalize_assessment(
        {
            "requested_operation": "read",
            "assistant_authority": "allowed",
            "is_complaint": False,
            "explicit_human_preference": False,
            "verification_level": "catalog",
            "protected_resource_scope": "none",
            "safety_domains": [],
            "response_requirements": [],
        }
    )

    assert assessment["assistant_authority"] == "allowed"
    assert policy_module.triage_requires_handoff(assessment) is False


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
