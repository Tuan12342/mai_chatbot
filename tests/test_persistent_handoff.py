from langchain_core.messages import HumanMessage

from mai_agent.skills.handoff import node as handoff_node_module


def test_wrong_item_complaint_handoffs_on_first_report(monkeypatch) -> None:
    monkeypatch.setattr(
        handoff_node_module,
        "_assess_latest_message",
        lambda _state: {
            "is_complaint": True,
            "issue_type": "wrong_item",
            "severity": "mild",
            "assistant_authority": "needs_human",
            "explicit_human_preference": False,
            "issue_resolved": False,
            "reason": "Khách đặt Brightening Serum nhưng nhận Barrier Repair Serum.",
        },
    )

    result = handoff_node_module.handoff_guard_node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Tôi đặt Brightening Serum mà giao Barrier Repair, sai hàng rồi!"
                    )
                )
            ],
            "handoff": {"status": "inactive", "complaint_counts": {}},
        }
    )

    assert result["handoff_action"] == "create"
    assert result["handoff"]["issue_type"] == "wrong_item"
    assert result["handoff"]["complaint_count"] == 1


def test_mild_first_service_complaint_does_not_handoff(monkeypatch) -> None:
    monkeypatch.setattr(
        handoff_node_module,
        "_assess_latest_message",
        lambda _state: {
            "is_complaint": True,
            "issue_type": "service",
            "severity": "mild",
            "assistant_authority": "allowed",
            "explicit_human_preference": False,
            "issue_resolved": False,
            "reason": "Khách hơi thất vọng vì shop trả lời chậm.",
        },
    )

    result = handoff_node_module.handoff_guard_node(
        {
            "messages": [HumanMessage(content="Shop trả lời hơi chậm.")],
            "handoff": {"status": "inactive", "complaint_counts": {}},
        }
    )

    assert result["handoff_action"] == "continue"
    assert result["handoff"]["issue_type"] == "service"


def test_semantic_business_verification_creates_handoff(monkeypatch) -> None:
    monkeypatch.setattr(
        handoff_node_module,
        "_assess_latest_message",
        lambda _state: {
            "is_complaint": False,
            "issue_type": "order_change",
            "severity": "none",
            "assistant_authority": "needs_human",
            "explicit_human_preference": False,
            "issue_resolved": False,
            "reason": "Khách yêu cầu hủy đơn đã xác nhận.",
        },
    )

    result = handoff_node_module.handoff_guard_node(
        {
            "messages": [HumanMessage(content="Hủy đơn đã xác nhận này giúp tôi")],
            "handoff": {"status": "inactive", "complaint_counts": {}},
        }
    )

    assert result["handoff_action"] == "create"


def test_non_complaint_without_business_verification_continues(monkeypatch) -> None:
    monkeypatch.setattr(
        handoff_node_module,
        "_assess_latest_message",
        lambda _state: {
            "is_complaint": False,
            "issue_type": "other",
            "severity": "none",
            "assistant_authority": "allowed",
            "explicit_human_preference": False,
            "issue_resolved": False,
            "reason": "Khách hỏi chính sách đổi trả.",
        },
    )

    result = handoff_node_module.handoff_guard_node(
        {
            "messages": [HumanMessage(content="Chính sách đổi trả của shop thế nào?")],
            "handoff": {"status": "inactive", "complaint_counts": {}},
        }
    )

    assert result["handoff_action"] == "continue"


def test_handoff_guard_reuses_multi_label_triage_without_second_model_call(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        handoff_node_module,
        "_assess_latest_message",
        lambda _state: (_ for _ in ()).throw(
            AssertionError("handoff must reuse the existing triage result")
        ),
    )

    result = handoff_node_module.handoff_guard_node(
        {
            "messages": [HumanMessage(content="Tôi muốn gặp quản lý.")],
            "triage": {
                "user_goals": ["request_human"],
                "requested_operation": "none",
                "assistant_authority": "allowed",
                "is_complaint": False,
                "issue_type": "other",
                "severity": "none",
                "explicit_human_preference": True,
                "safety_domains": [],
                "response_requirements": [],
                "issue_resolved": False,
                "reason": "Khách yêu cầu người thật tiếp nhận.",
            },
            "handoff": {"status": "inactive", "complaint_counts": {}},
        }
    )

    assert result["handoff_action"] == "create"
