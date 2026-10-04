from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from mai_agent.graph import route_after_handoff_guard
from mai_agent.handoff import (
    build_handoff_package,
    handoff_guard_node,
    human_agent_update,
)


def test_gemini_decision_requests_handoff_with_context_package() -> None:
    state = {
        "messages": [
            AIMessage(content="Em có thể hỗ trợ gì thêm ạ?"),
            HumanMessage(content="Cho tôi gặp nhân viên, SĐT 0901234567"),
        ],
        "session": {
            "current_step": "confirming_order",
            "cart": [
                {
                    "product_id": "OA001",
                    "product_name": "Sữa rửa mặt",
                    "quantity": 2,
                    "unit_price": 100_000,
                }
            ],
        },
    }

    with patch(
        "mai_agent.handoff._interpret_handoff",
        return_value=(True, "Khách yêu cầu gặp nhân viên.", "direct_request"),
    ):
        update = handoff_guard_node(state)

    assert update["response_owner"] == "human"
    assert update["session"]["handoff_status"] == "requested"
    package = update["session"]["handoff_package"]
    assert package["reason"] == "Khách yêu cầu gặp nhân viên."
    assert package["order_context"]["cart"][0]["product_id"] == "OA001"
    assert "0901234567" not in str(package)
    assert "[đã ẩn số điện thoại]" in str(package)


def test_active_handoff_monitors_new_messages_without_ai_reply() -> None:
    state = {
        "messages": [HumanMessage(content="Tôi bổ sung là đơn bị giao thiếu.")],
        "session": {
            "handoff_status": "active",
            "handoff_reason": "Đơn giao thiếu.",
            "handoff_trigger": "repeated_unresolved_complaint",
        },
    }

    with patch("mai_agent.handoff._interpret_handoff") as classifier:
        update = handoff_guard_node(state)

    classifier.assert_not_called()
    assert update["reply"] == ""
    assert update["response_owner"] == "human"
    assert update["session"]["handoff_status"] == "active"
    assert update["session"]["handoff_package"]["conversation_history"][-1][
        "content"
    ] == "Tôi bổ sung là đơn bị giao thiếu."


def test_resolved_handoff_returns_control_to_ai() -> None:
    state = {
        "messages": [AIMessage(content="Shop đã xử lý xong.")],
        "session": {"handoff_status": "resolved"},
    }

    update = handoff_guard_node(state)
    merged_state = {"session": {**state["session"], **update["session"]}}

    assert update["response_owner"] == "ai"
    assert update["session"]["handoff_status"] == "none"
    assert route_after_handoff_guard(merged_state) == "classify_intent"


def test_human_agent_update_controls_resume_explicitly() -> None:
    active = human_agent_update("Nhân viên đang kiểm tra.")
    resolved = human_agent_update("Đã xử lý xong đơn giao thiếu.", resume_ai=True)

    assert active["session"]["handoff_status"] == "active"
    assert resolved["session"]["handoff_status"] == "resolved"
    assert resolved["response_owner"] == "ai"
    assert resolved["messages"][0].additional_kwargs["sender"] == "human_agent"


def test_context_package_excludes_customer_profile_and_shipping_address() -> None:
    state = {
        "messages": [HumanMessage(content="Đơn của tôi đang ở đâu?")],
        "customer": {"name": "Lan", "allergies": ["A"]},
        "active_order": {
            "order_id": "ORD-1",
            "customer_id": "zalo-secret",
            "shipping_address": {"address_id": "A1", "address_line": "Địa chỉ riêng"},
            "items": [],
            "status": "processing",
        },
        "session": {"current_step": "idle", "cart": []},
    }

    package = build_handoff_package(state, reason="Cần kiểm tra đơn", trigger="direct_request")

    assert package["order_context"]["active_order"]["order_id"] == "ORD-1"
    assert "zalo-secret" not in str(package)
    assert "Địa chỉ riêng" not in str(package)
    assert "allergies" not in str(package)
