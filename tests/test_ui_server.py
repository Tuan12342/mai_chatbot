from langchain_core.messages import AIMessage, HumanMessage

from mai_agent.ui_server import _public_state


def test_public_state_only_exposes_displayable_messages() -> None:
    state = _public_state(
        {
            "messages": [
                HumanMessage(content="Cho tôi gặp nhân viên"),
                AIMessage(content="Em đã chuyển yêu cầu."),
                AIMessage(
                    content="Nhân viên đang kiểm tra.",
                    additional_kwargs={"sender": "human_agent"},
                ),
            ],
            "response_owner": "human",
            "session": {
                "handoff_status": "active",
                "current_intent": "order_lookup",
                "current_step": "idle",
                "handoff_package": {"reason": "Khách yêu cầu gặp nhân viên."},
            },
        }
    )

    assert [message["sender"] for message in state["messages"]] == [
        "customer",
        "assistant",
        "human_agent",
    ]
    assert state["handoff_status"] == "active"
    assert state["response_owner"] == "human"
    assert state["handoff_package"]["reason"] == "Khách yêu cầu gặp nhân viên."


def test_public_state_has_safe_defaults_for_a_new_session() -> None:
    state = _public_state({})

    assert state == {
        "messages": [],
        "handoff_status": "none",
        "response_owner": "ai",
        "current_intent": None,
        "current_step": "idle",
        "handoff_package": {},
    }
