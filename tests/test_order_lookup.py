from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph
from mai_agent.skills.order_lookup.tools import verify_customer_identity


def _conversation(user_id: str):
    graph = create_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"lookup-{user_id}"}}
    return graph, config


def _invoke(graph, config, user_id: str, text: str):
    return graph.invoke(
        {"user_id": user_id, "messages": [HumanMessage(content=text)]},
        config=config,
    )


def test_order_lookup_requires_verification_before_returning_orders() -> None:
    graph, config = _conversation("zalo-user-001")
    result = _invoke(graph, config, "zalo-user-001", "Cho tôi tra đơn cũ")

    assert result["session"]["current_step"] == "verifying_order_lookup"
    assert "ORDER-001" not in result["reply"]
    assert "4 số cuối" in result["reply"]


def test_verified_customer_can_read_only_their_orders() -> None:
    graph, config = _conversation("zalo-user-001")
    _invoke(graph, config, "zalo-user-001", "Cho tôi tra đơn cũ")
    result = _invoke(graph, config, "zalo-user-001", "Lan, 4567")

    assert result["session"]["identity_verified"] is True
    assert "ORDER-001" in result["reply"]


def test_wrong_zalo_user_cannot_verify_with_correct_name_and_phone() -> None:
    assert not verify_customer_identity(
        zalo_user_id="attacker-user",
        provided_name="Lan",
        phone_last_four="4567",
    )


def test_failed_verification_does_not_disclose_order() -> None:
    graph, config = _conversation("zalo-user-001")
    _invoke(graph, config, "zalo-user-001", "Cho tôi tra đơn cũ")
    result = _invoke(graph, config, "zalo-user-001", "Lan, 0000")

    assert result["session"]["identity_verified"] is False
    assert "ORDER-001" not in result["reply"]
    assert "chưa khớp" in result["reply"].lower()
