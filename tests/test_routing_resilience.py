from langchain_core.messages import HumanMessage

from mai_agent import routing


def test_scope_tool_failure_continues_to_intent_classifier(monkeypatch) -> None:
    monkeypatch.setattr(
        routing,
        "_is_out_of_scope",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("missing scope tool call")),
    )
    monkeypatch.setattr(
        routing,
        "_interpret_intent",
        lambda *_args: "product_question",
    )
    monkeypatch.setattr(routing, "find_customer_by_zalo_id", lambda _user_id: None)

    result = routing.classify_intent_node(
        {
            "user_id": "routing-test",
            "messages": [HumanMessage(content="Sản phẩm này có thành phần gì?")],
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert result["session"]["current_intent"] == "product_question"


def test_intent_tool_failure_uses_unknown_fallback(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_is_out_of_scope", lambda *_args: False)
    monkeypatch.setattr(
        routing,
        "_interpret_intent",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("missing intent tool call")),
    )
    monkeypatch.setattr(routing, "find_customer_by_zalo_id", lambda _user_id: None)

    result = routing.classify_intent_node(
        {
            "user_id": "routing-test",
            "messages": [HumanMessage(content="Nội dung khó phân loại")],
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert result["session"]["current_intent"] == "unknown"
