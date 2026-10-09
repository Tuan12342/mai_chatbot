from langchain_core.messages import HumanMessage

from mai_agent import routing


def _state(text: str, current_step: str) -> dict:
    return {
        "user_id": "test-user",
        "messages": [HumanMessage(content=text)],
        "session": {
            "session_id": "test-session",
            "language_code": "vi",
            "current_step": current_step,
            "cart": [],
            "pending_product_reference": "OA002",
        },
    }


def test_product_question_can_interrupt_quantity_collection(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_is_out_of_scope", lambda _text, _step: False)
    monkeypatch.setattr(
        routing,
        "_interpret_intent",
        lambda _text, _step, _context: "product_question",
    )
    monkeypatch.setattr(routing, "find_customer_by_zalo_id", lambda _user_id: None)

    result = routing.classify_intent_node(
        _state("Cho tôi thông tin về sản phẩm này", "collecting_quantity")
    )

    assert result["session"]["current_intent"] == "product_question"
    assert result["session"]["current_step"] == "collecting_quantity"


def test_short_quantity_stays_in_order_flow(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_is_out_of_scope", lambda _text, _step: False)
    captured = {}

    def interpret(_text, step, context):
        captured["step"] = step
        captured["context"] = context
        return "order"

    monkeypatch.setattr(
        routing,
        "_interpret_intent",
        interpret,
    )
    monkeypatch.setattr(routing, "find_customer_by_zalo_id", lambda _user_id: None)

    result = routing.classify_intent_node(_state("2", "collecting_quantity"))

    assert result["session"]["current_intent"] == "order"
    assert result["session"]["current_step"] == "collecting_quantity"
    assert captured == {
        "step": "collecting_quantity",
        "context": {"pending_product_reference": "OA002"},
    }


def test_unknown_message_is_not_forced_into_order(monkeypatch) -> None:
    monkeypatch.setattr(routing, "_is_out_of_scope", lambda _text, _step: False)
    monkeypatch.setattr(
        routing,
        "_interpret_intent",
        lambda _text, _step, _context: "unknown",
    )
    monkeypatch.setattr(routing, "find_customer_by_zalo_id", lambda _user_id: None)

    result = routing.classify_intent_node(
        _state("Tôi muốn hỏi một chuyện khác", "collecting_quantity")
    )

    assert result["session"]["current_intent"] == "unknown"
    assert result["session"]["current_step"] == "collecting_quantity"
