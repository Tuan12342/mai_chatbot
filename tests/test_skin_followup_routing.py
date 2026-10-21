from mai_agent import routing
from mai_agent.graph import route_by_intent


def test_short_yes_after_skin_recommendation_uses_model_context(monkeypatch) -> None:
    step = "awaiting_skin_guidance_confirmation"
    decisions = iter(
        [
            {"name": "route_in_scope", "args": {}},
            {"name": "route_product_question", "args": {}},
        ]
    )
    monkeypatch.setattr(routing, "invoke_tool", lambda *_args, **_kwargs: next(decisions))

    assert routing._is_out_of_scope("có", step) is False
    assert routing._interpret_intent("có", step, {}) == "product_question"


def test_natural_decline_after_skin_recommendation_uses_model_context(monkeypatch) -> None:
    step = "awaiting_skin_guidance_confirmation"
    decisions = iter(
        [
            {"name": "route_in_scope", "args": {}},
            {"name": "route_product_question", "args": {}},
        ]
    )
    monkeypatch.setattr(routing, "invoke_tool", lambda *_args, **_kwargs: next(decisions))

    assert routing._is_out_of_scope("thôi, phần hướng dẫn vậy là đủ rồi", step) is False
    assert (
        routing._interpret_intent("thôi, phần hướng dẫn vậy là đủ rồi", step, {})
        == "product_question"
    )


def test_recent_recommendations_are_passed_to_order_intent_router(monkeypatch) -> None:
    captured: dict = {}

    def invoke(_tools, messages, **_kwargs):
        captured["prompt"] = messages[-1].content
        return {"name": "route_order", "args": {}}

    monkeypatch.setattr(routing, "invoke_tool", invoke)
    context = routing._pending_context(
        {
            "last_product_candidates": [
                {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
            ],
            "cart": [],
        }
    )

    assert routing._interpret_intent("cho tôi gel rửa mặt", "idle", context) == "order"
    assert "recently_recommended_products" in captured["prompt"]
    assert "OA002" in captured["prompt"]


def test_short_informal_greeting_has_its_own_intent(monkeypatch) -> None:
    decisions = iter(
        [
            {"name": "route_in_scope", "args": {}},
            {"name": "route_greeting", "args": {}},
        ]
    )
    monkeypatch.setattr(routing, "invoke_tool", lambda *_args, **_kwargs: next(decisions))

    assert routing._is_out_of_scope("lô", "idle") is False
    assert routing._interpret_intent("lô", "idle", {}) == "greeting"
    assert route_by_intent({"session": {"current_intent": "greeting"}}) == "greeting"


def test_greeting_reply_is_generated_naturally(monkeypatch) -> None:
    captured: dict = {}

    def generate(business_context, *, response_language):
        captured["context"] = business_context
        captured["language"] = response_language
        return "Chào bạn! Mai có thể hỗ trợ gì cho bạn hôm nay?"

    monkeypatch.setattr(routing, "generate_reply", generate)

    result = routing.greeting_node(
        {
            "session": {"language_code": "vi"},
            "customer": {"name": "Lan"},
        }
    )

    assert result["reply"] == "Chào bạn! Mai có thể hỗ trợ gì cho bạn hôm nay?"
    assert "Lan" in captured["context"]
    assert captured["language"] == "vi"
