from langchain_core.messages import AIMessage, HumanMessage

from mai_agent import routing
from mai_agent.state import merge_mapping


class _LanguageModelStub:
    selected_tool = "keep_current_language"

    def __init__(self, **_: object) -> None:
        pass

    def bind_tools(self, *_: object, **__: object) -> "_LanguageModelStub":
        return self

    def invoke(self, _: object) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.selected_tool,
                    "args": {},
                    "id": "language-test",
                    "type": "tool_call",
                }
            ],
        )


def test_language_switch_preserves_business_session(monkeypatch) -> None:
    monkeypatch.setattr(routing, "ChatGoogleGenerativeAI", _LanguageModelStub)
    _LanguageModelStub.selected_tool = "use_english"
    original_session = {
        "session_id": "session-1",
        "language_code": "vi",
        "current_step": "collecting_address",
        "cart": [
            {
                "product_id": "OA002",
                "product_name": "Gel rửa mặt OA Purifying Gel",
                "quantity": 2,
                "unit_price": 219_000,
            }
        ],
        "pending_phone": "0123456789",
    }

    update = routing.detect_language_node(
        {
            "messages": [HumanMessage(content="Please send it to my new address")],
            "session": original_session,
        }
    )
    merged = merge_mapping(original_session, update["session"])

    assert merged["language_code"] == "en"
    assert merged["current_step"] == "collecting_address"
    assert merged["cart"] == original_session["cart"]
    assert merged["pending_phone"] == "0123456789"


def test_neutral_message_keeps_current_language(monkeypatch) -> None:
    monkeypatch.setattr(routing, "ChatGoogleGenerativeAI", _LanguageModelStub)
    _LanguageModelStub.selected_tool = "keep_current_language"

    update = routing.detect_language_node(
        {
            "messages": [HumanMessage(content="OA002")],
            "session": {"session_id": "session-1", "language_code": "en"},
        }
    )

    assert update == {"session": {"language_code": "en"}}
