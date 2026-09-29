from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from mai_agent.skills.product import node


class FakeGemini:
    response = AIMessage(content="")
    received_messages = []
    received_tools = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def bind_tools(self, tools):
        type(self).received_tools = tools
        return self

    def invoke(self, messages):
        type(self).received_messages = messages
        return type(self).response


def _configure_fake_gemini(monkeypatch, response: AIMessage) -> None:
    FakeGemini.response = response
    FakeGemini.received_messages = []
    FakeGemini.received_tools = []
    monkeypatch.setattr(node, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda: SimpleNamespace(google_api_key="test-key", google_model="gemini-test"),
    )


def test_assistant_preserves_gemini_tool_calls(monkeypatch) -> None:
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_product_knowledge",
                "args": {"query": "Công dụng OA004", "product_references": ["OA004"]},
                "id": "tool-call-1",
                "type": "tool_call",
            }
        ],
    )
    _configure_fake_gemini(monkeypatch, response)

    result = node.assistant_node(
        {"messages": [HumanMessage(content="OA004 có công dụng gì?")]}
    )

    assert result["messages"] == [response]
    assert result["messages"][0].tool_calls[0]["name"] == "search_product_knowledge"
    assert result["reply"] == ""
    assert FakeGemini.received_tools == node.PRODUCT_TOOLS
    assert isinstance(FakeGemini.received_messages[0], SystemMessage)
    assert isinstance(FakeGemini.received_messages[1], HumanMessage)


def test_assistant_returns_gemini_text(monkeypatch) -> None:
    response = AIMessage(content="OA004 phù hợp với da dầu.")
    _configure_fake_gemini(monkeypatch, response)

    result = node.assistant_node(
        {"messages": [HumanMessage(content="OA004 hợp da nào?")]}
    )

    assert result["reply"] == "OA004 phù hợp với da dầu."
    assert result["messages"] == [response]
