from langchain_core.messages import AIMessage, HumanMessage

from mai_agent import llm as llm_module


class _RetryingBoundModel:
    def __init__(self) -> None:
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        if len(self.messages) == 1:
            return AIMessage(content="Tôi sẽ trả lời bằng văn bản.")
        return AIMessage(
            content="",
            tool_calls=[{"name": "chosen_tool", "args": {"value": 1}, "id": "call-1"}],
        )


class _FakeModel:
    def __init__(self, bound_model: _RetryingBoundModel) -> None:
        self.bound_model = bound_model

    def bind_tools(self, _tools, **_kwargs):
        return self.bound_model


class _RefusingBoundModel:
    def invoke(self, _messages):
        return AIMessage(
            content="Không thể thực hiện yêu cầu.",
            additional_kwargs={"refusal": "Yêu cầu bị từ chối bởi policy của provider."},
        )


def test_invoke_tool_retries_once_when_model_returns_plain_text(monkeypatch) -> None:
    bound_model = _RetryingBoundModel()
    monkeypatch.setattr(
        llm_module,
        "create_chat_model",
        lambda: _FakeModel(bound_model),
    )

    result = llm_module.invoke_tool(
        [],
        [HumanMessage(content="Chọn tool phù hợp")],
        error_message="missing tool call",
    )

    assert result["name"] == "chosen_tool"
    assert len(bound_model.messages) == 2
    assert "bắt buộc" in bound_model.messages[1][0].content


def test_invoke_tool_preserves_provider_refusal(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_module,
        "create_chat_model",
        lambda: _FakeModel(_RefusingBoundModel()),
    )

    try:
        llm_module.invoke_tool(
            [],
            [HumanMessage(content="Yêu cầu cần được kiểm tra")],
            error_message="missing tool call",
        )
    except llm_module.ModelRefusalError as error:
        assert "provider" in str(error)
    else:
        raise AssertionError("Provider refusal must not be discarded")
