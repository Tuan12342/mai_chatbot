from langchain_core.messages import HumanMessage

from mai_agent import routing


def _detect(
    monkeypatch,
    text: str,
    *,
    tool_name: str,
    language: str | None = None,
    current: str | None = None,
) -> dict[str, str]:
    monkeypatch.setattr(
        routing,
        "invoke_tool",
        lambda *_args, **_kwargs: {
            "name": tool_name,
            "args": ({"language": language} if language else {}),
        },
    )
    session = {"language_code": current} if current else {}
    update = routing.detect_language_node(
        {"messages": [HumanMessage(content=text)], "session": session}
    )
    return update["session"]


def test_clear_sentence_changes_to_its_language(monkeypatch) -> None:
    assert _detect(
        monkeypatch,
        "How much is the cleanser?",
        tool_name="clear_message_language",
        language="en",
        current="vi",
    ) == {"language_code": "en", "language_signal": "clear_language"}

    assert _detect(
        monkeypatch,
        "Da dầu thì nên dùng sản phẩm nào?",
        tool_name="clear_message_language",
        language="vi",
        current="en",
    ) == {"language_code": "vi", "language_signal": "clear_language"}


def test_explicit_request_uses_requested_not_spoken_language(monkeypatch) -> None:
    assert _detect(
        monkeypatch,
        "Từ bây giờ hãy trả lời tôi bằng tiếng Anh",
        tool_name="explicit_language_request",
        language="en",
        current="vi",
    ) == {"language_code": "en", "language_signal": "explicit_request"}

    assert _detect(
        monkeypatch,
        "Please continue answering me in Vietnamese",
        tool_name="explicit_language_request",
        language="vi",
        current="en",
    ) == {"language_code": "vi", "language_signal": "explicit_request"}


def test_ambiguous_ok_keeps_current_language(monkeypatch) -> None:
    assert _detect(
        monkeypatch,
        "okay",
        tool_name="ambiguous_language_signal",
        current="vi",
    ) == {"language_code": "vi", "language_signal": "ambiguous"}


def test_unsupported_language_keeps_current_supported_language(monkeypatch) -> None:
    assert _detect(
        monkeypatch,
        "你们有防晒霜吗？",
        tool_name="ambiguous_language_signal",
        current="en",
    ) == {"language_code": "en", "language_signal": "ambiguous"}


def test_ambiguous_signal_defaults_to_profile_language(monkeypatch) -> None:
    monkeypatch.setattr(
        routing,
        "invoke_tool",
        lambda *_args, **_kwargs: {"name": "ambiguous_language_signal", "args": {}},
    )
    update = routing.detect_language_node(
        {
            "messages": [HumanMessage(content="OK")],
            "session": {},
            "customer": {"preferred_language": "en"},
        }
    )

    assert update["session"] == {
        "language_code": "en",
        "language_signal": "ambiguous",
    }
