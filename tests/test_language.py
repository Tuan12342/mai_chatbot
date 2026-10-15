from langchain_core.messages import HumanMessage

from mai_agent.routing import detect_language_node


def _detect(text: str, *, current: str | None = None) -> str:
    session = {"language_code": current} if current else {}
    update = detect_language_node({"messages": [HumanMessage(content=text)], "session": session})
    return update["session"]["language_code"]


def test_detects_english_and_vietnamese() -> None:
    assert _detect("How much is the cleanser?") == "en"
    assert _detect("Da dầu nên dùng sản phẩm nào?") == "vi"


def test_detects_short_english_greeting() -> None:
    assert _detect("Hi") == "en"


def test_unsupported_language_keeps_current_supported_language() -> None:
    assert _detect("你们有防晒霜吗？", current="en") == "en"


def test_unsupported_language_defaults_to_vietnamese() -> None:
    assert _detect("선크림 있어요?") == "vi"
