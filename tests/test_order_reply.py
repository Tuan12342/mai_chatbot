from types import SimpleNamespace

from langchain_core.messages import AIMessage

from mai_agent.skills.order import node


class FakeGemini:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        return AIMessage(content="Dạ, sản phẩm đã hết hàng. Mình xem OA002 nhé?")


def test_gemini_writes_order_reply_after_business_check(monkeypatch) -> None:
    monkeypatch.setattr(node, "ChatGoogleGenerativeAI", FakeGemini)
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda: SimpleNamespace(google_api_key="test-key", google_model="gemini-test"),
    )

    reply = node._generate_order_reply(
        "OA Gentle Cleanser hết hàng; chỉ được gợi ý OA002."
    )

    assert reply == "Dạ, sản phẩm đã hết hàng. Mình xem OA002 nhé?"


def test_order_reply_falls_back_when_gemini_fails(monkeypatch) -> None:
    class BrokenGemini(FakeGemini):
        def invoke(self, messages):
            raise RuntimeError("Gemini unavailable")

    monkeypatch.setattr(node, "ChatGoogleGenerativeAI", BrokenGemini)
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda: SimpleNamespace(google_api_key="test-key", google_model="gemini-test"),
    )

    required_content = "OA Gentle Cleanser hết hàng và chưa có sản phẩm thay thế."
    reply = node._generate_order_reply(required_content)

    assert reply == required_content
