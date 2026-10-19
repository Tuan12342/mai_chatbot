from types import SimpleNamespace

from mai_agent.config import Settings
from mai_agent.vision.google_provider import GoogleSkinVisionProvider


def test_vlm_parses_structured_skin_observation() -> None:
    response = SimpleNamespace(
        parsed={
            "image_subject": "facial_skin",
            "face_count": 1,
            "facial_skin_visible": True,
            "relevance_confidence": 0.97,
            "relevance_reason": "Một khuôn mặt nhìn rõ.",
            "image_quality": "good",
            "lighting_quality": "good",
            "visible_shine": "t_zone",
            "visible_dryness": "mild",
            "confidence": 0.91,
        },
        text=None,
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    provider = GoogleSkinVisionProvider(
        settings=Settings(_env_file=None, google_api_key="test-key"),
        client=client,
    )

    observation = provider.analyze_image(b"image", "image/jpeg")

    assert observation.visible_shine == "t_zone"
    assert observation.visible_dryness == "mild"
    assert observation.confidence == 0.91


def test_vlm_rechecks_and_clears_false_widespread_rash() -> None:
    responses = iter(
        [
            SimpleNamespace(
                parsed={
                    "image_subject": "facial_skin",
                    "face_count": 1,
                    "facial_skin_visible": True,
                    "relevance_confidence": 0.98,
                    "relevance_reason": "Một khuôn mặt nhìn rõ.",
                    "image_quality": "good",
                    "lighting_quality": "good",
                    "visible_dryness": "noticeable",
                    "visible_redness": "noticeable",
                    "widespread_rash": True,
                    "widespread_rash_confidence": 0.95,
                    "widespread_rash_reason": "Đỏ và bong nhẹ ở hai má.",
                    "confidence": 0.95,
                },
                text=None,
            ),
            SimpleNamespace(
                parsed={
                    "confirmed": False,
                    "confidence": 0.96,
                    "reason": "Chỉ thấy đỏ đều và bong vảy mịn ở hai má.",
                },
                text=None,
            ),
        ]
    )
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return next(responses)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GoogleSkinVisionProvider(
        settings=Settings(_env_file=None, google_api_key="test-key"),
        client=client,
    )

    observation = provider.analyze_image(b"image", "image/jpeg")

    assert len(calls) == 2
    assert observation.widespread_rash is False
    assert observation.widespread_rash_confidence == 0.96
    assert "bong vảy mịn" in observation.widespread_rash_reason


def test_vlm_keeps_verified_widespread_rash() -> None:
    responses = iter(
        [
            SimpleNamespace(
                parsed={
                    "image_subject": "facial_skin",
                    "face_count": 1,
                    "facial_skin_visible": True,
                    "relevance_confidence": 0.97,
                    "relevance_reason": "Một khuôn mặt nhìn rõ.",
                    "image_quality": "good",
                    "lighting_quality": "good",
                    "widespread_rash": True,
                    "confidence": 0.9,
                },
                text=None,
            ),
            SimpleNamespace(
                parsed={
                    "confirmed": True,
                    "confidence": 0.93,
                    "reason": "Nhiều tổn thương riêng biệt ở trán và hai má.",
                },
                text=None,
            ),
        ]
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: next(responses))
    )
    provider = GoogleSkinVisionProvider(
        settings=Settings(_env_file=None, google_api_key="test-key"),
        client=client,
    )

    observation = provider.analyze_image(b"image", "image/jpeg")

    assert observation.widespread_rash is True
    assert observation.widespread_rash_confidence == 0.93
    assert observation.widespread_rash_reason


def test_vlm_does_not_analyze_skin_features_for_unrelated_image() -> None:
    response = SimpleNamespace(
        parsed={
            "image_subject": "unrelated",
            "face_count": 0,
            "facial_skin_visible": False,
            "relevance_confidence": 0.99,
            "relevance_reason": "Ảnh phong cảnh, không có khuôn mặt.",
            "image_quality": "good",
            "lighting_quality": "good",
            "visible_shine": "unclear",
            "visible_dryness": "unclear",
            "visible_redness": "unclear",
            "visible_spots": "unclear",
            "confidence": 0,
        },
        text=None,
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    provider = GoogleSkinVisionProvider(
        settings=Settings(_env_file=None, google_api_key="test-key"),
        client=client,
    )

    observation = provider.analyze_image(b"image", "image/jpeg")

    assert observation.image_subject == "unrelated"
    assert observation.face_count == 0
    assert observation.confidence == 0
