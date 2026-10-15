from types import SimpleNamespace

import pytest

from mai_agent.config import Settings
from mai_agent.vision.google_provider import (
    GoogleSkinVisionProvider,
    SkinResearchUnavailable,
)
from mai_agent.vision.models import SkinObservation


def test_grounding_sources_are_restricted_to_allowed_domains() -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(
                    grounding_chunks=[
                        SimpleNamespace(
                            web=SimpleNamespace(uri="https://www.aad.org/page", title="AAD")
                        ),
                        SimpleNamespace(
                            web=SimpleNamespace(
                                uri="https://untrusted.example/blog",
                                title="Untrusted blog",
                            )
                        ),
                    ]
                )
            )
        ]
    )

    domains = GoogleSkinVisionProvider._grounded_allowed_domains(
        response,
        ["aad.org", "nhs.uk"],
    )

    assert domains == ["aad.org"]


def test_search_api_error_becomes_safe_unavailable_status() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("quota"))
        )
    )
    provider = GoogleSkinVisionProvider(
        settings=Settings(_env_file=None, google_api_key="test-key"),
        client=client,
    )
    observation = SkinObservation(
        image_quality="good",
        lighting_quality="good",
        confidence=0.9,
    )

    with pytest.raises(SkinResearchUnavailable, match="quota"):
        provider.research_skin_care(observation, "da dầu")
