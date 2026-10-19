from langchain_core.messages import HumanMessage

from mai_agent import profile_memory
from mai_agent.profile_memory import (
    profile_confirmation_node,
    profile_memory_node,
    route_after_profile_confirmation,
    route_after_profile_memory,
)
from mai_agent.skills.product import recommendation as recommendation_module


def _product(
    product_id: str,
    name: str,
    *,
    category: str = "Sữa rửa mặt",
    skin_types: list[str] | None = None,
) -> dict:
    return {
        "id": product_id,
        "name": name,
        "category": category,
        "skin_types": skin_types or ["da dầu"],
        "ingredients": ["Zinc PCA"],
        "benefits": ["Hỗ trợ làm sạch dầu thừa"],
        "usage": "Dùng trên da ẩm rồi rửa sạch.",
        "price_vnd": 219_000,
        "stock": 35,
    }


def _action(**overrides) -> dict:
    action = {
        "action": "recommend",
        "request": "sản phẩm cho da mụn",
        "subject": "self",
        "skin_type": None,
        "concerns": ["mụn"],
        "product_category": "unspecified",
        "excluded_ingredients": [],
    }
    action.update(overrides)
    return action


def _knowledge(product_ids: list[str]) -> list[dict]:
    return [
        {
            "product_id": product_id,
            "routine_role": "Làm sạch dầu thừa",
            "allowed_care_modes": ["standard"],
            "why": ["Phù hợp khi cần hỗ trợ làm sạch dầu thừa."],
            "cautions": ["Không mô tả như sản phẩm điều trị."],
        }
        for product_id in product_ids
    ]


def test_clear_recommendation_request_returns_grounded_products(monkeypatch) -> None:
    product = _product("OA002", "Gel rửa mặt OA Purifying Gel")
    monkeypatch.setattr(
        recommendation_module,
        "_interpret_recommendation_action",
        lambda *_args: _action(),
    )
    monkeypatch.setattr(
        recommendation_module,
        "eligible_catalog_products",
        lambda *_args, **_kwargs: [product],
    )
    monkeypatch.setattr(recommendation_module, "knowledge_for_products", _knowledge)
    monkeypatch.setattr(
        recommendation_module,
        "_select_recommendation_products",
        lambda *_args, **_kwargs: [
            {"product_id": "OA002", "evidence_keys": ["why_1"]}
        ],
    )

    result = recommendation_module.recommendation_node(
        {
            "messages": [HumanMessage(content="sản phẩm cho da mụn")],
            "customer": {"skin_type": "da dầu", "excluded_ingredients": ["Alcohol"]},
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert result["session"]["current_step"] == "idle"
    assert result["session"]["pending_recommendation"] is None
    assert result["session"]["last_product_candidates"] == [
        {"product_id": "OA002", "product_name": "Gel rửa mặt OA Purifying Gel"}
    ]
    assert "OA002" in result["reply"]
    assert "219.000 VNĐ" in result["reply"]
    assert "Phù hợp khi cần hỗ trợ làm sạch dầu thừa." in result["reply"]


def test_other_person_request_does_not_use_senders_profile(monkeypatch) -> None:
    product = _product(
        "OA006",
        "Kem dưỡng OA Ceramide Cream",
        category="Kem dưỡng",
        skin_types=["da khô"],
    )
    captured: dict = {}
    monkeypatch.setattr(
        recommendation_module,
        "_interpret_recommendation_action",
        lambda *_args: _action(
            request="kem dưỡng cho mẹ da khô",
            subject="other",
            skin_type="da khô",
            concerns=["khô căng"],
            product_category="moisturizer",
        ),
    )

    def eligible(profile, skin_type, **_kwargs):
        captured["profile"] = profile
        captured["skin_type"] = skin_type
        return [product]

    monkeypatch.setattr(recommendation_module, "eligible_catalog_products", eligible)
    monkeypatch.setattr(recommendation_module, "knowledge_for_products", _knowledge)
    monkeypatch.setattr(
        recommendation_module,
        "_select_recommendation_products",
        lambda *_args, **_kwargs: [
            {"product_id": "OA006", "evidence_keys": ["why_1"]}
        ],
    )

    result = recommendation_module.recommendation_node(
        {
            "messages": [HumanMessage(content="kem dưỡng cho mẹ tôi da khô")],
            "customer": {
                "skin_type": "da dầu",
                "excluded_ingredients": ["Ceramide"],
                "disliked_product_ids": ["OA006"],
            },
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert captured["skin_type"] == "da khô"
    assert captured["profile"]["excluded_ingredients"] == []
    assert captured["profile"]["disliked_product_ids"] == []
    assert "OA006" in result["reply"]


def test_current_self_skin_type_overrides_saved_skin_type() -> None:
    profile, skin_type = recommendation_module._effective_profile(
        _action(subject="self", skin_type="da khô"),
        {"skin_type": "da dầu", "excluded_ingredients": []},
    )

    assert skin_type == "da khô"
    assert profile["skin_type"] == "da dầu"


def test_unclear_request_is_saved_with_clarification_context(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendation_module,
        "_interpret_recommendation_action",
        lambda *_args: {
            "action": "clarify",
            "question": "Bạn muốn chăm sóc vấn đề nào?",
            "request": "Tư vấn giúp tôi",
            "missing_fields": ["concern"],
        },
    )

    result = recommendation_module.recommendation_node(
        {
            "messages": [HumanMessage(content="Tư vấn giúp tôi")],
            "customer": {},
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert result["session"] == {
        "current_step": "awaiting_recommendation_clarification",
        "pending_recommendation": {
            "request": "Tư vấn giúp tôi",
            "missing_fields": ["concern"],
            "last_question": "Bạn muốn chăm sóc vấn đề nào?",
        },
    }


def test_invalid_sku_or_evidence_is_discarded(monkeypatch) -> None:
    decisions = iter(
        [
            {
                "name": "choose_recommended_products",
                "args": {
                    "recommendations": [
                        {"product_id": "OA999", "evidence_keys": ["why_1"]},
                        {"product_id": "OA002", "evidence_keys": ["invented"]},
                    ]
                },
            }
        ]
    )
    monkeypatch.setattr(
        recommendation_module,
        "invoke_tool",
        lambda *_args, **_kwargs: next(decisions),
    )

    selected = recommendation_module._select_recommendation_products(
        {
            "request_context": {"product_category": "unspecified"},
            "candidate_products": [{"id": "OA002", "category": "Sữa rửa mặt"}],
            "product_knowledge": [
                {
                    "product_id": "OA002",
                    "routine_role": "Làm sạch",
                    "evidence": {"why_1": "Bằng chứng hợp lệ"},
                }
            ],
        },
        max_products=4,
    )

    assert selected == []


def test_duplicate_category_is_removed_for_general_routine(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendation_module,
        "invoke_tool",
        lambda *_args, **_kwargs: {
            "name": "choose_recommended_products",
            "args": {
                "recommendations": [
                    {"product_id": "OA001", "evidence_keys": ["why_1"]},
                    {"product_id": "OA002", "evidence_keys": ["why_1"]},
                ]
            },
        },
    )
    knowledge = [
        {
            "product_id": product_id,
            "routine_role": "Làm sạch",
            "evidence": {"why_1": "Bằng chứng hợp lệ"},
        }
        for product_id in ("OA001", "OA002")
    ]

    selected = recommendation_module._select_recommendation_products(
        {
            "request_context": {"product_category": "routine"},
            "candidate_products": [
                {"id": "OA001", "category": "Sữa rửa mặt"},
                {"id": "OA002", "category": "Sữa rửa mặt"},
            ],
            "product_knowledge": knowledge,
        },
        max_products=4,
    )

    assert selected == [{"product_id": "OA001", "evidence_keys": ["why_1"]}]


def test_empty_candidates_do_not_call_product_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendation_module,
        "_interpret_recommendation_action",
        lambda *_args: _action(),
    )
    monkeypatch.setattr(
        recommendation_module,
        "eligible_catalog_products",
        lambda *_args, **_kwargs: [],
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Không được gọi model chọn sản phẩm khi không có ứng viên.")

    monkeypatch.setattr(
        recommendation_module,
        "_select_recommendation_products",
        fail_if_called,
    )

    result = recommendation_module.recommendation_node(
        {
            "messages": [HumanMessage(content="sản phẩm cho da mụn")],
            "customer": {},
            "session": {"language_code": "vi", "current_step": "idle"},
        }
    )

    assert "chưa tìm thấy sản phẩm" in result["reply"]
    assert result["session"]["last_product_candidates"] == []


def test_profile_confirmation_resumes_original_recommendation(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_memory,
        "update_customer_profile",
        lambda _user_id, updates: {"zalo_user_id": "z001", **updates},
    )
    result = profile_confirmation_node(
        {
            "user_id": "z001",
            "messages": [HumanMessage(content="đúng")],
            "customer": {"skin_type": "da dầu"},
            "session": {
                "language_code": "vi",
                "current_intent": "recommendation",
                "pending_profile_update": {"skin_type": "da dầu"},
                "pending_profile_reason": "stale",
                "pending_resume_intent": "recommendation",
                "pending_resume_request": "tìm sản phẩm cho da dầu",
            },
        }
    )

    assert route_after_profile_confirmation(result) == "recommendation"
    assert result["session"]["pending_recommendation"] == {
        "request": "tìm sản phẩm cho da dầu"
    }
    assert result["session"]["pending_resume_request"] is None
    assert result["profile_memory_resume_intent"] == "recommendation"


def test_resumed_recommendation_interprets_original_request_not_confirmation(
    monkeypatch,
) -> None:
    captured: dict = {}

    def interpret(text, *_args):
        captured["text"] = text
        return {
            "action": "clarify",
            "question": "Bạn đang quan tâm vấn đề da nào?",
            "request": text,
            "missing_fields": ["concern"],
        }

    monkeypatch.setattr(
        recommendation_module,
        "_interpret_recommendation_action",
        interpret,
    )

    recommendation_module.recommendation_node(
        {
            "messages": [HumanMessage(content="đúng")],
            "profile_memory_resume_intent": "recommendation",
            "customer": {"skin_type": "da dầu"},
            "session": {
                "language_code": "vi",
                "pending_recommendation": {"request": "tư vấn sản phẩm cho da dầu"},
            },
        }
    )

    assert captured["text"] == "tư vấn sản phẩm cho da dầu"


def test_stale_profile_pause_saves_original_recommendation(monkeypatch) -> None:
    monkeypatch.setattr(profile_memory, "_extract_profile_information", lambda _text: {})
    monkeypatch.setattr(profile_memory, "_profile_is_stale", lambda _customer: True)

    result = profile_memory_node(
        {
            "user_id": "z001",
            "messages": [HumanMessage(content="sản phẩm cho da dầu")],
            "customer": {"skin_type": "da dầu"},
            "session": {
                "language_code": "vi",
                "current_intent": "recommendation",
            },
        }
    )

    assert result["session"]["pending_resume_intent"] == "recommendation"
    assert result["session"]["pending_resume_request"] == "sản phẩm cho da dầu"


def test_third_party_request_does_not_pause_for_senders_stale_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_memory,
        "_extract_profile_information",
        lambda _text: {"_profile_subject": "other"},
    )
    monkeypatch.setattr(profile_memory, "_profile_is_stale", lambda _customer: True)

    result = profile_memory_node(
        {
            "user_id": "z001",
            "messages": [HumanMessage(content="tìm kem dưỡng cho mẹ tôi da khô")],
            "customer": {"skin_type": "da dầu"},
            "session": {
                "language_code": "vi",
                "current_intent": "recommendation",
            },
        }
    )

    assert result["profile_memory_handled"] is False
    assert "pending_profile_update" not in result.get("session", {})


def test_general_skin_type_question_does_not_change_saved_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        profile_memory,
        "_extract_profile_information",
        lambda _text: {"_consultation_context_only": True},
    )
    monkeypatch.setattr(profile_memory, "_profile_is_stale", lambda _customer: True)

    result = profile_memory_node(
        {
            "user_id": "z001",
            "messages": [HumanMessage(content="da khô dùng sản phẩm gì")],
            "customer": {"skin_type": "da dầu"},
            "session": {
                "language_code": "vi",
                "current_intent": "recommendation",
            },
        }
    )

    assert result["profile_memory_handled"] is False
    assert result["customer"]["skin_type"] == "da dầu"
    assert "pending_profile_update" not in result.get("session", {})


def test_profile_memory_routes_recommendation_to_dedicated_node() -> None:
    assert (
        route_after_profile_memory(
            {
                "profile_memory_handled": False,
                "session": {"current_intent": "recommendation"},
            }
        )
        == "recommendation"
    )
