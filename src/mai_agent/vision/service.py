import re
from typing import Any

from langfuse import Langfuse
from underthesea import lang_detect

from mai_agent.config import Settings, get_settings
from mai_agent.observability import langfuse_observation
from mai_agent.vision.google_provider import GoogleSkinVisionProvider
from mai_agent.vision.knowledge_base import knowledge_for_products
from mai_agent.vision.models import SkinConsultationResult
from mai_agent.vision.prompt import generate_skin_reply, select_skin_products
from mai_agent.vision.recommendation import (
    eligible_catalog_products,
    extract_skin_type_from_message,
)
from mai_agent.vision.safety import asks_for_medical_advice, classify_skin_scope


def _language(message: str) -> str:
    try:
        return "en" if lang_detect(message) == "en" else "vi"
    except Exception:
        return "vi"


def _fixed_reply(kind: str, language: str) -> str:
    replies = {
        "medical_out_of_scope": {
            "vi": (
                "Mình không thể chẩn đoán bệnh da, xác định bệnh qua ảnh hoặc kê thuốc. "
                "Bạn nên gặp bác sĩ da liễu để được thăm khám chính xác và an toàn."
            ),
            "en": (
                "I can't diagnose a skin condition from an image or prescribe medicine. "
                "Please consult a dermatologist for a proper assessment."
            ),
        },
        "medical_red_flag": {
            "vi": (
                "Ảnh có dấu hiệu vượt quá phạm vi tư vấn mỹ phẩm nên mình không đề xuất "
                "sản phẩm lúc này. Bạn nên liên hệ bác sĩ hoặc cơ sở y tế; nếu có sưng "
                "mắt/môi kèm khó thở hoặc khó nuốt, hãy tìm hỗ trợ khẩn cấp."
            ),
            "en": (
                "The image contains a sign outside cosmetic-care scope, so I won't "
                "recommend a product. Please contact a clinician; seek urgent help if "
                "eye or lip swelling occurs with trouble breathing or swallowing."
            ),
        },
        "need_better_image": {
            "vi": (
                "Ảnh hiện chưa đủ rõ để mình quan sát đáng tin cậy. Bạn hãy chụp lại dưới "
                "ánh sáng tự nhiên, không dùng filter và để toàn bộ vùng da cần xem trong ảnh."
            ),
            "en": (
                "The image isn't clear enough for a reliable observation. Please retake "
                "it in natural light, without filters, and include the full area of concern."
            ),
        },
        "need_single_face": {
            "vi": (
                "Ảnh có nhiều hơn một khuôn mặt nên mình chưa thể biết cần quan sát ai. "
                "Bạn hãy gửi ảnh chỉ có một người, chụp chính diện, đủ sáng và không dùng filter."
            ),
            "en": (
                "The image contains more than one face, so I can't determine whose skin "
                "to assess. Please send a well-lit, unfiltered photo containing one person."
            ),
        },
        "irrelevant_image": {
            "vi": (
                "Ảnh này không cho thấy rõ vùng da mặt phù hợp để mình tư vấn mỹ phẩm. "
                "Bạn hãy gửi ảnh chính diện của một khuôn mặt, đủ sáng, không dùng filter "
                "và để vùng da cần xem hiện rõ trong ảnh nhé."
            ),
            "en": (
                "This image does not show suitable facial skin for cosmetic advice. "
                "Please send a clear, well-lit, unfiltered front-facing photo of one face."
            ),
        },
        "product_image": {
            "vi": (
                "Ảnh này có vẻ là sản phẩm mỹ phẩm, không phải ảnh da mặt nên mình không "
                "thực hiện phân tích da. Bạn hãy gửi tên hoặc mã SKU trên sản phẩm kèm câu "
                "hỏi; mình sẽ chuyển sang hỗ trợ thông tin và cách sử dụng sản phẩm."
            ),
            "en": (
                "This appears to be a cosmetic product rather than facial skin, so I "
                "won't run a skin assessment. Please send the product name or SKU with "
                "your question so I can help with product information and usage."
            ),
        },
        "unsupported_skin_region": {
            "vi": (
                "Ảnh này có vẻ là vùng da không phải da mặt. Hiện tại tính năng chỉ hỗ "
                "trợ quan sát da mặt để tư vấn mỹ phẩm; bạn hãy gửi ảnh chính diện của một "
                "khuôn mặt, đủ sáng và không dùng filter nhé."
            ),
            "en": (
                "This appears to show skin outside the face. This feature currently only "
                "supports facial-skin observations; please send one clear, well-lit, "
                "unfiltered front-facing photo."
            ),
        },
        "need_confirmation": {
            "vi": (
                "Mình chỉ có thể quan sát bề mặt qua ảnh và chưa thể kết luận loại da. "
                "Bạn cho mình biết sau khi rửa mặt da có căng không, da đổ dầu toàn mặt "
                "hay chủ yếu vùng chữ T, và bạn có dị ứng hoặc đang kích ứng không nhé."
            ),
            "en": (
                "I can only observe surface features from the image and can't determine "
                "your skin type from it alone. Does your skin feel tight after cleansing, "
                "where does it become oily, and do you have allergies or active irritation?"
            ),
        },
        "no_matching_product": {
            "vi": (
                "Mình chưa tìm thấy sản phẩm còn hàng đáp ứng đồng thời loại da và các "
                "thành phần bạn cần tránh. Mình sẽ không gợi ý sản phẩm không phù hợp."
            ),
            "en": (
                "I couldn't find an in-stock product that matches your confirmed skin type "
                "and ingredient exclusions, so I won't suggest an unsuitable product."
            ),
        },
        "insufficient_knowledge": {
            "vi": (
                "Mình đã quan sát ảnh nhưng các đặc điểm hiện chưa đủ rõ để đối chiếu "
                "chắc chắn với kho tri thức sản phẩm. Bạn có thể cho biết da đang thiên "
                "khô, bóng dầu, đỏ hay có nhiều nốt nhìn thấy nhất không?"
            ),
            "en": (
                "I inspected the image, but the visible features are not clear enough "
                "for a reliable match against the product knowledge base. Is the skin "
                "mainly dry, shiny, visibly red, or showing multiple visible spots?"
            ),
        },
    }
    return replies[kind][language]


def _remove_source_attribution(text: str) -> str:
    text = re.sub(r"\[([^\]]+)]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(
        r"(?im)^\s*(nguồn|nguon|sources?|references?|tham khảo|tham khao)\s*:.*$",
        "",
        text,
    )
    text = re.sub(
        r"(?i)củng cố (?:hàng rào bảo vệ da|hàng rào da)",
        "hỗ trợ hàng rào bảo vệ da",
        text,
    )
    text = re.sub(
        r"(?i)strengthen(?:s|ing)? the skin barrier",
        "support the skin barrier",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_ROUTINE_CATEGORY_ORDER = {
    "Tẩy trang": 0,
    "Sữa rửa mặt": 1,
    "Toner": 2,
    "Serum": 3,
    "Kem dưỡng": 4,
    "Mặt nạ": 5,
    "Kem chống nắng": 6,
}


def _routine_order(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        products,
        key=lambda product: (
            _ROUTINE_CATEGORY_ORDER.get(product["category"], 99),
            product["id"],
        ),
    )


class SkinConsultationService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: GoogleSkinVisionProvider | None = None,
        langfuse_client: Langfuse | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or GoogleSkinVisionProvider(self.settings)
        self.langfuse = langfuse_client

    def consult(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        message: str,
        customer: dict[str, Any],
    ) -> SkinConsultationResult:
        language = _language(message)
        with langfuse_observation(
            self.langfuse,
            name="medical-safety",
            as_type="guardrail",
            input={"message": message},
        ) as safety:
            medical_request = asks_for_medical_advice(message)
            if safety:
                safety.update(output={"medical_request": medical_request})
        if medical_request:
            return SkinConsultationResult(
                status="medical_out_of_scope",
                reply=_fixed_reply("medical_out_of_scope", language),
            )

        with langfuse_observation(
            self.langfuse,
            name="google-vlm",
            as_type="generation",
            input={"mime_type": mime_type, "size_bytes": len(image_bytes)},
            model=self.settings.google_vlm_model,
        ) as generation:
            observation = self.provider.analyze_image(image_bytes, mime_type)
            if generation:
                generation.update(output=observation.model_dump())
        with langfuse_observation(
            self.langfuse,
            name="image-safety-classification",
            as_type="guardrail",
            input=observation.model_dump(),
        ) as safety:
            scope = classify_skin_scope(
                message,
                observation,
                minimum_confidence=self.settings.skin_vision_min_confidence,
                minimum_relevance_confidence=(
                    self.settings.skin_image_relevance_min_confidence
                ),
                minimum_red_flag_confidence=self.settings.skin_red_flag_min_confidence,
            )
            if safety:
                safety.update(output={"scope": scope})
        if scope == "medical_red_flag":
            return SkinConsultationResult(
                status="medical_red_flag",
                reply=_fixed_reply("medical_red_flag", language),
                observation=observation,
            )
        medical_review = scope == "medical_review"
        if scope == "irrelevant_image":
            return SkinConsultationResult(
                status="irrelevant_image",
                reply=_fixed_reply("irrelevant_image", language),
                observation=observation,
            )
        if scope == "product_image":
            return SkinConsultationResult(
                status="product_image",
                reply=_fixed_reply("product_image", language),
                observation=observation,
            )
        if scope == "unsupported_skin_region":
            return SkinConsultationResult(
                status="unsupported_skin_region",
                reply=_fixed_reply("unsupported_skin_region", language),
                observation=observation,
            )
        if scope == "need_single_face":
            return SkinConsultationResult(
                status="need_single_face",
                reply=_fixed_reply("need_single_face", language),
                observation=observation,
            )
        if scope == "need_more_information":
            return SkinConsultationResult(
                status="need_better_image",
                reply=_fixed_reply("need_better_image", language),
                observation=observation,
            )

        # Tư vấn ảnh chỉ dùng thông tin trong request hiện tại. Không áp hồ sơ của
        # người gửi lên người xuất hiện trong ảnh vì hai người có thể khác nhau.
        skin_type = extract_skin_type_from_message(message)
        care_mode = "medical_review" if medical_review else "standard"
        max_products = (
            max(1, min(self.settings.skin_recommendation_max_products, 2))
            if medical_review
            else max(1, min(self.settings.skin_recommendation_max_products, 5))
        )
        with langfuse_observation(
            self.langfuse,
            name="local-product-knowledge",
            as_type="retriever",
            input={
                "skin_type_from_current_message": skin_type or "not_provided",
                "visible_observations": observation.model_dump(),
            },
        ) as retrieval:
            candidate_products = eligible_catalog_products(
                {},
                skin_type,
                care_mode=care_mode,
            )
            candidate_ids = [item["id"] for item in candidate_products]
            product_knowledge = knowledge_for_products(candidate_ids)
            if retrieval:
                retrieval.update(
                    output={
                        "candidate_product_ids": candidate_ids,
                        "knowledge": product_knowledge,
                    }
                )
        if not candidate_products:
            return SkinConsultationResult(
                status="no_matching_product",
                reply=_fixed_reply("no_matching_product", language),
                observation=observation,
            )

        safe_candidates = [
            {
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "skin_types": product["skin_types"],
                "ingredients": product["ingredients"],
                "benefits": product["benefits"],
                "usage": product["usage"],
                "price_vnd": product["price_vnd"],
            }
            for product in candidate_products
        ]
        selection_context = {
            "care_mode": care_mode,
            "max_products": max_products,
            "visible_observations": observation.model_dump(),
            "skin_type_from_current_message": skin_type or "not_provided",
            "candidate_products": safe_candidates,
            "candidate_product_knowledge": product_knowledge,
        }
        with langfuse_observation(
            self.langfuse,
            name="misa-product-selection",
            as_type="generation",
            input=selection_context,
            model=self.settings.misa_model,
        ) as generation:
            selected_ids, selection_reason = select_skin_products(
                selection_context,
                max_products=max_products,
            )
            if generation:
                generation.update(
                    output={
                        "selected_product_ids": selected_ids,
                        "selection_reason": selection_reason,
                    }
                )
        if not selected_ids:
            return SkinConsultationResult(
                status="no_matching_product",
                reply=_fixed_reply("no_matching_product", language),
                observation=observation,
            )

        products_by_id = {product["id"]: product for product in candidate_products}
        products = [products_by_id[product_id] for product_id in selected_ids]

        routine_products = _routine_order(products)
        routine_product_ids = [product["id"] for product in routine_products]
        routine_knowledge = knowledge_for_products(routine_product_ids)
        safe_products = [
            {
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "skin_types": product["skin_types"],
                "ingredients": product["ingredients"],
                "benefits": product["benefits"],
                "usage": product["usage"],
                "price_vnd": product["price_vnd"],
            }
            for product in routine_products
        ]
        context = {
            "care_mode": care_mode,
            "medical_review_context": (
                {
                    "visible_reason": observation.widespread_rash_reason,
                    "confidence": observation.widespread_rash_confidence,
                    "doctor_review_recommended": True,
                    "products_are_supportive_not_treatment": True,
                    "use_only_on_intact_skin": True,
                }
                if medical_review
                else None
            ),
            "visible_observations": observation.model_dump(
                exclude={
                    "eye_or_lip_swelling",
                    "open_wound",
                    "blistering",
                    "pus_or_crusting",
                    "widespread_rash",
                    "widespread_rash_confidence",
                    "widespread_rash_reason",
                }
            ),
            "skin_type_from_current_message": skin_type or "not_provided",
            "selection_policy": {
                "decision_owner": "misa_model",
                "sku_validation_owner": "application_code",
                "selection_reason": selection_reason,
                "final_response_model_must_not_reselect": True,
            },
            "routine_order_product_ids": routine_product_ids,
            "retrieved_product_knowledge": routine_knowledge,
            "selected_catalog_products": safe_products,
            "required_constraints": [
                "Nói rõ quan sát ảnh không phải chẩn đoán.",
                "Chỉ đề xuất sản phẩm trong selected_catalog_products.",
                "Không nhắc hoặc hiển thị nguồn, URL hay quá trình tìm kiếm web.",
                "Chỉ dùng retrieved_product_knowledge để giải thích lý do chọn, vai trò "
                "routine và lưu ý. Không tự bổ sung công dụng ngoài kho tri thức.",
                "Không gọi tên bệnh, không kê thuốc, không hứa điều trị hoặc chữa khỏi.",
                "Khuyên thử sản phẩm trên vùng nhỏ trước.",
                "Không dùng hoặc suy luận hồ sơ da lịch sử cho ảnh hiện tại. Nếu câu hỏi "
                "không nói loại da, phải nói rõ sản phẩm được chọn theo đặc điểm bề mặt "
                "nhìn thấy và chưa phải kết luận loại da.",
                "Mở đầu bằng các đặc điểm thực sự quan sát được ở ảnh hiện tại; không "
                "dùng một đoạn nhận xét chung cho mọi ảnh.",
                "Với mỗi sản phẩm, giải thích ngắn lý do chọn gắn với đặc điểm ảnh hoặc "
                "loại da đã xác nhận. Không nói rằng đã xác định loại da chỉ từ ảnh.",
                *(
                    [
                        "Đây là care mode cần đánh giá y tế: nói rõ sản phẩm chỉ hỗ trợ "
                        "chăm sóc, không điều trị nguyên nhân và nên sắp xếp khám bác sĩ da liễu.",
                        "Chỉ hướng dẫn dùng trên vùng da còn nguyên vẹn; ngừng dùng nếu "
                        "châm chích, đau, đỏ hoặc khó chịu tăng.",
                        "Không làm nhẹ mức độ cảnh báo và không gọi đây là routine điều trị.",
                    ]
                    if medical_review
                    else []
                ),
            ],
        }
        with langfuse_observation(
            self.langfuse,
            name="misa-final-response",
            as_type="generation",
            input=context,
            model=self.settings.misa_model,
        ) as generation:
            reply = _remove_source_attribution(
                generate_skin_reply(context, response_language=language)
            )
            if generation:
                generation.update(output={"reply": reply})
        return SkinConsultationResult(
            status=(
                "recommended_with_medical_review" if medical_review else "recommended"
            ),
            reply=reply,
            observation=observation,
            product_ids=[product["id"] for product in products],
        )
