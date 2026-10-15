import re
from urllib.parse import urlparse

from google import genai
from google.genai import types

from mai_agent.config import Settings, get_settings
from mai_agent.vision.models import SkinObservation, SkinResearch


class SkinResearchUnavailable(RuntimeError):
    pass


class GoogleSkinVisionProvider:
    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY chưa được cấu hình cho VLM.")
        self.client = client or genai.Client(api_key=self.settings.google_api_key)

    def analyze_image(self, image_bytes: bytes, mime_type: str) -> SkinObservation:
        prompt = (
            "Quan sát ảnh da để hỗ trợ lựa chọn mỹ phẩm. Chỉ mô tả đặc điểm bề mặt "
            "thực sự nhìn thấy. Không chẩn đoán, không gọi tên bệnh, không kê thuốc, "
            "không suy luận danh tính, tuổi, giới tính hay dân tộc. Nếu không chắc, "
            "dùng giá trị unclear và giảm confidence. filter_suspected=true nếu ảnh "
            "có dấu hiệu dùng bộ lọc làm đẹp. Các cờ nguy cơ chỉ phản ánh dấu hiệu "
            "nhìn thấy, không phải kết luận y khoa. Trả đúng schema được cung cấp."
        )
        response = self.client.models.generate_content(
            model=self.settings.google_vlm_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=SkinObservation,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SkinObservation):
            return parsed
        if isinstance(parsed, dict):
            return SkinObservation.model_validate(parsed)
        if not response.text:
            raise RuntimeError("Google VLM không trả kết quả quan sát ảnh.")
        return SkinObservation.model_validate_json(response.text)

    def research_skin_care(
        self,
        observation: SkinObservation,
        skin_type: str,
    ) -> SkinResearch:
        if not self.settings.skin_web_search_enabled:
            raise SkinResearchUnavailable("Tìm kiếm web cho tư vấn da đang bị tắt.")

        allowed_domains = self._allowed_domains()
        prompt = (
            "Tìm hướng dẫn chăm sóc mỹ phẩm chung, không chẩn đoán và không đề xuất "
            "thuốc. Chỉ sử dụng nguồn từ các domain sau: "
            f"{', '.join(allowed_domains)}. "
            f"Loại da người dùng đã tự xác nhận: {skin_type}. "
            "Các quan sát bề mặt đã ẩn danh từ ảnh: "
            f"độ bóng={observation.visible_shine}; "
            f"khô={observation.visible_dryness}; "
            f"đỏ={observation.visible_redness}; "
            f"nốt nhìn thấy={observation.visible_spots}. "
            "Tóm tắt ngắn các nguyên tắc về làm sạch, dưỡng ẩm và chống nắng phù hợp. "
            "Không gọi tên bệnh, không đưa phác đồ điều trị, không kê hoạt chất như thuốc."
        )
        try:
            response = self.client.models.generate_content(
                model=self.settings.google_search_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        except Exception as error:
            raise SkinResearchUnavailable(
                "Không thể gọi Google Search; hãy kiểm tra quota, billing và model."
            ) from error
        source_domains = self._grounded_allowed_domains(response, allowed_domains)
        if not source_domains:
            raise SkinResearchUnavailable(
                "Kết quả tìm kiếm không có nguồn thuộc danh sách được phép."
            )
        summary = re.sub(r"https?://\S+", "", response.text or "").strip()
        if not summary:
            raise SkinResearchUnavailable("Google Search không trả nội dung tham khảo.")
        return SkinResearch(summary=summary[:3000], source_domains=source_domains)

    def _allowed_domains(self) -> list[str]:
        domains = [
            value.strip().casefold()
            for value in self.settings.skin_search_allowed_domains.split(",")
            if value.strip()
        ]
        if not domains:
            raise SkinResearchUnavailable("Chưa cấu hình domain nguồn được phép.")
        return domains

    @staticmethod
    def _grounded_allowed_domains(response, allowed_domains: list[str]) -> list[str]:
        candidates = getattr(response, "candidates", None) or []
        found: list[str] = []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                uri = str(getattr(web, "uri", "") or "")
                title = str(getattr(web, "title", "") or "").casefold()
                hostname = (urlparse(uri).hostname or "").casefold()
                for allowed in allowed_domains:
                    if (
                        hostname == allowed
                        or hostname.endswith(f".{allowed}")
                        or allowed in title
                    ) and allowed not in found:
                        found.append(allowed)
        return found
