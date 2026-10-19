from io import BytesIO

from PIL import Image
from starlette.datastructures import Headers, UploadFile

from mai_agent import api
from mai_agent.vision.models import SkinConsultationResult


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (320, 320), "#d39b87").save(output, format="JPEG")
    return output.getvalue()


class FakeConsultationService:
    def __init__(self, **kwargs) -> None:
        pass

    def consult(self, **kwargs) -> SkinConsultationResult:
        assert kwargs["mime_type"] == "image/jpeg"
        assert kwargs["image_bytes"]
        return SkinConsultationResult(
            status="recommended",
            reply="Bạn có thể tham khảo OA002.",
            product_ids=["OA002"],
        )


def _patch_api(monkeypatch) -> None:
    monkeypatch.setattr(api, "_require_test_user", lambda user_id: None)
    monkeypatch.setattr(api, "get_chat_session", lambda session_id, user_id: {"id": session_id})
    monkeypatch.setattr(api, "find_customer_by_zalo_id", lambda user_id: {"skin_type": "da dầu"})
    monkeypatch.setattr(api, "SkinConsultationService", FakeConsultationService)
    monkeypatch.setattr(api, "_record_skin_exchange", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "touch_chat_session", lambda *args: None)
    monkeypatch.setattr(api, "_read_session_state", lambda session_id: {})
    monkeypatch.setattr(api, "get_langfuse_client", lambda: None)


def _upload() -> UploadFile:
    return UploadFile(
        file=BytesIO(_image_bytes()),
        filename="skin.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


def test_skin_analysis_endpoint_accepts_multipart_image(monkeypatch) -> None:
    _patch_api(monkeypatch)
    response = api.skin_analysis(
        session_id="session-1",
        user_id="user-1",
        message="Da dầu này phù hợp sản phẩm nào?",
        image=_upload(),
    )

    assert response.analysis_status == "recommended"
    assert response.recommended_product_ids == ["OA002"]
