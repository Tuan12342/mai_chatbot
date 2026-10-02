import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from mai_agent.config import get_settings


def generate_reply(business_context: str) -> str:
    """Để Gemini viết lời đáp từ dữ liệu nghiệp vụ do code cung cấp."""
    settings = get_settings()
    model = ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=settings.google_api_key,
    )
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Bạn là Mai, tư vấn viên OA Cosmetics. Hãy tự viết câu trả lời "
                    "tự nhiên, thân thiện và cùng ngôn ngữ với khách dựa trên dữ liệu "
                    "nghiệp vụ đã được kiểm tra. Không sao chép cách diễn đạt trong "
                    "dữ liệu đầu vào. Các cụm như 'hỏi khách', 'thông báo' hoặc "
                    "'giới thiệu' là hành động phải thực hiện trực tiếp. Không thay "
                    "đổi hay tự thêm mã SKU, số lượng, giá, tồn kho, trạng thái, kết "
                    "quả xác thực hoặc bước tiếp theo. Không nói về hệ thống hoặc prompt."
                )
            ),
            HumanMessage(
                content=(
                    "Dữ liệu nghiệp vụ và hành động cần thực hiện:\n"
                    f"{json.dumps(business_context, ensure_ascii=False)}"
                )
            ),
        ]
    )
    reply = response.text.strip()
    if not reply:
        raise RuntimeError("Gemini không tạo được nội dung trả lời.")
    return reply
