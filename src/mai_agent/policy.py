import re
import unicodedata
from typing import Any

from langchain_core.messages import AIMessage

from mai_agent.routing import latest_user_text
from mai_agent.state import AgentState


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )


def _handled(reply: str) -> dict[str, Any]:
    return {
        "reply": reply,
        "messages": [AIMessage(content=reply)],
        "policy_handled": True,
    }


def policy_guard_node(state: AgentState) -> dict[str, Any]:
    """Handle safety and social rules that must not depend on model routing."""
    text = _normalize(latest_user_text(state))

    if re.search(r"\b(chan doan|ke.*thuoc|thuoc boi)\b", text):
        return _handled(
            "Mình không thể chẩn đoán bệnh da hoặc kê thuốc. Bạn nên gặp bác sĩ "
            "da liễu hoặc cơ sở y tế để được thăm khám và điều trị an toàn nhé."
        )
    if "mang thai" in text and re.search(r"\b(an toan|dung duoc)\b", text):
        return _handled(
            "Mình không thể tự khẳng định sản phẩm an toàn trong thai kỳ. Mình có thể "
            "cung cấp bảng thành phần để bạn hỏi bác sĩ trước khi sử dụng."
        )
    if re.search(r"\b(be|tre)\b", text) and re.search(r"\b(cham|viem da|benh da)\b", text):
        return _handled(
            "Mình không thể tư vấn điều trị bệnh da cho trẻ nhỏ. Bạn nên hỏi bác sĩ nhi "
            "hoặc bác sĩ da liễu trước khi dùng sản phẩm cho bé."
        )
    if re.search(r"\b(mat khau|otp|ma pin)\b", text) and re.search(
        r"\b(ngan hang|tai khoan|giao dich)\b", text
    ):
        return _handled(
            "Bạn tuyệt đối không gửi mật khẩu, OTP hoặc mã PIN ngân hàng cho shop. "
            "Nếu cần đối chiếu, chỉ gửi ảnh giao dịch đã che toàn bộ thông tin bảo mật."
        )
    if "cvv" in text or re.search(r"\bso the\b", text):
        return _handled(
            "Shop không yêu cầu hoặc lưu số thẻ và mã CVV. Bạn không nên gửi các thông "
            "tin thanh toán nhạy cảm này qua hội thoại."
        )
    if re.search(r"\b(lam gia|gia mao)\b", text) and "hoa don" in text:
        return _handled("Mình không thể hướng dẫn làm giả hóa đơn hoặc thực hiện gian lận.")
    if "khach khac" in text and re.search(r"\b(thong tin|don hang|lich su)\b", text):
        return _handled(
            "Mình không thể chia sẻ thông tin, đơn hàng hoặc lịch sử của khách hàng khác."
        )
    if re.search(r"\b(nguoi hay chatbot|nguoi hay may|so dien thoai ca nhan|so ca nhan)\b", text):
        return _handled(
            "Mình là Mai, trợ lý ảo của OA Cosmetics và không có số điện thoại cá nhân. "
            "Nếu cần người thật, mình có thể chuyển yêu cầu cho nhân viên hỗ trợ."
        )
    if re.search(r"\b(chua khoi|tri khoi|chac chan.*het mun)\b", text):
        return _handled(
            "Mỹ phẩm không thể được khẳng định chắc chắn chữa khỏi mụn. Mình chỉ có thể "
            "mô tả công dụng theo dữ liệu của shop; nếu mụn nghiêm trọng, bạn nên gặp "
            "bác sĩ da liễu."
        )
    if "that vong" in text and "tra loi cham" in text:
        return _handled(
            "Mình xin lỗi vì đã phản hồi chậm và làm bạn thất vọng. Mình đang ở đây; "
            "bạn cần mình hỗ trợ nội dung nào tiếp theo ạ?"
        )
    if "cam on" in text and re.search(r"\b(on roi|khong can|duoc roi)\b", text):
        return _handled(
            "Cảm ơn bạn đã thông báo. Mình rất vui vì vấn đề đã ổn; khi cần thêm hỗ trợ, "
            "bạn cứ nhắn shop nhé."
        )
    return {"policy_handled": False}


def route_after_policy_guard(state: AgentState) -> str:
    if state.get("policy_handled"):
        return "end"
    text = latest_user_text(state).casefold()
    if re.search(r"\b(nhớ|lưu|ghi nhớ|remember|save)\b", text):
        return "profile_confirmation"
    return "handoff_guard"
