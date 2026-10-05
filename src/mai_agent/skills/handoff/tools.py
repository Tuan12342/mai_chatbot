from typing import Literal

from langchain_core.tools import tool

IssueType = Literal[
    "product_quality",
    "wrong_item",
    "delivery",
    "payment",
    "refund",
    "allergy_safety",
    "service",
    "order_change",
    "other",
    "none",
]

NegativeIntensity = Literal["none", "mild", "strong"]


@tool
def assess_handoff(
    is_complaint: bool,
    issue_type: IssueType,
    negative_intensity: NegativeIntensity,
    requests_human: bool,
    issue_resolved: bool,
    reason: str,
) -> dict[str, object]:
    """Phân tích liệu tin nhắn mới nhất có cần chuyển cho người thật hay không.

    is_complaint chỉ đúng khi khách đang phản ánh một vấn đề chưa hài lòng, không phải
    chỉ từ chối một gợi ý. issue_type phải phản ánh bản chất vấn đề để nhận ra khách
    đang lặp lại cùng phàn nàn. negative_intensity là strong khi khách thể hiện tức giận,
    mất niềm tin. requests_human đúng khi khách muốn nói chuyện với nhân viên, quản lý,
    chủ shop hay bất kỳ người thật nào,
    dù cách diễn đạt không trùng một câu mẫu. issue_resolved chỉ đúng khi chính khách xác
    nhận vấn đề trước đó đã được giải quyết.
    """
    return {
        "is_complaint": is_complaint,
        "issue_type": issue_type,
        "negative_intensity": negative_intensity,
        "requests_human": requests_human,
        "issue_resolved": issue_resolved,
        "reason": reason,
    }


HANDOFF_TOOLS = [assess_handoff]
