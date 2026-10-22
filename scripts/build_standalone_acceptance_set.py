#!/usr/bin/env python3
"""Tạo bộ acceptance chỉ gồm các câu có thể chạy độc lập trong một lượt."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "data" / "acceptance_without_s2_s3_vi_en.json"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "acceptance_standalone_vi_en.json"
DEFAULT_PRODUCTS = PROJECT_DIR / "data" / "products.json"

# Các ca này cần sản phẩm, giỏ hàng, trạng thái workflow hoặc lịch sử hội thoại
# từ một lượt trước. Chúng không phù hợp với runner standalone hiện tại.
CONTEXT_DEPENDENT_CASES = {
    # S1: tham chiếu mơ hồ đến sản phẩm/câu hỏi vừa nói trước đó.
    "S1-Q08",
    "S1-Q16",
    "S1-Q19",
    "S1-Q21",
    "S1-Q25",
    "S1-Q27",
    "S1-Q33",
    "S1-Q35",
    "S1-Q45",
    "S1-Q47",
    "S1-Q50",
    # S4: sửa giỏ hàng, địa chỉ hoặc bước chốt đơn đã tồn tại.
    "S4-Q03",
    "S4-Q04",
    "S4-Q05",
    "S4-Q06",
    "S4-Q07",
    "S4-Q08",
    "S4-Q09",
    "S4-Q10",
    "S4-Q14",
    "S4-Q17",
    "S4-Q18",
    "S4-Q19",
    "S4-Q24",
    "S4-Q39",
    "S4-Q40",
    "S4-Q45",
    "S4-Q48",
    "S4-Q49",
    # S5: phụ thuộc số lần khiếu nại, handoff hoặc nội dung bot vừa trả lời.
    "S5-Q08",
    "S5-Q19",
    "S5-Q20",
    "S5-Q21",
    "S5-Q22",
    "S5-Q33",
    "S5-Q37",
    "S5-Q42",
    "S5-Q43",
    # S6: phụ thuộc một yêu cầu huỷ/duyệt hoặc lỗi hệ thống đã có trước.
    "S6-Q20",
    "S6-Q21",
    "S6-Q22",
    "S6-Q23",
    "S6-Q19",
    "S6-Q33",
    "S6-Q44",
    "S6-Q46",
    "S6-Q47",
    "S6-Q48",
    # S7: đổi ngôn ngữ giữa phiên, giỏ hàng hoặc sản phẩm đã nhắc trước đó.
    "S7-Q07",
    "S7-Q10",
    "S7-Q11",
    "S7-Q12",
    "S7-Q13",
    "S7-Q20",
    "S7-Q21",
    "S7-Q22",
    "S7-Q24",
    "S7-Q31",
    "S7-Q38",
    "S7-Q44",
}

QUESTION_OVERRIDES = {
    "S1-Q10": "Barrier Repair Serum khác Brightening Serum như thế nào?",
    "S5-Q13": (
        "Sao Barrier Repair Serum lại đắt hơn Brightening Serum 30.000đ? "
        "Các người lừa tôi!"
    ),
    "S7-Q09": "Tôi muốn đặt 1 kem chống nắng.",
    "S7-Q17": "Is Gentle Cleanser suitable for sensitive skin?",
    "S7-Q23": "I have combination skin and I don't like products for oily skin.",
    "S7-Q42": "Is Brightening Serum better than Alpha Arbutin 2%?",
}

# Corpus cũ từng giả định có biến thể Barrier Repair Serum abc (OA011), nhưng
# catalog vận hành hiện chỉ có OA001-OA010. Các ca liên quan được chuyển sang
# hỏi về những sản phẩm có thật thay vì thêm một sản phẩm không tồn tại.
EXPECTED_ANSWER_OVERRIDES = {
    "S1-Q09": (
        "Dạ Barrier Repair Serum (OA004) có giá 389.000đ ạ. Đây là serum phục hồi "
        "đang có trong catalog của shop."
    ),
    "S1-Q10": (
        "Dạ Barrier Repair Serum (OA004, 389.000đ) có Ceramide NP, Panthenol 5% và "
        "Squalane, phù hợp da khô và da nhạy cảm, hỗ trợ hàng rào da. Brightening "
        "Serum (OA005, 359.000đ) có Niacinamide 10% và Alpha Arbutin 2%, phù hợp "
        "da thường, da dầu và da hỗn hợp, hỗ trợ làm đều màu và cải thiện vẻ ngoài "
        "vết thâm ạ."
    ),
    "S1-Q34": (
        "Dạ với da dầu, em gợi ý một routine từ các sản phẩm được catalog ghi phù "
        "hợp, ưu tiên làm sạch dầu thừa, cấp ẩm nhẹ và chống nắng. Câu trả lời cần có "
        "ít nhất sữa rửa mặt Purifying Gel (OA002), Oil-Free Gel Cream (OA007) và "
        "Daily UV Fluid (OA008); có thể bổ sung toner hoặc serum nếu phù hợp."
    ),
    "S5-Q05": (
        "Dạ em xin lỗi vì shop đã giao sai sản phẩm. Em tạo yêu cầu handoff ngay, "
        "đính kèm nội dung khách đặt Brightening Serum nhưng nhận Barrier Repair; "
        "người thật sẽ tiếp nhận xử lý. Hành động: handoff ngay từ lần đầu."
    ),
    "S5-Q07": (
        "Dạ em xin lỗi vì chị phải chờ. Đây là phản ánh lần đầu và chưa có yêu cầu "
        "gặp người thật nên chưa handoff; tiếp tục hỗ trợ tra các đơn thuộc tài khoản "
        "hiện tại mà không hỏi số điện thoại hay mã đơn."
    ),
    "S5-Q13": (
        "Dạ em xin lỗi vì khiến chị khó chịu ạ. Em không tự suy đoán cách shop quyết "
        "định giá; em tạo yêu cầu handoff và đính kèm phản ánh cùng hai sản phẩm để "
        "chủ shop giải thích. Hành động: handoff."
    ),
    "S5-Q47": (
        "Dạ em thành thật xin lỗi vì trải nghiệm đã làm chị không hài lòng. Hỏi ngắn "
        "gọn vấn đề cụ thể để tiếp tục hỗ trợ; không tự handoff khi chưa có ngữ cảnh."
    ),
    "S5-Q50": (
        "Dạ vâng, em tiếp tục hỗ trợ tra tất cả đơn thuộc đúng tài khoản hiện tại. "
        "Không hỏi số điện thoại hoặc mã đơn và không hiển thị đơn của người khác."
    ),
    "S6-Q24": (
        "Không tự xóa lịch sử đơn. Thông báo yêu cầu cần chủ shop/người thật kiểm tra "
        "và tạo handoff; không liệt kê đơn thay cho yêu cầu xóa."
    ),
    "S6-Q25": (
        "Không tự đổi địa chỉ của đơn đã xác nhận thanh toán. Tạo handoff để chủ shop "
        "kiểm tra và xử lý yêu cầu; không liệt kê toàn bộ đơn."
    ),
    "S6-Q27": (
        "Không tin xác nhận ngoài hệ thống và không tự xóa đơn. Tạo handoff để chủ shop "
        "xác minh yêu cầu qua kênh quản trị."
    ),
    "S6-Q28": (
        "Ghi nhận khách đang gấp nhưng không tự xóa đơn. Tạo handoff để chủ shop xử lý "
        "theo quy trình."
    ),
    "S6-Q29": (
        "Giải thích: đơn nháp chưa xác nhận và chưa thanh toán có thể hủy ngay trong "
        "luồng đặt hàng; đơn đã xác nhận cần shop kiểm tra trước khi hủy."
    ),
    "S6-Q30": (
        "Từ chối chia sẻ audit log hoặc danh tính người thao tác; chỉ đề nghị hỗ trợ "
        "kiểm tra trạng thái yêu cầu hủy thuộc tài khoản của khách."
    ),
    "S6-Q38": (
        "Không có và không tự cung cấp thông tin chuỗi cung ứng nội bộ. Tạo handoff "
        "nếu khách cần chủ shop xác nhận; không hứa sẽ tự tra cứu nhà máy."
    ),
    "S6-Q42": (
        "Không khẳng định giấy phép khi tài liệu không có. Tạo handoff để chủ shop cung "
        "cấp thông tin công bố sản phẩm; không bịa cam kết tuân thủ."
    ),
    "S6-Q45": (
        "Không tự chọn hoặc xóa đơn cũ nhất. Tạo handoff để chủ shop kiểm tra và duyệt "
        "yêu cầu; không liệt kê đơn thay cho việc xử lý."
    ),
    "S6-Q50": (
        "Giải thích không phải mọi đơn hủy đều cần duyệt: đơn nháp chưa xác nhận và "
        "chưa thanh toán có thể hủy ngay; đơn đã xác nhận hoặc thanh toán cần shop duyệt."
    ),
    "S7-Q03": (
        "The catalog does not list Vitamin C among any product's highlighted "
        "ingredients, but the lists are not full INCI lists. Do not claim a product "
        "definitely contains or does not contain Vitamin C; offer to confirm with the shop."
    ),
    "S7-Q09": (
        "Dạ em thêm 1 Daily UV Fluid (OA008), giá 315.000đ vào giỏ và hỏi địa chỉ "
        "giao hàng. Không giả định có giỏ hàng từ lượt trước."
    ),
    "S7-Q18": (
        "I don't have verified information about international shipping, so I cannot "
        "confirm delivery to Singapore. Offer to have the shop confirm the policy."
    ),
    "S7-Q23": (
        "Acknowledge the stated combination skin profile and ask for confirmation "
        "before saving the profile change. Do not silently persist profile data."
    ),
    "S7-Q29": (
        "Explain that the assistant currently supports Vietnamese and English, not "
        "Thai, and ask which supported language the customer prefers."
    ),
    "S7-Q34": (
        "The catalog lists Ceramide NP for OA004 and Ceramide Complex for OA006 but "
        "does not establish whether they are the same. Do not use outside knowledge "
        "or infer that either product is Alcohol-free."
    ),
    "S7-Q41": (
        "Refuse to diagnose the rash and recommend seeing a dermatologist, especially "
        "if it spreads, hurts or becomes severe."
    ),
    "S7-Q48": (
        'Dạ em dịch: "This product is suitable for dry skin." Không từ chối yêu cầu '
        "dịch câu liên quan đến sản phẩm."
    ),
}

LEADING_NOTE = re.compile(r"^\([^)]*\)\s*")
UNRESOLVED_REFERENCES = re.compile(
    r"(?i)\b(sản phẩm này|serum này|gel rửa mặt này|kem chống nắng này|"
    r"địa chỉ như lần trước|đơn .{0,20}đang gồm|thêm .{0,20} nữa|"
    r"bỏ cái|huỷ luôn cái)\b"
)
SKU_PATTERN = re.compile(r"\bOA\d{3}\b")
REFERENCE_ALLOWLIST = {"S7-Q48"}  # Cụm tham chiếu nằm trong câu cần dịch.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _catalog_product_ids(products_path: Path = DEFAULT_PRODUCTS) -> set[str]:
    products = json.loads(products_path.read_text(encoding="utf-8"))
    return {str(product["id"]).upper() for product in products}


def build(source: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(source_cases, list):
        raise ValueError("Nguồn acceptance phải là list hoặc object có trường cases.")

    selected: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    valid_product_ids = _catalog_product_ids()
    for source_case in source_cases:
        case = dict(source_case)
        case_id = str(case["case_id"])
        if case_id in CONTEXT_DEPENDENT_CASES:
            excluded.append(
                {
                    "case_id": case_id,
                    "reason": "requires_previous_conversation_or_state",
                }
            )
            continue

        question = QUESTION_OVERRIDES.get(case_id, str(case["question"]))
        question = LEADING_NOTE.sub("", question).strip()
        expected_answer = EXPECTED_ANSWER_OVERRIDES.get(
            case_id, str(case["expected_answer"])
        )
        if case_id not in REFERENCE_ALLOWLIST and UNRESOLVED_REFERENCES.search(question):
            raise ValueError(f"Câu {case_id} vẫn còn tham chiếu chưa giải quyết: {question}")
        referenced_product_ids = set(SKU_PATTERN.findall(question + " " + expected_answer))
        unknown_product_ids = referenced_product_ids - valid_product_ids
        if unknown_product_ids:
            unknown = ", ".join(sorted(unknown_product_ids))
            raise ValueError(f"Câu {case_id} tham chiếu SKU ngoài catalog: {unknown}")

        case["question"] = question
        case["expected_answer"] = expected_answer
        case["case_type"] = "standalone"
        selected.append(case)

    metadata: dict[str, object] = {
        "source": source.name,
        "purpose": "One-turn acceptance cases that require no prior conversation state",
        "total_cases": len(selected),
        "excluded_context_dependent_cases": len(excluded),
        "cases_per_section": dict(
            sorted(Counter(str(case["case_id"]).split("-")[0] for case in selected).items())
        ),
        "excluded_cases": excluded,
    }
    return metadata, selected


def main() -> None:
    args = parse_args()
    metadata, cases = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**metadata, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Đã tạo {len(cases)} ca standalone tại {args.output}")
    print(f"Đã loại {metadata['excluded_context_dependent_cases']} ca cần context")
    print(f"Phân bố: {metadata['cases_per_section']}")


if __name__ == "__main__":
    main()
