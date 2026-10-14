
## Chạy giao diện web

```bash
uv sync
uv run mai-agent-web
```

Mở `http://127.0.0.1:8000`. Giao diện thử nghiệm dùng một người dùng cố định và chưa có
đăng nhập; lịch sử được giữ trong RAM khi server còn chạy.

Nếu port 8000 đang được dùng:

```bash
MAI_AGENT_WEB_PORT=8765 uv run mai-agent-web
```

## Luồng graph

```text
START
  → detect_language → handoff_guard → profile_confirmation → classify_intent
      ├── recommendation → profile_memory → assistant → END
      ├── product_question → assistant → END
      ├── order → order → END
      ├── order_lookup → order_lookup → END
      └── unknown → fallback → END
```

LangGraph quản lý state, routing và vòng lặp product tool. LangChain gọi Gemini trong node
`assistant`.

## Hồ sơ da và memory dài hạn

Hồ sơ khách được gắn với `zalo_user_id` trong `data/customers.json`, gồm loại da,
độ nhạy cảm/dị ứng, vấn đề quan tâm, thành phần cần tránh, SKU không thích và
`last_updated`. Thông tin khách nói rõ trong luồng tư vấn được lưu dùng chung giữa
các phiên. Khi loại da mới mâu thuẫn với hồ sơ hoặc hồ sơ đã quá 90 ngày, Mai phải
hỏi xác nhận trước khi dùng/cập nhật. Công cụ gợi ý luôn lọc sản phẩm hết hàng,
không phù hợp loại da, chứa thành phần bị loại trừ hoặc nằm trong danh sách không thích.

## Cài đặt

```powershell
cd E:\1\mai-agent-langgraph
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Mặc định ứng dụng dùng Google Gemini. Để kiểm thử với MISA AI Gateway, cấu hình:

```dotenv
LLM_PROVIDER=misa
MISA_API_KEY=<token của bạn>
MISA_BASE_URL=https://ai.misa.vn/nlp/llm-gateway/v1
MISA_MODEL=misa-ai-1.1-plus
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=1
```

Không commit file `.env` hoặc token thật. Product embeddings vẫn dùng
`GOOGLE_EMBEDDING_MODEL`; các câu hỏi cần semantic search vì vậy vẫn cần Google API key.

## Luồng nghiệp vụ có cấu trúc

Agent dùng model để nhận diện intent và trích entity, nhưng không để model tự quyết
định SKU, giá, tồn kho hoặc trạng thái đơn:

```text
Tin nhắn → intent/entity → catalog resolver → action/state machine → reply generator
```

- `catalog.py` chuẩn hóa alias Việt/Anh/Trung/Hàn/Tây Ban Nha về SKU.
- `skills/order/tools.py` kiểm tra lại entity của model bằng catalog.
- `skills/order/node.py` xử lý giỏ hàng, tồn kho, địa chỉ và xác nhận đơn bằng Python.
- `profile_memory.py` chỉ lưu dữ liệu khách nói rõ, kèm nguồn và thời điểm xác nhận.
- Model ở bước cuối chỉ diễn đạt dữ liệu nghiệp vụ đã được kiểm tra.

## Chạy bộ acceptance

Chạy mới toàn bộ bộ câu hỏi DOCX bằng provider đang chọn trong `.env`:

```bash
uv run python scripts/run_acceptance.py --fresh --timeout 90
```

Nếu tiến trình bị dừng, chạy lại không có `--fresh` để tiếp tục từ checkpoint:

```bash
uv run python scripts/run_acceptance.py --timeout 90
```

Có thể chạy từng nhóm hoặc thử một batch nhỏ trước:

```bash
uv run python scripts/run_acceptance.py --section 1 --fresh --timeout 90
uv run python scripts/run_acceptance.py --limit 10 --fresh --timeout 90
```

Mặc định runner áp dụng `data/acceptance_scope.json`: loại S2 và các chức năng S4
chưa triển khai. Phạm vi hiện tại có 272 ca:

```bash
uv run python scripts/run_acceptance.py \
  --output tests/results/acceptance_without_s2.jsonl \
  --fresh --timeout 90
```

Kết quả chi tiết nằm tại `tests/results/acceptance_misa.jsonl`, bản tóm tắt tại
`tests/results/acceptance_misa.md`. Mỗi ca dùng bản sao tạm của dữ liệu khách và handoff,
vì vậy không sửa dữ liệu vận hành. Trạng thái `NEEDS_SETUP` nghĩa là câu hỏi cần hội thoại
hoặc trạng thái giả lập trước đó; không nên xem kết quả chạy độc lập là pass/fail cuối cùng.

Để model chấm semantic và chuyển `REVIEW` thành `PASS` hoặc `FAIL_SEMANTIC`, thêm
`--judge` khi bắt đầu một run mới:

```bash
uv run python scripts/run_acceptance.py \
  --output tests/results/acceptance_without_s2_judged.jsonl \
  --fresh --judge --timeout 90
```

Judge không được phép nâng `FAIL_CONTRACT`, `ERROR`, `TIMEOUT` hay `NEEDS_SETUP`
thành `PASS`. Chế độ này tạo thêm một lần gọi model cho mỗi ca đủ điều kiện chấm.

Chỉ khi cần kiểm tra cả chức năng chưa triển khai, truyền `--include-out-of-scope`.

Chạy và chấm một ca cụ thể:

```bash
uv run python scripts/run_acceptance.py \
  --case-id S6-Q17 --output tests/results/s6-q17.jsonl \
  --fresh --judge --timeout 90
```

Để hiển thị mã thanh toán VietQR sau khi khách xác nhận đơn, cấu hình các biến
`VIETQR_BANK_ID`, `VIETQR_ACCOUNT_NUMBER` và `VIETQR_ACCOUNT_NAME` trong `.env`.
`VIETQR_BANK_ID` có thể là mã ngân hàng như `MB`, `VCB` hoặc `ACB`. Mã QR sẽ tự
điền đúng tổng tiền và nội dung chuyển khoản theo mã đơn; ứng dụng không hiển thị
QR nếu chưa cấu hình đủ thông tin để tránh khách chuyển nhầm tài khoản.

## Chạy agent

```powershell
mai-agent
```
