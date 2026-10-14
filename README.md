
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
  → detect_language → policy_guard → handoff_guard → profile_confirmation → classify_intent
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

## Handoff cho nhân viên

`handoff_guard` chạy trước routing nghiệp vụ và để Gemini đánh giá ngữ nghĩa của lịch sử hội
thoại. Handoff chỉ được tạo khi khách yêu cầu trực tiếp gặp người thật, có phản ứng tiêu cực mạnh
với vấn đề chưa được xử lý, hoặc lặp lại cùng một khiếu nại ít nhất hai lần mà vẫn chưa được giải
quyết. Không có danh sách từ khóa cố định.

Khi handoff, `session.handoff_package` chứa lý do ngắn gọn, lịch sử gần nhất đã che số điện thoại/
email và ngữ cảnh giỏ hàng/đơn cần thiết. Trong trạng thái `active`, bot vẫn ghi nhận tin nhắn mới
vào package nhưng không sinh phản hồi. Adapter inbox dùng `human_agent_update(...)` để ghi nhận
phản hồi nhân viên; truyền `resume_ai=True` khi nhân viên đã xử lý xong để lượt tiếp theo trả quyền
cho bot. Payload này cần được ghi vào checkpoint bằng `graph.update_state(config, payload)`; không
đưa nó qua `graph.invoke`, vì phản hồi của nhân viên không phải một lượt chạy của bot.

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

Hoặc:

```powershell
python -m mai_agent.runner
```

Trước khi chạy, thêm Gemini API key vào `GOOGLE_API_KEY` trong file `.env`.

## Chạy giao diện test handoff

Giao diện chạy hoàn toàn ở máy local và hiện chỉ hiển thị khung chat khách hàng. Logic handoff
vẫn hoạt động ở backend; bảng điều khiển nội bộ cho nhân viên chưa được hiển thị ở giai đoạn này.

```powershell
cd E:\1\mai_agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Mở `.env`, điền `GOOGLE_API_KEY`, sau đó chạy:

```powershell
mai-agent-ui
```

Hoặc chạy trực tiếp module nếu terminal chưa nhận console script:

```powershell
python -m mai_agent.ui_server
```

Trình duyệt sẽ tự mở tại `http://127.0.0.1:8000`. Nếu không muốn tự mở trình duyệt:

```powershell
mai-agent-ui --no-browser
```

Kịch bản kiểm tra nhanh:

1. Gửi một câu bình thường như `Da dầu nên dùng sản phẩm nào?` — trạng thái vẫn là
   **Mai đang trả lời**.
2. Gửi `Cho tôi gặp nhân viên của shop` — trạng thái phía trên chuyển thành
   **Nhân viên đang phụ trách**.
3. Tiếp tục gửi một tin từ phía khách — tin vẫn được ghi nhận nhưng Mai không tự trả lời.

Nút **Phiên mới** tạo một thread test hoàn toàn mới. Dữ liệu UI chỉ nằm trong bộ nhớ và mất khi
dừng server.

## Chạy test

```powershell
python -m pytest
python -m ruff check .
```

## Thứ tự đọc source

1. `src/mai_agent/state.py`
2. `src/mai_agent/handoff.py`
3. `src/mai_agent/routing.py`
4. `src/mai_agent/graph.py`
5. `src/mai_agent/runner.py`
6. `tests/test_handoff.py`

Giao diện handoff này dùng backend state theo session; giao diện nghiệp vụ đầy đủ
dùng lệnh `mai-agent-web` ở đầu tài liệu.
