# Mai Agent với LangChain và LangGraph

Folder này chỉ chứa lõi agent đầu tiên. Chưa có MongoDB, FastAPI, RAG, giỏ hàng hay giao diện.

## Luồng graph

```text
START
  → handoff_guard (Gemini đọc ngữ cảnh và quyết định có cần người thật)
      ├── handoff → human_active → END
      └── classify_intent
          ├── product_question/recommendation → assistant → END
          ├── order → order → END
          └── unknown → fallback → END
```

LangGraph quản lý state, routing và vòng lặp product tool. LangChain gọi Gemini trong node
`assistant`.

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

Giai đoạn tiếp theo có thể bổ sung product catalog và RAG, sau đó mới thêm memory MongoDB và state machine đặt hàng.
