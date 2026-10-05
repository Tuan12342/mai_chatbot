# Mai Agent với LangChain và LangGraph

Folder này chứa lõi agent Mai dùng LangGraph và Gemini.

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

## Human handoff

Gemini đánh giá ngữ nghĩa của tin nhắn trước khi router nghiệp vụ chạy, không dò danh sách
từ khóa cố định. Hệ thống chuyển người thật khi khách yêu cầu trực tiếp, thể hiện tiêu cực
mạnh, hoặc lặp lại cùng loại phàn nàn chưa được giải quyết từ hai lần. Context package được
lưu tại `data/handoffs.json`, gồm lịch sử hội thoại, giỏ hàng, đơn liên quan và lý do khách
bức xúc. Mai sau đó tạm dừng phản hồi để không chồng chéo với người thật và kiểm tra trạng
thái handoff ở các tin nhắn tiếp theo để biết khi nào được resume. Giao diện quản trị sẽ
được bổ sung ở giai đoạn sau.

## Luồng graph

```text
START
  → classify_intent
      ├── product_question/recommendation → assistant → END
      ├── order → order → END
      └── unknown → fallback → END
```

LangGraph quản lý state, routing và vòng lặp product tool. LangChain gọi Gemini trong node
`assistant`.

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

## Chạy test

```powershell
pytest
ruff check .
```

## Thứ tự đọc source

1. `src/mai_agent/state.py`
2. `src/mai_agent/nodes.py`
3. `src/mai_agent/graph.py`
4. `src/mai_agent/runner.py`
5. `tests/test_graph.py`

Giai đoạn tiếp theo có thể bổ sung product catalog và RAG, sau đó mới thêm memory MongoDB và state machine đặt hàng.
