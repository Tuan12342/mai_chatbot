# Mai Agent với LangChain và LangGraph

Folder này chỉ chứa lõi agent đầu tiên. Chưa có MongoDB, FastAPI, RAG, giỏ hàng hay giao diện.

## Luồng graph

```text
START
  → classify_intent
      ├── greeting → greeting → END
      ├── product_question/recommendation → assistant → END
      ├── order → order → END
      └── unknown → fallback → END
```

LangGraph quản lý state và routing. LangChain được dùng qua message objects và `ChatOpenAI` trong node `assistant`.

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

Mặc định `DEMO_MODE=true`, vì vậy chưa cần OpenAI API key.

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
