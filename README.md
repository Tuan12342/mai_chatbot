
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

Để hiển thị mã thanh toán VietQR sau khi khách xác nhận đơn, cấu hình các biến
`VIETQR_BANK_ID`, `VIETQR_ACCOUNT_NUMBER` và `VIETQR_ACCOUNT_NAME` trong `.env`.
`VIETQR_BANK_ID` có thể là mã ngân hàng như `MB`, `VCB` hoặc `ACB`. Mã QR sẽ tự
điền đúng tổng tiền và nội dung chuyển khoản theo mã đơn; ứng dụng không hiển thị
QR nếu chưa cấu hình đủ thông tin để tránh khách chuyển nhầm tài khoản.

## Chạy agent

```powershell
mai-agent
```







