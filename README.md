# Mai Agent

## Cách chạy

Yêu cầu: Python 3.11 trở lên và [uv](https://docs.astral.sh/uv/).

### 1. Cài đặt thư viện

```bash
uv sync
```

### 2. Cấu hình môi trường

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Mở file `.env` và điền Gemini API key:

```env
GOOGLE_API_KEY=your_api_key
```

### 3. Chạy giao diện web

```bash
uv run mai-agent-web
```

Mở `http://127.0.0.1:8000` trên trình duyệt.

Để sử dụng cổng khác:

```bash
MAI_AGENT_WEB_PORT=8765 uv run mai-agent-web
```

Trên Windows PowerShell:

```powershell
$env:MAI_AGENT_WEB_PORT=8765
uv run mai-agent-web
```

### Chạy trong terminal

```bash
uv run mai-agent
```
