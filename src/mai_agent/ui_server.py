import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from mai_agent.graph import create_agent_graph
from mai_agent.handoff import human_agent_update

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 32_000


def _message_content(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(getattr(message, "content", ""))


def _public_state(values: dict[str, Any]) -> dict[str, Any]:
    session = values.get("session", {})
    messages: list[dict[str, str]] = []
    for message in values.get("messages", []):
        if isinstance(message, HumanMessage):
            sender = "customer"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            sender = (
                "human_agent"
                if message.additional_kwargs.get("sender") == "human_agent"
                else "assistant"
            )
        else:
            continue
        content = _message_content(message).strip()
        if content:
            messages.append({"sender": sender, "content": content})

    return {
        "messages": messages,
        "handoff_status": session.get("handoff_status", "none"),
        "response_owner": values.get("response_owner", "ai"),
        "current_intent": session.get("current_intent"),
        "current_step": session.get("current_step", "idle"),
        "handoff_package": session.get("handoff_package", {}),
    }


class MaiUIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, MaiUIHandler)
        self.graph = create_agent_graph(
            checkpointer=InMemorySaver(), handoff_backend="session"
        )
        self.graph_lock = threading.Lock()

    def config_for(self, session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}

    def read_state(self, session_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.config_for(session_id))
        return _public_state(dict(snapshot.values))


class MaiUIHandler(BaseHTTPRequestHandler):
    server: MaiUIServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[UI] {self.address_string()} - {format % args}")

    def _json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length không hợp lệ.") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Nội dung rỗng hoặc vượt quá giới hạn 32 KB.")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("JSON không hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Payload phải là một JSON object.")
        return payload

    @staticmethod
    def _required_text(payload: dict[str, Any], field: str, limit: int) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Thiếu trường {field}.")
        value = value.strip()
        if len(value) > limit:
            raise ValueError(f"Trường {field} vượt quá {limit} ký tự.")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            session_id = self.headers.get("X-Session-ID", "").strip()
            if not session_id:
                self._json_response({"error": "Thiếu X-Session-ID."}, HTTPStatus.BAD_REQUEST)
                return
            with self.server.graph_lock:
                state = self.server.read_state(session_id)
            self._json_response(state)
            return

        asset = ASSETS.get(path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        body = files("mai_agent").joinpath("web", filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            session_id = self._required_text(payload, "session_id", 100)
            if self.path == "/api/customer-message":
                self._handle_customer_message(payload, session_id)
            elif self.path == "/api/agent-message":
                self._handle_agent_message(payload, session_id)
            else:
                self._json_response({"error": "API không tồn tại."}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # Giao diện dev cần hiển thị lỗi Gemini/cấu hình ngay tại chỗ.
            self._json_response({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_customer_message(self, payload: dict[str, Any], session_id: str) -> None:
        message = self._required_text(payload, "message", 4_000)
        config = self.server.config_for(session_id)
        with self.server.graph_lock:
            self.server.graph.invoke(
                {
                    "user_id": session_id,
                    "messages": [HumanMessage(content=message)],
                },
                config=config,
            )
            state = self.server.read_state(session_id)
        self._json_response(state)

    def _handle_agent_message(self, payload: dict[str, Any], session_id: str) -> None:
        message = self._required_text(payload, "message", 4_000)
        resume_ai = payload.get("resume_ai", False)
        if not isinstance(resume_ai, bool):
            raise ValueError("resume_ai phải là boolean.")
        config = self.server.config_for(session_id)
        with self.server.graph_lock:
            current = self.server.read_state(session_id)
            if current["handoff_status"] not in {"active", "requested"}:
                raise ValueError("Phiên này chưa được handoff cho nhân viên.")
            self.server.graph.update_state(
                config,
                human_agent_update(message, resume_ai=resume_ai),
            )
            state = self.server.read_state(session_id)
        self._json_response(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Giao diện test Mai Agent và human handoff")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = MaiUIServer((args.host, args.port))
    url = f"http://{args.host}:{args.port}"
    print(f"Mai Agent UI đang chạy tại {url}")
    print("Nhấn Ctrl+C để dừng.")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng Mai Agent UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
