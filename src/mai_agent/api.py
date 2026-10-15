import logging
import os
import sqlite3
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from mai_agent.chat_sessions import (
    create_chat_session,
    get_chat_session,
    list_chat_sessions,
    touch_chat_session,
)
from mai_agent.config import get_settings
from mai_agent.customer_store import find_customer_by_zalo_id, load_customers
from mai_agent.graph import create_agent_graph
from mai_agent.observability import create_langfuse_handler
from mai_agent.order_store import save_order
from mai_agent.payment import build_vietqr_payment
from mai_agent.vision import InvalidSkinImage, SkinConsultationService, sanitize_skin_image

LOGGER = logging.getLogger(__name__)
GRAPH_LOCK = Lock()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DATABASE = DATA_DIR / "chat_checkpoints.sqlite"
CHECKPOINT_CONNECTION = sqlite3.connect(CHECKPOINTS_DATABASE, check_same_thread=False)

graph = create_agent_graph(checkpointer=SqliteSaver(CHECKPOINT_CONNECTION))
app = FastAPI(title="Mai — OA Cosmetics", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class ChatMessage(BaseModel):
    role: str
    content: str


class PaymentResponse(BaseModel):
    order_id: str
    amount: int
    bank_id: str
    account_number: str
    account_name: str
    transfer_content: str
    qr_image_url: str


class ChatResponse(BaseModel):
    reply: str
    current_step: str
    cart: list[dict[str, Any]]
    handoff_status: str
    payment: PaymentResponse | None = None
    analysis_status: str | None = None
    recommended_product_ids: list[str] = Field(default_factory=list)


class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class TestUser(BaseModel):
    user_id: str
    name: str


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)


def _require_test_user(user_id: str) -> None:
    if find_customer_by_zalo_id(user_id) is None:
        raise HTTPException(status_code=400, detail="User ID kiểm thử không hợp lệ.")


def _graph_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _traced_graph_config(session_id: str, user_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        **_graph_config(session_id),
        "run_name": "mai-chat-turn",
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": session_id,
            "langfuse_tags": ["mai-agent", "web"],
        },
    }
    if handler := create_langfuse_handler():
        config["callbacks"] = [handler]
    return config


def _invoke_graph(message: str, session_id: str, user_id: str) -> dict[str, Any]:
    with GRAPH_LOCK:
        return graph.invoke(
            {
                "user_id": user_id,
                "messages": [HumanMessage(content=message)],
            },
            config=_traced_graph_config(session_id, user_id),
        )


def _read_history(session_id: str) -> list[ChatMessage]:
    with GRAPH_LOCK:
        snapshot = graph.get_state(_graph_config(session_id))
    state = snapshot.values or {}
    history: list[ChatMessage] = []
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = str(message.content).strip()
        if content:
            history.append(ChatMessage(role=role, content=content))
    return history


def _payment_for_order(order: dict[str, Any]) -> dict[str, Any] | None:
    if order.get("status") != "confirmed" or order.get("payment_status") != "pending":
        return None
    return build_vietqr_payment(order)


def _read_payment(session_id: str) -> dict[str, Any] | None:
    with GRAPH_LOCK:
        snapshot = graph.get_state(_graph_config(session_id))
    return _payment_for_order((snapshot.values or {}).get("active_order", {}))


def _record_skin_exchange(session_id: str, user_id: str, message: str, reply: str) -> None:
    """Chỉ lưu text và dấu mốc ảnh; tuyệt đối không đưa bytes/base64 vào checkpoint."""
    safe_message = f"[Đã gửi ảnh da để phân tích riêng tư] {message}"
    with GRAPH_LOCK:
        graph.update_state(
            _graph_config(session_id),
            {
                "user_id": user_id,
                "messages": [
                    HumanMessage(content=safe_message),
                    AIMessage(content=reply),
                ],
                "reply": reply,
            },
        )


def _read_session_state(session_id: str) -> dict[str, Any]:
    with GRAPH_LOCK:
        snapshot = graph.get_state(_graph_config(session_id))
    return dict(snapshot.values or {})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = files("mai_agent.web").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/test-users", response_model=list[TestUser])
def test_users() -> list[dict[str, str]]:
    """Danh tính giả lập cho giao diện kiểm thử; không dùng làm auth production."""
    return [
        {"user_id": customer["zalo_user_id"], "name": customer.get("name", "Khách")}
        for customer in load_customers()
    ]


@app.get("/api/history", response_model=list[ChatMessage])
def history(session_id: str, user_id: str) -> list[ChatMessage]:
    _require_test_user(user_id)
    if get_chat_session(session_id, user_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return _read_history(session_id)


@app.get("/api/payment", response_model=PaymentResponse | None)
def payment(session_id: str, user_id: str) -> dict[str, Any] | None:
    _require_test_user(user_id)
    if get_chat_session(session_id, user_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return _read_payment(session_id)


@app.get("/api/sessions", response_model=list[ChatSession])
def sessions(user_id: str) -> list[dict[str, Any]]:
    _require_test_user(user_id)
    return list_chat_sessions(user_id)


@app.post("/api/sessions", response_model=ChatSession, status_code=201)
def create_session(payload: CreateSessionRequest) -> dict[str, Any]:
    _require_test_user(payload.user_id)
    return create_chat_session(payload.user_id)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    _require_test_user(payload.user_id)
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")
    if get_chat_session(payload.session_id, payload.user_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    try:
        result = _invoke_graph(message, payload.session_id, payload.user_id)
        active_order = result.get("active_order", {})
        if active_order.get("status") == "confirmed":
            save_order(active_order)
        touch_chat_session(payload.session_id, payload.user_id, message)
    except Exception as error:
        LOGGER.exception("Không thể xử lý tin nhắn", exc_info=error)
        raise HTTPException(
            status_code=502,
            detail="Mai chưa thể xử lý tin nhắn lúc này. Vui lòng thử lại.",
        ) from error

    session = result.get("session", {})
    handoff = result.get("handoff", {})
    active_order = result.get("active_order", {})
    return ChatResponse(
        reply=result.get("reply", ""),
        current_step=session.get("current_step", "idle"),
        cart=session.get("cart", []),
        handoff_status=handoff.get("status", "inactive"),
        payment=_payment_for_order(active_order),
    )


@app.post("/api/skin-analysis", response_model=ChatResponse)
def skin_analysis(
    session_id: Annotated[str, Form(min_length=1, max_length=100)],
    user_id: Annotated[str, Form(min_length=1, max_length=100)],
    message: Annotated[str, Form(min_length=1, max_length=4000)],
    consent: Annotated[bool, Form()],
    image: Annotated[UploadFile, File()],
) -> ChatResponse:
    _require_test_user(user_id)
    if get_chat_session(session_id, user_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Cần xác nhận người trong ảnh đã đồng ý cho phân tích.",
        )

    settings = get_settings()
    uploaded = image.file.read(settings.skin_image_max_bytes + 1)
    try:
        image_bytes, mime_type = sanitize_skin_image(
            uploaded,
            image.content_type or "",
            settings.skin_image_max_bytes,
        )
    except InvalidSkinImage as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    customer = find_customer_by_zalo_id(user_id) or {}
    try:
        consultation = SkinConsultationService(settings=settings).consult(
            image_bytes=image_bytes,
            mime_type=mime_type,
            message=message.strip(),
            customer=customer,
        )
        _record_skin_exchange(session_id, user_id, message.strip(), consultation.reply)
        touch_chat_session(session_id, user_id, message.strip())
    except Exception as error:
        LOGGER.exception("Không thể xử lý yêu cầu phân tích ảnh da", exc_info=error)
        raise HTTPException(
            status_code=502,
            detail="Mai chưa thể phân tích ảnh lúc này. Vui lòng thử lại sau.",
        ) from error
    finally:
        # Không giữ tham chiếu đến dữ liệu ảnh lâu hơn vòng đời request.
        image_bytes = b""
        uploaded = b""

    state = _read_session_state(session_id)
    session = state.get("session", {})
    handoff = state.get("handoff", {})
    return ChatResponse(
        reply=consultation.reply,
        current_step=session.get("current_step", "idle"),
        cart=session.get("cart", []),
        handoff_status=handoff.get("status", "inactive"),
        payment=_payment_for_order(state.get("active_order", {})),
        analysis_status=consultation.status,
        recommended_product_ids=consultation.product_ids,
    )


def main() -> None:
    import uvicorn

    host = os.getenv("MAI_AGENT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("MAI_AGENT_WEB_PORT", "8000"))
    uvicorn.run("mai_agent.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
