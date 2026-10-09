import logging
import os
import sqlite3
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from mai_agent.chat_sessions import (
    create_chat_session,
    get_chat_session,
    list_chat_sessions,
    touch_chat_session,
)
from mai_agent.graph import create_agent_graph
from mai_agent.payment import build_vietqr_payment

LOGGER = logging.getLogger(__name__)
USER_ID = "local-user"
GRAPH_LOCK = Lock()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DATABASE = DATA_DIR / "chat_checkpoints.sqlite"
CHECKPOINT_CONNECTION = sqlite3.connect(CHECKPOINTS_DATABASE, check_same_thread=False)

graph = create_agent_graph(checkpointer=SqliteSaver(CHECKPOINT_CONNECTION))
app = FastAPI(title="Mai — OA Cosmetics", version="0.1.0")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
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


class ChatSession(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


def _graph_config(session_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": session_id,
        },
        "callbacks": [
            CallbackHandler(),
        ],
        "run_name": "mai-chat-turn",
        "metadata": {
            "langfuse_user_id": USER_ID,
            "langfuse_session_id": session_id,
            "langfuse_tags": ["mai-agent", "web"],
        },
    }


def _invoke_graph(message: str, session_id: str) -> dict[str, Any]:
    with GRAPH_LOCK:
        return graph.invoke(
            {
                "user_id": USER_ID,
                "messages": [HumanMessage(content=message)],
            },
            config=_graph_config(session_id),
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


def _read_payment(session_id: str) -> dict[str, Any] | None:
    with GRAPH_LOCK:
        snapshot = graph.get_state(_graph_config(session_id))
    order = (snapshot.values or {}).get("active_order", {})
    if order.get("status") != "confirmed" or order.get("payment_status") != "pending":
        return None
    return build_vietqr_payment(order)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = files("mai_agent.web").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history", response_model=list[ChatMessage])
def history(session_id: str) -> list[ChatMessage]:
    if get_chat_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return _read_history(session_id)


@app.get("/api/payment", response_model=PaymentResponse | None)
def payment(session_id: str) -> dict[str, Any] | None:
    if get_chat_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    return _read_payment(session_id)


@app.get("/api/sessions", response_model=list[ChatSession])
def sessions() -> list[dict[str, Any]]:
    return list_chat_sessions()


@app.post("/api/sessions", response_model=ChatSession, status_code=201)
def create_session() -> dict[str, Any]:
    return create_chat_session()


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")
    if get_chat_session(payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên chat.")
    try:
        result = _invoke_graph(message, payload.session_id)
        touch_chat_session(payload.session_id, message)
    except Exception as error:
        LOGGER.exception("Không thể xử lý tin nhắn", exc_info=error)
        raise HTTPException(
            status_code=502,
            detail="Mai chưa thể xử lý tin nhắn lúc này. Vui lòng thử lại.",
        ) from error

    session = result.get("session", {})
    handoff = result.get("handoff", {})
    active_order = result.get("active_order", {})
    payment_info = None
    if (
        active_order.get("status") == "confirmed"
        and active_order.get("payment_status") == "pending"
    ):
        payment_info = build_vietqr_payment(active_order)
    return ChatResponse(
        reply=result.get("reply", ""),
        current_step=session.get("current_step", "idle"),
        cart=session.get("cart", []),
        handoff_status=handoff.get("status", "inactive"),
        payment=payment_info,
    )


def main() -> None:
    import uvicorn

    host = os.getenv("MAI_AGENT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("MAI_AGENT_WEB_PORT", "8000"))
    uvicorn.run("mai_agent.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
