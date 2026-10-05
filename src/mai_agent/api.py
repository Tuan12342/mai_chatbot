import logging
import os
from importlib.resources import files
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from mai_agent.graph import create_agent_graph

LOGGER = logging.getLogger(__name__)
USER_ID = "local-user"
THREAD_ID = "web-local-user"
GRAPH_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
GRAPH_LOCK = Lock()

graph = create_agent_graph(checkpointer=InMemorySaver())
app = FastAPI(title="Mai — OA Cosmetics", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    reply: str
    current_step: str
    cart: list[dict[str, Any]]
    handoff_status: str


def _invoke_graph(message: str) -> dict[str, Any]:
    with GRAPH_LOCK:
        return graph.invoke(
            {
                "user_id": USER_ID,
                "messages": [HumanMessage(content=message)],
            },
            config=GRAPH_CONFIG,
        )


def _read_history() -> list[ChatMessage]:
    with GRAPH_LOCK:
        snapshot = graph.get_state(GRAPH_CONFIG)
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


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = files("mai_agent.web").joinpath("index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/history", response_model=list[ChatMessage])
def history() -> list[ChatMessage]:
    return _read_history()


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Tin nhắn không được để trống.")
    try:
        result = _invoke_graph(message)
    except Exception as error:
        LOGGER.exception("Không thể xử lý tin nhắn", exc_info=error)
        raise HTTPException(
            status_code=502,
            detail="Mai chưa thể xử lý tin nhắn lúc này. Vui lòng thử lại.",
        ) from error

    session = result.get("session", {})
    handoff = result.get("handoff", {})
    return ChatResponse(
        reply=result.get("reply", ""),
        current_step=session.get("current_step", "idle"),
        cart=session.get("cart", []),
        handoff_status=handoff.get("status", "inactive"),
    )


def main() -> None:
    import uvicorn

    host = os.getenv("MAI_AGENT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("MAI_AGENT_WEB_PORT", "8000"))
    uvicorn.run("mai_agent.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
