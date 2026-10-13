import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SESSIONS_DATABASE = DATA_DIR / "chat_sessions.sqlite"
DEFAULT_TITLE = "Cuộc trò chuyện mới"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SESSIONS_DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(chat_sessions)").fetchall()
    }
    if "user_id" not in columns:
        connection.execute(
            "ALTER TABLE chat_sessions ADD COLUMN user_id TEXT NOT NULL "
            "DEFAULT 'zalo-user-001'"
        )
    return connection


def create_chat_session(user_id: str, title: str = DEFAULT_TITLE) -> dict[str, Any]:
    session_id = str(uuid4())
    timestamp = _now()
    normalized_title = title.strip() or DEFAULT_TITLE
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, normalized_title, timestamp, timestamp),
        )
    return {
        "session_id": session_id,
        "user_id": user_id,
        "title": normalized_title,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def list_chat_sessions(user_id: str) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT session_id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat_session(session_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    with _connect() as connection:
        if user_id is None:
            row = connection.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT session_id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
    return dict(row) if row else None


def touch_chat_session(session_id: str, user_id: str, first_message: str) -> None:
    """Cập nhật thời gian và dùng tin nhắn đầu tiên làm tiêu đề phiên."""
    session = get_chat_session(session_id, user_id)
    if session is None:
        raise ValueError(f"Không tìm thấy phiên chat {session_id}.")

    title = session["title"]
    if title == DEFAULT_TITLE:
        compact_message = " ".join(first_message.split())
        title = compact_message[:60] or DEFAULT_TITLE

    with _connect() as connection:
        connection.execute(
            """
            UPDATE chat_sessions
            SET title = ?, updated_at = ?
            WHERE session_id = ? AND user_id = ?
            """,
            (title, _now(), session_id, user_id),
        )
