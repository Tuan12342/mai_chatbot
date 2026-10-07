from mai_agent import chat_sessions


def test_create_list_and_touch_chat_session(tmp_path, monkeypatch) -> None:
    database = tmp_path / "chat_sessions.sqlite"
    monkeypatch.setattr(chat_sessions, "SESSIONS_DATABASE", database)

    first = chat_sessions.create_chat_session()
    second = chat_sessions.create_chat_session("Phiên có tên")

    assert chat_sessions.get_chat_session(first["session_id"]) == first
    assert {item["session_id"] for item in chat_sessions.list_chat_sessions()} == {
        first["session_id"],
        second["session_id"],
    }

    chat_sessions.touch_chat_session(
        first["session_id"],
        "  Tôi muốn tư vấn serum cho da dầu  ",
    )

    updated = chat_sessions.get_chat_session(first["session_id"])
    assert updated is not None
    assert updated["title"] == "Tôi muốn tư vấn serum cho da dầu"


def test_touch_keeps_existing_title(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        chat_sessions,
        "SESSIONS_DATABASE",
        tmp_path / "chat_sessions.sqlite",
    )
    session = chat_sessions.create_chat_session("Tư vấn trước đó")

    chat_sessions.touch_chat_session(session["session_id"], "Tin nhắn mới")

    updated = chat_sessions.get_chat_session(session["session_id"])
    assert updated is not None
    assert updated["title"] == "Tư vấn trước đó"
