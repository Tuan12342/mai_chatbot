import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

HANDOFFS_FILE = Path(__file__).resolve().parents[4] / "data" / "handoffs.json"


def _load_handoffs() -> list[dict[str, Any]]:
    if not HANDOFFS_FILE.exists():
        return []
    with HANDOFFS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def _save_handoffs(handoffs: list[dict[str, Any]]) -> None:
    HANDOFFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=HANDOFFS_FILE.parent,
        delete=False,
    ) as file:
        json.dump(handoffs, file, ensure_ascii=False, indent=2)
        temporary_path = Path(file.name)
    temporary_path.replace(HANDOFFS_FILE)


def save_handoff(package: dict[str, Any]) -> None:
    handoffs = _load_handoffs()
    handoffs.append(package)
    _save_handoffs(handoffs)


def get_handoff(handoff_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in _load_handoffs() if item["handoff_id"] == handoff_id),
        None,
    )


def list_handoffs(*, status: str | None = None) -> list[dict[str, Any]]:
    handoffs = _load_handoffs()
    if status is None:
        return handoffs
    return [item for item in handoffs if item.get("status") == status]


def update_handoff(
    handoff_id: str,
    *,
    status: str,
    assigned_to: str | None = None,
    resolution_summary: str | None = None,
) -> dict[str, Any]:
    handoffs = _load_handoffs()
    for handoff in handoffs:
        if handoff["handoff_id"] != handoff_id:
            continue
        handoff["status"] = status
        handoff["updated_at"] = datetime.now().astimezone().isoformat()
        if assigned_to is not None:
            handoff["assigned_to"] = assigned_to
        if resolution_summary is not None:
            handoff["resolution_summary"] = resolution_summary
        _save_handoffs(handoffs)
        return handoff
    raise ValueError(f"Không tìm thấy handoff {handoff_id}.")


def append_customer_message(handoff_id: str, content: str) -> None:
    handoffs = _load_handoffs()
    for handoff in handoffs:
        if handoff["handoff_id"] != handoff_id:
            continue
        handoff.setdefault("messages_after_handoff", []).append(
            {
                "role": "user",
                "content": content,
                "created_at": datetime.now().astimezone().isoformat(),
            }
        )
        handoff["updated_at"] = datetime.now().astimezone().isoformat()
        _save_handoffs(handoffs)
        return
