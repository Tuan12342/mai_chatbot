import re
from typing import Any, TypedDict
from urllib.parse import quote, urlencode

from mai_agent.config import Settings, get_settings


class PaymentInfo(TypedDict):
    order_id: str
    amount: int
    bank_id: str
    account_number: str
    account_name: str
    transfer_content: str
    qr_image_url: str


def build_vietqr_payment(
    order: dict[str, Any],
    settings: Settings | None = None,
) -> PaymentInfo | None:
    """Tạo thông tin VietQR cho một đơn đang chờ thanh toán.
    """
    settings = settings or get_settings()
    bank_id = settings.vietqr_bank_id.strip()
    account_number = settings.vietqr_account_number.strip()
    account_name = " ".join(settings.vietqr_account_name.split())
    order_id = str(order.get("order_id", "")).strip()
    amount = order.get("total_amount")

    if not all((bank_id, account_number, account_name, order_id)):
        return None
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._-]+", bank_id):
        return None
    if not re.fullmatch(r"[A-Za-z0-9._-]+", account_number):
        return None

    transfer_content = f"THANHTOAN {order_id}"
    template = settings.vietqr_template.strip() or "compact2"
    path = "-".join(quote(value, safe="") for value in (bank_id, account_number, template))
    query = urlencode(
        {
            "amount": amount,
            "addInfo": transfer_content,
            "accountName": account_name,
        }
    )
    return {
        "order_id": order_id,
        "amount": amount,
        "bank_id": bank_id.upper(),
        "account_number": account_number,
        "account_name": account_name,
        "transfer_content": transfer_content,
        "qr_image_url": f"https://img.vietqr.io/image/{path}.png?{query}",
    }
