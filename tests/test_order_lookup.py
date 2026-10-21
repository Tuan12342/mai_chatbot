from mai_agent.skills.order_lookup import node


def _order(order_id: str, customer_id: str) -> dict:
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": "processing",
        "payment_status": "paid",
        "total_amount": 219000,
        "shipping_address": {
            "address_line": "Dữ liệu riêng không được hiển thị",
            "phone": "0900000000",
        },
        "items": [
            {
                "product_id": "OA002",
                "product_name": "Gel rửa mặt OA Purifying Gel",
                "quantity": 1,
                "unit_price": 219000,
            }
        ],
    }


def test_lookup_lists_all_orders_owned_by_current_user_only(monkeypatch) -> None:
    captured: dict = {}

    def find_orders(customer_id):
        captured["customer_id"] = customer_id
        return [
            _order("ORDER-001", customer_id),
            _order("ORDER-002", customer_id),
            _order("ORDER-PRIVATE", "another-user"),
        ]

    def generate(instruction, *, response_language):
        captured["instruction"] = instruction
        captured["language"] = response_language
        return "Danh sách đơn của bạn"

    monkeypatch.setattr(node, "find_orders_by_customer_id", find_orders)
    monkeypatch.setattr(node, "generate_reply", generate)

    result = node.order_lookup_node(
        {"user_id": "zalo-user-001", "session": {"language_code": "vi"}}
    )

    assert captured["customer_id"] == "zalo-user-001"
    assert "ORDER-001" in captured["instruction"]
    assert "ORDER-002" in captured["instruction"]
    assert "ORDER-PRIVATE" not in captured["instruction"]
    assert "0900000000" not in captured["instruction"]
    assert "Dữ liệu riêng" not in captured["instruction"]
    assert captured["language"] == "vi"
    assert result["session"]["current_step"] == "idle"
    assert result["session"]["pending_order_lookup_id"] is None


def test_lookup_does_not_ask_for_phone_or_order_id_when_no_orders(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(node, "find_orders_by_customer_id", lambda _customer_id: [])

    def generate(instruction, *, response_language):
        captured["instruction"] = instruction
        return "Bạn chưa có đơn hàng nào."

    monkeypatch.setattr(node, "generate_reply", generate)

    result = node.order_lookup_node(
        {"user_id": "zalo-user-new", "session": {"language_code": "vi"}}
    )

    assert "không hỏi mã đơn hoặc số điện thoại" in captured["instruction"]
    assert result["reply"] == "Bạn chưa có đơn hàng nào."


def test_lookup_without_authenticated_user_does_not_query_orders(monkeypatch) -> None:
    def unexpected_query(_customer_id):
        raise AssertionError("Không được truy vấn khi thiếu user_id")

    monkeypatch.setattr(node, "find_orders_by_customer_id", unexpected_query)
    monkeypatch.setattr(
        node,
        "generate_reply",
        lambda _instruction, **_kwargs: "Không thể tải lịch sử đơn.",
    )

    result = node.order_lookup_node({"session": {"language_code": "vi"}})

    assert result["reply"] == "Không thể tải lịch sử đơn."
