

1. Người dùng gửi tin nhắn:

Người dùng nhập văn bản thuần vào ô chat, ví dụ `Sản phẩm này dùng thế nào?`.
Khi người dùng bấm Gửi, JavaScript trong [web/index.html](src/mai_agent/web/index.html#L192) gửi yêu cầu `POST /api/chat` với `Content-Type: application/json`. Payload ví dụ:

json
{
  "session_id": "phien-chat-vi-du",
  "user_id": "khach-hang-vi-du",
  "message": "OA001 dùng thế nào?"
}


 `session_id` :Mã phiên đang mở, lấy từ API tạo hoặc liệt kê phiên. Chuỗi từ 1 đến 100 ký tự.
 `user_id`  Mã người dùng đang chọn trong giao diện thử nghiệm. Chuỗi từ 1 đến 100 :ký tự.
`message` :Nội dung khách nhập. Chuỗi từ 1 đến 4.000 ký tự theo schema API.



2. API kiểm tra người dùng và phiên thế nào


 FastAPI và Pydantic kiểm tra các trường bắt buộc, kiểu dữ liệu và độ dài theo `ChatRequest`.
 `_require_test_user()` tìm `user_id` trong hồ sơ khách ở `data/customers.json` theo `zalo_user_id`.
 `chat()` dùng `.strip()` để bỏ khoảng trắng đầu cuối và kiểm tra nội dung còn lại.
 `get_chat_session(session_id, user_id)` truy vấn SQLite theo cả hai mã.





3. Policy guard kiểm tra bằng cách nào

Sau khi `routing.py` xác định ngôn ngữ, [graph.py](src/mai_agent/graph.py#L87) gọi `policy_guard_node()` trong [policy.py](src/mai_agent/policy.py#L486). Bước này dùng Gemini để phân tích ngữ nghĩa và quy tắc Python để quyết định đường đi.

1. Phân tích yêu cầu: `_assess_policy()` gửi tin nhắn mới nhất và phần hội thoại lấy từ tối đa 12 phần tử gần nhất trong `state.messages`. Prompt yêu cầu Gemini gọi tool `assess_message` và trả dữ liệu có cấu trúc, gồm mục tiêu, thao tác được yêu cầu, thẩm quyền của bot, phạm vi dữ liệu riêng tư, mức độ khiếu nại, nhu cầu xác minh và yêu cầu an toàn. Prompt hướng dẫn model dựa trên ngữ nghĩa hội thoại.
2. Kiểm tra riêng quyền truy cập: Python chuẩn hóa kết quả. Nếu lượt đầu đã xác định thao tác không được phép trên tài nguyên của người khác, nó dùng kết quả đó ngay. Với các trường hợp còn lại, Gemini được yêu cầu gọi `assess_protected_resource_access` bằng nội dung tin nhắn mới nhất. Python gộp kết quả kiểm tra này vào đánh giá ban đầu.
3. Áp dụng quy tắc: Code xét tín hiệu chuyển nhân viên trước khi soạn phản hồi chính sách. Tín hiệu có thể là khách muốn gặp người thật, yêu cầu cần thẩm quyền nhân viên, khiếu nại mạnh hoặc cùng loại khiếu nại lặp lại ít nhất hai lần và chưa được giải quyết. Kết quả từ chối của model hoặc từ chối quyền truy cập loại trừ việc chuyển nhân viên theo đánh giá này.
4. Chọn bước kế tiếp: Nếu cần phản hồi theo chính sách, `reply_generator.py` dùng Gemini để diễn đạt nội dung theo chỉ dẫn của `policy.py`. Node đặt `policy_handled=True` và graph kết thúc lượt. Với yêu cầu cần chuyển nhân viên hoặc cần xử lý nghiệp vụ tiếp, node trả `policy_handled=False` kèm `triage`, rồi graph gọi `handoff_guard`.

4.Luồng tiếp tục


Kiểm tra chuyển nhân viên [skills/handoff/node.py](src/mai_agent/skills/handoff/node.py) tạo yêu cầu, chờ nhân viên đang phụ trách hoặc cho bot tiếp tục.
 Xác nhận hồ sơ da : [profile_memory.py](src/mai_agent/profile_memory.py) xử lý hồ sơ cần xác nhận trước khi tiếp tục gợi ý hoặc phân loại yêu cầu. |
 Hỏi sản phẩm : [skills/product/node.py](src/mai_agent/skills/product/node.py) gọi [tools.py](src/mai_agent/skills/product/tools.py) khi cần tra catalog, rồi soạn trả lời dựa trên kết quả.
 Gợi ý mỹ phẩm : [profile_memory.py](src/mai_agent/profile_memory.py) xử lý ngữ cảnh hồ sơ, rồi graph có thể chuyển sang [recommendation.py](src/mai_agent/skills/product/recommendation.py) hoặc trợ lý sản phẩm.
 Đặt hàng : [skills/order/node.py](src/mai_agent/skills/order/node.py) xử lý giỏ, tồn kho, địa chỉ và xác nhận. API lưu đơn khi trạng thái là `confirmed`.
 Tra đơn : [skills/order_lookup/node.py](src/mai_agent/skills/order_lookup/node.py) lấy đơn từ [order_store.py](src/mai_agent/order_store.py).


`api.py` cập nhật phiên và trả JSON về trình duyệt với `reply`, `current_step`, `cart`, `handoff_status` và `payment` nếu có. Nếu lỗi xảy ra trong khối chạy graph, lưu đơn hoặc cập nhật phiên, API ghi log và trả HTTP 502.

5. Đường đi theo từng file và hàm





1. [web/index.html](src/mai_agent/web/index.html#L180) → `sendMessage()` lấy câu chữ khách nhập và gửi `POST /api/chat` với `message`, `user_id` và `session_id`.
2. [api.py](src/mai_agent/api.py#L264) → `chat()` nhận payload qua `ChatRequest`. Hàm gọi [customer_store.py](src/mai_agent/customer_store.py) → `find_customer_by_zalo_id()` để tìm người dùng trong `data/customers.json`, rồi gọi [chat_sessions.py](src/mai_agent/chat_sessions.py#L76) → `get_chat_session()` để kiểm tra cặp mã phiên và người dùng trong SQLite.
3. [api.py](src/mai_agent/api.py#L118) → `_invoke_graph()` chuyển nội dung thành `HumanMessage` và gọi graph được dựng trong [graph.py](src/mai_agent/graph.py#L58). Mã phiên trở thành `thread_id` để LangGraph dùng checkpoint của đúng cuộc trò chuyện.
4. [graph.py](src/mai_agent/graph.py#L87) → [routing.py](src/mai_agent/routing.py#L98) → `detect_language_node()` cập nhật `session.language_code`. Graph chuyển sang `policy_guard`.

Luồng rẽ nhánh

1. [graph.py](src/mai_agent/graph.py#L87) → [policy.py](src/mai_agent/policy.py#L486) → `policy_guard_node()` gọi `_assess_policy()` với tin nhắn mới và lịch sử gần đây.
2. `policy.py` → [llm.py](src/mai_agent/llm.py) → `invoke_tool()` gọi Gemini để lấy kết quả `assess_message`. Nếu lượt này chưa xác định rõ thao tác bị cấm trên dữ liệu của người khác, luồng bình thường gọi tiếp tool `assess_protected_resource_access` với tin nhắn mới nhất. `policy.py` chuẩn hóa và gộp các kết quả vào `triage`.
3. Nếu cần phản hồi theo chính sách, `policy.py` → [reply_generator.py](src/mai_agent/reply_generator.py#L8) → `llm.py` để soạn câu trả lời. Node đặt `policy_handled=True`, rồi `route_after_policy_guard()` trả `end`.
4. Nếu cần tiếp tục, `policy.py` trả `triage` cùng `policy_handled=False`. `graph.py` gọi [skills/handoff/node.py](src/mai_agent/skills/handoff/node.py#L83) → `handoff_guard_node()`. Nhánh `create` hoặc `wait` đi qua bước xử lý nhân viên rồi kết thúc lượt. Nhánh `continue` chuyển sang xác nhận hồ sơ.

 Trả kết quả về trình duyệt

[api.py](src/mai_agent/api.py#L264) nhận kết quả từ graph → cập nhật phiên qua `chat_sessions.py` → lấy thông tin thanh toán qua `payment.py` nếu có đơn đủ điều kiện → trả `ChatResponse` → [web/index.html](src/mai_agent/web/index.html#L194) đọc JSON và hiển thị câu trả lời hoặc thẻ thanh toán.

Ví dụ với **“OA001 dùng thế nào?”**, đường đi dự kiến là `web/index.html → api.py → graph.py → routing.py → policy.py → skills/handoff/node.py → profile_memory.py → routing.py → skills/product/node.py → skills/product/tools.py → skills/product/node.py → api.py → web/index.html`. Lần đầu qua `routing.py` là nhận diện ngôn ngữ. Lần sau là phân loại yêu cầu.
