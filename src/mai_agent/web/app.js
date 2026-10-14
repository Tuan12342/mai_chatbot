const elements = {
  chat: document.querySelector("#chat"),
  customerForm: document.querySelector("#customerForm"),
  customerInput: document.querySelector("#customerInput"),
  customerSend: document.querySelector("#customerSend"),
  newSessionButton: document.querySelector("#newSessionButton"),
  statusBadge: document.querySelector("#statusBadge"),
  statusText: document.querySelector("#statusText"),
  errorBox: document.querySelector("#errorBox"),
};

let sessionId = crypto.randomUUID();
let busy = false;
let lastState = { messages: [], handoff_status: "none", current_step: "idle" };

const labels = {
  customer: "Khách hàng",
  assistant: "Mai · AI",
  human_agent: "Nhân viên shop",
};

function setBusy(value) {
  busy = value;
  elements.customerInput.disabled = value;
  elements.customerSend.disabled = value;
  if (value) elements.customerSend.textContent = "Đang xử lý…";
  else elements.customerSend.innerHTML = 'Gửi <span aria-hidden="true">↗</span>';
  renderHandoff(lastState);
}

function showError(message = "") {
  elements.errorBox.hidden = !message;
  elements.errorBox.textContent = message;
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...payload }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Không thể kết nối máy chủ.");
  return data;
}

function renderMessages(messages) {
  elements.chat.replaceChildren();
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-chat";
    empty.innerHTML =
      '<div class="sparkle">✦</div><h3>Bắt đầu một tình huống</h3>' +
      "<p>Hỏi sản phẩm bình thường, phàn nàn lặp lại hoặc yêu cầu gặp nhân viên.</p>";
    elements.chat.append(empty);
    return;
  }

  messages.forEach((message) => {
    const item = document.createElement("div");
    item.className = `message ${message.sender}`;
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = labels[message.sender] || message.sender;
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.textContent = message.content;
    item.append(meta, bubble);
    elements.chat.append(item);
  });
  elements.chat.scrollTop = elements.chat.scrollHeight;
}

function renderHandoff(state) {
  const active = ["requested", "active"].includes(state.handoff_status);
  const resolved = state.handoff_status === "resolved";

  elements.statusBadge.classList.toggle("status-human", active);
  elements.statusText.textContent = active
    ? "Nhân viên đang phụ trách"
    : resolved
      ? "Sẵn sàng trả về Mai"
      : "Mai đang trả lời";
}

function render(state) {
  lastState = state;
  renderMessages(state.messages || []);
  renderHandoff(state);
}

elements.customerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const message = elements.customerInput.value.trim();
  if (!message) return;
  showError();
  setBusy(true);
  try {
    const state = await api("/api/customer-message", { message });
    elements.customerInput.value = "";
    render(state);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

elements.customerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.customerForm.requestSubmit();
  }
});

elements.newSessionButton.addEventListener("click", () => {
  sessionId = crypto.randomUUID();
  elements.customerInput.value = "";
  showError();
  render({ messages: [], handoff_status: "none", current_step: "idle" });
  elements.customerInput.focus();
});

render({ messages: [], handoff_status: "none", current_step: "idle" });
