let chats = JSON.parse(localStorage.getItem("chats")) || {};
let currentChatId = localStorage.getItem("currentChatId");

if (Object.keys(chats).length === 0) {
  createNewChat();
} else if (!currentChatId || !chats[currentChatId]) {
  currentChatId = Object.keys(chats)[0];
}

const chatListEl = document.getElementById("chatList");
const messagesDiv = document.getElementById("messages");
const chatTitleEl = document.getElementById("chatTitle");
const form = document.getElementById("inputForm");
const input = document.getElementById("queryInput");
const themeBtn = document.getElementById("themeToggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
  themeBtn.textContent = theme === "dark" ? "☀️" : "🌙";
}

applyTheme(localStorage.getItem("theme") || "light");

themeBtn.addEventListener("click", () => {
  const t = document.documentElement.getAttribute("data-theme");
  applyTheme(t === "dark" ? "light" : "dark");
});

function createNewChat() {
  const id = Date.now().toString();
  chats[id] = { title: "Новый чат", messages: [] };
  currentChatId = id;
  save();
  render();
}

document.getElementById("newChatBtn").addEventListener("click", createNewChat);

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  chats[currentChatId].messages.push({ role: "user", content: text });
  input.value = "";
  save();
  render();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: chats[currentChatId].messages })
    });

    const data = await res.json();
    chats[currentChatId].messages.push({
      role: "assistant",
      content: data.answer
    });

    save();
    render();
  } catch (err) {
    alert("Ошибка подключения к серверу");
  }
});

function renameChat(chatId) {
  const oldTitle = chats[chatId].title;
  const newTitle = prompt("Введите новое название чата:", oldTitle);

  if (newTitle && newTitle.trim()) {
    chats[chatId].title = newTitle.trim();
    save();
    render();
  }
}

function deleteChat(chatId) {
  const ok = confirm(`Удалить чат «${chats[chatId].title}»?`);
  if (!ok) return;

  delete chats[chatId];

  const ids = Object.keys(chats);
  if (ids.length === 0) {
    createNewChat();
    return;
  }

  if (chatId === currentChatId) {
    currentChatId = ids[0];
  }

  save();
  render();
}

function render() {
  chatListEl.innerHTML = "";

  for (const id in chats) {
    const item = document.createElement("div");
    item.className = `chat-list-item ${id === currentChatId ? "active" : ""}`;

    const title = document.createElement("span");
    title.className = "chat-title";
    title.textContent = chats[id].title;

    title.onclick = () => {
      currentChatId = id;
      save();
      render();
    };

    title.ondblclick = (e) => {
      e.stopPropagation();
      renameChat(id);
    };

     // Кнопка редактирования
    const editBtn = document.createElement("button");
    editBtn.className = "chat-edit-btn";
    editBtn.innerHTML = "✏️";
    editBtn.title = "Переименовать чат";

    const delBtn = document.createElement("button");
    delBtn.className = "chat-delete-btn";
    delBtn.innerHTML = "🗑";
    delBtn.title = "Удалить чат";

    editBtn.onclick = (e) => {
      e.stopPropagation();
      renameChat(id);
    };

    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteChat(id);
    }
    

    item.appendChild(title);
    item.appendChild(editBtn);
    item.appendChild(delBtn);
    chatListEl.appendChild(item);
  }

  chatTitleEl.textContent = chats[currentChatId].title;
  chatTitleEl.ondblclick = () => renameChat(currentChatId);

  messagesDiv.innerHTML = "";
  chats[currentChatId].messages.forEach(msg => {
    const el = document.createElement("div");
    el.className = `message ${msg.role === "user" ? "user" : "bot"}`;
    el.innerHTML =
      msg.role === "assistant" && window.marked
        ? marked.parse(msg.content)
        : msg.content;
    messagesDiv.appendChild(el);
  });

  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function save() {
  localStorage.setItem("chats", JSON.stringify(chats));
  localStorage.setItem("currentChatId", currentChatId);
}

render();