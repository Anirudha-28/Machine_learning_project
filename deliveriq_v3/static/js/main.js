document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("chat-toggle");
  const panel = document.getElementById("chat-panel");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messages = document.getElementById("chat-messages");
  const history = [];  // {role, content} pairs so follow-ups keep context

  toggle.addEventListener("click", () => panel.classList.toggle("hidden"));

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    appendMessage("you", text);
    input.value = "";

    const typingEl = appendTyping();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
      });
      const data = await res.json();
      typingEl.remove();
      appendMessage("bot", data.reply);
      history.push({ role: "user", content: text });
      history.push({ role: "assistant", content: data.reply });
    } catch (err) {
      typingEl.remove();
      appendMessage("bot", "Sorry, something went wrong.");
    }
  });

  function appendMessage(who, text) {
    const div = document.createElement("div");
    div.style.margin = "0.4rem 0";
    div.style.textAlign = who === "you" ? "right" : "left";
    div.innerHTML = `<span style="display:inline-block;padding:0.4rem 0.6rem;border-radius:6px;background:${who === "you" ? "#e4efe8" : "#f1efe9"}">${text}</span>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function appendTyping() {
    const div = document.createElement("div");
    div.className = "widget-typing";
    div.style.margin = "0.4rem 0";
    div.style.textAlign = "left";
    div.innerHTML = `<span class="widget-typing-bubble"><span></span><span></span><span></span></span>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }
});
