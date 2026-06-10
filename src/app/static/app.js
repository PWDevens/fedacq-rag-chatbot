async function send() {
  const textarea = document.getElementById("question");
  const q = textarea.value.trim();
  if (!q) return;

  const chat = document.getElementById("chat-container");
  const citationsDiv = document.getElementById("citations");
  const typingDiv = document.getElementById("typing");
  const form = document.getElementById("chat-form");
  const submitBtn = form.querySelector("button[type=submit]");

  citationsDiv.innerHTML = "";
  typingDiv.textContent = "Bot is thinking...";
  submitBtn.disabled = true;

  const userBubble = document.createElement("div");
  userBubble.className = "bubble user-bubble";
  userBubble.textContent = q;
  chat.appendChild(userBubble);

  textarea.value = "";

  const botBubble = document.createElement("div");
  botBubble.className = "bubble bot-bubble";
  botBubble.textContent = "";
  chat.appendChild(botBubble);

  chat.scrollTop = chat.scrollHeight;

  let response;
  try {
    response = await fetch("/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
  } catch (err) {
    botBubble.textContent = "[Network error]";
    typingDiv.textContent = "";
    submitBtn.disabled = false;
    return;
  }

  if (!response.ok) {
    botBubble.textContent = `[Error: HTTP ${response.status}]`;
    typingDiv.textContent = "";
    submitBtn.disabled = false;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.startsWith("data:")) continue;

        const data = part.slice(5).trim();
        if (!data) continue;

        // Citations payload
        if (data.startsWith("{")) {
          try {
            const parsed = JSON.parse(data);
            if (parsed.citations) {
              citationsDiv.innerHTML =
                "<b>Citations:</b><br>" +
                parsed.citations
                  .map((c) => "- " + JSON.stringify(c))
                  .join("<br>");
            }
          } catch {
            botBubble.textContent += "\n[Error parsing citations]";
          }
          continue;
        }

        // Token text
        botBubble.textContent += data;
      }

      chat.scrollTop = chat.scrollHeight;
    }
  } catch (err) {
    botBubble.textContent += "\n[Stream interrupted]";
  }

  typingDiv.textContent = "";
  submitBtn.disabled = false;
}

document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});

document.getElementById("question").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    this.form.requestSubmit();
  }
});
