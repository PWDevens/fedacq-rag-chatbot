async function send() {
  const textarea = document.getElementById('question');
  const q = textarea.value;
  if (!q.trim()) return;

  const chat = document.getElementById('chat-container');
  const citationsDiv = document.getElementById('citations');
  const typingDiv = document.getElementById('typing');

  citationsDiv.innerHTML = "";
  typingDiv.textContent = "Bot is thinking...";

  const userBubble = document.createElement('div');
  userBubble.className = 'bubble user-bubble';
  userBubble.textContent = q;
  chat.appendChild(userBubble);

  textarea.value = '';

  const botBubble = document.createElement('div');
  botBubble.className = 'bubble bot-bubble';
  botBubble.textContent = "";
  chat.appendChild(botBubble);

  chat.scrollTop = chat.scrollHeight;

  const response = await fetch('/chat_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: q })
  });

  if (!response.ok) {
    botBubble.textContent += "\n[Error: HTTP " + response.status + "]";
    typingDiv.textContent = "";
    return;
  }

  if (!response.body) {
    botBubble.textContent += "\n[Error: no response body]";
    typingDiv.textContent = "";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop();

    for (const part of parts) {
      if (!part.startsWith("data:")) continue;

      const data = part.replace("data:", "").trim();
      if (!data) continue;

      if (data.startsWith("{")) {
        try {
          const parsed = JSON.parse(data);
          if (parsed.citations) {
            citationsDiv.innerHTML = "<b>Citations:</b><br>" +
              parsed.citations.map(c => "- " + JSON.stringify(c)).join("<br>");
          }
        } catch (e) {
          botBubble.textContent += "\n[Error parsing citations]";
        }
      } else {
        botBubble.textContent += data;
      }
    }

    chat.scrollTop = chat.scrollHeight;
    window.scrollTo(0, document.body.scrollHeight);
  }

  typingDiv.textContent = "";
}

document.getElementById('chat-form').addEventListener('submit', function(e) {
  e.preventDefault();
  send();
});

document.getElementById('question').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    this.form.requestSubmit();
  }
});
