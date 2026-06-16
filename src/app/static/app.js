/**
 * FAR/DFARS Chatbot Frontend
 * Minimal, accurate UI for legal Q&A
 */

// State
let isLoading = false;
let currentBubble = null;
let currentCitations = [];

// Configure marked for safe markdown rendering
marked.setOptions({
  breaks: true,
  gfm: true,
  pedantic: false,
});

/**
 * Render markdown to HTML with basic safety
 */
function renderMarkdown(text) {
  let html = marked.parse(text);
  // Remove outer <p> tags if that's all there is
  if (html.startsWith("<p>") && html.endsWith("</p>\n")) {
    html = html.slice(3, -5);
  }
  return html;
}

/**
 * Create and append a user message
 */
function appendUserMessage(text) {
  const chat = document.getElementById("chat-container");
  const msg = document.createElement("div");
  msg.className = "message user";

  const bubble = document.createElement("div");
  bubble.className = "bubble user-bubble";
  bubble.textContent = text;

  msg.appendChild(bubble);
  chat.appendChild(msg);
  scrollToBottom();
}

/**
 * Create a bot message container
 */
function createBotMessage() {
  const chat = document.getElementById("chat-container");
  const msg = document.createElement("div");
  msg.className = "message bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble bot-bubble";
  bubble.innerHTML = "";

  msg.appendChild(bubble);
  chat.appendChild(msg);

  scrollToBottom();
  return bubble;
}

/**
 * Create loading indicator
 */
function createLoadingMessage() {
  const chat = document.getElementById("chat-container");
  const msg = document.createElement("div");
  msg.className = "message bot loading";

  const bubble = document.createElement("div");
  bubble.className = "bubble bot-bubble";
  bubble.innerHTML = `
    <div class="typing-dots">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;

  msg.appendChild(bubble);
  chat.appendChild(msg);

  scrollToBottom();
  return msg;
}

/**
 * Display citations
 */
function displayCitations(citations) {
  const panel = document.getElementById("citations-panel");
  const container = document.getElementById("citations");

  if (!citations || citations.length === 0) {
    panel.classList.remove("visible");
    return;
  }

  let html = '<div class="citations-header">Source Citations</div>';

  citations.forEach((cite, idx) => {
    const reg = cite.regulation || "FAR";
    const part = cite.part || "";
    const section = cite.section || "";
    const path = cite.source_path || "";

    const label = [reg, part, section].filter(Boolean).join(" ");
    const display = label || "Source";

    html += `
      <div class="citation-item">
        <div class="citation-label">${escapeHtml(display)}</div>
        <div class="citation-path">${escapeHtml(path)}</div>
      </div>
    `;
  });

  container.innerHTML = html;
  panel.classList.add("visible");
}

/**
 * HTML escape for safety
 */
function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (c) => map[c]);
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
  const chat = document.getElementById("chat-container");
  setTimeout(() => {
    chat.scrollTop = chat.scrollHeight;
  }, 0);
}

/**
 * Update bot message content (with incremental markdown rendering)
 */
function updateBotMessage(bubble, text) {
  if (!bubble.dataset.streaming) {
    bubble.innerHTML = "";
    bubble.dataset.streaming = "true";
  }

  // Render markdown incrementally for better visual experience
  try {
    const html = renderMarkdown(text);
    bubble.innerHTML = html;
  } catch (e) {
    // If markdown fails, fall back to escaped text
    bubble.textContent = text;
  }
}

/**
 * Finalize bot message (ensure proper markdown rendering)
 */
function finalizeBotMessage(bubble) {
  if (!bubble.textContent && !bubble.innerHTML) return;

  try {
    // Get the text content and re-render to ensure final formatting is correct
    const text = bubble.textContent;
    const html = renderMarkdown(text);
    bubble.innerHTML = html;
  } catch (e) {
    console.warn("Markdown render error:", e);
    // Keep as text if markdown fails
  }

  delete bubble.dataset.streaming;
  scrollToBottom();
}

/**
 * Handle sending a message
 */
async function send() {
  const textarea = document.getElementById("question");
  const q = textarea.value.trim();

  if (!q || isLoading) return;

  isLoading = true;

  // UI setup
  const form = document.getElementById("chat-form");
  const submitBtn = form.querySelector("button[type=submit]");
  submitBtn.disabled = true;

  // Clear previous citations
  currentCitations = [];
  document.getElementById("citations-panel").classList.remove("visible");

  // Add user message
  appendUserMessage(q);
  textarea.value = "";

  // Create loading indicator
  let loadingMsg = createLoadingMessage();
  let botBubble = null;

  try {
    const response = await fetch("/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });

    if (!response.ok) {
      loadingMsg.remove();
      botBubble = createBotMessage();
      botBubble.innerHTML = `<strong>Error:</strong> HTTP ${response.status}`;
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

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

        // Check for JSON (citations or structured data)
        if (data.startsWith("{")) {
          try {
            const parsed = JSON.parse(data);
            if (parsed.citations) {
              currentCitations = parsed.citations;
              displayCitations(parsed.citations);
            }
          } catch (e) {
            console.warn("Citation parse error:", e);
            // Treat as regular text if JSON parsing fails
            if (!botBubble) {
              loadingMsg.remove();
              botBubble = createBotMessage();
            }
            fullText += data;
            updateBotMessage(botBubble, fullText);
          }
          continue;
        }

        // Regular text token
        if (!botBubble) {
          loadingMsg.remove();
          botBubble = createBotMessage();
        }

        fullText += data;
        updateBotMessage(botBubble, fullText);
      }
    }

    // Finalize
    if (!botBubble) {
      loadingMsg.remove();
      botBubble = createBotMessage();
      botBubble.textContent = "(No response)";
    } else {
      finalizeBotMessage(botBubble);
    }

  } catch (err) {
    console.error("Chat error:", err);
    if (loadingMsg && loadingMsg.parentNode) {
      loadingMsg.remove();
    }
    if (!botBubble) {
      botBubble = createBotMessage();
    }
    if (err.name === "TypeError") {
      botBubble.innerHTML = "<strong>Connection Error:</strong> Unable to reach the server. Is it running?";
    } else if (err.message.includes("HTTP")) {
      botBubble.innerHTML = `<strong>Server Error:</strong> The server returned an error. Please try again.`;
    } else {
      botBubble.innerHTML = `<strong>Error:</strong> ${escapeHtml(err.message)}`;
    }
  } finally {
    isLoading = false;
    submitBtn.disabled = false;
    textarea.focus();
  }
}

/**
 * Initialize event listeners
 */
function init() {
  const form = document.getElementById("chat-form");
  const textarea = document.getElementById("question");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    send();
  });

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // Auto-resize textarea
  textarea.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  textarea.focus();
}

// Start on load
document.addEventListener("DOMContentLoaded", init);
