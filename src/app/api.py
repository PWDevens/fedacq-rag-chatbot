## app/api.py

from flask import Blueprint, request, Response
import json
import traceback

api_bp = Blueprint("api", __name__)

# ---------------------------------------------------------
# Lazy-load query engine
# ---------------------------------------------------------
_qe = None
def get_qe():
    global _qe
    if _qe is None:
        from rag.retrieval.query_engine import load_query_engine
        _qe = load_query_engine()
    return _qe

# ---------------------------------------------------------
# Embedded HTML/CSS/JS (ChatGPT-style UI)
# ---------------------------------------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>FAR / DFARS RAG Chatbot</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f0f2f5;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    }
    .bubble {
      max-width: 80%;
      padding: 12px 16px;
      margin-bottom: 12px;
      border-radius: 12px;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    .user-bubble {
      background: #d1e7ff;
      margin-left: auto;
      border-bottom-right-radius: 4px;
    }
    .bot-bubble {
      background: #ffffff;
      margin-right: auto;
      border-bottom-left-radius: 4px;
      border: 1px solid #ddd;
    }
    #input-bar {
      display: flex;
      padding: 12px;
      background: #ffffff;
      border-top: 1px solid #ccc;
      position: sticky;
      bottom: 0;
      z-index: 9999;
    }
    #question {
      flex: 1;
      padding: 10px;
      font-size: 16px;
      border: 1px solid #ccc;
      border-radius: 6px;
      resize: none;
      height: 60px;
      background: white;
    }
    #send-btn {
      margin-left: 10px;
      padding: 0 20px;
      font-size: 16px;
      border: none;
      background: #005bbb;
      color: white;
      border-radius: 6px;
      cursor: pointer;
    }
    #send-btn:hover {
      background: #004999;
    }
    #citations {
      padding: 10px 20px;
      font-size: 14px;
      color: #555;
      border-top: 1px solid #ddd;
      background: #fafafa;
    }
    #typing {
      padding: 0 20px 10px 20px;
      font-size: 13px;
      color: #777;
      font-style: italic;
    }
  </style>
</head>

<body>

<div id="chat-container"></div>
<div id="typing"></div>
<div id="citations"></div>

<form id="chat-form">
  <div id="input-bar">
    <textarea id="question" placeholder="Ask a question..."></textarea>
    <button id="send-btn" type="submit">Send</button>
  </div>
</form>

<script>
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
</script>

</body>
</html>
"""

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@api_bp.route("/")
def home():
    return HTML


@api_bp.route("/chat_stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("question") or "").strip()
        print(f"[chat_stream] Received question: {q}")

        if not q:
            def err_gen():
                yield "data: Please enter a question.\n\n"
            return Response(err_gen(), mimetype="text/event-stream")

        # Load query engine
        try:
            print("[chat_stream] Loading query engine...")
            qe = get_qe()
            print("[chat_stream] Query engine loaded successfully")
        except Exception as e:
            print(f"[chat_stream] ERROR loading query engine: {e}")
            traceback.print_exc()
            def err_gen():
                yield f"data: Error loading query engine: {str(e)}\n\n"
            return Response(err_gen(), mimetype="text/event-stream", status=500)

        system_prompt = (
            "You are a compliance assistant specializing in FAR and DFARS. "
            "Answer ONLY using retrieved context. "
            "If unsure, say you are not certain."
        )
        full_prompt = f"{system_prompt}\n\nUser question: {q}"

        def generate():
            try:
                print("[generate] Starting stream...")
                stream = qe.stream_query(full_prompt)
                print("[generate] Stream created successfully")

                # Stream response tokens
                for token in stream.response_gen:
                    print(f"[generate] Token: {token}")
                    yield f"data: {token}\n\n"

                # Get final response for citations
                print("[generate] Getting final response...")
                final_response = stream.get_response()
                print(f"[generate] Final response: {final_response}")
                print(f"[generate] Source nodes: {final_response.source_nodes}")

                # Extract citations
                cites = []
                for i, sn in enumerate(final_response.source_nodes, start=1):
                    meta = sn.metadata or {}
                    cites.append({
                        "index": i,
                        "regulation": meta.get("regulation"),
                        "part": meta.get("part"),
                        "section": meta.get("section"),
                        "source_path": meta.get("source_path"),
                    })

                print(f"[generate] Citations: {cites}")
                yield f"data: {json.dumps({'citations': cites})}\n\n"

            except Exception as e:
                print(f"[generate] ERROR: {e}")
                traceback.print_exc()
                yield f"data: Error during generation: {str(e)}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        print(f"[chat_stream] OUTER ERROR: {e}")
        traceback.print_exc()
        def err_gen():
            yield f"data: Server error: {str(e)}\n\n"
        return Response(err_gen(), mimetype="text/event-stream", status=500)
