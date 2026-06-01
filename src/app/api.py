# src/app/api.py

from flask import Blueprint, request, Response, send_from_directory, current_app
import json
import traceback
import os

api_bp = Blueprint("api", __name__, static_folder="static")

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
# Routes
# ---------------------------------------------------------

@api_bp.route("/")
def home():
    """Serve external HTML file."""
    static_dir = os.path.join(current_app.root_path, "static")
    return send_from_directory(static_dir, "index.html")


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
                print("[generate] Starting streaming query...")
                response = qe.query(full_prompt)
                print("[generate] Streaming response received")

                # Stream tokens
                if hasattr(response, "response_gen"):
                    for token in response.response_gen:
                        yield f"data: {token}\n\n"

                # Citations
                source_nodes = getattr(response, "source_nodes", [])
                cites = []
                for i, sn in enumerate(source_nodes, start=1):
                    meta = sn.metadata or {}
                    cites.append({
                        "index": i,
                        "regulation": meta.get("regulation"),
                        "part": meta.get("part"),
                        "section": meta.get("section"),
                        "source_path": meta.get("source_path"),
                    })

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
