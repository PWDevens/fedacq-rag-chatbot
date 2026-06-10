## src/app/api.py

from flask import Blueprint, request, Response
import json
import traceback

api_bp = Blueprint(
    "api",
    __name__,
    static_folder="static",
    static_url_path="/static",
)

_qe = None


def get_qe():
    global _qe
    if _qe is None:
        from rag.retrieval.query_engine import load_query_engine
        _qe = load_query_engine()
    return _qe


def err_gen(msg: str):
    yield f"data: {msg}\n\n"


@api_bp.route("/")
def home():
    return api_bp.send_static_file("index.html")


@api_bp.route("/chat_stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json(silent=True) or {}
        q = (data.get("question") or "").strip()
        print(f"[chat_stream] Received question: {q}")

        if not q:
            return Response(err_gen("Please enter a question."),
                            mimetype="text/event-stream")

        # Load query engine
        try:
            print("[chat_stream] Loading query engine...")
            qe = get_qe()
            print("[chat_stream] Query engine loaded successfully")
        except Exception as e:
            traceback.print_exc()
            return Response(
                err_gen(f"Error loading query engine: {str(e)}"),
                mimetype="text/event-stream",
                status=500,
            )

        system_prompt = (
            "You are a U.S. federal acquisition compliance assistant specializing in FAR and DFARS.\n"
            "- Prefer FAR when possible; use DFARS only as a supplement.\n"
            "- Answer concisely in no more than 3 short paragraphs.\n"
            "- No pleasantries or chit-chat.\n"
            "- If unsure, say: 'I am not certain.'\n"
        )

        full_prompt = f"{system_prompt}\nUser question: {q}"

        def generate():
            try:
                print("[generate] Starting streaming query...")
                response = qe.query(full_prompt)

                # Case 1: LlamaIndex streaming response
                if hasattr(response, "response_gen"):
                    for token in response.response_gen:
                        if token:
                            t = str(token)
                            print(f"[generate] Token: {t}")
                            yield f"data: {t}\n\n"

                # Case 2: ONNX LLM returns a plain string
                else:
                    t = str(response)
                    print(f"[generate] Full response: {t}")
                    yield f"data: {t}\n\n"

                # Citations
                cites = []
                for i, sn in enumerate(getattr(response, "source_nodes", []), start=1):
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
                traceback.print_exc()
                yield f"data: Error during generation: {str(e)}\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        return Response(
            err_gen(f"Server error: {str(e)}"),
            mimetype="text/event-stream",
            status=500,
        )


@api_bp.route("/health")
def health():
    return {"status": "ok"}
