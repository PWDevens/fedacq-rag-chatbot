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
            return Response(
                err_gen("Please enter a question."),
                mimetype="text/event-stream",
            )

        try:
            print("[chat_stream] Loading query engine...")
            retriever = get_qe()
            print("[chat_stream] Query engine loaded successfully")
        except Exception as e:
            traceback.print_exc()
            return Response(
                err_gen(f"Error loading query engine: {str(e)}"),
                mimetype="text/event-stream",
                status=500,
            )

        system_prompt = (
            "You are a U.S. federal acquisition compliance assistant specialized in FAR and DFARS.\n"
            "- Cite all regulations with their section numbers, e.g., [FAR 15.404], [DFARS 215.404].\n"
            "- Prefer FAR regulations; use DFARS only where FAR does not apply or as a supplement.\n"
            "- Keep responses to 3 short paragraphs or fewer.\n"
            "- Provide direct, factual answers without unnecessary elaboration.\n"
            "- If uncertain, state: 'I cannot provide a definitive answer based on the available regulations.'\n"
        )

        def generate():
            try:
                print("[generate] Retrieving context for question...")
                source_nodes = retriever.retrieve(q)

                if not source_nodes:
                    yield f"data: No relevant regulations found. Try rephrasing your question.\n\n"
                    return

                context_chunks = []
                cites = []
                for i, sn in enumerate(source_nodes, start=1):
                    context_chunks.append(sn.text)
                    meta = sn.metadata or {}
                    cites.append(
                        {
                            "index": i,
                            "regulation": meta.get("regulation"),
                            "part": meta.get("part"),
                            "section": meta.get("section"),
                            "source_path": meta.get("source_path"),
                        }
                    )

                context_str = (
                    "\n\n".join(context_chunks)
                    if context_chunks
                    else "No relevant context retrieved."
                )

                full_prompt = (
                    f"{system_prompt}\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"User question: {q}\n\n"
                    f"Answer:"
                )

                print("[generate] Loading ONNX LLM...")
                from rag.llm.models import init_models
                llm, _ = init_models()

                print("[generate] Starting ONNX streaming...")
                for token in llm.stream_complete(full_prompt):
                    if token:
                        t = str(token)
                        print(f"[generate] Token: {t}")
                        yield f"data: {t}\n\n"

                # Send citations as JSON at the end
                yield f"data: {json.dumps({'citations': cites})}\n\n"

            except Exception as e:
                traceback.print_exc()
                error_msg = str(e)
                # Provide user-friendly error message
                if "ChromaDB" in error_msg or "vector" in error_msg.lower():
                    yield f"data: Error: Could not retrieve relevant regulations. Please try a different question.\n\n"
                else:
                    yield f"data: Error during generation: {error_msg}\n\n"

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
