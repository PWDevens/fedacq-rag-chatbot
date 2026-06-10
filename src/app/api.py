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
            "You are a U.S. federal acquisition compliance assistant specializing in FAR and DFARS.\n"
            "- Prefer FAR when possible; use DFARS only as a supplement.\n"
            "- Answer concisely in no more than 3 short paragraphs.\n"
            "- No pleasantries or chit-chat.\n"
            "- If unsure, say: 'I am not certain.'\n"
        )

        def generate():
            try:
                print("[generate] Retrieving context...")
                source_nodes = retriever.retrieve(q)

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

                # send citations as JSON at the end
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
