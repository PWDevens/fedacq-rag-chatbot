from flask import Blueprint, request, Response
import json
import traceback

api_bp = Blueprint(
    "api",
    __name__,
    static_folder="static",
    static_url_path="/static",
)

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
# Error generator (fixed)
# ---------------------------------------------------------
def err_gen(msg):
    yield f"data: {msg}\n\n"


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
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
            return Response(err_gen("Please enter a question."), mimetype="text/event-stream")

        # Load query engine
        try:
            print("[chat_stream] Loading query engine...")
            qe = get_qe()
            print("[chat_stream] Query engine loaded successfully")
        except Exception as e:
            print(f"[chat_stream] ERROR initializing query engine: {e}")
            traceback.print_exc()
            return Response(err_gen(f"Error loading query engine: {str(e)}"),
                            mimetype="text/event-stream",
                            status=500)

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

                # Async generator
                if hasattr(response, "__aiter__"):
                    import asyncio

                    async def stream_async():
                        async for chunk in response:
                            print(f"[generate] Chunk: {chunk}")
                            yield f"data: {chunk}\n\n"

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    async_gen = stream_async()

                    async def iterate():
                        async for item in async_gen:
                            yield item

                    for chunk in loop.run_until_complete(iterate().__anext__()):
                        yield chunk

                # Sync generator
                elif hasattr(response, "response_gen"):
                    for token in response.response_gen:
                        print(f"[generate] Token: {token}")
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
        return Response(err_gen(f"Server error: {str(e)}"),
                        mimetype="text/event-stream",
                        status=500)


@api_bp.route("/health")
def health():
    return {"status": "ok"}
