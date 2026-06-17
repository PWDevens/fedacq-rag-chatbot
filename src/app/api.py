"""
API Blueprint for the FAR/DFARS RAG chatbot.

Routes:
  GET  /           - Serves the frontend HTML.
  POST /chat_stream - Accepts {"question": "<text>"}, returns an SSE stream.
                      Token events: "data: <token text>\\n\\n"
                      Citation event (final): "data: {"citations": [...]}\n\n"
  GET  /health     - Returns {"status": "ok"} for liveness checks.
"""

import json
import logging
import traceback
import threading

from flask import Blueprint, request, Response

logger = logging.getLogger(__name__)

api_bp = Blueprint(
    "api",
    __name__,
    static_folder="static",
    static_url_path="/static",
)

_query_engine_instance = None
_query_engine_lock = threading.Lock()

_MAX_QUESTION_LENGTH = 2000


def get_query_engine():
    """
    Return the shared retriever, initializing it on first call.
    Thread-safe: uses a lock to prevent double-initialization under concurrency.
    """
    global _query_engine_instance
    if _query_engine_instance is None:
        with _query_engine_lock:
            if _query_engine_instance is None:
                from rag.retrieval.query_engine import load_query_engine
                _query_engine_instance = load_query_engine()
    return _query_engine_instance


def yield_error_event(msg: str):
    """Yield a single SSE-formatted error frame."""
    yield f"data: {msg}\n\n"


@api_bp.route("/")
def home():
    return api_bp.send_static_file("index.html")


@api_bp.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    Stream a RAG-generated answer for the submitted question.

    Request body (JSON): {"question": "<user question>"}
    Response: text/event-stream (SSE)
      - Zero or more token events: data: <token>
      - One final citation event:  data: {"citations": [...]}
      - On error:                  data: <user-safe error message>
    """
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        logger.info("[chat_stream] Received question (length=%d)", len(question))

        if not question:
            return Response(
                yield_error_event("Please enter a question."),
                mimetype="text/event-stream",
            )

        if len(question) > _MAX_QUESTION_LENGTH:
            return Response(
                yield_error_event(
                    f"Question is too long. Please limit your question to "
                    f"{_MAX_QUESTION_LENGTH} characters."
                ),
                mimetype="text/event-stream",
            )

        try:
            logger.info("[chat_stream] Loading query engine...")
            retriever = get_query_engine()
            logger.info("[chat_stream] Query engine loaded successfully")
        except Exception as e:
            logger.error("[chat_stream] Failed to load query engine: %s", e, exc_info=True)
            return Response(
                yield_error_event("Could not initialize the retrieval engine. Please try again later."),
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
                logger.info("[generate] Retrieving context for question...")
                source_nodes = retriever.retrieve(question)

                if not source_nodes:
                    yield "data: No relevant regulations found. Try rephrasing your question.\n\n"
                    return

                context_chunks = []
                cites = []
                for i, source_node in enumerate(source_nodes, start=1):
                    context_chunks.append(source_node.text)
                    meta = source_node.metadata or {}
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
                    f"User question: {question}\n\n"
                    f"Answer:"
                )

                logger.info("[generate] Loading ONNX LLM...")
                from rag.llm.models import init_models
                llm, _ = init_models()

                logger.info("[generate] Starting ONNX streaming...")
                for token in llm.stream_complete(full_prompt):
                    if token:
                        yield f"data: {token!s}\n\n"

                yield f"data: {json.dumps({'citations': cites})}\n\n"

            except Exception as e:
                logger.error("[generate] Error during generation: %s", e, exc_info=True)
                yield "data: An error occurred while generating a response. Please try again.\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        logger.error("[chat_stream] Unhandled error: %s", e, exc_info=True)
        return Response(
            yield_error_event("An unexpected server error occurred. Please try again."),
            mimetype="text/event-stream",
            status=500,
        )


@api_bp.route("/health")
def health():
    return {"status": "ok"}
