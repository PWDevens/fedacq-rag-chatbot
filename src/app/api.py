"""
API Blueprint for the FAR/DFARS RAG chatbot.

Routes:
  GET  /           - Serves the frontend HTML.
  POST /chat_stream - Accepts {"question": "<text>"}, returns an SSE stream.
                      Token events: "data: <token text>\\n\\n"
                      Citation event (final): "data: {"citations": [...]}\n\n"
  GET  /health     - Returns {"status": "ok"} for liveness checks.

Retrieval strategy is selected by RagConfig.RAG_MODE (naive | hybrid | graph)
via the engine factory; all modes share this single generation + citation path.
A persistent exact-match answer cache replays repeated questions instantly.
"""

import json
import logging
import threading

from flask import Blueprint, request, Response

from rag import cache

logger = logging.getLogger(__name__)

api_bp = Blueprint(
    "api",
    __name__,
    static_folder="static",
    static_url_path="/static",
)

_engine_instance = None
_engine_lock = threading.Lock()

_MAX_QUESTION_LENGTH = 2000

_NO_CONTEXT = "No relevant context retrieved."


def get_engine():
    """
    Return the shared RAG engine for the active mode, initializing on first call.
    Thread-safe: a lock prevents double-initialization under concurrency.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                from rag.retrieval.factory import get_engine as build_engine
                _engine_instance = build_engine()
    return _engine_instance


def warm_start():
    """
    Eagerly load the LLM/embedder and build the retrieval engine so the first
    request doesn't pay the cold-start cost. Best-effort; callers should not let
    a failure here prevent the server from starting.
    """
    from rag.llm.models import init_models
    init_models()
    get_engine()


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

        # Fast path: exact-match answer cache (survives restarts).
        cached = cache.get(question)
        if cached is not None:
            logger.info("[chat_stream] Cache hit — replaying stored answer")

            def replay():
                answer = cached.get("answer") or ""
                if answer:
                    yield f"data: {answer}\n\n"
                yield f"data: {json.dumps({'citations': cached.get('citations', [])})}\n\n"

            return Response(replay(), mimetype="text/event-stream")

        try:
            logger.info("[chat_stream] Loading RAG engine...")
            engine = get_engine()
            logger.info("[chat_stream] RAG engine ready")
        except Exception as e:
            logger.error("[chat_stream] Failed to load RAG engine: %s", e, exc_info=True)
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

        logger.info("[chat_stream] Loading ONNX LLM...")
        from rag.llm.models import init_models
        llm, _ = init_models()

        def generate():
            try:
                logger.info("[generate] Retrieving context for question...")
                context_str, cites = engine.retrieve_context(question)

                if context_str == _NO_CONTEXT:
                    yield "data: No relevant regulations found. Try rephrasing your question.\n\n"
                    return

                full_prompt = (
                    f"{system_prompt}\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"User question: {question}\n\n"
                    f"Answer:"
                )

                logger.info("[generate] Starting ONNX streaming...")
                pieces = []
                for token in llm.stream_complete(full_prompt):
                    if token:
                        pieces.append(str(token))
                        yield f"data: {token!s}\n\n"

                yield f"data: {json.dumps({'citations': cites})}\n\n"

                # Cache the whitespace-normalized answer for instant replay. The
                # frontend concatenates newline-free token frames, so normalizing
                # whitespace keeps the replay protocol-safe and visually close.
                answer = " ".join("".join(pieces).split())
                if answer:
                    cache.put(question, answer, cites)

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
