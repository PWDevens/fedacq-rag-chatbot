FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# System deps (curl for healthcheck, git for any repo ops)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps using lock file (cacheable layer)
COPY requirements.lock .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Install project
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Bake the embedding model into the image so the first query does NOT download
# it at runtime. The model (~90MB) is cached under HF_HOME during build; at
# runtime the app reads it from cache with no network access.
ENV HF_HOME=/opt/hf \
    EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
RUN python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# After the build-time download, run fully offline at runtime: the embedding
# model is in the image cache and never needs Hugging Face again.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The 135MB Chroma index and 4.6GB Phi-4 ONNX model are NOT baked into the
# image. They are mounted at runtime via docker-compose (see
# docker/docker-compose.yml), keeping the image small and builds fast.
ENV CHROMA_PATH=/code/data/chroma \
    PHI4_MODEL_DIR=/code/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4

EXPOSE 7860

HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1

CMD ["hypercorn", "--bind", "0.0.0.0:7860", "src.app.asgi:app"]
