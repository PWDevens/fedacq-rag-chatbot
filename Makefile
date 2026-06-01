# -------------------------
# Environment
# -------------------------
VENV=.venv
PYTHON=$(VENV)/bin/python

# -------------------------
# Setup
# -------------------------
setup:
    python -m venv $(VENV)
    $(PYTHON) -m pip install --upgrade pip
    $(PYTHON) -m pip install -r requirements.txt
    $(PYTHON) -m pip install -e .

# -------------------------
# Unified Run (ASGI preferred)
# -------------------------
run:
    @echo "Starting RAG chatbot..."
    @if command -v hypercorn >/dev/null 2>&1; then \
        echo "→ Running with Hypercorn (ASGI)"; \
        hypercorn --bind 0.0.0.0:7860 src.app.asgi:app; \
    else \
        echo "→ Hypercorn not found, falling back to Flask dev server"; \
        python -m flask --app src.app run --host=0.0.0.0 --port=7860; \
    fi

# -------------------------
# Dev Mode (Flask only)
# -------------------------
dev:
    @echo "Running Flask development server..."
    python -m flask --app src.app run --host=0.0.0.0 --port=7860 --debug

# -------------------------
# Build Index
# -------------------------
index:
    $(PYTHON) -m scripts.build_index

# -------------------------
# Tests
# -------------------------
test:
    pytest -q

# -------------------------
# Docker
# -------------------------
docker-build:
    docker build -t fedacq-rag-chatbot .

docker-run:
    docker run -p 7860:7860 --name ragbot fedacq-rag-chatbot

# -------------------------
# Clean (pyc, cache, build artifacts)
# -------------------------
clean:
    @echo "Cleaning project..."
    find . -name "__pycache__" -exec rm -rf {} +
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete
    rm -rf build dist *.egg-info
    @echo "Done."
