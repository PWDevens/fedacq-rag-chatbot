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
# Run App
# -------------------------
run:
    $(PYTHON) -m hypercorn src.app.asgi:asgi_app --bind 0.0.0.0:7860

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
    docker run -p 7860:7860 fedacq-rag-chatbot
