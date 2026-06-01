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
    $(PYTHON) -m flask --app src.app run --host=0.0.0.0 --port=7860

hypercorn:
    hypercorn --bind 0.0.0.0:7860 src.app.asgi:app

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
