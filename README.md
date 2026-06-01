# fedacq‑rag‑chatbot
### Federal Acquisition Regulation Retrieval‑Augmented Generation (RAG) Chatbot

A production‑ready Retrieval‑Augmented Generation (RAG) system that provides fast, accurate, citation‑backed answers to questions about the Federal Acquisition Regulation (FAR) and Defense Federal Acquisition Regulation Supplement (DFARS). Designed for federal contractors, acquisition professionals, and businesses navigating the federal market.

---

## Background

Federal contracting regulations are complex, distributed across thousands of pages of FAR/DFARS text, and updated frequently. Professionals need fast, reliable, context‑aware answers to support:

- Capture strategy  
- Proposal development  
- Compliance reviews  
- Contract administration  
- Market entry decisions  

This project automates that research using a modern RAG pipeline.

---

## User Story

**As a federal contractor, federal employee, or business entering the federal market, I need quick, accurate answers to questions about the regulatory landscape so that I can make informed business strategy decisions.**

---

## Acceptance Criteria

- Natural‑language question interface  
- Retrieval of relevant FAR/DFARS sections  
- Accurate, citation‑backed responses  
- Up‑to‑date regulatory text  
- Reproducible end‑to‑end pipeline  
- Deployable locally, via Docker, or via CI/CD  

---

## Technical Approach

### Data Source
- FAR and DFARS pulled from official `.dita` XML repositories  
- Parsed into structured documents  
- Metadata normalized for retrieval  

### Embeddings + Vector Store
- HuggingFace Embeddings (BGE‑small)  
- ChromaDB persistent vector store  
- Chunking via LlamaIndex `SentenceSplitter`  

### Retrieval‑Augmented Generation
- LlamaIndex orchestration  
- HuggingFace LLM (Qwen) for generation  
- ChromaVectorStore for retrieval  
- Query engine configured with top‑k similarity search  

### Application Layer
- ASGI application (src.app.asgi)
- /chat_stream endpoint with async token streaming
- Static HTML/JS/CSS served from src/app/static/
- Frontend  loads external JS + CSS instead of embedding them in Python

### Deployment
- Local Python environment  
- Docker container  
- GitHub Actions CI pipeline  
- CI job verifies index exists before deployment  

---

## Architecture Overview

```
User → ASGI API → Query Engine → LlamaIndex → ChromaDB → FAR/DFARS DITA Source
```

Pipeline:

1. Clone FAR/DFARS repos  
2. Parse `.dita` XML  
3. Chunk + embed  
4. Store in ChromaDB  
5. Serve via ASGI API  
6. LLM generates answers with citations  

---

## Repository Structure

```
fedacq-rag-chatbot/
│
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── config.py
│   │   ├── asgi.py
│   │   └── static/
│   │       ├── index.html
│   │       ├── app.js
│   │       └── style.css
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   │
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   └── loader.py
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   │
│   │   └── retrieval/
│   │       ├── __init__.py
│   │       ├── metadata.py
│   │       ├── parser_dita.py
│   │       └── query_engine.py
│   │
│   └── scripts/
│       └── build_index.py
│
├── data/
│   ├── chroma/      # Persistent ChromaDB index (Git LFS)
│   └── regs/        # FAR/DFARS cloned repositories (Git LFS)
│
├── tests/
│   ├── test_indexing.py
│   ├── test_llm.py
│   ├── test_metadata.py
│   ├── test_parser.py
│   └── test_query_engine.py
│
├── docker/
│   ├── docker-compose.yml
│   └── local.env
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── Makefile
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── requirements.lock
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/fedacq-rag-chatbot.git
cd fedacq-rag-chatbot
```

### 2. Install Git LFS (required for Chroma index)

```bash
git lfs install
git lfs pull
```

### 3. Create & activate environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install project + dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

### 5. Verify the prebuilt RAG index

Ensure these folders contain data:

```
data/chroma/
data/regs/
```

If empty:

```bash
git lfs pull
```

---

## Building the RAG Index (Local Only)

The RAG index is **not** built in CI due to the size of FAR/DFARS and the cost of embedding.

To rebuild locally:

```bash
python -m scripts.build_index
```

This will:

- Clone FAR + DFARS into `data/regs/`
- Parse `.dita` XML files
- Chunk and embed text
- Write a new ChromaDB index into `data/chroma/`

After rebuilding, commit the updated index using Git LFS:

```bash
git add data/chroma data/regs
git commit -m "Rebuild RAG index"
git push
```

---

## Running the Application

### ASGI (Local Development)

```bash
uvicorn src.app.asgi:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t fedacq-rag-chatbot .
docker run -p 7860:7860 fedacq-rag-chatbot
```

---

## Testing

```bash
pytest -q
```

---

## CI/CD Workflow (GitHub Actions)

The CI pipeline performs:

- LFS checkout (downloads prebuilt index)
- Python environment setup
- Dependency installation
- Test execution
- Docker image build
- Docker artifact upload

The CI pipeline **does not** rebuild the RAG index.  
Index building is performed locally and versioned via Git LFS.

---

## Usage Example

Send a POST request:

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does FAR 15.404 say about price analysis?"}'
```

Expected response:

- Summary of the regulation  
- Citations  
- Retrieved sections  
- LLM‑generated explanation  

