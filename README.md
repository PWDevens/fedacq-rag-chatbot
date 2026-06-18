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
- Deployable locally or via Docker  

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
- Custom ONNX Runtime GenAI LLM (Phi‑4‑mini‑instruct‑onnx)
- ChromaDB for retrieval  
- BGE-small embeddings
- Query engine configured with top‑k similarity search  

### Why ONNX Runtime GenAI?
- Fast inference on CPU (no GPU required)
- No PyTorch dependency
- No external API calls (fully local)
- Smaller memory footprint
- Production-ready kernels optimized by Microsoft
- Works well with smaller models like Phi-4-mini-instruct-onnx

### Application Layer

- Flask application (`src.app`)  
- Served as an ASGI app via Hypercorn  
- `/chat_stream` endpoint with token streaming (Server‑Sent Events)  
- Lightweight HTML/JS/CSS UI served from `src/app/static/`  

### Deployment

- Local Python environment  
- Docker container (Hypercorn ASGI server)  
- GitHub Actions CI pipeline (optional)  

---

## Architecture Overview

```text
User → Flask (ASGI via Hypercorn) → Query Engine → LlamaIndex → ChromaDB → FAR/DFARS DITA Source
```

Pipeline:

1. Clone FAR/DFARS repos  
2. Parse `.dita` XML  
3. Chunk + embed  
4. Store in ChromaDB  
5. Serve via Flask API (ASGI)  
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
│   │       └── styles.css
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
git clone https://github.com/PWDevens/fedacq-rag-chatbot.git
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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install project + dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

```bash
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx \
  --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
  --local-dir .
```
This creates cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/ with the ONNX model files.

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

First, edit .gitignore and comment out 'data/chroma/chroma.sqlite3'; then, 

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
git add data/chroma 
git commit -m "Rebuild RAG index"
git push
```

---

## Running the Application

### Flask Development Server (local)

```bash
python -m flask --app src.app run --host=0.0.0.0 --port=7860
```

### Hypercorn (ASGI, local production-like)
```bash
hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
```

### Docker

```bash
docker build -t fedacq-rag-chatbot .
docker run -d -p 7860:7860 --name ragbot fedacq-rag-chatbot
```
The container uses Hypercorn to serve the ASGI‑wrapped Flask app.

### ChromaDB HTTP Server (Production-Ready)

To run the application with ChromaDB as a separate HTTP service:

**Terminal 1: Start ChromaDB server**
```bash
chroma run --host 127.0.0.1 --port 8000
```

**Terminal 2: Run the Flask application**
```bash
export CHROMA_HOST=127.0.0.1
export CHROMA_PORT=8000
python -m flask --app src.app run --host=0.0.0.0 --port=7860
```

Or with Hypercorn:
```bash
export CHROMA_HOST=127.0.0.1
export CHROMA_PORT=8000
hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
```

**Benefits:**
- Separates vector database from application server
- Supports horizontal scaling
- Enables containerized deployments
- Production-ready architecture

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
curl -X POST http://localhost:7860/chat_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FAR 15.404 say about price analysis?"}'
```

Expected response:

- Summary of the regulation  
- Citations  
- Retrieved sections  
- LLM‑generated explanation  

