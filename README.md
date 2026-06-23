# fedacq‑rag‑chatbot
### Federal Acquisition Regulation Retrieval‑Augmented Generation (RAG) Chatbot

A production‑ready Retrieval‑Augmented Generation (RAG) system that provides fast, accurate, citation‑backed answers to questions about the Federal Acquisition Regulation (FAR) and Defense Federal Acquisition Regulation Supplement (DFARS). Designed for federal contractors, acquisition professionals, and businesses navigating the federal market. Runs entirely locally on a consumer laptop — no GPU, no cloud API, no subscription required.

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
 
 ___
#### Landing Page
![Landing-Page](demo_artifacts/Landing.png)

#### Query Response
![Response-to-Query](demo_artifacts/MAS.png)

#### Query Loading
![Pending-Response-to-Query](demo_artifacts/IDIQ-pre.png)
---

## Technical Approach

### Data Source

- FAR and DFARS pulled from official `.dita` XML repositories  
- Parsed into structured documents  
- Metadata normalized for retrieval  

### Embeddings + Vector Store

- HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)  
- ChromaDB persistent (on‑disk) vector store  
- Chunking via LlamaIndex `SentenceSplitter` (512-token chunks, 64 overlap)  

### Retrieval‑Augmented Generation

- LlamaIndex orchestration  
- Custom ONNX Runtime GenAI LLM (Phi‑4‑mini‑instruct‑onnx)
- ChromaDB for retrieval  
- `BAAI/bge-small-en-v1.5` embeddings (must match the model used to build the index)
- **Selectable retrieval strategy** via `RAG_MODE` (`naive` | `hybrid` | `graph`)
- **Cross-encoder reranker** (`RERANK`, on by default) layered over naive/hybrid
- See [Retrieval Modes](#retrieval-modes-rag_mode) below

All modes share one generation + citation path: each produces `(context,
citations)` and the existing Phi‑4 streaming endpoint does the rest.

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

## Screenshots

<!-- Add a screenshot of the chat UI here, e.g.: -->
<!-- ![FAR/DFARS chatbot UI](docs/screenshot.png) -->

_UI screenshot to be added._

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
│   │       ├── purify.min.js
│   │       └── styles.css
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── config.py
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
│   ├── test_query_engine.py
│   └── test_query_engine_load.py
│
├── docker/
│   ├── docker-compose.yml
│   └── local.env.example
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── CODEOWNERS
├── Dockerfile
├── Makefile
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── requirements.lock
```

---

## Prerequisites

> **Hardware:** this project runs entirely on CPU. No GPU is required.
> It has been validated on a consumer laptop (Intel Core i5, 16 GB RAM, integrated graphics).

| Requirement | Minimum | Notes |
|---|---|---|
| **RAM** | 16 GB | Phi-4 int4 model loads ~4.6 GB into RAM at startup; 16 GB is the practical floor |
| **CPU** | Any modern x86-64 or Apple Silicon | Inference is CPU-bound; expect 10–60 s per response depending on question length |
| **GPU** | None required | ONNX Runtime runs on CPU; integrated graphics are not used |
| **Disk** | ~8 GB free | ~4.6 GB model + ~135 MB index + ~500 MB venv + FAR/DFARS repos |
| **OS** | Windows 10/11, Linux, macOS | Windows users: use the Flask dev server or Docker (see Running the Application) |
| **Python** | 3.10–3.12 | 3.12 recommended; matches the `python:3.12-slim` Docker base image |
| **Git LFS** | Required | Pulls the prebuilt ChromaDB index (`git lfs install` before cloning) |
| **Docker** | Optional | Only for the containerized run path; not needed for local Python setup |

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

To rebuild locally (from the repository root):

```bash
python -m scripts.build_index
```

This will:

- Clone FAR + DFARS into `data/regs/`
- Parse `.dita` XML files
- Chunk and embed text
- Write a new ChromaDB index into `data/chroma/`

After rebuilding, commit the updated index via Git LFS (the whole
`data/chroma/` directory, including `chroma.sqlite3`, is LFS‑tracked and is
committed so clones run without rebuilding):

```bash
git add data/chroma
git commit -m "Rebuild RAG index"
git push
```

---

## Running the Application

The app reads the ChromaDB index **directly from disk** (`data/chroma/`) — no
separate database server is required. You need two things present locally
before starting: the prebuilt index (`data/chroma/`, via Git LFS) and the
Phi‑4 ONNX model (`cpu_and_mobile/...`, downloaded in Setup step 4).

> **Performance note:** on first startup, the app maps the Phi-4 model into memory — expect 15–30 seconds before the first response token appears. Subsequent queries in the same session are faster (model stays loaded). This is normal on CPU-only hardware.

### Option A — Docker (recommended, one command)

```bash
cd docker
docker compose up --build
```

Then open <http://localhost:7860>.

The Compose file mounts the model and index from your local checkout into the
container **read‑only**, so the image stays small and rebuilds are fast (the
4.6 GB model and 135 MB index are never copied into the image). Stop with
`docker compose down`.

### Option B — Local Python

After completing Setup & Installation, from the repository root:

```bash
# Flask dev server (simplest; recommended on Windows)
python -m flask --app src.app run --host=0.0.0.0 --port=7860
```

```bash
# Hypercorn (ASGI, production-like) — recommended on Linux/macOS/Docker
hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
```

Then open <http://localhost:7860>.

> **Windows note:** Hypercorn's multiprocessing worker model can fail to start
> on native Windows. Use the Flask dev server (above) or Docker on Windows;
> use Hypercorn on Linux/macOS or inside the container.

### Configuration (environment variables)

All have working defaults for local use; override only if needed.

| Variable | Default | Purpose |
|---|---|---|
| `CHROMA_PATH` | `./data/chroma` | On-disk ChromaDB index directory |
| `PHI4_MODEL_DIR` | `./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4` | Phi‑4 ONNX model directory |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model (must match the index) |
| `FLASK_ENV` | `local` | Set to `production` to require a real `SECRET_KEY` |
| `SECRET_KEY` | dev fallback | Required only when `FLASK_ENV=production` |
| `CHROMA_HOST` | _(unset)_ | If set, connect to a remote ChromaDB HTTP server instead of on-disk |
| `CHROMA_PORT` | `8000` | Port for the remote ChromaDB server (only used when `CHROMA_HOST` is set) |
| `RAG_MODE` | `naive` | Retrieval strategy: `naive`, `hybrid`, or `graph` |
| `RERANK` | `true` | Cross-encoder reranker over naive/hybrid results |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (small, CPU) |
| `RETRIEVAL_TOP_K` | `20` | Candidate pool retrieved before reranking |
| `RERANK_TOP_N` | `5` | Final chunks fed to the LLM |
| `MAX_NEW_TOKENS` | `128` | Generation length cap (latency lever) |
| `ANSWER_CACHE` | `true` | Persistent exact-match answer cache (`data/cache.db`) |
| `LIGHTRAG_WORKING_DIR` | `./data/lightrag` | GraphRAG artifacts directory |
| `GRAPH_BUILD_LLM` | _(unset)_ | Optional API model for offline graph extraction (else Phi‑4) |

### Retrieval Modes (`RAG_MODE`)

Set `RAG_MODE` in your environment (or `docker/local.env`) and restart.

| Mode | What it does | Cost on CPU |
|---|---|---|
| `naive` | Dense vector search over Chroma (the baseline). | Cheapest |
| `hybrid` | Dense **+** BM25 (`rank-bm25`) fused with Reciprocal Rank Fusion. Catches exact-term matches (clause numbers, defined terms) that pure embeddings miss. | + a few ms (BM25 is pure CPU) |
| `graph` | **GraphRAG via LightRAG** — retrieves over a prebuilt knowledge graph of entities/relations (dual-level local+global). Requires an offline build first. | Highest (one extra LLM call for keyword extraction per query) |

The reranker (`RERANK=true`) applies to `naive` and `hybrid`: it pulls a larger
candidate pool (`RETRIEVAL_TOP_K`) and re-scores it with a cross-encoder down to
`RERANK_TOP_N`, improving both the injected context and the citations.

**Graph mode setup** (one-time, offline):

```bash
# Builds a knowledge graph from a subset of the committed index using the local
# Phi-4 model. Runs LLM extraction on CPU — start small.
GRAPH_BUILD_MAX_DOCS=40 python -m scripts.build_graph
# Optional: better extraction quality with an API model (needs OPENAI_API_KEY):
# GRAPH_BUILD_LLM=gpt-4o-mini python -m scripts.build_graph
RAG_MODE=graph make run
```

Graph mode answers from graph-assembled context; it does not map back to
discrete FAR sections, so its citation panel is intentionally sparse.

### Latency notes

The chatbot is CPU-bound; generation dominates the 10–60 s/response range.

- **Warm start:** models + retrieval engine are loaded at server startup (in
  `asgi.py`), so the first request no longer pays the 15–30 s cold start. Set
  `RAG_NO_WARM=1` to opt out.
- **Answer cache:** `ANSWER_CACHE=true` replays repeated questions instantly
  from `data/cache.db` (survives restarts).
- **Tunable knobs:** lower `MAX_NEW_TOKENS` and `RERANK_TOP_N` to cut generation
  time; the reranker shrinks the prompt to the most relevant chunks, which also
  helps prefill.
- **Further wins (not enabled by default):** pin ONNX intra-op threads to your
  physical core count; batch SSE token flushes; add an in-process LRU cache for
  repeat-query embeddings.

> **Note on the committed index:** the app reads from a gitignored runtime copy
> (`data/chroma_runtime/`) so query-time WAL checkpoints never dirty the
> LFS-tracked `data/chroma/`. After rebuilding the index, delete
> `data/chroma_runtime/` so the fresh copy is picked up.

### Optional — Remote ChromaDB server (advanced / scaling)

The default embedded mode is recommended. To instead run ChromaDB as a
separate HTTP service (e.g. for horizontal scaling), set `CHROMA_HOST` — this
switches both the builder and the query engine to the HTTP client. You must
(re)build the index *through the server* so its storage holds the vectors;
the on-disk `data/chroma/` index is not used in this mode.

```bash
# Terminal 1
chroma run --host 127.0.0.1 --port 8000

# Terminal 2
export CHROMA_HOST=127.0.0.1 CHROMA_PORT=8000
python -m scripts.build_index          # ingest into the running server
hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
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
curl -X POST http://localhost:7860/chat_stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does FAR 15.404 say about price analysis?"}'
```

Expected response:

- Summary of the regulation  
- Citations  
- Retrieved sections  
- LLM‑generated explanation  

---

## API / SSE Response Format

`POST /chat_stream` — accepts JSON, returns a Server-Sent Events stream.

**Request:**

```json
{"question": "What does FAR 15.404 say about price analysis?"}
```

**Response stream** (`text/event-stream`):

- Zero or more token events: `data: <token text>\n\n`
- One final citation event: `data: {"citations": [{"index": 1, "regulation": "FAR", "part": "15", "section": "15.404", "source_path": "..."}]}\n\n`
- On error: `data: <user-safe error message>\n\n`

---

## Data & Model Attribution

- **FAR / DFARS text** is a work of the U.S. Government and is in the public
  domain. It is sourced from the official `.dita` XML repositories.
- **Phi-4-mini-instruct-onnx** is provided by Microsoft under its own model
  license. It is **not** redistributed in this repository; download it directly
  from Hugging Face (see Setup) and comply with Microsoft's license terms.
- **BAAI/bge-small-en-v1.5** (embeddings) is downloaded from Hugging Face at
  build time and is governed by its own license.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
