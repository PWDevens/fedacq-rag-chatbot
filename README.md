# Federal Acquisition Regulation Retrieval‑Augmented Generation (RAG) Chatbot
A production‑grade Retrieval‑Augmented Generation (RAG) system designed to provide fast, accurate answers to questions about the Federal Acquisition Regulation (FAR) and Defense Federal Acquisition Regulation Supplement (DFARS).
Built for federal contractors, acquisition professionals, and businesses navigating the federal market.
## Background
Federal contracting regulations are complex, distributed across thousands of pages of FAR/DFARS text, and updated frequently.
Professionals need fast, reliable, citation‑backed answers to support:
- Capture strategy
- Proposal development
- Compliance reviews
- Contract administration
- Market entry decisions
- This project automates that research using a modern RAG pipeline.`
### User Story
As a federal contractor, federal employee, or business entering the federal market, I need quick, accurate answers to questions about the regulatory landscape so that I can make informed business strategy decisions.
### Acceptance Criteria
- A chatbot interface that:
 - Accepts natural‑language questions
 - Retrieves relevant FAR/DFARS sections
 - Generates accurate, citation‑backed responses
 - Uses up‑to‑date regulatory text
- End‑to‑end reproducible pipeline
- Deployable locally, via Docker, or via CI/CD
## Technical Appproach
### Data Source
- FAR and DFARS pulled from official .dita XML repositories
- Parsed into structured documents
- Metadata normalized for retrieval
### Embeddings + Vector Store
- HuggingFace Embeddings (BGE‑small or similar)
- ChromaDB persistent vector store
- Chunking via LlamaIndex SentenceSplitter
### Retrieval‑Augmented Generation
- LlamaIndex orchestration
- HuggingFace LLM (Qwen) for generation
- ChromaVectorStore for retrieval
- Query engine configured with top‑k similarity search
### Application Layer
- Flask API
- /chat and /chat_stream endpoints
- Token streaming for responsive UI
### Deployment
- Local Python environment
- Docker container
- GitHub Actions CI pipeline
- Future: Cloud Run / Azure Web App / ECS
## Architecture Overview
User → Flask API → Query Engine → LlamaIndex → ChromaDB → FAR/DFARS DITA Source
### Pipeline:
- Clone FAR/DFARS repos
- Parse .dita XML
- Chunk + embed
- Store in ChromaDB
- Serve via Flask API
- LLM generates answers with citations
## Repository Structure
fedacq-rag-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   └── wsgi.py
│
├── data/
│   ├── chroma/          # Persistent ChromaDB index
│   └── regs/            # FAR/DFARS cloned repositories
│
├── docker/
│   ├── docker-compose.yml
│   └── local.env
│
├── rag/
│   ├── __init__.py
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── builder.py      # Builds the RAG index
│   │   └── loader.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── metadata.py
│   │   ├── parser_dita.py  # DITA parsing logic
│   │   └── query_engine.py # Loads Chroma + LLM
│   │
│   └── retrieval/
│       ├── __init__.py
│       └── models.py
│
├── scripts/
│   └── build_index.py      # CLI entrypoint for index building
│
├── tests/
│   ├── test_indexing.py
│   ├── test_llm.py
│   ├── test_metadata.py
│   ├── test_parser.py
│   └── test_query_engine.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── README.md
└── requirements.txt
## Setup & Installation
1. Clone the repository
git clone https://github.com/your-org/fedacq-rag-chatbot.git
cd fedacq-rag-chatbot
2. Create & activate environment
python -m venv .venv
source .venv/bin/activate
3. Install package + dependencies
pip install -e .
pip install -r requirements.txt
4. Build the RAG Index
python -m scripts.build_index
5. Run the Application
- Flask (Windows‑friendly)
export FLASK_APP=app
flask run --host=0.0.0.0 --port=8080
- Waitress (production on Windows)
waitress-serve --host=0.0.0.0 --port=8080 app.wsgi:app
- Docker
docker compose up --build
6. Testing
pytest -q
7. GitHub Actions CI Pipeline
Actions → Local RAG Build & App Launch → Run workflow
