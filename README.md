# fedacq-rag-chatbot
Federal Acquisition Regulation Retrieval-Augmented Generation (RAG) Chatbot

## Background
### User Story
As a [federal contractor/federal employee/business seeking GTM in federal market], I need quick information based on a number of questions I have about the regulatory landscape, so that I can make informed business strategy decisions.
### Acceptance Criteria
Chatbot interface that accurately answers questions about federal contracting laws, policies, and regulations
### Technical Appproach
#### Vector Embeddings
API > .dita files > ChromaDB 
#### Retrieval Augmented Generation
LlamaIndex > HuggingFaceLLM (Qwen) > HuggingFaceEmbedding > ChromaVectorStore 
#### Deployment
Flask > TBD

## Repo Root
<details>
<summary><strong>Project Structure (click to expand)</strong></summary>
fedacq-rag-chatbot/
│
├── app/
│   ├── init.py
│   ├── api.py
│   ├── config.py
│   └── wsgi.py
│
├── data/
│
├── docker/
│   ├── docker-compose.yml
│   └── local.env
│
├── rag/
│   ├── indexing/
│   │   ├── init.py
│   │   ├── builder.py
│   │   └── loader.py
│   │
│   ├── llm/
│   │   ├── init.py
│   │   ├── metadata.py
│   │   ├── parser_dita.py
│   │   └── query_engine.py
│   │
│   └── retrieval/
│       ├── init.py
│       └── models.py
│
├── scripts/
│   └── build_index.py
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

</details>

