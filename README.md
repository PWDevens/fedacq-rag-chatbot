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
<summary><strong>📁 Project Structure (click to expand)</strong></summary>

