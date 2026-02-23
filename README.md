# Promtior RAG Chatbot

A chatbot assistant built with RAG (Retrieval-Augmented Generation) architecture that answers questions about Promtior using content from the company's website.

## Tech Stack

- **LangChain** + **LangServe** - LLM orchestration and API serving
- **FAISS** - Vector store for semantic search
- **OpenAI** - GPT-3.5-turbo (LLM) + text-embedding-3-small (embeddings)
- **FastAPI** - Web framework
- **Docker** - Containerization

## Quick Start

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env
# Add your OPENAI_API_KEY to .env

# Run (auto-ingests on first start)
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the chat UI or http://localhost:8000/chat/playground for the LangServe playground.

## Docker

```bash
docker compose up --build
```

## Documentation

- [Project Overview](doc/overview.md)
- [Component Diagram](doc/components.md)
