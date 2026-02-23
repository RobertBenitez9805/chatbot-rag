# Project Overview

## Summary

This project implements a RAG (Retrieval-Augmented Generation) chatbot assistant for Promtior, an AI consulting company. The chatbot answers questions about Promtior's services, founding, and company information by retrieving relevant content from the company's website and using it as context for an LLM to generate accurate responses.

## Approach

The challenge was to build a chatbot that provides factual answers about Promtior without hallucinating. The RAG architecture solves this by grounding the LLM's responses in actual content scraped from Promtior's website.

### Implementation Logic

1. **Data Ingestion**: The system scrapes Promtior's website using LangChain's `WebBaseLoader`, splits the content into chunks, generates vector embeddings using OpenAI's `text-embedding-3-small` model, and stores them in a FAISS vector index.

2. **Query Processing**: When a user asks a question, the system converts it to a vector embedding, performs a similarity search against the FAISS index to find the most relevant chunks, and passes them as context to the LLM.

3. **Response Generation**: The LLM (GPT-3.5-turbo) generates a response based solely on the retrieved context, following a prompt template that prevents hallucination.

4. **API Serving**: The RAG chain is served as a REST API using LangServe, which provides `/invoke`, `/stream`, and `/playground` endpoints automatically.

### Main Challenges

- **Web Content Quality**: Raw HTML from websites contains navigation menus, footers, and other noise. The `RecursiveCharacterTextSplitter` helps by breaking content into meaningful chunks, and the retriever's similarity search filters out irrelevant sections.

- **Chunk Size Tuning**: A chunk size of 500 characters with 50 character overlap was chosen to balance between having enough context per chunk for meaningful retrieval while keeping chunks small enough for precise matching.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Runtime |
| LangChain | LLM orchestration framework |
| LangServe | Serve chains as REST API |
| FastAPI | Web framework (base for LangServe) |
| FAISS | Vector store for similarity search |
| OpenAI API | LLM (gpt-3.5-turbo) and embeddings (text-embedding-3-small) |
| Docker | Containerization |
| AWS App Runner | Cloud deployment |

## How to Run Locally

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone https://github.com/RobertBenitez9805/chatbot-rag.git
cd chatbot-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env
# Edit .env and add your OPENAI_API_KEY

# Run the server (auto-ingests on first run)
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
docker compose up --build
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI |
| `/chat/invoke` | POST | Send a question, get a response |
| `/chat/stream` | POST | Stream a response via SSE |
| `/chat/playground` | GET | LangServe interactive playground |

### API Usage

```bash
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "What services does Promtior offer?"}'
```
