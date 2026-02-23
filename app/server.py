import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes

from app.chain import create_chain
from app.config import VECTORSTORE_PATH
from app.ingest import ingest

app = FastAPI(
    title="Promtior RAG Chatbot",
    version="1.0",
    description="A RAG chatbot that answers questions about Promtior",
)

# Auto-ingest if vectorstore doesn't exist
if not os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
    ingest()

chain = create_chain()

add_routes(app, chain, path="/chat")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
