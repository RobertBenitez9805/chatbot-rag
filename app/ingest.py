from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL, PROMTIOR_URLS, VECTORSTORE_PATH


def ingest():
    """Load Promtior web pages, split into chunks, embed, and persist to FAISS."""
    loader = WebBaseLoader(PROMTIOR_URLS)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Ingested {len(chunks)} chunks from {len(documents)} pages into {VECTORSTORE_PATH}/")
    return vectorstore


if __name__ == "__main__":
    ingest()
