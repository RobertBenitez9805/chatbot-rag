import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = "text-embedding-3-small"
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")

PROMTIOR_URLS = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://cie.ort.edu.uy/emprendimientos/promptior"
]
