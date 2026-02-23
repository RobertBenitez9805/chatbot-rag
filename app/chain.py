from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH

PROMPT_TEMPLATE = """You are a helpful assistant for Promtior, an AI consulting company.
Answer the question based only on the following context. If you cannot find
the answer in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""


def _get_retriever():
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_chain():
    """Create the RAG chain using LCEL."""
    retriever = _get_retriever()
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
