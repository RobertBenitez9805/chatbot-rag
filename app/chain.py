from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH

CONDENSE_TEMPLATE = """Given the following chat history and a follow-up question,
rephrase the follow-up question to be a standalone question about Promtior.
If the question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

PROMPT_TEMPLATE = """You are a helpful assistant for Promtior, an AI consulting company.
Answer the question based only on the following context. If you cannot find
the answer in the context, say "I don't have that information."

Context:
{context}

Question: {standalone_question}

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
    """Create the RAG chain with conversational context using LCEL."""
    retriever = _get_retriever()
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )

    condense_prompt = ChatPromptTemplate.from_template(CONDENSE_TEMPLATE)
    answer_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Condense chat_history + question into a standalone question
    condense_chain = condense_prompt | llm | StrOutputParser()

    # Use the standalone question to retrieve context and generate answer
    def build_rag_input(standalone_question):
        docs = retriever.invoke(standalone_question)
        return {
            "context": _format_docs(docs),
            "standalone_question": standalone_question,
        }

    chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", ""),
        }
        | condense_chain
        | RunnableLambda(build_rag_input)
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    return chain
