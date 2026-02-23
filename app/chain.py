from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH


class ChatInput(BaseModel):
    """Input schema for the chat chain."""
    question: str = Field(description="The user's question")
    chat_history: str = Field(default="", description="Previous conversation history")

CONDENSE_TEMPLATE = """Given the following chat history and a follow-up question,
rephrase the follow-up question to be a standalone question.
If the follow-up references something from the chat history (like "it", "they",
"the company", "their services"), resolve the reference to make it standalone.
If the question is already standalone or is a greeting/casual message, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

PROMPT_TEMPLATE = """You are a friendly and helpful assistant for Promtior, an AI consulting company.

Rules:
- For greetings (hi, hello, hola, etc.), respond warmly and let them know you can answer questions about Promtior.
- For questions about Promtior, answer based on the context provided below.
- If the context doesn't contain the answer, say "I don't have that information about Promtior."
- Keep responses conversational and natural.

Context:
{context}

Chat History:
{chat_history}

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

    def process_input(input_dict):
        """Condense the question, retrieve context, and build the final prompt input."""
        chat_history = input_dict.get("chat_history", "")
        question = input_dict["question"]

        # Condense question with history
        standalone_question = condense_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        # Retrieve relevant docs
        docs = retriever.invoke(standalone_question)

        return {
            "context": _format_docs(docs),
            "chat_history": chat_history,
            "standalone_question": standalone_question,
        }

    chain = (
        RunnableLambda(process_input)
        | answer_prompt
        | llm
        | StrOutputParser()
    ).with_types(input_type=ChatInput)
    return chain
