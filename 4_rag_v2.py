# pip install -U langchain langchain-community langchain-core \
# langchain-text-splitters langchain-groq langchain-huggingface \
# faiss-cpu pypdf python-dotenv langsmith huggingface-hub

import os
from dotenv import load_dotenv

from langsmith import traceable  # ⭐ key import

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# Environment
# --------------------------------------------------
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG VERSION 2"
# LANGCHAIN_TRACING_V2=true must be set in .env
# LANGCHAIN_API_KEY=...
# HF_API_TOKEN=...
# GROQ_API_KEY=...


PDF_PATH = "islr.pdf"
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


# --------------------------------------------------
# ----------- Traced setup steps -------------------
# --------------------------------------------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    embeddings = HuggingFaceEndpointEmbeddings(
        model=HF_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
    )

    vs = FAISS.from_documents(splits, embeddings)
    return vs


@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs


# --------------------------------------------------
# ----------- RAG pipeline --------------------------
# --------------------------------------------------

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say you don't know.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# Build vector store under traced setup
vectorstore = setup_pipeline(PDF_PATH)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = parallel | prompt | llm | StrOutputParser()


# --------------------------------------------------
# ----------- Interactive query (traced) ------------
# --------------------------------------------------

print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

while True:
    try:
        q = input("\nQ: ").strip()
        if not q:
            continue

        # 👇 Visible run name in LangSmith
        config = {
            "run_name": "pdf_rag_query",
            "tags": ["rag", "hf-embeddings", "groq"],
            "metadata": {
                "embedding_model": HF_MODEL,
                "llm": GROQ_MODEL,
                "retriever_k": 4,
            },
        }

        answer = chain.invoke(q, config=config)
        print("\nA:", answer)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
