import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# LangSmith project
os.environ["LANGCHAIN_PROJECT"] = "RAG VERSION 1"

# -----------------------------
# Imports (latest LangChain)
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Hugging Face Inference API embeddings (NEW package)
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Config
# -----------------------------
PDF_PATH = "islr.pdf"  # ensure this file exists
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


# -----------------------------
# 1) Load PDF
# -----------------------------
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


# -----------------------------
# 2) Chunk documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
splits = splitter.split_documents(docs)


# -----------------------------
# 3) Hugging Face embeddings (REMOTE, FREE)
# -----------------------------
embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_MODEL,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
)


# -----------------------------
# 4) Vector store + retriever
# -----------------------------
vs = FAISS.from_documents(splits, embeddings)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# -----------------------------
# 5) Prompt
# -----------------------------
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


# -----------------------------
# 6) LLM (Groq – LLaMA 3.1)
# -----------------------------
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# -----------------------------
# 7) Parallel retrieval
# -----------------------------
parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)


# -----------------------------
# 8) RAG chain
# -----------------------------
chain = parallel | prompt | llm | StrOutputParser()


# -----------------------------
# 9) Interactive loop
# -----------------------------
print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

while True:
    try:
        q = input("\nQ: ").strip()
        if not q:
            continue

        answer = chain.invoke(q)
        print("\nA:", answer)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
