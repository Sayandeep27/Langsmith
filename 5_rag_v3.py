# pip install -U langchain langchain-community langchain-core \
# langchain-text-splitters langchain-groq langchain-huggingface \
# faiss-cpu pypdf python-dotenv langsmith huggingface-hub


'''
Why this is production-grade

Zero re-embedding unless PDF or config changes

Deterministic index reuse

Clean LangSmith observability

Provider-agnostic design

Ready for evaluation, CI, experiments

This is exactly how real RAG systems are built in production.
'''


import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

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

os.environ["LANGCHAIN_PROJECT"] = "RAG VERSION 3"
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# HF_API_TOKEN=...
# GROQ_API_KEY=...

PDF_PATH = "islr.pdf"
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# --------------------------------------------------
# ----------- helpers (traced) ----------------------
# --------------------------------------------------

@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits, embedding_model: str):
    embeddings = HuggingFaceEndpointEmbeddings(
        model=embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
    )
    return FAISS.from_documents(splits, embeddings)


# --------------------------------------------------
# ----------- index fingerprinting -----------------
# --------------------------------------------------

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }


def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embedding_model: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------
# ----------- explicitly traced index ops -----------
# --------------------------------------------------

@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path, embedding_model: str):
    embeddings = HuggingFaceEndpointEmbeddings(
        model=embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
    )
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@traceable(name="build_index", tags=["index"])
def build_index_run(
    pdf_path: str,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    vs = build_vectorstore(splits, embedding_model)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "pdf_path": os.path.abspath(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding_model,
            },
            indent=2,
        )
    )
    return vs


# --------------------------------------------------
# ----------- dispatcher (not traced) ---------------
# --------------------------------------------------

def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = HF_MODEL,
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embedding_model)
    index_dir = INDEX_ROOT / key

    if index_dir.exists() and not force_rebuild:
        return load_index_run(index_dir, embedding_model)
    else:
        return build_index_run(
            pdf_path,
            index_dir,
            chunk_size,
            chunk_overlap,
            embedding_model,
        )


# --------------------------------------------------
# ----------- model, prompt -------------------------
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


# --------------------------------------------------
# ----------- setup + query (ROOT RUN) --------------
# --------------------------------------------------

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(
    pdf_path: str,
    chunk_size=1000,
    chunk_overlap=150,
    embedding_model=HF_MODEL,
    force_rebuild=False,
):
    return load_or_build_index(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embedding_model,
        force_rebuild,
    )


@traceable(
    name="pdf_rag_full_run",
    tags=["rag", "hf-embeddings", "groq"],
)
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size=1000,
    chunk_overlap=150,
    embedding_model=HF_MODEL,
    force_rebuild=False,
):
    vectorstore = setup_pipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embedding_model,
        force_rebuild,
    )

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

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {
                "k": 4,
                "embedding_model": embedding_model,
                "llm": GROQ_MODEL,
            },
        },
    )


# --------------------------------------------------
# ---------------- CLI ------------------------------
# --------------------------------------------------

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)
