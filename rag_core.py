import os
from typing import List, Dict, Any

# --- SQLite shim for Chroma on Streamlit Cloud ---
# If the system sqlite3 is too old, use pysqlite3-binary.
try:
    import pysqlite3  # noqa: F401
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb

from chromadb.config import Settings

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from uuid import uuid4
# Choose embeddings/LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# If you want Gemini later:
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def get_embeddings():
    # swap here for Gemini if desired:
    # return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

def get_llm():
    model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
    # Gemini alternative:
    # return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)
    return ChatOpenAI(model=model, temperature=0.1)

def chunk_text(text: str, doc_title: str, source_path: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800, chunk_overlap=200,
        separators=["\n\n", "\n", "Section ", "Sec.", ".", " "]
    )
    chunks = splitter.split_text(text)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": f"{doc_title}-{i}-{uuid4().hex}",  # ensure unique IDs across pages/runs
            "text": c,
            "metadata": {
                "doc_title": doc_title,
                "source_path": source_path,
            }
        })

    return out

def init_chroma(persist_dir: str = "./chroma_db"):
    client = chromadb.Client(Settings(persist_directory=persist_dir, is_persistent=True))
    coll = client.get_or_create_collection("big_bill_collection", metadata={"hnsw:space": "cosine"})
    return client, coll

def add_to_vectorstore(coll, embeddings, chunks: List[Dict[str, Any]]):
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    vecs = embeddings.embed_documents(texts)
    coll.add(ids=ids, embeddings=vecs, metadatas=metas, documents=texts)

def retrieve(coll, embeddings, query: str, k: int = 6) -> List[Dict[str, Any]]:
    # Vector recall
    q_emb = embeddings.embed_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=max(k, 10))
    docs = [
        {"text": d, "metadata": m} for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]
    # Lightweight lexical rerank (BM25) to reduce hallucination
    bm25 = BM25Okapi([d["text"].split() for d in docs])
    scores = bm25.get_scores(query.split())
    reranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return reranked[:k]

def format_context(docs: List[Dict[str, Any]]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        title = meta.get("doc_title", "Unknown")
        page = meta.get("page", "?")
        sec = meta.get("section", "?")
        out.append(f"[{i}] ({title}, p.{page} §{sec})\n{d['text']}\n")
    return "\n".join(out)
