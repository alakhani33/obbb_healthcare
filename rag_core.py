# rag_core.py
import os
from typing import List, Dict, Any

# --- Backend selector ---
USE_FAISS = os.getenv("VECTOR_BACKEND", "chroma").lower() == "faiss"

# Common: embeddings factory (LangChain OpenAI)
from langchain_openai import OpenAIEmbeddings

def get_embeddings():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))

# -----------------------------
# FAISS backend (Cloud-friendly)
# -----------------------------
if USE_FAISS:
    from langchain_community.vectorstores import FAISS

    _faiss_store: FAISS | None = None

    def init_chroma(persist_dir: str = "./chroma_db"):
        # Keep signature for the app; FAISS doesn't need a client/collection
        return None, None

    def add_to_vectorstore(_coll, _embeddings, chunks: List[Dict[str, Any]]):
        """Build or extend an in-memory FAISS index."""
        global _faiss_store
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        embeddings = get_embeddings()
        if _faiss_store is None:
            _faiss_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
        else:
            _faiss_store.add_texts(texts=texts, metadatas=metas, embedding=embeddings)

    def retrieve(_coll, _embeddings, query: str, k: int = 6):
        if _faiss_store is None:
            return []
        docs = _faiss_store.similarity_search(query, k=k)
        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

# -----------------------------
# Chroma backend (local dev)
# -----------------------------
else:
    # SQLite shim for Cloud/Linux if present; harmless elsewhere
    try:
        import pysqlite3  # installed only on Linux (requirements has a platform marker)
        import sys
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        pass

    import chromadb

    def init_chroma(persist_dir: str = "./chroma_db"):
        # Use 0.5.x PersistentClient API to avoid _type issues with Settings
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            coll = client.get_or_create_collection(
                "big_bill_collection",
                metadata={"hnsw:space": "cosine"},
            )
            return client, coll
        except Exception as e:
            raise RuntimeError(
                "Vector store initialization failed (Chroma). "
                "Make sure runtime.txt pins Python 3.11 and chromadb==0.5.5. "
                f"Original error: {e}"
            )

    def add_to_vectorstore(coll, _embeddings, chunks: List[Dict[str, Any]]):
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        coll.add(ids=ids, documents=texts, metadatas=metadatas)

    def retrieve(coll, embeddings, query: str, k: int = 6):
        # Simple similarity search
        res = coll.query(query_texts=[query], n_results=k, include=["metadatas", "documents"])
        docs = []
        if res and res.get("documents"):
            for text, meta in zip(res["documents"][0], res["metadatas"][0]):
                docs.append({"text": text, "metadata": meta})
        return docs

# -----------------------------
# Your existing helpers
# -----------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

def chunk_text(text: str, doc_title: str, source_path: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_text(text)
    out = []
    from uuid import uuid4
    for i, c in enumerate(chunks):
        out.append({
            "id": f"{doc_title}-{i}-{uuid4().hex}",
            "text": c,
            "metadata": {"doc_title": doc_title, "source_path": source_path},
        })
    return out

def format_context(docs: List[Dict[str, Any]]) -> str:
    blocks = []
    for d in docs:
        meta = d.get("metadata", {})
        page = meta.get("page", "")
        sec = meta.get("section", "")
        head = f"{meta.get('doc_title','')}"
        if page or sec: head += f" (p.{page} {sec})"
        blocks.append(f"[{head}]\n{d['text']}")
    return "\n\n---\n\n".join(blocks)

def get_llm():
    # your existing OpenAI chat model factory (left as-is)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("MODEL_CHOICE", "gpt-4o-mini"), temperature=0.2)
