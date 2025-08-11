# rag_core.py
import os
from typing import List, Dict, Any
from uuid import uuid4

# Shared deps
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


# Decide vector backend from env.
# Default to FAISS (cloud-friendly). Set VECTOR_BACKEND=chroma to use Chroma locally.
USE_FAISS = os.getenv("VECTOR_BACKEND", "faiss").lower() == "faiss"

if USE_FAISS:
    try:
        from langchain_community.vectorstores import FAISS  # modern path
    except ImportError:
        from langchain.vectorstores import FAISS           # legacy path




# -----------------------------
# Embeddings & LLM factories
# -----------------------------
def get_embeddings():
    return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))


def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
    )


# -----------------------------
# FAISS backend (default)
# -----------------------------
if USE_FAISS:
    from langchain_community.vectorstores import FAISS

    _faiss_store: FAISS | None = None

    def init_chroma(_persist_dir: str = "./chroma_db"):
        """Keep signature compatibility; FAISS doesn't need a client/collection."""
        return None, None

    def add_to_vectorstore(_coll, _embeddings, chunks: List[Dict[str, Any]]):
        """Build or extend an in-memory FAISS index."""
        global _faiss_store
        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        embs = get_embeddings()
        if _faiss_store is None:
            _faiss_store = FAISS.from_texts(texts=texts, embedding=embs, metadatas=metas)
        else:
            _faiss_store.add_texts(texts=texts, metadatas=metas, embedding=embs)

    def _bm25_rerank(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Optional lexical re-rank for sharper relevance."""
        if not docs:
            return docs
        corpus = [d["text"] for d in docs]
        bm25 = BM25Okapi([t.split() for t in corpus])
        scores = bm25.get_scores(query.split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in order]

    def retrieve(_coll, _embeddings, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Vector search (FAISS) + light BM25 re-rank."""
        if _faiss_store is None:
            return []
        # Vector search
        lc_docs = _faiss_store.similarity_search(query, k=k * 2)  # overfetch a bit
        docs = [{"text": d.page_content, "metadata": d.metadata} for d in lc_docs]
        # Lexical re-rank to tighten results
        docs = _bm25_rerank(query, docs, top_k=k)
        return docs

# -----------------------------
# Chroma backend (local dev)
# -----------------------------
else:
    # Optional SQLite shim (harmless if not present). Helps on some Linux hosts.
    try:
        import pysqlite3  # installed only on Linux if present in requirements
        import sys
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        pass

    # Import Chroma lazily (so Cloud builds using FAISS never touch it)
    import chromadb

    def init_chroma(persist_dir: str = "./chroma_db"):
        """Initialize Chroma 0.5.x using PersistentClient to avoid _type/Settings issues."""
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
                "Ensure runtime.txt pins Python 3.11 and chromadb==0.5.5. "
                f"Original error: {e}"
            )

    def add_to_vectorstore(coll, _embeddings, chunks: List[Dict[str, Any]]):
        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        coll.add(ids=ids, documents=docs, metadatas=metas)

    def _bm25_rerank(query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return docs
        corpus = [d["text"] for d in docs]
        bm25 = BM25Okapi([t.split() for t in corpus])
        scores = bm25.get_scores(query.split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in order]

    def retrieve(coll, _embeddings, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Chroma similarity + BM25 re-rank."""
        res = coll.query(query_texts=[query], n_results=k * 2, include=["metadatas", "documents"])
        docs: List[Dict[str, Any]] = []
        if res and res.get("documents"):
            for text, meta in zip(res["documents"][0], res["metadatas"][0]):
                docs.append({"text": text, "metadata": meta})
        return _bm25_rerank(query, docs, top_k=k)


# -----------------------------
# Chunking & formatting helpers
# -----------------------------
def chunk_text(text: str, doc_title: str, source_path: str) -> List[Dict[str, Any]]:
    """Split long text into overlapping chunks and attach metadata + unique IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks):
        out.append({
            "id": f"{doc_title}-{i}-{uuid4().hex}",
            "text": c,
            "metadata": {"doc_title": doc_title, "source_path": source_path},
        })
    return out


def format_context(docs: List[Dict[str, Any]]) -> str:
    """Turn retrieved docs into a readable context block for prompting."""
    blocks = []
    for d in docs:
        meta = d.get("metadata", {})
        page = meta.get("page", "")
        sec = meta.get("section", "")
        head = f"{meta.get('doc_title','')}"
        if page or sec:
            head += f" (p.{page} {sec})"
        blocks.append(f"[{head}]\n{d['text']}")
    return "\n\n---\n\n".join(blocks)
