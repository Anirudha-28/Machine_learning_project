import os
import pickle
import numpy as np

from llm import embed_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
INDEX_PATH = os.path.join(BASE_DIR, "vector_store.pkl")

# Embeddings now come from Gemini's free text-embedding-004 model (see
# llm.embed_text) instead of a local Ollama/nomic-embed-text call — one
# less local service to keep running, and no embedding-side cold start.


def get_embedding(text: str) -> np.ndarray:
    return np.array(embed_text(text), dtype=np.float32)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
    """Splits text into overlapping word chunks (simple, dependency-free)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index():
    """
    Reads every .txt file in knowledge_base/, splits into chunks, embeds each
    chunk with Gemini embeddings, and saves the result to vector_store.pkl.
    Run this once, and again any time you edit the knowledge_base/ files.
    """
    records = []  # list of {"text": chunk, "source": filename, "embedding": vector}

    for filename in sorted(os.listdir(KB_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        for chunk in chunk_text(text):
            print(f"Embedding chunk from {filename} ({len(chunk)} chars)...")
            embedding = get_embedding(chunk)
            records.append({"text": chunk, "source": filename, "embedding": embedding})

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(records, f)

    print(f"\nSaved {len(records)} chunks to {INDEX_PATH}")


def _load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "vector_store.pkl not found. Run 'python rag.py' once to build it."
        )
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def _keyword_search(query: str, top_k: int = 3) -> str:
    """
    Simple keyword-based fallback search over raw .txt files.
    Used when vector_store.pkl hasn't been built yet, or Gemini is unreachable.
    Scores each chunk by how many query words appear in it.
    """
    query_words = set(query.lower().split())
    scored = []

    for filename in sorted(os.listdir(KB_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into paragraphs for chunking
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for w in query_words if w in para_lower)
            if score > 0:
                scored.append((score, para))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [text for _, text in scored[:top_k]]
    if not top_chunks:
        return ""
    return "\n\n---\n\n".join(top_chunks)


def search(query: str, top_k: int = 3) -> str:
    """
    Embeds the query, ranks knowledge_base chunks by cosine similarity,
    and returns the top_k chunks joined into one context string.
    Falls back to keyword search if vector_store.pkl not built yet.
    """
    # Try vector search first
    try:
        records   = _load_index()
        query_vec = get_embedding(query)

        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        scored    = [(cosine_sim(query_vec, r["embedding"]), r["text"]) for r in records]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [text for score, text in scored[:top_k]]
        return "\n\n---\n\n".join(top_chunks)

    except (FileNotFoundError, Exception):
        # Vector store not built, or Gemini embedding call failed — use keyword fallback
        print("[RAG] Vector store unavailable, falling back to keyword search.")
        return _keyword_search(query, top_k=top_k)


if __name__ == "__main__":
    build_index()
