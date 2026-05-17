"""
architectures/vanilla_rag.py — Standard RAG pipeline (Baseline).

Pipeline:
  1. Index: chunk documents → embed chunks → store in ChromaDB
  2. Query: embed query → retrieve top-k chunks → LLM generates answer

This is the simplest possible RAG setup and acts as our performance floor.
All other architectures are measured relative to this baseline.

Can be run independently:
    python architectures/vanilla_rag.py
"""
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.llm_client import LLMClient
from utils.embedder import embed, embed_single

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Document chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping character-level chunks.

    Overlap exists because RAG retrieval is based on chunk similarity —
    if a key fact falls near a chunk boundary, the overlap ensures it
    appears in at least one complete chunk rather than being split across two.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Advance by (chunk_size - overlap) so adjacent chunks share `overlap` chars
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_collection(collection_name: str = config.COLLECTION_NAME):
    """
    Return a persistent ChromaDB collection.
    Using 'cosine' distance so similarity scores are intuitive (higher = more similar).
    """
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_papers(papers: List[Dict], collection_name: str = config.COLLECTION_NAME, force: bool = False):
    """
    Chunk all papers, embed the chunks, and store them in ChromaDB.

    Skips indexing if the collection already has documents — set force=True
    to re-index (e.g. after changing chunk size).
    """
    collection = get_chroma_collection(collection_name)

    if collection.count() > 0 and not force:
        logger.info(f"Collection '{collection_name}' already has {collection.count()} chunks. Skipping indexing.")
        return collection

    logger.info(f"Indexing {len(papers)} papers into '{collection_name}' …")
    all_texts, all_embeddings, all_metadatas, all_ids = [], [], [], []

    for paper in papers:
        # Use full_text if available, otherwise abstract
        text = paper.get("full_text") or paper["abstract"]
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{paper['arxiv_id']}_chunk_{i}"
            all_texts.append(chunk)
            all_embeddings.append(embed_single(chunk))
            all_metadatas.append({
                "arxiv_id": paper["arxiv_id"],
                "title":    paper["title"],
                "authors":  ", ".join(paper["authors"][:3]),  # truncate long author lists
                "chunk_i":  i,
            })
            all_ids.append(chunk_id)

    # ChromaDB supports batched upserts — use upsert so re-running is idempotent
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        collection.upsert(
            ids=all_ids[i:i+batch_size],
            embeddings=all_embeddings[i:i+batch_size],
            documents=all_texts[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
        )

    logger.info(f"Indexed {len(all_ids)} chunks across {len(papers)} papers.")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# Query & Answer
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """You are a research assistant. Answer the question using ONLY
the provided context passages. If the answer is not in the context, say so explicitly.
Do not fabricate details."""

_ANSWER_PROMPT = """Context passages retrieved from research papers:

{context}

---
Question: {question}

Answer the question based solely on the context above. Be specific and cite
which passage supports your answer."""


def retrieve(query: str, collection, top_k: int = config.TOP_K) -> List[Dict]:
    """
    Embed the query and return the top-k most similar chunks from ChromaDB.
    Returns a list of dicts with keys: text, arxiv_id, title, score.
    """
    query_embedding = embed_single(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "arxiv_id": meta.get("arxiv_id", ""),
            "title":    meta.get("title", ""),
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score: higher = more similar.
            "score":    round(1 - dist, 4),
        })
    return chunks


def answer(
    question: str,
    collection,
    llm: LLMClient,
    top_k: int = config.TOP_K,
) -> Dict[str, Any]:
    """
    Core Vanilla RAG answer step.
    Returns: {answer, contexts, sources}
    """
    # Step 1: retrieve relevant chunks using the raw question as the query
    chunks = retrieve(question, collection, top_k)

    # Step 2: format chunks into a single context block for the LLM
    context_str = "\n\n---\n\n".join(
        f"[Source: {c['title']} ({c['arxiv_id']})]\n{c['text']}"
        for c in chunks
    )

    # Step 3: ask the LLM to synthesise an answer from the retrieved context
    llm_answer = llm.complete(
        _ANSWER_PROMPT.format(context=context_str, question=question),
        system=_ANSWER_SYSTEM,
        temperature=0.1,
        max_tokens=512,
    )

    return {
        "answer":   llm_answer,
        "contexts": [c["text"] for c in chunks],   # plain text for RAGAS
        "sources":  [{"arxiv_id": c["arxiv_id"], "title": c["title"], "score": c["score"]} for c in chunks],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

class VanillaRAG:
    """Stateful wrapper that initialises once and answers many questions."""

    def __init__(self, papers: List[Dict]):
        self.llm = LLMClient()
        self.collection = index_papers(papers)

    def answer(self, question: str) -> Dict[str, Any]:
        return answer(question, self.collection, self.llm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.data_loader import load_papers, load_questions

    papers = load_papers()
    rag    = VanillaRAG(papers)
    questions = load_questions(papers)

    q = questions[0]["question"]
    print(f"\nQuestion: {q}")
    result = rag.answer(q)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {[s['arxiv_id'] for s in result['sources']]}")
