"""
architectures/hyde_rag.py — HyDE (Hypothetical Document Embeddings) RAG.

Key insight: raw user questions and technical document chunks live in
very different parts of the embedding space. A question like
"What dataset was used?" has a very different embedding from a paper
chunk saying "We evaluated on the SQuAD 2.0 benchmark …".

HyDE bridges this gap by first asking the LLM to write a *hypothetical*
answer, then embedding THAT instead of the question. A hypothetical answer
is phrased like a document chunk, so it ends up close to real relevant
chunks in embedding space.

Pipeline:
  1. [Same index as Vanilla RAG — shared ChromaDB collection]
  2. Query:
     a. Ask LLM to write a hypothetical answer to the question
     b. Embed the hypothetical answer (not the question)
     c. Use that embedding to retrieve real chunks
     d. Pass real chunks + original question to LLM for final answer

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)

Can be run independently:
    python architectures/hyde_rag.py
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.llm_client import LLMClient
from utils.embedder import embed_single
from architectures.vanilla_rag import index_papers, get_chroma_collection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Generate a hypothetical answer
# ─────────────────────────────────────────────────────────────────────────────

_HYPOTHETICAL_SYSTEM = """You are a research scientist writing a section of a paper.
Write a plausible, specific answer to the question as if you were drafting the
relevant paragraph of a research paper. Use technical language and be precise.
Do NOT say 'I think' or hedge — write as if stating known facts."""

_HYPOTHETICAL_PROMPT = """Write a hypothetical research paper passage that would
directly answer this question. The passage should sound like it was extracted
from a real AI/ML paper.

Question: {question}

Write 2-3 sentences in an academic, factual style:"""


def generate_hypothetical_answer(question: str, llm: LLMClient) -> str:
    """
    Generate a plausible 'fake' answer that reads like a paper excerpt.

    Why this works: the hypothetical answer uses vocabulary and phrasing
    typical of paper text, so its embedding lands near real paper chunks
    that contain the actual answer — even though the hypothetical content
    may be factually wrong. Retrieval quality improves because we're
    matching document-style text to document-style text.
    """
    return llm.complete(
        _HYPOTHETICAL_PROMPT.format(question=question),
        system=_HYPOTHETICAL_SYSTEM,
        temperature=0.4,   # Slightly higher temp for diverse hypotheticals
        max_tokens=256,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Steps 2-4: Retrieve with hypothetical embedding, answer with real context
# ─────────────────────────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """You are a research assistant. Answer the question using ONLY
the provided context passages. If the answer is not in the context, say so explicitly.
Do not fabricate details."""

_ANSWER_PROMPT = """Context passages retrieved from research papers:

{context}

---
Original question: {question}

Answer the question based solely on the context above. Be specific and cite
which passage supports your answer."""


def retrieve_with_hyde(
    question: str,
    hypothetical: str,
    collection,
    top_k: int = config.TOP_K,
) -> List[Dict]:
    """
    Use the hypothetical answer's embedding (not the question's) to retrieve chunks.
    This is the single key difference between HyDE and Vanilla RAG.
    """
    # Embed the *hypothetical answer* — not the question
    hyp_embedding = embed_single(hypothetical)

    results = collection.query(
        query_embeddings=[hyp_embedding],
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
    Full HyDE answer pipeline.
    Returns: {answer, contexts, sources, hypothetical_answer}
    """
    # Step 1: generate a hypothetical document that might contain the answer
    hypothetical = generate_hypothetical_answer(question, llm)
    logger.debug(f"HyDE hypothetical: {hypothetical[:100]}…")

    # Step 2: use the hypothetical embedding to find real matching chunks
    chunks = retrieve_with_hyde(question, hypothetical, collection, top_k)

    # Step 3: build context from REAL retrieved chunks
    context_str = "\n\n---\n\n".join(
        f"[Source: {c['title']} ({c['arxiv_id']})]\n{c['text']}"
        for c in chunks
    )

    # Step 4: answer the ORIGINAL question using the REAL context
    llm_answer = llm.complete(
        _ANSWER_PROMPT.format(context=context_str, question=question),
        system=_ANSWER_SYSTEM,
        temperature=0.1,
        max_tokens=512,
    )

    return {
        "answer":              llm_answer,
        "contexts":            [c["text"] for c in chunks],
        "sources":             [{"arxiv_id": c["arxiv_id"], "title": c["title"], "score": c["score"]} for c in chunks],
        "hypothetical_answer": hypothetical,  # keep for analysis / debugging
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HyDERAG:
    """Uses the same ChromaDB collection as Vanilla RAG (shared index)."""

    def __init__(self, papers: List[Dict]):
        self.llm = LLMClient()
        # Re-use or create the shared collection — no re-indexing needed
        self.collection = index_papers(papers)

    def answer(self, question: str) -> Dict[str, Any]:
        return answer(question, self.collection, self.llm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.data_loader import load_papers, load_questions

    papers = load_papers()
    rag    = HyDERAG(papers)
    questions = load_questions(papers)

    q = questions[0]["question"]
    print(f"\nQuestion: {q}")
    result = rag.answer(q)
    print(f"\nHypothetical answer used for retrieval:\n{result['hypothetical_answer']}")
    print(f"\nFinal answer:\n{result['answer']}")
