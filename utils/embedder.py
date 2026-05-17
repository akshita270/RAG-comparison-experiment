"""
utils/embedder.py — Singleton wrapper around a sentence-transformers model.

All 4 RAG architectures use the SAME embedding model so that retrieval
comparisons are fair — the only variable between architectures is the
query embedding strategy, not the embedding model itself.

The model is loaded once (at first use) and reused, because loading
sentence-transformers from disk takes ~2-3 seconds.
"""
import sys
from pathlib import Path
from typing import List, Union
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

_model_instance = None  # Module-level singleton


def get_model():
    """
    Lazy-load the embedding model. Returns the same instance on
    subsequent calls to avoid re-loading 80MB of weights every query.
    """
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer
        _model_instance = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model_instance


def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Embed one string or a list of strings.
    Returns a 2-D numpy array of shape (n, embedding_dim).

    For all-MiniLM-L6-v2, embedding_dim = 384.
    Normalise=True gives unit vectors so cosine similarity = dot product,
    which is both faster and what ChromaDB's cosine space expects.
    """
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_single(text: str) -> List[float]:
    """
    Convenience wrapper: embed one text and return a plain Python list.
    ChromaDB's .query() and .add() expect lists, not numpy arrays.
    """
    return embed(text)[0].tolist()
