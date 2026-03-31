"""Embedding generation and similarity utilities for CLM."""

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer


# Global model instance (lazy-loaded)
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_name)
    return _model


@lru_cache(maxsize=128)
def embed(text: str) -> list[float]:
    """
    Generate embedding vector using sentence-transformers.
    
    Uses all-MiniLM-L6-v2 model (384-dimensional embeddings).
    Results are cached to avoid redundant computation for repeated texts
    (especially useful for root intent embeddings).
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector (384 dimensions)
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding.tolist()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    
    Returns value in range [-1, 1], where:
    - 1.0 = identical vectors
    - 0.0 = orthogonal vectors
    - -1.0 = opposite vectors
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity clamped to [-1, 1]
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    similarity = dot_product / (norm_v1 * norm_v2)
    
    # Clamp to [-1, 1] to handle floating point errors
    return float(max(-1.0, min(1.0, similarity)))


def clear_cache() -> None:
    """
    Clear the embedding cache.
    
    Useful for testing or when memory needs to be freed.
    """
    embed.cache_clear()
