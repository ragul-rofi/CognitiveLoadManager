"""Embedding generation and similarity utilities for CLM."""

import logging
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from clm.exceptions import EmbeddingError

logger = logging.getLogger("clm.embeddings")

# Global model instance (lazy-loaded)
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """
    Get or initialize the sentence transformer model.
    
    Raises:
        EmbeddingError: If model loading fails
    """
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {_model_name}")
            _model = SentenceTransformer(_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load embedding model {_model_name}: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
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
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    try:
        model = _get_model()
        embedding = model.encode(text, convert_to_tensor=False)
        logger.debug(f"Generated embedding for text (length={len(text)})")
        return embedding.tolist()
    except EmbeddingError:
        raise
    except Exception as e:
        error_msg = f"Failed to generate embedding: {e}"
        logger.error(error_msg)
        raise EmbeddingError(error_msg) from e


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
        
    Raises:
        EmbeddingError: If similarity computation fails
    """
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            logger.warning("Zero-norm vector encountered in cosine similarity")
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        
        # Clamp to [-1, 1] to handle floating point errors
        result = float(max(-1.0, min(1.0, similarity)))
        logger.debug(f"Computed cosine similarity: {result:.3f}")
        return result
        
    except Exception as e:
        error_msg = f"Failed to compute cosine similarity: {e}"
        logger.error(error_msg)
        raise EmbeddingError(error_msg) from e


def clear_cache() -> None:
    """
    Clear the embedding cache.
    
    Useful for testing or when memory needs to be freed.
    """
    embed.cache_clear()
