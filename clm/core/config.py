"""Configuration for Cognitive Load Manager."""

import logging
from dataclasses import dataclass, field

from clm.exceptions import ConfigurationError

logger = logging.getLogger("clm.config")

# Check if sentence-transformers is available
try:
    import sentence_transformers
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class CLMConfig:
    """Configuration for Cognitive Load Manager."""
    
    # Signal thresholds
    branching_threshold: int = 7
    repetition_threshold: float = 0.85
    uncertainty_threshold: float = 0.15
    
    # Scoring weights [branching, repetition, uncertainty, goal_distance]
    weights: list[float] = field(default_factory=lambda: [0.30, 0.25, 0.25, 0.20])
    
    # Zone boundaries
    green_max: float = 40.0
    amber_max: float = 70.0
    
    # Storage configuration
    storage_type: str = "sqlite"
    storage_params: dict = field(default_factory=dict)
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # No embed mode for restricted environments
    no_embed: bool = False  # If True, skip sentence-transformers entirely
                            # repetition + goal_distance fall back to keyword overlap
    
    # Hedged tokens
    hedged_tokens: list[str] = field(default_factory=lambda: [
        "maybe", "perhaps", "possibly", "might", "could",
        "uncertain", "unclear", "probably", "likely", "seems"
    ])
    
    def __post_init__(self):
        """Auto-detect and configure no_embed mode if sentence-transformers is not available."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE and not self.no_embed:
            self.no_embed = True
            print("[CLM] sentence-transformers not found — using keyword-based signals. Install with: pip install clm-plugin[embed]")
    
    def validate(self) -> None:
        """
        Validate configuration constraints.
        
        Raises:
            ConfigurationError: If validation fails
        """
        logger.debug(f"Validating configuration: weights={self.weights}, green_max={self.green_max}, amber_max={self.amber_max}")
        
        # Validate weights sum to 1.0
        weights_sum = sum(self.weights)
        if abs(weights_sum - 1.0) > 0.01:
            error_msg = f"Weights must sum to 1.0 (±0.01), got {weights_sum}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        # Validate zone boundaries
        if not (0 < self.green_max < self.amber_max < 100):
            error_msg = f"Zone boundaries must satisfy: 0 < green_max < amber_max < 100, got green_max={self.green_max}, amber_max={self.amber_max}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        # Validate storage type
        if self.storage_type not in ["sqlite", "redis"]:
            error_msg = f"Invalid storage_type: {self.storage_type}. Must be 'sqlite' or 'redis'"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.info("Configuration validated successfully")
