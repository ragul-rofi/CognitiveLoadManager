"""Configuration for Cognitive Load Manager."""

from dataclasses import dataclass, field


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
    
    # Hedged tokens
    hedged_tokens: list[str] = field(default_factory=lambda: [
        "maybe", "perhaps", "possibly", "might", "could",
        "uncertain", "unclear", "probably", "likely", "seems"
    ])
    
    def validate(self) -> None:
        """Validate configuration constraints."""
        if abs(sum(self.weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")
        if not (0 < self.green_max < self.amber_max < 100):
            raise ValueError("Zone boundaries must satisfy: 0 < green_max < amber_max < 100")
