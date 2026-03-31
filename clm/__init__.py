"""
Cognitive Load Manager (CLM) v1.0

A metacognitive middleware layer for LLM-based agent loops that monitors
cognitive load, detects overload conditions, and intervenes through task
compression, goal anchoring, and clarification requests.
"""

__version__ = "1.0.0"

from clm.cognitive_load_manager import CognitiveLoadManager
from clm.core.config import CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode, InterventionResponse
from clm.exceptions import (
    CLMError,
    ConfigurationError,
    StorageError,
    EmbeddingError,
    ValidationError,
    ExpansionError,
)

__all__ = [
    "CognitiveLoadManager",
    "CLMConfig",
    "TaskState",
    "TaskTree",
    "TaskNode",
    "InterventionResponse",
    "CLMError",
    "ConfigurationError",
    "StorageError",
    "EmbeddingError",
    "ValidationError",
    "ExpansionError",
]
