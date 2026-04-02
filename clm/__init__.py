"""
Cognitive Load Manager (CLM) v0.1

A metacognitive middleware layer for LLM-based agent loops that monitors
cognitive load, detects overload conditions, and intervenes through task
compression, goal anchoring, and clarification requests.
"""

__version__ = "0.1.3"

from clm.cognitive_load_manager import CognitiveLoadManager
from clm.core.config import CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode, InterventionResponse
from clm.utils.auto_state import AutoStateBuilder
from clm.exceptions import (
    CLMError,
    ConfigurationError,
    StorageError,
    EmbeddingError,
    ValidationError,
    ExpansionError,
)

# Shorthand alias — users write CLM() not CognitiveLoadManager()
CLM = CognitiveLoadManager

# Import adapters with graceful fallback for missing optional dependencies
try:
    from clm.adapters.langchain_adapter import CLMCallbackHandler
except ImportError:
    pass  # LangChain not installed, adapter unavailable

try:
    from clm.adapters.openai_adapter import CLMOpenAIHook
except ImportError:
    pass

from clm.adapters.loop_adapter import CLMLoop

__all__ = [
    "CognitiveLoadManager",
    "CLM",
    "CLMConfig",
    "TaskState",
    "TaskTree",
    "TaskNode",
    "InterventionResponse",
    "AutoStateBuilder",
    "CLMError",
    "ConfigurationError",
    "StorageError",
    "EmbeddingError",
    "ValidationError",
    "ExpansionError",
    "CLMCallbackHandler",
    "CLMOpenAIHook",
    "CLMLoop",
]

