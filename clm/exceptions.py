"""Custom exception classes for CLM error handling."""


class CLMError(Exception):
    """Base exception for all CLM errors."""
    pass


class ConfigurationError(CLMError):
    """
    Raised when configuration validation fails.
    
    Examples:
    - Weights do not sum to 1.0
    - Invalid zone boundaries
    - Invalid storage type
    """
    pass


class StorageError(CLMError):
    """
    Raised when sidecar storage operations fail.
    
    Examples:
    - Database connection failure
    - Disk space exhaustion
    - Data corruption
    """
    pass


class EmbeddingError(CLMError):
    """
    Raised when embedding generation fails.
    
    Examples:
    - Model loading failure
    - Inference failure
    - Invalid input text
    """
    pass


class ValidationError(CLMError):
    """
    Raised when input validation fails.
    
    Examples:
    - Invalid task state
    - Malformed task tree
    - Missing required fields
    """
    pass


class ExpansionError(CLMError):
    """
    Raised when task expansion fails.
    
    Examples:
    - Task ID not found in sidecar store
    - Task ID not found in task tree
    - Corrupted task data
    """
    pass
