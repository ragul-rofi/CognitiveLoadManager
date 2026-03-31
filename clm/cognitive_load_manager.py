"""Cognitive Load Manager facade - main entry point for agent loop integration."""

from clm.core.config import CLMConfig
from clm.core.models import TaskState, InterventionResponse
from clm.core.signal_collector import SignalCollector
from clm.core.scorer import CLMScorer
from clm.core.chunking_engine import ChunkingEngine
from clm.core.action_dispatcher import ActionDispatcher
from clm.storage.sidecar_store import SidecarStore


class CognitiveLoadManager:
    """
    Metacognitive middleware for LLM-based agent loops.
    
    Monitors cognitive load in real-time, detects overload conditions,
    and intervenes through task compression, goal anchoring, and clarification.
    
    Usage:
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        # In agent loop after each LLM call:
        response = clm.observe(llm_output, task_state)
        
        if response.action == "pass":
            # Continue normally
            pass
        elif response.action == "patch":
            # Replace context with response.context
            pass
        elif response.action == "interrupt":
            # Request clarification: response.clarification
            pass
    """
    
    def __init__(self, config: CLMConfig):
        """
        Initialize CLM with configuration.
        
        Args:
            config: Configuration object with thresholds, weights, storage settings
        """
        # Validate configuration
        config.validate()
        
        self.config = config
        
        # Initialize storage backend
        self.sidecar_store = SidecarStore(
            storage_type=config.storage_type,
            connection_params=config.storage_params
        )
        
        # Initialize signal collector
        self.signal_collector = SignalCollector(
            branching_threshold=config.branching_threshold,
            repetition_threshold=config.repetition_threshold,
            uncertainty_threshold=config.uncertainty_threshold,
            hedged_tokens=config.hedged_tokens
        )
        
        # Initialize scorer
        self.scorer = CLMScorer(
            weights=config.weights,
            green_max=config.green_max,
            amber_max=config.amber_max
        )
        
        # Initialize chunking engine
        self.chunking_engine = ChunkingEngine(
            sidecar_store=self.sidecar_store
        )
        
        # Initialize action dispatcher
        self.action_dispatcher = ActionDispatcher(
            chunking_engine=self.chunking_engine
        )
        
        # State tracking
        self._current_score: float = 0.0
        self._current_zone: str = "Green"
    
    def observe(self, llm_output: str, task_state: TaskState) -> InterventionResponse:
        """
        Process LLM output and task state, return intervention if needed.
        
        Orchestrates the full CLM workflow:
        1. Extract cognitive load signals
        2. Compute CLM score
        3. Classify zone
        4. Dispatch intervention
        5. Auto-expand if score drops below 40
        
        Args:
            llm_output: Raw text output from LLM
            task_state: Current task tree and root intent
            
        Returns:
            InterventionResponse with action, context, and clarification fields
        """
        # Step 1: Extract signals
        signals = self.signal_collector.extract_signals(llm_output, task_state)
        
        # Step 2: Compute score
        clm_score = self.scorer.compute_score(signals)
        
        # Step 3: Classify zone
        zone = self.scorer.classify_zone(clm_score)
        
        # Step 4: Dispatch intervention
        response = self.action_dispatcher.dispatch(clm_score, zone, task_state)
        
        # Step 5: Auto-expand if score drops below 40
        if clm_score < 40:
            task_state.task_tree = self.chunking_engine.auto_expand(
                task_state.task_tree, 
                clm_score
            )
        
        # Update state tracking
        self._current_score = clm_score
        self._current_zone = zone
        
        return response
    
    def get_score(self) -> float:
        """
        Return current CLM score (0-100).
        
        Returns:
            Current cognitive load score
        """
        return self._current_score
    
    def get_zone(self) -> str:
        """
        Return current zone classification (Green/Amber/Red).
        
        Returns:
            Current zone classification
        """
        return self._current_zone
    
    def get_sidecar_stats(self) -> dict:
        """
        Return storage statistics (count, size, etc).
        
        Returns:
            Dictionary with storage statistics:
            - count: Total number of stored chunks
            - total_size: Total size of stored data
            - compressed_count: Number of compressed tasks
            - expanded_count: Number of expanded tasks
            - db_path: Path to database file
        """
        return self.sidecar_store.get_stats()
    
    def close(self) -> None:
        """
        Close storage connections and cleanup resources.
        
        Should be called when CLM is no longer needed.
        """
        self.sidecar_store.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
