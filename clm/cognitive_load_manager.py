"""Cognitive Load Manager facade - main entry point for agent loop integration."""

import logging
from datetime import datetime
from clm.core.config import CLMConfig
from clm.core.models import TaskState, InterventionResponse
from clm.core.signal_collector import SignalCollector
from clm.core.scorer import CLMScorer
from clm.core.chunking_engine import ChunkingEngine
from clm.core.action_dispatcher import ActionDispatcher
from clm.storage.sidecar_store import SidecarStore
from clm.exceptions import CLMError, ConfigurationError, StorageError, EmbeddingError, ValidationError

logger = logging.getLogger("clm")


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
    
    def __init__(self, config: CLMConfig = None, verbose: bool = False):
        """
        Initialize CLM with configuration.
        
        Args:
            config: Configuration object with thresholds, weights, storage settings.
                   If None, uses default configuration.
            verbose: If True, prints CLM observations to stdout
            
        Raises:
            ConfigurationError: If configuration validation fails
            StorageError: If storage initialization fails
        """
        if config is None:
            config = CLMConfig()
        self.verbose = verbose
        self._history: list[dict] = []
        self._step = 0
        
        logger.info("Initializing Cognitive Load Manager")
        
        # Validate configuration
        try:
            config.validate()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        self.config = config
        
        # Initialize storage backend
        try:
            self.sidecar_store = SidecarStore(
                storage_type=config.storage_type,
                connection_params=config.storage_params
            )
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise
        
        # Initialize signal collector
        try:
            self.signal_collector = SignalCollector(
                branching_threshold=config.branching_threshold,
                repetition_threshold=config.repetition_threshold,
                uncertainty_threshold=config.uncertainty_threshold,
                hedged_tokens=config.hedged_tokens,
                no_embed=config.no_embed
            )
            logger.debug("Signal collector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize signal collector: {e}")
            raise
        
        # Initialize scorer
        try:
            self.scorer = CLMScorer(
                weights=config.weights,
                green_max=config.green_max,
                amber_max=config.amber_max
            )
            logger.debug("CLM scorer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize scorer: {e}")
            raise
        
        # Initialize chunking engine
        try:
            self.chunking_engine = ChunkingEngine(
                sidecar_store=self.sidecar_store
            )
            logger.debug("Chunking engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize chunking engine: {e}")
            raise
        
        # Initialize action dispatcher
        try:
            self.action_dispatcher = ActionDispatcher(
                chunking_engine=self.chunking_engine
            )
            logger.debug("Action dispatcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize action dispatcher: {e}")
            raise
        
        # State tracking
        self._current_score: float = 0.0
        self._current_zone: str = "Green"
        
        logger.info("Cognitive Load Manager initialized successfully")
    
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
            
        Note:
            If component failures occur, CLM will attempt graceful degradation
            and return a "pass" action to avoid disrupting the agent loop.
        """
        logger.info("CLM observe() called")
        
        try:
            # Step 1: Extract signals
            try:
                signals = self.signal_collector.extract_signals(llm_output, task_state)
            except ValidationError as e:
                logger.error(f"Signal extraction failed: {e}. Returning pass action.")
                return InterventionResponse(
                    action="pass",
                    clm_score=0.0,
                    zone="Green"
                )
            
            # Step 2: Compute score
            clm_score = self.scorer.compute_score(signals)
            
            # Step 3: Classify zone
            zone = self.scorer.classify_zone(clm_score)
            
            # Step 4: Dispatch intervention
            try:
                response = self.action_dispatcher.dispatch(clm_score, zone, task_state)
            except StorageError as e:
                logger.warning(f"Storage error during intervention: {e}. Attempting graceful degradation.")
                # Graceful degradation: return pass action if storage fails
                response = InterventionResponse(
                    action="pass",
                    clm_score=clm_score,
                    zone=zone
                )
            
            # Step 4.5: Record history entry
            self._step += 1
            entry = {
                "step": self._step,
                "score": round(clm_score, 2),
                "zone": zone,
                "action": response.action,
                "compressed_tasks": response.compressed_tasks,
                "signals": {
                    "branching": round(signals.branching_factor, 3),
                    "repetition": round(signals.repetition_rate, 3),
                    "uncertainty": round(signals.uncertainty_density, 3),
                    "goal_distance": round(signals.goal_distance, 3),
                },
                "timestamp": datetime.now().isoformat(),
            }
            self._history.append(entry)
            
            if self.verbose:
                zone_icon = {"Green": "✓", "Amber": "⚠", "Red": "✗"}.get(zone, "?")
                print(
                    f"[CLM] step={self._step:03d} | {zone_icon} {zone:6s} | "
                    f"score={clm_score:5.1f} | action={response.action:9s} | "
                    f"branch={signals.branching_factor:.2f} "
                    f"repeat={signals.repetition_rate:.2f} "
                    f"uncert={signals.uncertainty_density:.2f} "
                    f"drift={signals.goal_distance:.2f}",
                    flush=True
                )
            
            # Step 5: Auto-expand if score drops below 40
            if clm_score < 40:
                try:
                    task_state.task_tree = self.chunking_engine.auto_expand(
                        task_state.task_tree, 
                        clm_score
                    )
                except Exception as e:
                    logger.warning(f"Auto-expand failed: {e}. Continuing without expansion.")
            
            # Update state tracking
            self._current_score = clm_score
            self._current_zone = zone
            
            logger.info(f"CLM observe() complete: action={response.action}, zone={zone}, score={clm_score:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in observe(): {e}. Returning pass action to avoid disrupting agent loop.")
            # Graceful degradation: return pass action on any unexpected error
            return InterventionResponse(
                action="pass",
                clm_score=self._current_score,
                zone=self._current_zone
            )
    
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
    
    def get_history(self) -> list[dict]:
        """Return full step-by-step intervention log."""
        return self._history
    
    def summary(self) -> dict:
        """Return a human-readable performance summary."""
        if not self._history:
            return {"steps": 0, "message": "No observations yet."}
        
        actions = [e["action"] for e in self._history]
        zones = [e["zone"] for e in self._history]
        scores = [e["score"] for e in self._history]
        
        return {
            "steps": len(self._history),
            "avg_score": round(sum(scores) / len(scores), 2),
            "peak_score": round(max(scores), 2),
            "interventions": {
                "pass": actions.count("pass"),
                "patch": actions.count("patch"),
                "interrupt": actions.count("interrupt"),
            },
            "zone_distribution": {
                "Green": zones.count("Green"),
                "Amber": zones.count("Amber"),
                "Red": zones.count("Red"),
            },
            "total_compressed": sum(len(e["compressed_tasks"]) for e in self._history),
            "sidecar": self.get_sidecar_stats(),
        }
    
    def observe_raw(self, llm_output: str) -> InterventionResponse:
        """
        Minimal observe() — no TaskState required.
        
        Automatically builds and maintains an internal task tree from outputs.
        Perfect for simple agent loops where you just want CLM to work.
        
        Usage:
            clm = CLM(verbose=True)
            for step in range(max_steps):
                output = call_llm(prompt)
                result = clm.observe_raw(output)
                if result.action == "interrupt":
                    break
        """
        if not hasattr(self, '_auto_builder'):
            from clm.utils.auto_state import AutoStateBuilder
            self._auto_builder = AutoStateBuilder()
        
        self._auto_builder.observe(llm_output)
        task_state = self._auto_builder.get_state()
        return self.observe(llm_output, task_state)
    
    def reset_session(self) -> None:
        """Reset the internal auto-state builder for a new agent session."""
        if hasattr(self, '_auto_builder'):
            self._auto_builder.reset()
        self._history = []
        self._step = 0
        self._current_score = 0.0
        self._current_zone = "Green"
    
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
