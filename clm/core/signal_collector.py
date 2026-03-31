"""Signal extraction for cognitive load monitoring."""

import logging
from clm.core.models import Signals, TaskState
from clm.utils.embeddings import embed, cosine_similarity
from clm.exceptions import EmbeddingError, ValidationError

logger = logging.getLogger("clm.signal_collector")


class SignalCollector:
    """
    Extracts cognitive load signals from LLM outputs and task state.
    
    Signals extracted:
    - branching_factor: Count of active concurrent sub-tasks (normalized)
    - repetition_rate: Cosine similarity between recent reasoning steps
    - uncertainty_density: Frequency of hedged language in outputs
    - goal_distance: Semantic distance from root intent
    """
    
    def __init__(self, branching_threshold: int = 7, 
                 repetition_threshold: float = 0.85,
                 uncertainty_threshold: float = 0.15,
                 hedged_tokens: list[str] | None = None,
                 no_embed: bool = False):
        """
        Initialize SignalCollector with thresholds and hedged token list.
        
        Args:
            branching_threshold: Max active tasks before normalization (default 7)
            repetition_threshold: Threshold for repetition detection (default 0.85)
            uncertainty_threshold: Threshold for uncertainty density (default 0.15)
            hedged_tokens: List of hedged tokens to detect uncertainty
            no_embed: If True, skip embeddings and use keyword-based fallback
        """
        self.branching_threshold = branching_threshold
        self.repetition_threshold = repetition_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.no_embed = no_embed
        
        # Default hedged tokens if not provided
        self.hedged_tokens = hedged_tokens or [
            "maybe", "perhaps", "possibly", "might", "could",
            "uncertain", "unclear", "probably", "likely", "seems"
        ]
        
        # Convert to lowercase for case-insensitive matching
        self.hedged_tokens_lower = [token.lower() for token in self.hedged_tokens]
    
    def extract_signals(self, llm_output: str, task_state: TaskState) -> Signals:
        """
        Extract all cognitive load signals from LLM output and task state.
        
        Args:
            llm_output: Raw text output from LLM
            task_state: Current task tree and reasoning history
            
        Returns:
            Signals object with normalized values (0-1) for all four signals
            
        Raises:
            ValidationError: If task_state is invalid
        """
        logger.debug("Extracting cognitive load signals")
        
        try:
            branching_factor = self._compute_branching_factor(task_state)
            repetition_rate = self._compute_repetition_rate(task_state.reasoning_history)
            uncertainty_density = self._compute_uncertainty_density(llm_output)
            goal_distance = self._compute_goal_distance(task_state)
            
            signals = Signals(
                branching_factor=branching_factor,
                repetition_rate=repetition_rate,
                uncertainty_density=uncertainty_density,
                goal_distance=goal_distance
            )
            
            logger.debug(f"Extracted signals: branching={branching_factor:.3f}, repetition={repetition_rate:.3f}, uncertainty={uncertainty_density:.3f}, goal_distance={goal_distance:.3f}")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to extract signals: {e}")
            raise ValidationError(f"Failed to extract signals: {e}") from e
    
    def _compute_branching_factor(self, task_state: TaskState) -> float:
        """
        Count active concurrent sub-tasks and normalize to [0, 1].
        
        Formula: min(active_task_count / branching_threshold, 1.0)
        
        Args:
            task_state: Current task state with task tree
            
        Returns:
            Normalized branching factor in [0, 1]
        """
        active_tasks = task_state.task_tree.get_active_tasks()
        active_count = len(active_tasks)
        
        # Normalize by threshold, cap at 1.0
        branching_factor = min(active_count / self.branching_threshold, 1.0)
        
        return branching_factor
    
    def _compute_repetition_rate(self, reasoning_history: list[str]) -> float:
        """
        Compute cosine similarity between last 3 reasoning steps.
        
        Takes the maximum pairwise similarity between consecutive steps
        in the reasoning history.
        
        Args:
            reasoning_history: List of recent reasoning steps (last 3)
            
        Returns:
            Maximum cosine similarity in [0, 1], or 0.0 if insufficient history
        """
        if len(reasoning_history) < 2:
            logger.debug("Insufficient reasoning history for repetition rate")
            return 0.0
        
        try:
            # Take last 3 steps
            recent_steps = reasoning_history[-3:]
            
            if self.no_embed:
                # Fallback: keyword Jaccard similarity
                similarities = []
                for i in range(len(recent_steps) - 1):
                    sim = self._jaccard_similarity(recent_steps[i], recent_steps[i + 1])
                    similarities.append(sim)
                
                if similarities:
                    return max(similarities)
                return 0.0
            
            # Generate embeddings for each step
            embeddings = [embed(step) for step in recent_steps]
            
            # Compute pairwise similarities between consecutive steps
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(embeddings[i], embeddings[i + 1])
                similarities.append(sim)
            
            # Return maximum similarity (highest repetition)
            if similarities:
                max_sim = max(similarities)
                # Cosine similarity can be negative, clamp to [0, 1]
                return max(0.0, min(1.0, max_sim))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to compute repetition rate, using default 0.0: {e}")
            return 0.0  # Graceful degradation
    
    def _compute_uncertainty_density(self, llm_output: str) -> float:
        """
        Count hedged tokens per 500 output tokens and normalize.
        
        Formula: min((hedged_count / token_count) * 500 / uncertainty_threshold, 1.0)
        
        Args:
            llm_output: Raw LLM output text
            
        Returns:
            Normalized uncertainty density in [0, 1]
        """
        if not llm_output:
            return 0.0
        
        # Tokenize (simple whitespace split)
        tokens = llm_output.lower().split()
        token_count = len(tokens)
        
        if token_count == 0:
            return 0.0
        
        # Count hedged tokens (case-insensitive)
        hedged_count = sum(1 for token in tokens if token in self.hedged_tokens_lower)
        
        # Normalize per 500 tokens
        uncertainty_density_raw = (hedged_count / token_count) * 500
        
        # Normalize by threshold, cap at 1.0
        uncertainty_density = min(uncertainty_density_raw / self.uncertainty_threshold, 1.0)
        
        return uncertainty_density
    
    def _compute_goal_distance(self, task_state: TaskState) -> float:
        """
        Compute cosine similarity between current sub-task and root intent.
        
        Returns semantic distance (1 - similarity) to measure how far
        the current task has drifted from the original goal.
        
        Args:
            task_state: Current task state with task tree and current task ID
            
        Returns:
            Goal distance in [0, 1], where 0 = aligned, 1 = completely diverged
        """
        try:
            # Find current task node
            current_task = task_state.task_tree.find_node(task_state.current_task_id)
            
            if not current_task:
                logger.warning(f"Current task {task_state.current_task_id} not found in tree")
                return 0.0
            
            if self.no_embed:
                # Fallback: keyword Jaccard similarity
                goal_similarity = self._jaccard_similarity(
                    current_task.description,
                    task_state.task_tree.root_intent
                )
                goal_distance = 1.0 - goal_similarity
                return max(0.0, min(1.0, goal_distance))
            
            # Get or compute root intent embedding
            if task_state.task_tree.root_intent_embedding is None:
                task_state.task_tree.root_intent_embedding = embed(
                    task_state.task_tree.root_intent
                )
            
            # Compute current task embedding
            current_embedding = embed(current_task.description)
            
            # Compute similarity
            goal_similarity = cosine_similarity(
                current_embedding,
                task_state.task_tree.root_intent_embedding
            )
            
            # Convert similarity to distance (1 - similarity)
            # Clamp to [0, 1] range
            goal_distance = 1.0 - goal_similarity
            goal_distance = max(0.0, min(1.0, goal_distance))
            
            return goal_distance
            
        except Exception as e:
            logger.warning(f"Failed to compute goal distance, using default 0.0: {e}")
            return 0.0  # Graceful degradation
    
    def _jaccard_similarity(self, a: str, b: str) -> float:
        """
        Compute Jaccard similarity between two texts using keyword overlap.
        
        Args:
            a: First text
            b: Second text
            
        Returns:
            Jaccard similarity in [0, 1]
        """
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
