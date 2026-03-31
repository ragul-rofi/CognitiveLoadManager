"""Action dispatcher for cognitive load interventions."""

import logging
from clm.core.models import TaskState, InterventionResponse, TaskNode, TaskTree
from clm.core.chunking_engine import ChunkingEngine

logger = logging.getLogger("clm.action_dispatcher")


class ActionDispatcher:
    """
    Selects and executes interventions based on CLM score and zone.
    
    Routes to zone-specific handlers:
    - Green Zone (0-40): No intervention
    - Amber Zone (40-70): Compress deepest branches
    - Red Zone (70-100): Full compression + anchor + clarification
    """
    
    def __init__(self, chunking_engine: ChunkingEngine):
        """
        Initialize with chunking engine for compression/anchoring operations.
        
        Args:
            chunking_engine: ChunkingEngine instance for task compression and anchoring
        """
        self.chunking_engine = chunking_engine
        self.amber_counter: int = 0
        self.red_counter: int = 0
    
    def dispatch(self, clm_score: float, zone: str, task_state: TaskState) -> InterventionResponse:
        """
        Determine intervention action based on zone.
        
        Green Zone (0-40): Return "pass" with no intervention
        Amber Zone (40-70): Compress deepest branches, return "patch" with context
        Red Zone (70-100): Full compression + anchor + clarification interrupt
        
        Args:
            clm_score: Current cognitive load score (0-100)
            zone: Zone classification ("Green", "Amber", or "Red")
            task_state: Current task state with task tree
            
        Returns:
            InterventionResponse with action, context, and clarification fields
        """
        logger.info(f"Dispatching intervention for zone={zone}, score={clm_score:.2f}")
        
        if zone == "Green":
            return self._handle_green(task_state, clm_score)
        elif zone == "Amber":
            return self._handle_amber(task_state, clm_score)
        else:  # Red zone
            return self._handle_red(task_state, clm_score)
    
    def _handle_green(self, task_state: TaskState, clm_score: float) -> InterventionResponse:
        """
        No intervention needed.
        
        Args:
            task_state: Current task state
            clm_score: Current CLM score
            
        Returns:
            InterventionResponse with action="pass"
        """
        logger.debug("Green zone: No intervention needed")
        return InterventionResponse(
            action="pass",
            clm_score=clm_score,
            zone="Green"
        )
    
    def _handle_amber(self, task_state: TaskState, clm_score: float) -> InterventionResponse:
        """
        Compress deepest sub-task branches, patch context.
        
        Args:
            task_state: Current task state
            clm_score: Current CLM score
            
        Returns:
            InterventionResponse with action="patch" and context_patch
        """
        logger.info("Amber zone: Compressing deepest branches")
        
        # Find deepest branches
        deepest_branches = self._find_deepest_branches(task_state.task_tree)
        logger.debug(f"Found {len(deepest_branches)} deepest branches to compress")
        
        compressed_ids = []
        
        # Compress each deepest branch
        for node in deepest_branches:
            try:
                summary_node = self.chunking_engine.compress(node, clm_score)
                
                # Update node in tree
                tree_node = task_state.task_tree.find_node(node.task_id)
                if tree_node:
                    tree_node.description = summary_node.description
                    tree_node.status = "compressed"
                    compressed_ids.append(node.task_id)
            except Exception as e:
                logger.warning(f"Failed to compress task {node.task_id}: {e}")
        
        # Generate context patch (serialized task tree)
        context_patch = self._serialize_task_tree(task_state.task_tree)
        
        logger.info(f"Amber zone intervention complete: compressed {len(compressed_ids)} tasks")
        
        return InterventionResponse(
            action="patch",
            context=context_patch,
            clm_score=clm_score,
            zone="Amber",
            compressed_tasks=compressed_ids
        )
    
    def _handle_red(self, task_state: TaskState, clm_score: float) -> InterventionResponse:
        """
        Full compression, anchor injection, clarification request.
        
        Args:
            task_state: Current task state
            clm_score: Current CLM score
            
        Returns:
            InterventionResponse with action="interrupt", context with anchor, and clarification
        """
        logger.warning("Red zone: Full compression and clarification interrupt")
        
        compressed_ids = []
        
        # Compress all active tasks except root
        for node in task_state.task_tree.traverse_dfs():
            if node.parent_id is not None and node.status == "active":
                try:
                    summary_node = self.chunking_engine.compress(node, clm_score)
                    
                    # Update node in tree
                    node.description = summary_node.description
                    node.status = "compressed"
                    compressed_ids.append(node.task_id)
                except Exception as e:
                    logger.warning(f"Failed to compress task {node.task_id}: {e}")
        
        # Generate anchor
        anchor = self.chunking_engine.anchor(task_state.task_tree.root_intent)
        logger.debug(f"Generated anchor: {anchor[:50]}...")
        
        # Generate context with anchor
        task_tree_serialized = self._serialize_task_tree(task_state.task_tree)
        context_patch = f"{anchor}\n\n{task_tree_serialized}"
        
        # Generate clarification request
        clarification = (
            "Cognitive load is very high. Please clarify: "
            "What is the most critical sub-task to focus on right now?"
        )
        
        logger.critical(f"Red zone intervention complete: compressed {len(compressed_ids)} tasks, requesting clarification")
        
        return InterventionResponse(
            action="interrupt",
            context=context_patch,
            clarification=clarification,
            clm_score=clm_score,
            zone="Red",
            compressed_tasks=compressed_ids
        )
    
    def _find_deepest_branches(self, task_tree: TaskTree) -> list[TaskNode]:
        """
        Identify leaf nodes at maximum depth for targeted compression.
        
        Algorithm:
        1. Compute depth for all nodes
        2. Find maximum depth
        3. Return all leaf nodes at maximum depth with status="active"
        
        Args:
            task_tree: Task tree to analyze
            
        Returns:
            List of leaf nodes at maximum depth
        """
        # Compute depths using BFS
        max_depth = 0
        for node in task_tree.traverse_bfs():
            if node.parent_id is None:
                node.depth = 0
            else:
                parent = task_tree.find_node(node.parent_id)
                if parent:
                    node.depth = parent.depth + 1
            max_depth = max(max_depth, node.depth)
        
        # Find all leaf nodes at max depth with status="active"
        deepest_branches = [
            node for node in task_tree.traverse_dfs()
            if node.depth == max_depth and node.is_leaf() and node.status == "active"
        ]
        
        return deepest_branches
    
    def _serialize_task_tree(self, task_tree: TaskTree) -> str:
        """
        Serialize task tree to string representation for context patching.
        
        Args:
            task_tree: Task tree to serialize
            
        Returns:
            String representation of task tree
        """
        lines = []
        
        def _serialize_node(node: TaskNode, indent: int = 0):
            """Recursively serialize node and children."""
            prefix = "  " * indent
            status_marker = "✓" if node.status == "completed" else "○" if node.status == "compressed" else "•"
            lines.append(f"{prefix}{status_marker} [{node.task_id}] {node.description}")
            
            for child in node.children:
                _serialize_node(child, indent + 1)
        
        _serialize_node(task_tree.root)
        
        return "\n".join(lines)
