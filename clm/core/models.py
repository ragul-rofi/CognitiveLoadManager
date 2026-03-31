"""Core data models for Cognitive Load Manager."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass
class Signals:
    """Normalized cognitive load signals (0-1 range)."""
    
    branching_factor: float  # Count of active concurrent sub-tasks
    repetition_rate: float   # Cosine similarity between last 3 reasoning steps
    uncertainty_density: float  # Hedged tokens per 500 output tokens
    goal_distance: float     # Cosine similarity between current sub-task and root intent
    
    def __post_init__(self):
        """Validate all signals are in 0-1 range."""
        for field_name, value in self.__dict__.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in range [0, 1], got {value}")


@dataclass
class TaskNode:
    """Node in the task tree hierarchy."""
    
    task_id: str
    parent_id: str | None
    description: str
    status: str  # "active", "compressed", "completed"
    depth: int = 0
    children: list[TaskNode] = field(default_factory=list)
    
    def is_leaf(self) -> bool:
        """Check if node has no children."""
        return len(self.children) == 0
    
    def compute_depth(self) -> int:
        """Compute depth from root (root has depth 0)."""
        if self.parent_id is None:
            return 0
        # Depth computed during tree traversal
        return self.depth


@dataclass
class TaskTree:
    """Hierarchical representation of active tasks."""
    
    root: TaskNode
    root_intent: str
    root_intent_embedding: list[float] | None = None
    
    def find_node(self, task_id: str) -> TaskNode | None:
        """Find node by task_id using BFS."""
        from collections import deque
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.task_id == task_id:
                return node
            queue.extend(node.children)
        return None
    
    def get_active_tasks(self) -> list[TaskNode]:
        """Return all nodes with status='active'."""
        active_tasks = []
        for node in self.traverse_dfs():
            if node.status == "active":
                active_tasks.append(node)
        return active_tasks
    
    def get_deepest_nodes(self) -> list[TaskNode]:
        """Return nodes with maximum depth."""
        # First compute all depths
        max_depth = 0
        for node in self.traverse_bfs():
            if node.parent_id is None:
                node.depth = 0
            else:
                parent = self.find_node(node.parent_id)
                if parent:
                    node.depth = parent.depth + 1
            max_depth = max(max_depth, node.depth)
        
        # Find all nodes at max depth
        deepest = []
        for node in self.traverse_dfs():
            if node.depth == max_depth:
                deepest.append(node)
        return deepest
    
    def traverse_dfs(self) -> Iterator[TaskNode]:
        """Depth-first traversal from root to leaves."""
        def _dfs(node: TaskNode) -> Iterator[TaskNode]:
            yield node
            for child in node.children:
                yield from _dfs(child)
        
        yield from _dfs(self.root)
    
    def traverse_bfs(self) -> Iterator[TaskNode]:
        """Breadth-first traversal from root to leaves."""
        from collections import deque
        
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)


@dataclass
class TaskState:
    """Current state passed to CLM.observe()."""
    
    task_tree: TaskTree
    current_task_id: str
    reasoning_history: list[str]  # Last 3 reasoning steps


@dataclass
class TaskChunk:
    """Compressed task stored in sidecar."""
    
    task_id: str
    parent_id: str | None
    summary: str  # ≤200 tokens
    full_detail: str
    clm_score_at_compression: float
    compressed_at: datetime
    status: str  # "compressed", "expanded"


@dataclass
class InterventionResponse:
    """Response from CLM.observe() to agent loop."""
    
    action: str  # "pass", "patch", "interrupt"
    context: str | None = None  # Patched context for "patch" action
    clarification: str | None = None  # Clarification request for "interrupt" action
    clm_score: float = 0.0
    zone: str = "Green"
    compressed_tasks: list[str] = field(default_factory=list)  # IDs of compressed tasks
