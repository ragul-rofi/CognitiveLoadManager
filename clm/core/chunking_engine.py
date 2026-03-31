"""Chunking engine for task compression, anchoring, and expansion."""

from datetime import datetime
from typing import Optional, Callable

from clm.core.models import TaskNode, TaskTree, TaskChunk
from clm.storage.sidecar_store import SidecarStore


class ChunkingEngine:
    """
    Manages task compression, anchoring, and expansion operations.
    
    Compresses task nodes to summaries, stores full detail in sidecar,
    and expands compressed tasks when cognitive load decreases.
    """
    
    def __init__(self, sidecar_store: SidecarStore, summarizer: Optional[Callable[[str, int], str]] = None):
        """
        Initialize with storage backend and summarization capability.
        
        Args:
            sidecar_store: Storage backend for compressed task details
            summarizer: Optional custom summarization function (text, max_tokens) -> summary
                       If None, uses default extractive summarization
        """
        self.sidecar_store = sidecar_store
        self.summarizer = summarizer or self._default_summarizer
    
    def compress(self, task_node: TaskNode, clm_score: float) -> TaskNode:
        """
        Compress task node to summary, store full detail in sidecar.
        
        Args:
            task_node: Full task node to compress
            clm_score: Current CLM score at compression time
            
        Returns:
            Summary node (≤200 tokens) with reference to sidecar storage
        """
        # Generate summary (≤200 tokens)
        summary = self._generate_summary(task_node.description)
        
        # Create task chunk for storage
        task_chunk = TaskChunk(
            task_id=task_node.task_id,
            parent_id=task_node.parent_id,
            summary=summary,
            full_detail=task_node.description,
            clm_score_at_compression=clm_score,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        # Store in sidecar
        self.sidecar_store.store(task_chunk)
        
        # Create summary node with sidecar reference
        summary_description = f"{summary}\n[Full detail in sidecar: {task_node.task_id}]"
        
        summary_node = TaskNode(
            task_id=task_node.task_id,
            parent_id=task_node.parent_id,
            description=summary_description,
            status="compressed",
            depth=task_node.depth,
            children=task_node.children  # Preserve tree structure
        )
        
        return summary_node
    
    def _generate_summary(self, full_detail: str, max_tokens: int = 200) -> str:
        """
        Create ≤200 token summary with sidecar reference.
        
        Args:
            full_detail: Full task description to summarize
            max_tokens: Maximum tokens in summary (default 200)
            
        Returns:
            Concise summary preserving key entities and actions
        """
        return self.summarizer(full_detail, max_tokens)
    
    def _default_summarizer(self, text: str, max_tokens: int) -> str:
        """
        Default extractive summarization strategy.
        
        Strategy:
        1. Extract first and last sentences (often contain key info)
        2. Truncate to max_tokens if needed
        
        Args:
            text: Text to summarize
            max_tokens: Maximum tokens in summary
            
        Returns:
            Summary string
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text[:max_tokens * 5]  # Rough estimate: 5 chars per token
        
        # Take first sentence and last sentence
        if len(sentences) <= 2:
            summary = '. '.join(sentences) + '.'
        else:
            summary = f"{sentences[0]}. ... {sentences[-1]}."
        
        # Truncate to max_tokens (rough estimate: 5 chars per token)
        tokens = summary.split()
        if len(tokens) > max_tokens:
            summary = ' '.join(tokens[:max_tokens]) + "..."
        
        return summary
    
    def anchor(self, root_intent: str, max_tokens: int = 100) -> str:
        """
        Format root intent as ≤100 token prefix.
        
        Args:
            root_intent: Original top-level goal
            max_tokens: Maximum tokens in anchor (default 100)
            
        Returns:
            Formatted anchor string for injection
        """
        # Format as persistent prefix
        anchor_text = f"[ROOT INTENT] {root_intent}"
        
        # Truncate to max_tokens if needed
        tokens = anchor_text.split()
        if len(tokens) > max_tokens:
            anchor_text = ' '.join(tokens[:max_tokens]) + "..."
        
        return anchor_text
    
    def expand(self, task_id: str, task_tree: TaskTree) -> TaskTree:
        """
        Retrieve full detail from sidecar and restore in task tree.
        
        Args:
            task_id: ID of compressed task to expand
            task_tree: Current task tree
            
        Returns:
            Updated task tree with expanded node
            
        Raises:
            ValueError: If task_id not found in sidecar or task tree
        """
        # Retrieve full detail from sidecar
        full_detail = self.sidecar_store.expand(task_id)
        
        if full_detail is None:
            raise ValueError(f"Task {task_id} not found in sidecar store")
        
        # Find compressed node in tree
        node = task_tree.find_node(task_id)
        
        if node is None:
            raise ValueError(f"Task {task_id} not found in task tree")
        
        # Restore full detail
        node.description = full_detail
        node.status = "active"
        
        # Update sidecar status
        task_chunk = self.sidecar_store.get(task_id)
        if task_chunk:
            task_chunk.status = "expanded"
            self.sidecar_store.store(task_chunk)
        
        return task_tree
    
    def auto_expand(self, task_tree: TaskTree, clm_score: float) -> TaskTree:
        """
        Automatically expand most recently compressed task when score < 40.
        
        Args:
            task_tree: Current task tree
            clm_score: Current CLM score
            
        Returns:
            Updated task tree (expanded if conditions met, unchanged otherwise)
        """
        # Only expand if score drops below 40
        if clm_score >= 40:
            return task_tree
        
        # Find all compressed tasks in tree
        compressed_tasks = [
            node for node in task_tree.traverse_dfs()
            if node.status == "compressed"
        ]
        
        if not compressed_tasks:
            return task_tree  # Nothing to expand
        
        # Get compression timestamps from sidecar
        task_chunks = []
        for node in compressed_tasks:
            chunk = self.sidecar_store.get(node.task_id)
            if chunk:
                task_chunks.append(chunk)
        
        if not task_chunks:
            return task_tree
        
        # Sort by compression time (most recent first)
        task_chunks.sort(key=lambda x: x.compressed_at, reverse=True)
        
        # Expand most recent
        most_recent = task_chunks[0]
        task_tree = self.expand(most_recent.task_id, task_tree)
        
        return task_tree
