"""Chunking engine for task compression, anchoring, and expansion."""

import re
import logging
from datetime import datetime
from typing import Optional, Callable

from clm.core.models import TaskNode, TaskTree, TaskChunk
from clm.storage.sidecar_store import SidecarStore
from clm.exceptions import ExpansionError, StorageError

logger = logging.getLogger("clm.chunking_engine")


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
            
        Raises:
            StorageError: If sidecar storage fails
        """
        logger.debug(f"Compressing task {task_node.task_id} with CLM score {clm_score}")
        
        try:
            # Generate summary (≤200 tokens)
            summary = self._generate_summary(task_node.description)
            logger.debug(f"Generated summary for task {task_node.task_id}: {len(summary.split())} tokens")
            
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
            logger.info(f"Compressed task {task_node.task_id} and stored in sidecar")
            
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
            
        except Exception as e:
            logger.error(f"Failed to compress task {task_node.task_id}: {e}")
            raise StorageError(f"Failed to compress task {task_node.task_id}: {e}") from e
    
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
        Improved extractive summarizer.
        Strategy:
        1. Extract all sentences
        2. Score each by: position (first/last weighted higher) +
           information density (unique word ratio)
        3. Take top-N sentences up to max_tokens
        4. Preserve original order
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
        
        if not sentences:
            words = text.split()
            return ' '.join(words[:max_tokens]) + ('...' if len(words) > max_tokens else '')
        
        if len(sentences) <= 3:
            summary = ' '.join(sentences)
            words = summary.split()
            return ' '.join(words[:max_tokens]) + ('...' if len(words) > max_tokens else '')
        
        # Score sentences
        def score(i, sent):
            words = sent.lower().split()
            if not words:
                return 0
            unique_ratio = len(set(words)) / len(words)  # info density
            position_bonus = 1.5 if i == 0 else (1.3 if i == len(sentences)-1 else 1.0)
            length_bonus = min(len(words) / 20, 1.0)  # reward substance
            return unique_ratio * position_bonus * length_bonus
        
        scored = sorted(enumerate(sentences), key=lambda x: score(x[0], x[1]), reverse=True)
        
        # Take top sentences up to max_tokens
        selected_indices = set()
        token_count = 0
        for i, sent in scored:
            sent_tokens = len(sent.split())
            if token_count + sent_tokens <= max_tokens:
                selected_indices.add(i)
                token_count += sent_tokens
            if token_count >= max_tokens * 0.85:
                break
        
        # Restore original order
        summary_sentences = [sentences[i] for i in sorted(selected_indices)]
        return ' '.join(summary_sentences)
    
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
            ExpansionError: If task_id not found in sidecar or task tree
        """
        logger.debug(f"Expanding task {task_id}")
        
        try:
            # Retrieve full detail from sidecar
            full_detail = self.sidecar_store.expand(task_id)
            
            if full_detail is None:
                error_msg = f"Task {task_id} not found in sidecar store"
                logger.error(error_msg)
                raise ExpansionError(error_msg)
            
            # Find compressed node in tree
            node = task_tree.find_node(task_id)
            
            if node is None:
                error_msg = f"Task {task_id} not found in task tree"
                logger.error(error_msg)
                raise ExpansionError(error_msg)
            
            # Restore full detail
            node.description = full_detail
            node.status = "active"
            logger.info(f"Expanded task {task_id} from sidecar")
            
            # Update sidecar status
            task_chunk = self.sidecar_store.get(task_id)
            if task_chunk:
                task_chunk.status = "expanded"
                self.sidecar_store.store(task_chunk)
                logger.debug(f"Updated sidecar status for task {task_id} to 'expanded'")
            
            return task_tree
            
        except ExpansionError:
            raise
        except Exception as e:
            error_msg = f"Failed to expand task {task_id}: {e}"
            logger.error(error_msg)
            raise ExpansionError(error_msg) from e
    
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
            logger.debug(f"Auto-expand skipped: CLM score {clm_score} >= 40")
            return task_tree
        
        logger.debug(f"Auto-expand triggered: CLM score {clm_score} < 40")
        
        try:
            # Find all compressed tasks in tree
            compressed_tasks = [
                node for node in task_tree.traverse_dfs()
                if node.status == "compressed"
            ]
            
            if not compressed_tasks:
                logger.debug("No compressed tasks to expand")
                return task_tree  # Nothing to expand
            
            # Get compression timestamps from sidecar
            task_chunks = []
            for node in compressed_tasks:
                chunk = self.sidecar_store.get(node.task_id)
                if chunk:
                    task_chunks.append(chunk)
            
            if not task_chunks:
                logger.warning("Compressed tasks found in tree but not in sidecar")
                return task_tree
            
            # Sort by compression time (most recent first)
            task_chunks.sort(key=lambda x: x.compressed_at, reverse=True)
            
            # Expand most recent
            most_recent = task_chunks[0]
            logger.info(f"Auto-expanding most recent compressed task: {most_recent.task_id}")
            task_tree = self.expand(most_recent.task_id, task_tree)
            
            return task_tree
            
        except Exception as e:
            logger.warning(f"Auto-expand failed: {e}. Continuing with compressed tree.")
            return task_tree  # Graceful degradation: return unchanged tree
