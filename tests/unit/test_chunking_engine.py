"""Unit tests for ChunkingEngine."""

import pytest
from datetime import datetime

from clm.core.chunking_engine import ChunkingEngine
from clm.core.models import TaskNode, TaskTree, TaskChunk
from clm.storage.sidecar_store import SidecarStore
from clm.exceptions import ExpansionError


@pytest.fixture
def sidecar_store():
    """Create in-memory sidecar store for testing."""
    return SidecarStore(storage_type="sqlite", connection_params={"db_path": ":memory:"})


@pytest.fixture
def chunking_engine(sidecar_store):
    """Create chunking engine with in-memory storage."""
    return ChunkingEngine(sidecar_store=sidecar_store)


@pytest.fixture
def sample_task_node():
    """Create a sample task node for testing."""
    return TaskNode(
        task_id="task-1",
        parent_id="root",
        description="This is a detailed task description. It contains multiple sentences. The task involves processing data and generating reports.",
        status="active",
        depth=1,
        children=[]
    )


@pytest.fixture
def sample_task_tree():
    """Create a sample task tree for testing."""
    root = TaskNode(
        task_id="root",
        parent_id=None,
        description="Root task",
        status="active",
        depth=0,
        children=[]
    )
    
    child1 = TaskNode(
        task_id="task-1",
        parent_id="root",
        description="Child task 1 with detailed description.",
        status="active",
        depth=1,
        children=[]
    )
    
    child2 = TaskNode(
        task_id="task-2",
        parent_id="root",
        description="Child task 2 with another description.",
        status="compressed",
        depth=1,
        children=[]
    )
    
    root.children = [child1, child2]
    
    return TaskTree(
        root=root,
        root_intent="Complete the main objective"
    )


class TestChunkingEngineInit:
    """Test ChunkingEngine initialization."""
    
    def test_init_with_sidecar_store(self, sidecar_store):
        """Test initialization with sidecar store."""
        engine = ChunkingEngine(sidecar_store=sidecar_store)
        assert engine.sidecar_store == sidecar_store
        assert engine.summarizer is not None  # Uses default summarizer
    
    def test_init_with_summarizer(self, sidecar_store):
        """Test initialization with custom summarizer."""
        def custom_summarizer(text):
            return text[:50]
        
        engine = ChunkingEngine(sidecar_store=sidecar_store, summarizer=custom_summarizer)
        assert engine.summarizer == custom_summarizer


class TestCompress:
    """Test compress method."""
    
    def test_compress_creates_summary_node(self, chunking_engine, sample_task_node):
        """Test that compress creates a summary node."""
        clm_score = 55.0
        summary_node = chunking_engine.compress(sample_task_node, clm_score)
        
        assert summary_node.task_id == sample_task_node.task_id
        assert summary_node.parent_id == sample_task_node.parent_id
        assert summary_node.status == "compressed"
        assert summary_node.depth == sample_task_node.depth
        assert "[Full detail in sidecar:" in summary_node.description
    
    def test_compress_stores_in_sidecar(self, chunking_engine, sample_task_node):
        """Test that compress stores full detail in sidecar."""
        clm_score = 55.0
        chunking_engine.compress(sample_task_node, clm_score)
        
        # Verify stored in sidecar
        chunk = chunking_engine.sidecar_store.get(sample_task_node.task_id)
        assert chunk is not None
        assert chunk.task_id == sample_task_node.task_id
        assert chunk.full_detail == sample_task_node.description
        assert chunk.clm_score_at_compression == clm_score
        assert chunk.status == "compressed"
    
    def test_compress_preserves_children(self, chunking_engine):
        """Test that compress preserves tree structure."""
        parent = TaskNode(
            task_id="parent",
            parent_id=None,
            description="Parent task",
            status="active",
            depth=0,
            children=[]
        )
        
        child = TaskNode(
            task_id="child",
            parent_id="parent",
            description="Child task",
            status="active",
            depth=1,
            children=[]
        )
        
        parent.children = [child]
        
        summary_node = chunking_engine.compress(parent, 50.0)
        assert len(summary_node.children) == 1
        assert summary_node.children[0].task_id == "child"


class TestGenerateSummary:
    """Test _generate_summary method."""
    
    def test_generate_summary_single_sentence(self, chunking_engine):
        """Test summary generation for single sentence."""
        text = "This is a single sentence"
        summary = chunking_engine._generate_summary(text)
        assert summary == "This is a single sentence"
    
    def test_generate_summary_multiple_sentences(self, chunking_engine):
        """Test summary generation for multiple sentences."""
        text = "First sentence. Middle sentence. Last sentence."
        summary = chunking_engine._generate_summary(text)
        # The summarizer should include at least some of the sentences
        assert "First sentence" in summary or "Last sentence" in summary
        # For short text with 3 sentences, all should be included without truncation
        assert len(summary.split()) <= 200
    
    def test_generate_summary_respects_token_limit(self, chunking_engine):
        """Test that summary is truncated to 200 tokens."""
        # Create text with more than 200 tokens
        text = " ".join([f"word{i}" for i in range(250)])
        summary = chunking_engine._generate_summary(text)
        
        tokens = summary.split()
        assert len(tokens) <= 201  # 200 tokens + "..." counts as 1 token
    
    def test_generate_summary_empty_text(self, chunking_engine):
        """Test summary generation for empty text."""
        text = ""
        summary = chunking_engine._generate_summary(text)
        assert summary == ""


class TestAnchor:
    """Test anchor method."""
    
    def test_anchor_formats_root_intent(self, chunking_engine):
        """Test that anchor formats root intent correctly."""
        root_intent = "Complete the main objective"
        anchor = chunking_engine.anchor(root_intent)
        
        assert anchor.startswith("[ROOT INTENT]")
        assert "Complete the main objective" in anchor
    
    def test_anchor_respects_token_limit(self, chunking_engine):
        """Test that anchor is truncated to 100 tokens."""
        # Create root intent with more than 100 tokens
        root_intent = " ".join([f"word{i}" for i in range(150)])
        anchor = chunking_engine.anchor(root_intent)
        
        # Remove the "[ROOT INTENT]: " prefix for token counting
        content = anchor.replace("[ROOT INTENT]: ", "")
        tokens = content.split()
        assert len(tokens) <= 101  # 100 tokens + "..." counts as 1 token
    
    def test_anchor_short_intent(self, chunking_engine):
        """Test anchor with short root intent."""
        root_intent = "Short goal"
        anchor = chunking_engine.anchor(root_intent)
        
        assert anchor == "[ROOT INTENT] Short goal"
        assert "..." not in anchor


class TestExpand:
    """Test expand method."""
    
    def test_expand_restores_full_detail(self, chunking_engine, sample_task_tree, sidecar_store):
        """Test that expand restores full detail from sidecar."""
        # First compress a task
        task_node = sample_task_tree.root.children[0]
        original_description = task_node.description
        chunking_engine.compress(task_node, 55.0)
        
        # Update the tree node to compressed state
        task_node.status = "compressed"
        task_node.description = "Summary..."
        
        # Now expand
        updated_tree = chunking_engine.expand(task_node.task_id, sample_task_tree)
        
        # Verify restoration
        expanded_node = updated_tree.find_node(task_node.task_id)
        assert expanded_node.description == original_description
        assert expanded_node.status == "active"
    
    def test_expand_updates_sidecar_status(self, chunking_engine, sample_task_tree):
        """Test that expand updates sidecar status to expanded."""
        # First compress a task
        task_node = sample_task_tree.root.children[0]
        chunking_engine.compress(task_node, 55.0)
        
        # Update the tree node to compressed state
        task_node.status = "compressed"
        
        # Now expand
        chunking_engine.expand(task_node.task_id, sample_task_tree)
        
        # Verify sidecar status
        chunk = chunking_engine.sidecar_store.get(task_node.task_id)
        assert chunk.status == "expanded"
    
    def test_expand_raises_error_for_missing_task_in_sidecar(self, chunking_engine, sample_task_tree):
        """Test that expand raises error if task not in sidecar."""
        with pytest.raises(ExpansionError, match="not found in sidecar store"):
            chunking_engine.expand("nonexistent-task", sample_task_tree)
    
    def test_expand_raises_error_for_missing_task_in_tree(self, chunking_engine, sample_task_tree, sidecar_store):
        """Test that expand raises error if task not in tree."""
        # Store a chunk in sidecar but not in tree
        chunk = TaskChunk(
            task_id="orphan-task",
            parent_id="root",
            summary="Summary",
            full_detail="Full detail",
            clm_score_at_compression=50.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        sidecar_store.store(chunk)
        
        with pytest.raises(ExpansionError, match="not found in task tree"):
            chunking_engine.expand("orphan-task", sample_task_tree)


class TestAutoExpand:
    """Test auto_expand method."""
    
    def test_auto_expand_when_score_below_40(self, chunking_engine, sample_task_tree):
        """Test that auto_expand expands when score < 40."""
        # Compress a task first
        task_node = sample_task_tree.root.children[1]
        original_description = "Original description for task-2"
        task_node.description = original_description
        chunking_engine.compress(task_node, 55.0)
        task_node.status = "compressed"
        
        # Auto expand with low score
        updated_tree = chunking_engine.auto_expand(sample_task_tree, clm_score=35.0)
        
        # Verify expansion
        expanded_node = updated_tree.find_node(task_node.task_id)
        assert expanded_node.status == "active"
        assert expanded_node.description == original_description
    
    def test_auto_expand_no_action_when_score_above_40(self, chunking_engine, sample_task_tree):
        """Test that auto_expand does nothing when score >= 40."""
        # Compress a task first
        task_node = sample_task_tree.root.children[1]
        chunking_engine.compress(task_node, 55.0)
        task_node.status = "compressed"
        task_node.description = "Compressed summary"
        
        # Auto expand with high score
        updated_tree = chunking_engine.auto_expand(sample_task_tree, clm_score=50.0)
        
        # Verify no expansion
        node = updated_tree.find_node(task_node.task_id)
        assert node.status == "compressed"
        assert "Compressed summary" in node.description
    
    def test_auto_expand_no_compressed_tasks(self, chunking_engine, sample_task_tree):
        """Test that auto_expand handles no compressed tasks gracefully."""
        # All tasks are active
        updated_tree = chunking_engine.auto_expand(sample_task_tree, clm_score=35.0)
        
        # Should return unchanged tree
        assert updated_tree == sample_task_tree
    
    def test_auto_expand_expands_most_recent(self, chunking_engine, sample_task_tree):
        """Test that auto_expand expands the most recently compressed task."""
        # Add another child for testing
        child3 = TaskNode(
            task_id="task-3",
            parent_id="root",
            description="Task 3 description",
            status="active",
            depth=1,
            children=[]
        )
        sample_task_tree.root.children.append(child3)
        
        # Compress two tasks at different times
        import time
        
        task1 = sample_task_tree.root.children[0]
        chunking_engine.compress(task1, 55.0)
        task1.status = "compressed"
        
        time.sleep(0.01)  # Small delay to ensure different timestamps
        
        task3 = sample_task_tree.root.children[2]
        original_desc_3 = task3.description
        chunking_engine.compress(task3, 60.0)
        task3.status = "compressed"
        
        # Auto expand - should expand task3 (most recent)
        updated_tree = chunking_engine.auto_expand(sample_task_tree, clm_score=35.0)
        
        # Verify task3 is expanded
        node3 = updated_tree.find_node("task-3")
        assert node3.status == "active"
        assert node3.description == original_desc_3
        
        # Verify task1 is still compressed
        node1 = updated_tree.find_node("task-1")
        assert node1.status == "compressed"
