"""Unit tests for SidecarStore."""

import pytest
from datetime import datetime
from clm.storage import SidecarStore
from clm.core.models import TaskChunk


class TestSidecarStore:
    """Test suite for SidecarStore SQLite backend."""
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving task chunks."""
        store = SidecarStore()
        
        # Create a task chunk
        chunk = TaskChunk(
            task_id="task-1",
            parent_id="root",
            summary="Test summary",
            full_detail="This is the full detail of the task",
            clm_score_at_compression=65.5,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        # Store it
        task_id = store.store(chunk)
        assert task_id == "task-1"
        
        # Retrieve it
        retrieved = store.get("task-1")
        assert retrieved is not None
        assert retrieved.task_id == "task-1"
        assert retrieved.parent_id == "root"
        assert retrieved.summary == "Test summary"
        assert retrieved.full_detail == "This is the full detail of the task"
        assert retrieved.clm_score_at_compression == 65.5
        assert retrieved.status == "compressed"
        
        store.close()
    
    def test_list_children(self):
        """Test list_children with multiple children."""
        store = SidecarStore()
        
        # Create parent and children
        parent_chunk = TaskChunk(
            task_id="parent-1",
            parent_id=None,
            summary="Parent task",
            full_detail="Parent task detail",
            clm_score_at_compression=50.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        child1 = TaskChunk(
            task_id="child-1",
            parent_id="parent-1",
            summary="Child 1",
            full_detail="Child 1 detail",
            clm_score_at_compression=60.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        child2 = TaskChunk(
            task_id="child-2",
            parent_id="parent-1",
            summary="Child 2",
            full_detail="Child 2 detail",
            clm_score_at_compression=70.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        # Store all
        store.store(parent_chunk)
        store.store(child1)
        store.store(child2)
        
        # List children
        children = store.list_children("parent-1")
        assert len(children) == 2
        assert {c.task_id for c in children} == {"child-1", "child-2"}
        
        store.close()
    
    def test_expand(self):
        """Test expand with valid task_id."""
        store = SidecarStore()
        
        chunk = TaskChunk(
            task_id="task-expand",
            parent_id="root",
            summary="Short summary",
            full_detail="This is the complete full detail that should be retrieved",
            clm_score_at_compression=55.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        store.store(chunk)
        
        # Expand
        full_detail = store.expand("task-expand")
        assert full_detail == "This is the complete full detail that should be retrieved"
        
        store.close()
    
    def test_expand_missing_task(self):
        """Test expand with invalid task_id returns None."""
        store = SidecarStore()
        
        full_detail = store.expand("non-existent-task")
        assert full_detail is None
        
        store.close()
    
    def test_get_stats(self):
        """Test get_stats accuracy."""
        store = SidecarStore()
        
        # Initially empty
        stats = store.get_stats()
        assert stats["count"] == 0
        assert stats["compressed_count"] == 0
        
        # Add some chunks
        chunk1 = TaskChunk(
            task_id="task-1",
            parent_id=None,
            summary="Summary 1",
            full_detail="Detail 1",
            clm_score_at_compression=50.0,
            compressed_at=datetime.now(),
            status="compressed"
        )
        
        chunk2 = TaskChunk(
            task_id="task-2",
            parent_id=None,
            summary="Summary 2",
            full_detail="Detail 2",
            clm_score_at_compression=60.0,
            compressed_at=datetime.now(),
            status="expanded"
        )
        
        store.store(chunk1)
        store.store(chunk2)
        
        # Check stats
        stats = store.get_stats()
        assert stats["count"] == 2
        assert stats["compressed_count"] == 1
        assert stats["expanded_count"] == 1
        assert stats["total_size"] > 0
        
        store.close()
    
    def test_context_manager(self):
        """Test SidecarStore works as context manager."""
        with SidecarStore() as store:
            chunk = TaskChunk(
                task_id="task-ctx",
                parent_id=None,
                summary="Context test",
                full_detail="Testing context manager",
                clm_score_at_compression=45.0,
                compressed_at=datetime.now(),
                status="compressed"
            )
            store.store(chunk)
            retrieved = store.get("task-ctx")
            assert retrieved is not None
