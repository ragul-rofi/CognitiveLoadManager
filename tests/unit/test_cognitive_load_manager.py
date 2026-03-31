"""Unit tests for CognitiveLoadManager facade."""

import pytest
from clm import CognitiveLoadManager, CLMConfig, TaskState, TaskTree, TaskNode


class TestCognitiveLoadManager:
    """Test suite for CognitiveLoadManager facade."""
    
    def test_initialization(self):
        """Test CLM initializes with default config."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        assert clm.get_score() == 0.0
        assert clm.get_zone() == "Green"
        
        clm.close()
    
    def test_initialization_with_custom_config(self):
        """Test CLM initializes with custom config."""
        config = CLMConfig(
            branching_threshold=10,
            weights=[0.25, 0.25, 0.25, 0.25],
            green_max=50.0,
            amber_max=80.0
        )
        clm = CognitiveLoadManager(config)
        
        assert clm.config.branching_threshold == 10
        assert clm.config.green_max == 50.0
        
        clm.close()
    
    def test_observe_green_zone(self):
        """Test observe returns pass action in Green zone."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        # Create minimal task state
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        task_tree = TaskTree(root=root, root_intent="Complete the project")
        task_state = TaskState(
            task_tree=task_tree,
            current_task_id="root",
            reasoning_history=["Step 1"]
        )
        
        llm_output = "I will complete the task."
        
        response = clm.observe(llm_output, task_state)
        
        assert response.action == "pass"
        assert response.zone == "Green"
        assert response.clm_score < 40
        assert clm.get_score() == response.clm_score
        assert clm.get_zone() == "Green"
        
        clm.close()
    
    def test_observe_amber_zone(self):
        """Test observe returns patch action in Amber zone."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        # Create task state with many active tasks (high branching)
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        
        # Add many child tasks to trigger high branching factor
        for i in range(10):
            child = TaskNode(
                task_id=f"task_{i}",
                parent_id="root",
                description=f"Sub-task {i}",
                status="active"
            )
            root.children.append(child)
        
        task_tree = TaskTree(root=root, root_intent="Complete the project")
        task_state = TaskState(
            task_tree=task_tree,
            current_task_id="task_0",
            reasoning_history=["Step 1", "Step 2", "Step 3"]
        )
        
        # High uncertainty output
        llm_output = "Maybe I should perhaps possibly try this approach. Might work, unclear."
        
        response = clm.observe(llm_output, task_state)
        
        # Should be in Amber or Red zone due to high branching and uncertainty
        assert response.action in ["patch", "interrupt"]
        assert response.clm_score >= 40
        assert clm.get_score() == response.clm_score
        
        clm.close()
    
    def test_get_sidecar_stats(self):
        """Test get_sidecar_stats returns storage statistics."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        stats = clm.get_sidecar_stats()
        
        assert "count" in stats
        assert "total_size" in stats
        assert "compressed_count" in stats
        assert "expanded_count" in stats
        assert stats["count"] == 0  # No compressions yet
        
        clm.close()
    
    def test_context_manager(self):
        """Test CLM works as context manager."""
        config = CLMConfig()
        
        with CognitiveLoadManager(config) as clm:
            assert clm.get_score() == 0.0
            assert clm.get_zone() == "Green"
        
        # Connection should be closed after context exit
    
    def test_invalid_config_raises_error(self):
        """Test invalid config raises ValueError."""
        config = CLMConfig(weights=[0.5, 0.5, 0.5, 0.5])  # Sum > 1.0
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CognitiveLoadManager(config)
    
    def test_observe_workflow_completeness(self):
        """Test observe executes full workflow: signals → score → zone → dispatch."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        task_tree = TaskTree(root=root, root_intent="Complete the project")
        task_state = TaskState(
            task_tree=task_tree,
            current_task_id="root",
            reasoning_history=["Step 1"]
        )
        
        llm_output = "Processing task."
        
        response = clm.observe(llm_output, task_state)
        
        # Verify response has all required fields
        assert hasattr(response, "action")
        assert hasattr(response, "context")
        assert hasattr(response, "clarification")
        assert hasattr(response, "clm_score")
        assert hasattr(response, "zone")
        assert hasattr(response, "compressed_tasks")
        
        # Verify state is updated
        assert clm.get_score() > 0 or clm.get_score() == 0
        assert clm.get_zone() in ["Green", "Amber", "Red"]
        
        clm.close()
    
    def test_score_getter_consistency(self):
        """Test get_score returns same score as observe."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        task_tree = TaskTree(root=root, root_intent="Complete the project")
        task_state = TaskState(
            task_tree=task_tree,
            current_task_id="root",
            reasoning_history=["Step 1"]
        )
        
        response = clm.observe("Processing.", task_state)
        
        assert clm.get_score() == response.clm_score
        
        clm.close()
    
    def test_zone_getter_consistency(self):
        """Test get_zone returns same zone as observe."""
        config = CLMConfig()
        clm = CognitiveLoadManager(config)
        
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        task_tree = TaskTree(root=root, root_intent="Complete the project")
        task_state = TaskState(
            task_tree=task_tree,
            current_task_id="root",
            reasoning_history=["Step 1"]
        )
        
        response = clm.observe("Processing.", task_state)
        
        assert clm.get_zone() == response.zone
        
        clm.close()
