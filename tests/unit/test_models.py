"""Unit tests for core data models."""

import pytest
from datetime import datetime
from clm.core.models import (
    Signals,
    TaskNode,
    TaskTree,
    TaskState,
    TaskChunk,
    InterventionResponse
)


class TestSignals:
    """Test Signals dataclass."""
    
    def test_valid_signals(self):
        """Test creating signals with valid values."""
        signals = Signals(
            branching_factor=0.5,
            repetition_rate=0.7,
            uncertainty_density=0.3,
            goal_distance=0.2
        )
        assert signals.branching_factor == 0.5
        assert signals.repetition_rate == 0.7
        assert signals.uncertainty_density == 0.3
        assert signals.goal_distance == 0.2
    
    def test_signals_at_boundaries(self):
        """Test signals at 0 and 1 boundaries."""
        signals_zero = Signals(
            branching_factor=0.0,
            repetition_rate=0.0,
            uncertainty_density=0.0,
            goal_distance=0.0
        )
        assert signals_zero.branching_factor == 0.0
        
        signals_one = Signals(
            branching_factor=1.0,
            repetition_rate=1.0,
            uncertainty_density=1.0,
            goal_distance=1.0
        )
        assert signals_one.branching_factor == 1.0
    
    def test_signals_out_of_range_raises_error(self):
        """Test that signals outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="branching_factor must be in range"):
            Signals(
                branching_factor=1.5,
                repetition_rate=0.5,
                uncertainty_density=0.5,
                goal_distance=0.5
            )
        
        with pytest.raises(ValueError, match="goal_distance must be in range"):
            Signals(
                branching_factor=0.5,
                repetition_rate=0.5,
                uncertainty_density=0.5,
                goal_distance=-0.1
            )


class TestTaskNode:
    """Test TaskNode dataclass."""
    
    def test_create_task_node(self):
        """Test creating a task node."""
        node = TaskNode(
            task_id="task-1",
            parent_id=None,
            description="Root task",
            status="active"
        )
        assert node.task_id == "task-1"
        assert node.parent_id is None
        assert node.description == "Root task"
        assert node.status == "active"
        assert node.depth == 0
        assert node.children == []
    
    def test_is_leaf_with_no_children(self):
        """Test is_leaf returns True for node with no children."""
        node = TaskNode(
            task_id="task-1",
            parent_id=None,
            description="Leaf task",
            status="active"
        )
        assert node.is_leaf() is True
    
    def test_is_leaf_with_children(self):
        """Test is_leaf returns False for node with children."""
        parent = TaskNode(
            task_id="task-1",
            parent_id=None,
            description="Parent task",
            status="active"
        )
        child = TaskNode(
            task_id="task-2",
            parent_id="task-1",
            description="Child task",
            status="active"
        )
        parent.children.append(child)
        
        assert parent.is_leaf() is False
        assert child.is_leaf() is True
    
    def test_compute_depth(self):
        """Test compute_depth method."""
        root = TaskNode(
            task_id="task-1",
            parent_id=None,
            description="Root",
            status="active"
        )
        assert root.compute_depth() == 0
        
        child = TaskNode(
            task_id="task-2",
            parent_id="task-1",
            description="Child",
            status="active",
            depth=1
        )
        assert child.compute_depth() == 1


class TestTaskTree:
    """Test TaskTree dataclass."""
    
    def test_create_task_tree(self):
        """Test creating a task tree."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        tree = TaskTree(
            root=root,
            root_intent="Complete the project"
        )
        assert tree.root == root
        assert tree.root_intent == "Complete the project"
        assert tree.root_intent_embedding is None
    
    def test_find_node_root(self):
        """Test finding the root node."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        tree = TaskTree(root=root, root_intent="Test")
        
        found = tree.find_node("root")
        assert found is not None
        assert found.task_id == "root"
    
    def test_find_node_child(self):
        """Test finding a child node."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        child = TaskNode(
            task_id="child-1",
            parent_id="root",
            description="Child task",
            status="active"
        )
        root.children.append(child)
        tree = TaskTree(root=root, root_intent="Test")
        
        found = tree.find_node("child-1")
        assert found is not None
        assert found.task_id == "child-1"
    
    def test_find_node_not_found(self):
        """Test finding a non-existent node returns None."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        tree = TaskTree(root=root, root_intent="Test")
        
        found = tree.find_node("non-existent")
        assert found is None
    
    def test_get_active_tasks(self):
        """Test getting all active tasks."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        child1 = TaskNode(
            task_id="child-1",
            parent_id="root",
            description="Active child",
            status="active"
        )
        child2 = TaskNode(
            task_id="child-2",
            parent_id="root",
            description="Compressed child",
            status="compressed"
        )
        root.children.extend([child1, child2])
        tree = TaskTree(root=root, root_intent="Test")
        
        active = tree.get_active_tasks()
        assert len(active) == 2
        assert root in active
        assert child1 in active
        assert child2 not in active
    
    def test_get_deepest_nodes(self):
        """Test getting deepest nodes in tree."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root task",
            status="active"
        )
        child1 = TaskNode(
            task_id="child-1",
            parent_id="root",
            description="Child 1",
            status="active"
        )
        grandchild = TaskNode(
            task_id="grandchild-1",
            parent_id="child-1",
            description="Grandchild",
            status="active"
        )
        child1.children.append(grandchild)
        root.children.append(child1)
        tree = TaskTree(root=root, root_intent="Test")
        
        deepest = tree.get_deepest_nodes()
        assert len(deepest) == 1
        assert deepest[0].task_id == "grandchild-1"
        assert deepest[0].depth == 2
    
    def test_traverse_dfs(self):
        """Test depth-first traversal."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root",
            status="active"
        )
        child1 = TaskNode(
            task_id="child-1",
            parent_id="root",
            description="Child 1",
            status="active"
        )
        child2 = TaskNode(
            task_id="child-2",
            parent_id="root",
            description="Child 2",
            status="active"
        )
        root.children.extend([child1, child2])
        tree = TaskTree(root=root, root_intent="Test")
        
        nodes = list(tree.traverse_dfs())
        assert len(nodes) == 3
        assert nodes[0].task_id == "root"
        assert nodes[1].task_id == "child-1"
        assert nodes[2].task_id == "child-2"
    
    def test_traverse_bfs(self):
        """Test breadth-first traversal."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root",
            status="active"
        )
        child1 = TaskNode(
            task_id="child-1",
            parent_id="root",
            description="Child 1",
            status="active"
        )
        grandchild = TaskNode(
            task_id="grandchild-1",
            parent_id="child-1",
            description="Grandchild",
            status="active"
        )
        child1.children.append(grandchild)
        root.children.append(child1)
        tree = TaskTree(root=root, root_intent="Test")
        
        nodes = list(tree.traverse_bfs())
        assert len(nodes) == 3
        assert nodes[0].task_id == "root"
        assert nodes[1].task_id == "child-1"
        assert nodes[2].task_id == "grandchild-1"


class TestTaskState:
    """Test TaskState dataclass."""
    
    def test_create_task_state(self):
        """Test creating a task state."""
        root = TaskNode(
            task_id="root",
            parent_id=None,
            description="Root",
            status="active"
        )
        tree = TaskTree(root=root, root_intent="Test")
        
        state = TaskState(
            task_tree=tree,
            current_task_id="root",
            reasoning_history=["step1", "step2", "step3"]
        )
        
        assert state.task_tree == tree
        assert state.current_task_id == "root"
        assert len(state.reasoning_history) == 3


class TestTaskChunk:
    """Test TaskChunk dataclass."""
    
    def test_create_task_chunk(self):
        """Test creating a task chunk."""
        now = datetime.now()
        chunk = TaskChunk(
            task_id="task-1",
            parent_id="root",
            summary="Short summary",
            full_detail="Full detailed description",
            clm_score_at_compression=65.5,
            compressed_at=now,
            status="compressed"
        )
        
        assert chunk.task_id == "task-1"
        assert chunk.parent_id == "root"
        assert chunk.summary == "Short summary"
        assert chunk.full_detail == "Full detailed description"
        assert chunk.clm_score_at_compression == 65.5
        assert chunk.compressed_at == now
        assert chunk.status == "compressed"


class TestInterventionResponse:
    """Test InterventionResponse dataclass."""
    
    def test_create_pass_response(self):
        """Test creating a pass intervention response."""
        response = InterventionResponse(
            action="pass",
            clm_score=25.0,
            zone="Green"
        )
        
        assert response.action == "pass"
        assert response.context is None
        assert response.clarification is None
        assert response.clm_score == 25.0
        assert response.zone == "Green"
        assert response.compressed_tasks == []
    
    def test_create_patch_response(self):
        """Test creating a patch intervention response."""
        response = InterventionResponse(
            action="patch",
            context="Compressed task tree context",
            clm_score=55.0,
            zone="Amber",
            compressed_tasks=["task-1", "task-2"]
        )
        
        assert response.action == "patch"
        assert response.context == "Compressed task tree context"
        assert response.clarification is None
        assert response.clm_score == 55.0
        assert response.zone == "Amber"
        assert len(response.compressed_tasks) == 2
    
    def test_create_interrupt_response(self):
        """Test creating an interrupt intervention response."""
        response = InterventionResponse(
            action="interrupt",
            context="Compressed context with anchor",
            clarification="Please clarify the most critical sub-task",
            clm_score=85.0,
            zone="Red",
            compressed_tasks=["task-1", "task-2", "task-3"]
        )
        
        assert response.action == "interrupt"
        assert response.context == "Compressed context with anchor"
        assert response.clarification == "Please clarify the most critical sub-task"
        assert response.clm_score == 85.0
        assert response.zone == "Red"
        assert len(response.compressed_tasks) == 3
