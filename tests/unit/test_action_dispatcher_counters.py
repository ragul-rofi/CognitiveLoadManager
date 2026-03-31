"""Unit tests for ActionDispatcher counter logic (Task 3)."""

import pytest
from clm.core.action_dispatcher import ActionDispatcher
from clm.core.chunking_engine import ChunkingEngine
from clm.core.models import TaskState, TaskTree, TaskNode
from clm.storage.sidecar_store import SidecarStore


@pytest.fixture
def chunking_engine():
    """Create a ChunkingEngine instance for testing."""
    sidecar = SidecarStore()
    return ChunkingEngine(sidecar_store=sidecar)


@pytest.fixture
def action_dispatcher(chunking_engine):
    """Create an ActionDispatcher instance for testing."""
    return ActionDispatcher(chunking_engine)


@pytest.fixture
def simple_task_state():
    """Create a simple task state for testing."""
    root = TaskNode(
        task_id="root",
        parent_id=None,
        description="Root task",
        status="active"
    )
    task_tree = TaskTree(root=root, root_intent="Complete the root task")
    return TaskState(
        task_tree=task_tree,
        current_task_id="root",
        reasoning_history=["Step 1", "Step 2", "Step 3"]
    )


class TestCounterInitialization:
    """Test counter initialization (Subtask 2.1)."""
    
    def test_counters_initialize_to_zero(self, action_dispatcher):
        """Test that amber_counter and red_counter initialize to 0."""
        assert action_dispatcher.amber_counter == 0
        assert action_dispatcher.red_counter == 0


class TestAmberEscalation:
    """Test Amber escalation logic (Subtask 3.1)."""
    
    def test_amber_escalation_at_threshold(self, action_dispatcher, simple_task_state):
        """Test that Amber escalates to Red after 3 consecutive triggers."""
        # First Amber trigger
        response1 = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert response1.action == "patch"
        assert action_dispatcher.amber_counter == 1
        
        # Second Amber trigger
        response2 = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert response2.action == "patch"
        assert action_dispatcher.amber_counter == 2
        
        # Third Amber trigger - should escalate to Red
        response3 = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert response3.action == "interrupt"  # Red zone behavior
        assert response3.zone == "Red"
        assert action_dispatcher.amber_counter == 0  # Reset after escalation
        assert action_dispatcher.red_counter == 1  # Incremented as Red


class TestRedAbort:
    """Test Red abort logic (Subtask 3.2)."""
    
    def test_red_abort_at_threshold(self, action_dispatcher, simple_task_state):
        """Test that Red triggers abort after 5 consecutive triggers."""
        # Trigger Red 4 times
        for i in range(4):
            response = action_dispatcher.dispatch(80.0, "Red", simple_task_state)
            assert response.action == "interrupt"
            assert action_dispatcher.red_counter == i + 1
        
        # Fifth Red trigger - should abort
        response5 = action_dispatcher.dispatch(80.0, "Red", simple_task_state)
        assert response5.action == "abort"
        assert action_dispatcher.red_counter == 5
        assert "Cognitive load remains critically high" in response5.clarification


class TestCounterUpdates:
    """Test counter increment and reset logic (Subtask 3.3)."""
    
    def test_amber_counter_increments(self, action_dispatcher, simple_task_state):
        """Test that amber_counter increments on Amber zone."""
        response = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert action_dispatcher.amber_counter == 1
        assert action_dispatcher.red_counter == 0
    
    def test_red_counter_increments(self, action_dispatcher, simple_task_state):
        """Test that red_counter increments on Red zone."""
        response = action_dispatcher.dispatch(80.0, "Red", simple_task_state)
        assert action_dispatcher.red_counter == 1
        assert action_dispatcher.amber_counter == 0
    
    def test_green_resets_both_counters(self, action_dispatcher, simple_task_state):
        """Test that Green zone resets both counters."""
        # Set up some counter values
        action_dispatcher.amber_counter = 2
        action_dispatcher.red_counter = 3
        
        # Dispatch Green
        response = action_dispatcher.dispatch(20.0, "Green", simple_task_state)
        assert action_dispatcher.amber_counter == 0
        assert action_dispatcher.red_counter == 0
    
    def test_amber_resets_red_counter(self, action_dispatcher, simple_task_state):
        """Test that Amber zone resets red_counter."""
        # Set up red_counter
        action_dispatcher.red_counter = 3
        
        # Dispatch Amber
        response = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert action_dispatcher.amber_counter == 1
        assert action_dispatcher.red_counter == 0
    
    def test_red_resets_amber_counter(self, action_dispatcher, simple_task_state):
        """Test that Red zone resets amber_counter."""
        # Set up amber_counter
        action_dispatcher.amber_counter = 2
        
        # Dispatch Red
        response = action_dispatcher.dispatch(80.0, "Red", simple_task_state)
        assert action_dispatcher.red_counter == 1
        assert action_dispatcher.amber_counter == 0


class TestCounterMetadata:
    """Test counter metadata injection (Subtask 3.4)."""
    
    def test_metadata_included_in_response(self, action_dispatcher, simple_task_state):
        """Test that counter values are included in InterventionResponse."""
        # Set up counters
        action_dispatcher.amber_counter = 1
        action_dispatcher.red_counter = 2
        
        # Dispatch and check metadata
        response = action_dispatcher.dispatch(20.0, "Green", simple_task_state)
        assert response.amber_counter == 0  # Reset by Green
        assert response.red_counter == 0  # Reset by Green
    
    def test_metadata_reflects_current_state(self, action_dispatcher, simple_task_state):
        """Test that metadata reflects counter state after dispatch."""
        # First Amber
        response1 = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert response1.amber_counter == 1
        assert response1.red_counter == 0
        
        # Second Amber
        response2 = action_dispatcher.dispatch(50.0, "Amber", simple_task_state)
        assert response2.amber_counter == 2
        assert response2.red_counter == 0


class TestAbortClarification:
    """Test abort clarification text (Subtask 3.2)."""
    
    def test_abort_clarification_text(self, action_dispatcher, simple_task_state):
        """Test that abort action includes correct clarification text."""
        # Set red_counter to 4 to trigger abort on next Red
        action_dispatcher.red_counter = 4
        
        response = action_dispatcher.dispatch(80.0, "Red", simple_task_state)
        assert response.action == "abort"
        assert response.clarification is not None
        assert "Cognitive load remains critically high after multiple interventions" in response.clarification
        assert "Consider simplifying the task or breaking it into smaller independent goals" in response.clarification
