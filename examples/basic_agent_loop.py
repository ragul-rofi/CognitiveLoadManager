"""
Basic Agent Loop Integration Example

This example demonstrates how to integrate the Cognitive Load Manager (CLM)
into a simple LLM-based agent loop. It shows:
- Configuration setup
- CLM initialization
- observe() calls after each LLM response
- Handling all three intervention types (pass, patch, interrupt)
"""

import logging
from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode

# Configure logging to see CLM's internal operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def mock_llm_call(prompt: str) -> str:
    """
    Mock LLM call for demonstration purposes.
    In a real agent, this would call an actual LLM API.
    """
    # Simulate different types of responses
    if "clarify" in prompt.lower():
        return "I need to focus on implementing the authentication module first."
    elif "compressed" in prompt.lower():
        return "Continuing with the simplified task list..."
    else:
        return "I'm working on multiple sub-tasks: database schema, API endpoints, authentication, caching, logging, monitoring, and testing."


def build_sample_task_tree() -> TaskTree:
    """
    Build a sample task tree for demonstration.
    In a real agent, this would be constructed from the agent's planning.
    """
    # Create root task
    root = TaskNode(
        task_id="root",
        parent_id=None,
        description="Build a web application with user authentication",
        status="active",
        depth=0
    )
    
    # Create sub-tasks (simulating high branching factor)
    subtasks = [
        TaskNode("task-1", "root", "Design database schema", "active", 1),
        TaskNode("task-2", "root", "Implement REST API endpoints", "active", 1),
        TaskNode("task-3", "root", "Build authentication module", "active", 1),
        TaskNode("task-4", "root", "Add caching layer", "active", 1),
        TaskNode("task-5", "root", "Set up logging", "active", 1),
        TaskNode("task-6", "root", "Configure monitoring", "active", 1),
        TaskNode("task-7", "root", "Write integration tests", "active", 1),
    ]
    
    root.children = subtasks
    
    return TaskTree(
        root=root,
        root_intent="Build a web application with user authentication"
    )


def run_agent_loop_with_clm():
    """
    Demonstrate a simple agent loop with CLM integration.
    """
    logger.info("=== Starting Agent Loop with CLM ===\n")
    
    # Step 1: Configure CLM
    config = CLMConfig(
        branching_threshold=7,
        repetition_threshold=0.85,
        uncertainty_threshold=0.15,
        weights=[0.30, 0.25, 0.25, 0.20],
        green_max=40.0,
        amber_max=70.0,
        storage_type="sqlite",
        storage_params={"db_path": "clm_sidecar.db"}
    )
    
    # Step 2: Initialize CLM (using context manager for automatic cleanup)
    with CognitiveLoadManager(config) as clm:
        
        # Step 3: Build initial task state
        task_tree = build_sample_task_tree()
        reasoning_history = []
        
        # Simulate agent loop iterations
        for iteration in range(1, 4):
            logger.info(f"\n--- Iteration {iteration} ---")
            
            # Build current task state
            task_state = TaskState(
                task_tree=task_tree,
                current_task_id="task-3",  # Currently working on authentication
                reasoning_history=reasoning_history
            )
            
            # Simulate LLM call
            if iteration == 1:
                llm_output = mock_llm_call("What should I work on?")
            elif iteration == 2:
                llm_output = mock_llm_call("Continue with compressed tasks")
            else:
                llm_output = mock_llm_call("Please clarify the priority")
            
            logger.info(f"LLM Output: {llm_output}")
            
            # Step 4: Call CLM.observe() after each LLM response
            response = clm.observe(llm_output, task_state)
            
            # Step 5: Handle intervention based on action type
            logger.info(f"CLM Response: action={response.action}, zone={response.zone}, score={response.clm_score:.2f}")
            
            if response.action == "pass":
                # Green Zone: No intervention needed, continue normally
                logger.info("✓ Green Zone - Continuing normally")
                reasoning_history.append(llm_output)
                
            elif response.action == "patch":
                # Amber Zone: Replace context with compressed task tree
                logger.info("⚠ Amber Zone - Applying context patch")
                logger.info(f"Compressed tasks: {response.compressed_tasks}")
                
                # In a real agent, you would replace the context window with response.context
                # For this demo, we just log it
                if response.context:
                    logger.info(f"New context (first 200 chars): {response.context[:200]}...")
                
                reasoning_history.append(llm_output)
                
            elif response.action == "interrupt":
                # Red Zone: Request clarification from user
                logger.info("🛑 Red Zone - Requesting clarification")
                logger.info(f"Clarification needed: {response.clarification}")
                logger.info(f"Compressed tasks: {response.compressed_tasks}")
                
                # In a real agent, you would pause execution and request user input
                # For this demo, we simulate a clarification response
                user_clarification = "Focus on the authentication module first"
                logger.info(f"User clarification: {user_clarification}")
                
                # Update task tree based on clarification
                # (In real implementation, this would involve re-planning)
                reasoning_history = [user_clarification]  # Reset with clarification
            
            # Step 6: Check CLM metrics
            current_score = clm.get_score()
            current_zone = clm.get_zone()
            logger.info(f"Current CLM Score: {current_score:.2f}, Zone: {current_zone}")
            
            # Step 7: Check sidecar storage stats
            stats = clm.get_sidecar_stats()
            logger.info(f"Sidecar Stats: {stats['compressed_count']} compressed, {stats['expanded_count']} expanded")
    
    logger.info("\n=== Agent Loop Complete ===")


def demonstrate_custom_configuration():
    """
    Demonstrate custom CLM configuration for different agent types.
    """
    logger.info("\n=== Custom Configuration Example ===\n")
    
    # Example 1: More aggressive intervention (lower thresholds)
    aggressive_config = CLMConfig(
        branching_threshold=5,  # Trigger earlier with fewer branches
        green_max=30.0,  # Smaller green zone
        amber_max=60.0,  # Smaller amber zone
        weights=[0.40, 0.20, 0.20, 0.20]  # Higher weight on branching
    )
    
    logger.info("Aggressive Config:")
    logger.info(f"  - Branching threshold: {aggressive_config.branching_threshold}")
    logger.info(f"  - Green zone: 0-{aggressive_config.green_max}")
    logger.info(f"  - Amber zone: {aggressive_config.green_max}-{aggressive_config.amber_max}")
    logger.info(f"  - Red zone: {aggressive_config.amber_max}-100")
    
    # Example 2: More tolerant configuration (higher thresholds)
    tolerant_config = CLMConfig(
        branching_threshold=10,  # Allow more branches before triggering
        green_max=50.0,  # Larger green zone
        amber_max=80.0,  # Larger amber zone
        weights=[0.25, 0.25, 0.25, 0.25]  # Equal weights
    )
    
    logger.info("\nTolerant Config:")
    logger.info(f"  - Branching threshold: {tolerant_config.branching_threshold}")
    logger.info(f"  - Green zone: 0-{tolerant_config.green_max}")
    logger.info(f"  - Amber zone: {tolerant_config.green_max}-{tolerant_config.amber_max}")
    logger.info(f"  - Red zone: {tolerant_config.amber_max}-100")
    
    # Example 3: Custom hedged tokens for domain-specific uncertainty
    domain_config = CLMConfig(
        hedged_tokens=[
            "maybe", "perhaps", "possibly", "might", "could",
            "uncertain", "unclear", "probably", "likely", "seems",
            # Add domain-specific hedged tokens
            "approximately", "roughly", "estimate", "assume", "guess"
        ]
    )
    
    logger.info("\nDomain-Specific Config:")
    logger.info(f"  - Custom hedged tokens: {len(domain_config.hedged_tokens)} tokens")
    logger.info(f"  - Additional tokens: approximately, roughly, estimate, assume, guess")


if __name__ == "__main__":
    # Run the basic agent loop demonstration
    run_agent_loop_with_clm()
    
    # Show custom configuration examples
    demonstrate_custom_configuration()
