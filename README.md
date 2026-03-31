# Cognitive Load Manager (CLM) v1.0

A metacognitive middleware layer for LLM-based agent loops that monitors cognitive load in real-time, detects overload conditions, and intervenes through task compression, goal anchoring, and clarification requests.

## Overview

The Cognitive Load Manager (CLM) wraps around agent execution loops to:
- **Monitor** cognitive load signals after every LLM call
- **Detect** overload conditions using normalized scoring (0-100)
- **Intervene** through adaptive strategies based on load zones (Green/Amber/Red)
- **Preserve** information through sidecar storage during compression
- **Recover** by auto-expanding tasks when cognitive load decreases

## Installation

### Requirements

- Python 3.10+
- sentence-transformers >= 2.2.0
- numpy >= 1.24.0
- pytest >= 7.4.0 (for testing)
- hypothesis >= 6.90.0 (for property-based testing)

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd cognitive-load-manager

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode

# 1. Configure CLM
config = CLMConfig()

# 2. Initialize CLM
clm = CognitiveLoadManager(config)

# 3. In your agent loop, after each LLM call:
response = clm.observe(llm_output, task_state)

# 4. Handle intervention
if response.action == "pass":
    # Green Zone: Continue normally
    continue
elif response.action == "patch":
    # Amber Zone: Replace context with compressed task tree
    context = response.context
elif response.action == "interrupt":
    # Red Zone: Request clarification from user
    print(response.clarification)
```

## Configuration

### Default Configuration

```python
from clm import CLMConfig

config = CLMConfig(
    # Signal thresholds
    branching_threshold=7,        # Max active sub-tasks before normalization
    repetition_threshold=0.85,    # Cosine similarity threshold for repetition
    uncertainty_threshold=0.15,   # Hedged tokens per 500 tokens threshold
    
    # Scoring weights [branching, repetition, uncertainty, goal_distance]
    weights=[0.30, 0.25, 0.25, 0.20],
    
    # Zone boundaries
    green_max=40.0,   # Green zone: 0-40
    amber_max=70.0,   # Amber zone: 40-70, Red zone: 70-100
    
    # Storage configuration
    storage_type="sqlite",
    storage_params={"db_path": "clm_sidecar.db"},
    
    # Embedding model
    embedding_model="all-MiniLM-L6-v2",
    
    # Hedged tokens for uncertainty detection
    hedged_tokens=[
        "maybe", "perhaps", "possibly", "might", "could",
        "uncertain", "unclear", "probably", "likely", "seems"
    ]
)
```

### Configuration Parameters

#### Signal Thresholds

- **branching_threshold** (int, default: 7): Number of active concurrent sub-tasks that triggers normalization to 1.0
- **repetition_threshold** (float, default: 0.85): Cosine similarity threshold for detecting repetitive reasoning
- **uncertainty_threshold** (float, default: 0.15): Hedged token density threshold (per 500 tokens)

#### Scoring Weights

- **weights** (list[float], default: [0.30, 0.25, 0.25, 0.20]): Weights for [branching_factor, repetition_rate, uncertainty_density, goal_distance]. Must sum to 1.0 (±0.01 tolerance).

#### Zone Boundaries

- **green_max** (float, default: 40.0): Upper bound for Green zone (observation only)
- **amber_max** (float, default: 70.0): Upper bound for Amber zone (soft intervention)
- Red zone is implicitly 70.0-100.0 (full intervention)

#### Storage Configuration

- **storage_type** (str, default: "sqlite"): Storage backend type ("sqlite" or "redis")
- **storage_params** (dict, default: {}): Backend-specific connection parameters
  - For SQLite: `{"db_path": "path/to/db.sqlite"}`
  - For Redis: `{"host": "localhost", "port": 6379, "db": 0}`

#### Embedding Configuration

- **embedding_model** (str, default: "all-MiniLM-L6-v2"): sentence-transformers model for similarity calculations

#### Hedged Tokens

- **hedged_tokens** (list[str]): List of tokens indicating uncertainty in LLM outputs

### Custom Configuration Examples

#### Aggressive Intervention (Lower Thresholds)

```python
config = CLMConfig(
    branching_threshold=5,   # Trigger earlier
    green_max=30.0,          # Smaller green zone
    amber_max=60.0,          # Smaller amber zone
    weights=[0.40, 0.20, 0.20, 0.20]  # Higher weight on branching
)
```

#### Tolerant Configuration (Higher Thresholds)

```python
config = CLMConfig(
    branching_threshold=10,  # Allow more branches
    green_max=50.0,          # Larger green zone
    amber_max=80.0,          # Larger amber zone
    weights=[0.25, 0.25, 0.25, 0.25]  # Equal weights
)
```

#### Domain-Specific Hedged Tokens

```python
config = CLMConfig(
    hedged_tokens=[
        "maybe", "perhaps", "possibly", "might", "could",
        "uncertain", "unclear", "probably", "likely", "seems",
        # Add domain-specific tokens
        "approximately", "roughly", "estimate", "assume", "guess"
    ]
)
```

## Usage

### Basic Agent Loop Integration

```python
import logging
from clm import CognitiveLoadManager, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize CLM
config = CLMConfig()
clm = CognitiveLoadManager(config)

# Build task tree (from your agent's planning)
task_tree = TaskTree(
    root=TaskNode(
        task_id="root",
        parent_id=None,
        description="Build a web application",
        status="active"
    ),
    root_intent="Build a web application with user authentication"
)

# Agent loop
reasoning_history = []
for iteration in range(max_iterations):
    # Your LLM call
    llm_output = call_llm(prompt)
    
    # Build task state
    task_state = TaskState(
        task_tree=task_tree,
        current_task_id="current-task-id",
        reasoning_history=reasoning_history[-3:]  # Last 3 steps
    )
    
    # CLM observation
    response = clm.observe(llm_output, task_state)
    
    # Handle intervention
    if response.action == "pass":
        # Continue normally
        reasoning_history.append(llm_output)
        
    elif response.action == "patch":
        # Replace context with compressed task tree
        context = response.context
        reasoning_history.append(llm_output)
        
    elif response.action == "interrupt":
        # Request clarification
        user_input = request_clarification(response.clarification)
        reasoning_history = [user_input]  # Reset with clarification

# Cleanup
clm.close()
```

### Using Context Manager

```python
with CognitiveLoadManager(config) as clm:
    # Your agent loop here
    response = clm.observe(llm_output, task_state)
    # ... handle response
# Automatic cleanup on exit
```

### Intervention Types

#### Green Zone (Score 0-40): Pass Action

No intervention needed. Agent continues normally.

```python
response = clm.observe(llm_output, task_state)

if response.action == "pass":
    print(f"✓ Green Zone (score: {response.clm_score:.2f})")
    print("Continuing normally...")
```

#### Amber Zone (Score 40-70): Patch Action

Soft intervention: Compress deepest task branches and patch context.

```python
response = clm.observe(llm_output, task_state)

if response.action == "patch":
    print(f"⚠ Amber Zone (score: {response.clm_score:.2f})")
    print(f"Compressed tasks: {response.compressed_tasks}")
    
    # Replace context window with compressed task tree
    context = response.context
    
    # Continue with patched context
    next_prompt = build_prompt(context)
```

#### Red Zone (Score 70-100): Interrupt Action

Full intervention: Compress all tasks, inject goal anchor, request clarification.

```python
response = clm.observe(llm_output, task_state)

if response.action == "interrupt":
    print(f"🛑 Red Zone (score: {response.clm_score:.2f})")
    print(f"Compressed tasks: {response.compressed_tasks}")
    
    # Request clarification from user
    print(f"Clarification needed: {response.clarification}")
    user_input = input("Your response: ")
    
    # Reset reasoning with clarification
    reasoning_history = [user_input]
    
    # Context includes goal anchor
    context = response.context
```

### Monitoring CLM Metrics

```python
# Get current CLM score
score = clm.get_score()
print(f"Current CLM Score: {score:.2f}")

# Get current zone
zone = clm.get_zone()
print(f"Current Zone: {zone}")

# Get sidecar storage statistics
stats = clm.get_sidecar_stats()
print(f"Compressed tasks: {stats['compressed_count']}")
print(f"Expanded tasks: {stats['expanded_count']}")
print(f"Total stored: {stats['count']}")
```

### Auto-Expansion

CLM automatically expands the most recently compressed task when the score drops below 40:

```python
# High load: tasks get compressed
response1 = clm.observe(high_load_output, task_state)
# response1.action == "patch", tasks compressed

# Load decreases: most recent task auto-expands
response2 = clm.observe(low_load_output, task_state)
# Most recently compressed task is automatically restored
```

## Architecture

### Components

- **Signal Collector**: Extracts cognitive load signals (branching, repetition, uncertainty, goal distance)
- **CLM Scorer**: Computes weighted score and classifies zones
- **Chunking Engine**: Compresses, anchors, and expands task representations
- **Action Dispatcher**: Selects and executes interventions based on zones
- **Sidecar Store**: Persistent storage for compressed task details (SQLite/Redis)

### Cognitive Load Signals

1. **Branching Factor**: Count of active concurrent sub-tasks (normalized by threshold)
2. **Repetition Rate**: Cosine similarity between last 3 reasoning steps
3. **Uncertainty Density**: Frequency of hedged tokens per 500 output tokens
4. **Goal Distance**: Cosine similarity between current sub-task and root intent

### CLM Score Formula

```
CLM Score = 100 × (w1×branching + w2×repetition + w3×uncertainty + w4×goal_distance)
```

Default weights: [0.30, 0.25, 0.25, 0.20]

## Examples

See `examples/basic_agent_loop.py` for a complete working example demonstrating:
- Configuration setup
- Agent loop integration
- Handling all three intervention types
- Custom configuration examples

Run the example:

```bash
python examples/basic_agent_loop.py
```

## Testing

### Run All Tests

```bash
# Run all tests (unit + property + integration)
pytest tests/

# Run with coverage
pytest tests/ --cov=clm --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/property/      # Property-based tests only
pytest tests/integration/   # Integration tests only
```

### Hypothesis Profiles

```bash
# Default profile (100 examples)
pytest tests/property/

# CI profile (500 examples)
pytest tests/property/ --hypothesis-profile=ci

# Debug profile (10 examples)
pytest tests/property/ --hypothesis-profile=debug
```

## Error Handling

CLM implements graceful degradation to avoid disrupting agent loops:

- **Signal extraction failures**: Return "pass" action with score 0
- **Storage failures**: Return "pass" action, log warning
- **Embedding failures**: Fall back to simple text similarity
- **Unexpected errors**: Return "pass" action, log error

```python
# CLM will never crash your agent loop
response = clm.observe(llm_output, task_state)
# Always returns a valid InterventionResponse
```

## Logging

Configure logging to see CLM's internal operations:

```python
import logging

# Set CLM logging level
logging.basicConfig(level=logging.INFO)

# Or configure specific loggers
logging.getLogger("clm").setLevel(logging.DEBUG)
logging.getLogger("clm.signal_collector").setLevel(logging.INFO)
```

Log levels:
- **DEBUG**: Detailed signal extraction and computation steps
- **INFO**: High-level operations (observe, compress, expand)
- **WARNING**: Graceful degradation events
- **ERROR**: Component failures

## Performance Considerations

- **Embedding generation**: Cached for root intent, computed on-demand for sub-tasks
- **Storage**: SQLite for local development, Redis for production/distributed systems
- **Signal extraction**: O(n) where n = number of active tasks
- **Compression**: O(1) per task, stores full detail in sidecar
- **Expansion**: O(1) retrieval from sidecar

## Limitations

- Requires structured task tree representation from agent
- Embedding model (all-MiniLM-L6-v2) requires ~100MB memory
- SQLite storage is single-process (use Redis for multi-process)
- Summary generation uses simple extractive approach (can be enhanced with LLM-based summarization)

## Contributing

Contributions are welcome! Please ensure:
- All tests pass: `pytest tests/`
- Code follows existing style
- New features include unit and property tests
- Documentation is updated

## License

[Your License Here]

## Citation

If you use CLM in your research, please cite:

```bibtex
@software{cognitive_load_manager,
  title = {Cognitive Load Manager: Metacognitive Middleware for LLM Agents},
  author = {[Your Name]},
  year = {2024},
  version = {1.0}
}
```

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
