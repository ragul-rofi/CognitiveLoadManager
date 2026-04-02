# CLM — Cognitive Load Manager

Real-time metacognitive middleware for LLM agents. Detects when your agent is cognitively overloaded and intervenes before it hallucinates, drifts, or crashes.

```bash
pip install clm-agent
```

## Quickstart — 3 lines

```python
from clm import CLM

clm = CLM(verbose=True)

# In your agent loop, replace nothing — just add one line:
result = clm.observe_raw(llm_output)
```

That's it. CLM automatically builds its own internal task tree from your agent's outputs.

## What it does

CLM wraps your agent loop and monitors 4 cognitive signals after every LLM call:

- **Branching** — how many tasks are in flight simultaneously
- **Repetition** — is the agent going in circles
- **Uncertainty** — is the agent hedging and guessing
- **Goal drift** — has the agent wandered from the original intent

It combines these into a single CLM score (0–100) and acts:

| Zone | Score | Action |
|------|-------|--------|
| Green | 0–40 | Pass through — no intervention |
| Amber | 40–70 | Compress deep task branches, patch context |
| Red | 70–100 | Full compression + goal re-anchor + clarification request |

## Integration patterns

### Important: Context Patch Behavior

⚠️ **response.context replaces only your task plan section, not your full conversation history**

When CLM returns `action="patch"`, the `response.context` field contains a compressed representation of your task tree. You should inject this into the task planning portion of your prompt, not replace your entire conversation history.

**Correct usage:**
```python
# Maintain conversation history, update only task section
conversation_history = [...]  # Your full conversation
task_section = result.context if result.action == "patch" else current_task_plan

prompt = f"""
Conversation so far:
{conversation_history}

Current task structure:
{task_section}

Continue working on the task.
"""
```

**Incorrect usage:**
```python
# DON'T DO THIS - overwrites entire conversation
if result.action == "patch":
    prompt = result.context  # ❌ Loses all conversation history
```

### Pattern 1 — Minimal (observe_raw)
No task state construction needed. Just feed outputs.

```python
from clm import CLM

clm = CLM(verbose=True)

while not done:
    output = call_llm(prompt)
    result = clm.observe_raw(output)
    
    if result.action == "interrupt":
        prompt = f"Clarification needed: {result.clarification}"
    elif result.action == "patch":
        context = result.context  # use compressed context in next call

print(clm.summary())
```

### Pattern 2 — LangChain (one line)

```python
from clm.adapters import CLMCallbackHandler

handler = CLMCallbackHandler(verbose=True)
agent.run("your task", callbacks=[handler])

print(handler.clm.summary())
```

### Pattern 3 — Decorator (raw loop)

```python
from clm.adapters import CLMLoop

loop = CLMLoop(verbose=True)

@loop
def agent_step(prompt: str) -> str:
    return openai_client.chat(prompt)

# Call normally — CLM wraps every call
for i in range(max_steps):
    output = agent_step(current_prompt)
    if loop.clm.get_zone() == "Red":
        break
```

### Pattern 4 — Full control (manual TaskState)

```python
from clm import CLM, CLMConfig
from clm.core.models import TaskState, TaskTree, TaskNode

clm = CLM(CLMConfig(verbose_signals=True), verbose=True)

task_state = TaskState(task_tree=your_tree, ...)
result = clm.observe(llm_output, task_state)
```

## Observability

```python
clm.get_score()       # current CLM score (0–100)
clm.get_zone()        # "Green" | "Amber" | "Red"
clm.get_history()     # full step-by-step intervention log
clm.summary()         # aggregate stats for the session
clm.get_sidecar_stats()  # compressed task storage stats
```

## No internet? No GPU? Use no_embed mode

```python
from clm import CLM, CLMConfig

clm = CLM(CLMConfig(no_embed=True))  # keyword-based signals, no model download
```

## Configuration

```python
from clm import CLMConfig

config = CLMConfig(
    branching_threshold=7,    # active tasks before normalising to 1.0
    repetition_threshold=0.85,
    uncertainty_threshold=0.15,
    weights=[0.30, 0.25, 0.25, 0.20],  # must sum to 1.0
    green_max=40.0,
    amber_max=70.0,
    no_embed=False,           # set True to skip model download
    storage_type="sqlite",
    storage_params={"db_path": "clm.db"},  # omit for in-memory
)
```

**Note:** The sidecar database (default: `clm_sidecar.db`) stores compressed task state and should not be committed to version control. It's already excluded in the default `.gitignore` pattern (`*.db`).

### Tuning Weights for Your Domain

⚠️ **Default weights are informed heuristics, not empirically validated. Tune them for your domain using `CLMConfig(weights=[...])`.**

The default weights `[0.30, 0.25, 0.25, 0.20]` (branching, repetition, uncertainty, goal_distance) are based on reasoning about general agent behavior, not empirical validation across domains.

**How to tune:**

1. **Start with defaults** and observe your agent's behavior with `verbose=True`
2. **Identify false positives**: If CLM interrupts when your agent is working correctly, reduce the weight of the signal that triggered the intervention
3. **Identify false negatives**: If CLM doesn't intervene when your agent is struggling, increase the weight of the signal that should have triggered intervention

**Example: Reducing false positives from branching**

If your agent legitimately needs to track many parallel tasks (e.g., data pipeline orchestration), but CLM keeps interrupting:

```python
# Default: branching weight = 0.30
config = CLMConfig(weights=[0.15, 0.30, 0.30, 0.25])  # Reduce branching to 0.15
```

**Example: Increasing sensitivity to goal drift**

If your agent frequently wanders off-task but CLM doesn't catch it:

```python
# Default: goal_distance weight = 0.20
config = CLMConfig(weights=[0.25, 0.20, 0.20, 0.35])  # Increase goal_distance to 0.35
```

Remember: weights must sum to 1.0.

## Domain-Specific Configuration

CLM's default weights are tuned for general-purpose agent tasks. Different domains benefit from different signal priorities:

### Medical Diagnosis Assistant

Prioritize goal distance (staying on diagnostic protocol) and minimize false interruptions:

```python
from clm import CLM, CLMConfig

config = CLMConfig(
    weights=[0.20, 0.20, 0.15, 0.45],  # Heavy weight on goal_distance
    green_max=50.0,  # Higher tolerance before intervention
    amber_max=75.0,
    branching_threshold=5,  # Medical protocols are often sequential
)

clm = CLM(config, verbose=True)
```

**Rationale:** Medical diagnosis requires strict adherence to diagnostic protocols. Goal drift is the most critical signal, while branching is less concerning since medical workflows are often linear.

### Legal Document Analysis

Prioritize repetition detection (circular reasoning) and uncertainty (hedging language):

```python
config = CLMConfig(
    weights=[0.15, 0.35, 0.35, 0.15],  # Heavy weight on repetition and uncertainty
    repetition_threshold=0.75,  # Lower threshold for detecting circular reasoning
    uncertainty_threshold=0.20,  # Higher tolerance for legal hedging language
)

clm = CLM(config, verbose=True)
```

**Rationale:** Legal analysis must avoid circular reasoning and excessive hedging. Repetition and uncertainty are critical signals, while branching (considering multiple legal precedents) is expected behavior.

### Voice Assistant

Prioritize branching (context switching) and goal distance (staying on user intent):

```python
config = CLMConfig(
    weights=[0.40, 0.15, 0.15, 0.30],  # Heavy weight on branching and goal_distance
    branching_threshold=3,  # Voice interactions should stay focused
    green_max=35.0,  # Lower tolerance for intervention
)

clm = CLM(config, verbose=True)
```

**Rationale:** Voice assistants must maintain tight focus on user intent and avoid context switching. Branching and goal distance are critical, while repetition and uncertainty are less concerning in conversational contexts.

## Architecture

5 layers, each independently testable:

```
Agent loop output
      ↓
Signal Collector   — extracts 4 cognitive signals
      ↓
CLM Scorer         — weighted score → zone classification
      ↓
Action Dispatcher  — routes to Green / Amber / Red handler
      ↓
Chunking Engine    — compress · anchor · expand
      ↓
Sidecar Store      — SQLite persistence for compressed tasks
```

## License

MIT
