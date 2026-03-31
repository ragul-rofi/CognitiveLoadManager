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
