# CLM — Cognitive Load Manager

Real-time metacognitive middleware for LLM agents. Detects when your agent is cognitively overloaded and intervenes before it hallucinates, drifts, or crashes.

## Installation

```bash
# Lightweight install (numpy only, ~10MB)
pip install clm-plugin

# Full install with embedding support (~1.5GB)
pip install clm-plugin[embed]
```

## Quickstart

```python
from clm import CLM

clm = CLM(verbose=True)

# In your agent loop, add one line:
result = clm.observe_raw(llm_output)

if result.action == "interrupt":
    # CLM detected cognitive overload
    prompt = f"Clarification needed: {result.clarification}"
elif result.action == "patch":
    # Use compressed context in next call
    context = result.context
```

## What it monitors

CLM tracks 4 cognitive signals:

- **Branching** — too many tasks in flight
- **Repetition** — agent going in circles
- **Uncertainty** — excessive hedging language
- **Goal drift** — wandering from original intent

Combines these into a CLM score (0–100) and acts:

| Zone | Score | Action |
|------|-------|--------|
| Green | 0–40 | Pass through |
| Amber | 40–70 | Compress task branches, patch context |
| Red | 70–100 | Full compression + goal re-anchor |

## Configuration

```python
from clm import CLM, CLMConfig

config = CLMConfig(
    weights=[0.30, 0.25, 0.25, 0.20],  # [branching, repetition, uncertainty, goal_distance]
    green_max=40.0,
    amber_max=70.0,
    no_embed=False,  # Auto-set to True if sentence-transformers not installed
)

clm = CLM(config, verbose=True)
```

## Adapters

### LangChain

```python
from clm.adapters import CLMCallbackHandler

handler = CLMCallbackHandler(verbose=True)
agent.run("your task", callbacks=[handler])
```

### Loop Decorator

```python
from clm.adapters import CLMLoop

loop = CLMLoop(verbose=True)

@loop
def agent_step(prompt: str) -> str:
    return openai_client.chat(prompt)
```

## Observability

```python
clm.get_score()       # Current CLM score (0–100)
clm.get_zone()        # "Green" | "Amber" | "Red"
clm.get_history()     # Step-by-step intervention log
clm.summary()         # Session statistics
```

## Links

- **Documentation:** [GitHub README](https://github.com/ragul-rofi/CognitiveLoadManager#readme)
- **Source:** [GitHub Repository](https://github.com/ragul-rofi/CognitiveLoadManager)
- **Issues:** [Bug Reports](https://github.com/ragul-rofi/CognitiveLoadManager/issues)

## License

MIT
