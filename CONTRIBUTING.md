# Contributing to CLM

CLM is an open research project. Contributions, bug reports, and real-world usage reports are the most valuable thing you can give right now.

## What we need most

1. **Real-world usage reports** — Did CLM fire when it shouldn't? Did it miss an overload? Open an issue with your domain, agent type, and what happened.
2. **Domain-specific weight tuning** — Found weights that work well for medical, legal, financial, or other domains? Submit a PR with a config example.
3. **Bug reports** — Especially integration issues with specific agent frameworks.
4. **Adapter contributions** — CrewAI, AutoGen, LlamaIndex, smolagents adapters welcome.

## Getting started

```bash
git clone https://github.com/ragul-rofi/CognitiveLoadManager
cd CognitiveLoadManager
pip install -e ".[dev]"
pytest tests/unit/ -q          # fast, no internet needed
pytest tests/integration/ -q   # also offline-safe
```

## Running tests

```bash
pytest tests/unit/        # unit tests — always offline
pytest tests/integration/ # integration tests — offline-safe (uses no_embed=True)
pytest tests/ --cov=clm   # with coverage
```

## Principles

- CLM must never crash an agent loop. Every failure path must return a valid InterventionResponse.
- Every new public method needs a docstring.
- New signals or intervention types go through an issue first — discuss before building.
- The core scoring formula is intentionally tunable. Don't hardcode domain assumptions.

## Roadmap (planned for v0.2)

- Empirically validated default weights from real agent failure data
- Explicit `clm.expand(task_id)` API for agent-initiated context recovery
- Async `aobserve()` and `aobserve_raw()` for async agent frameworks
- CrewAI and AutoGen adapters
- Compression cooldown to prevent amber loop
