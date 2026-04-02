# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] — 2025-04-02

### Fixed
- README now displays correctly on PyPI dashboard (re-release of 0.1.2)

## [0.1.2] — 2025-04-02

### Fixed
- README now displays correctly on PyPI dashboard

## [0.1.1] — 2025-04-02

### Changed

**Dependency optimization**
- Made `sentence-transformers` optional — base install now only requires `numpy` (~10MB vs ~1.5GB)
- Added `pip install clm-plugin[embed]` for full embedding support
- Auto-detect missing `sentence-transformers` and enable `no_embed=True` with helpful message

**PyPI metadata improvements**
- Added authors and maintainers fields
- Added Repository and Changelog URLs
- Created PyPI-specific README (README_PYPI.md) — short, install-focused, under 100 lines
- Full README.md remains on GitHub

**Documentation**
- Added note about sidecar database (*.db) exclusion from version control
- Clarified that `*.db` files should not be committed

### Fixed
- Prevented crashes for users who install lean version without reading docs

## [0.1.0] — 2025-03-31

### First public release

**Core architecture**
- 5-layer cognitive load management: Signal Collector, CLM Scorer, Chunking Engine, Action Dispatcher, Sidecar Store
- 4 cognitive load signals: branching factor, repetition rate, uncertainty density, goal distance
- 3-zone intervention system: Green (pass), Amber (compress), Red (interrupt)
- Abort action for structurally unresolvable tasks (5 consecutive Red triggers)
- Amber escalation protection (3 consecutive Amber triggers → Red)

**Integration**
- `CLM()` — zero-argument instantiation with sensible defaults
- `observe_raw()` — single-line integration, no TaskState construction required
- `AutoStateBuilder` — automatic task tree inference from LLM outputs
- LangChain adapter: `CLMCallbackHandler`
- OpenAI Agents SDK adapter: `CLMOpenAIHook`
- Generic loop adapter: `CLMLoop` with decorator and context manager support

**Observability**
- `verbose=True` — real-time step-by-step output
- `get_history()` — full intervention log
- `summary()` — session aggregate stats
- `get_score()`, `get_zone()`, `get_sidecar_stats()`

**Configuration**
- `no_embed=True` — keyword-based fallback, zero model download, works offline
- Fully tunable weights, thresholds, and zone boundaries
- Domain-specific configuration examples: medical, legal, voice

**Storage**
- SQLite sidecar store, auto-created on first use
- In-memory mode (default) for ephemeral sessions

**Known limitations**
- Default weights `[0.30, 0.25, 0.25, 0.20]` are heuristic, not empirically validated
- AutoStateBuilder uses regex heuristics for task tree inference
- `response.context` replaces task plan section only, not full conversation history
- Embedding model requires ~90MB download on first use (avoidable with `no_embed=True`)
