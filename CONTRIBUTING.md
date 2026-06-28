# Contributing to Darkclaw

Welcome. Whether you're a developer, a designer, or someone who opened a business and wants to help others do the same — there's a place for you here.

## Three ways in

### 1. Code — new healing strategy
Open `core/heal_engine.py`. Add a new entry to `STRATEGY_MAP` or a new `FailureType`. Write a handler in `_apply_strategy()`. Open a PR. See the existing strategies for examples.

### 2. Code — new memory backend
Implement the `MemoryBackend` interface (see `memory/darkclaw_core.py`). ChromaDB, Neo4j, and Redis are all wanted. The interface is small — `ingest()`, `query()`, `stats()`.

### 3. Art — new Maple Creek sprite
Draw a new character or tile in the Zelda village palette:
- 16×16 pixels
- 4 colors + transparent (warm brown outline, no hard black)
- 3-frame walk cycle × 4 directions = 48×64px sheet
- Any pixel art tool: Aseprite, LibreSprite, even Paint

Drop the PNG in `daydream/sprites/` and open a PR. Non-developers welcome.

## Palette reference
```
Outline:  #402810  (warm dark brown — never pure black)
Grass:    #60a838  #508828  #70b848
Path:     #d8c098  #c4ac84
Water:    #4890d8  #3878c0  #78b8f0
Roof:     #b86048  #984830
Gold:     #f0c838  #c89818
Skin:     #f8d098  #e0b070
```

## Code style
- Python 3.12+
- Type hints everywhere
- Dataclasses over dicts for structured data
- Every public method gets a docstring
- Emit events to the bus for anything the UI should know about

## Running tests
```bash
pip install -r requirements.txt
python memory/darkclaw_core.py      # Darkclaw self-test (8/8)
python -m pytest tests/ -v          # Full suite
```

## Questions
Open an Issue. Or leave a message at the Inn (daydream.html → Feedback button).
