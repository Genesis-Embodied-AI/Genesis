# Testing Guide

## Environment Setup

Use `uv` for running tests:

```bash
# Setup environment (if not already done)
uv sync

# Install PyTorch for your platform (see README.md)
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # NVIDIA
```

## Running Tests

```bash
# Run all tests (parallel, excludes benchmarks and examples)
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_rigid_physics.py

# Run with GPU backend
uv run pytest tests/ --backend=gpu

# Run with visualization (disables parallelism)
uv run pytest tests/ --vis

# Run in debug mode
uv run pytest tests/ --dev

# Run specific markers
uv run pytest tests/ -m required
uv run pytest tests/ -m "not slow"

# In restricted environments (sandboxes, containers), disable retry plugins:
uv run pytest tests/ -p no:pytest-retry -p no:rerunfailures
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `required` | Minimal test set that must pass before merging |
| `slow` | Tests taking >100s |
| `examples` | Example scripts |
| `benchmarks` | Performance benchmarks |

## Key Fixtures

From `tests/conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `initialize_genesis` | function | Auto-initializes and destroys Genesis per test |
| `backend` | session | Returns configured backend (gs.cpu/gs.gpu) |
| `precision` | function | Returns precision ("32" or "64") |
| `show_viewer` | session | Whether viewer is enabled |
| `tol` | function | Tolerance based on precision |

## Writing Tests

```python
import pytest
import genesis as gs

def test_example(initialize_genesis, backend):
    """Test runs with auto-initialized Genesis."""
    scene = gs.Scene()
    entity = scene.add_entity(gs.morphs.Box(size=(1, 1, 1)))
    scene.build()
    scene.step()
    assert entity.get_pos()[2] > 0

@pytest.mark.slow
def test_long_simulation(initialize_genesis):
    """Mark slow tests explicitly."""
    pass

@pytest.mark.required
def test_critical_feature(initialize_genesis):
    """Mark tests that must always pass."""
    pass
```

## CI Requirements

Before submitting a PR, ensure tests pass locally:

```bash
uv run pytest -v -m required tests/
```
