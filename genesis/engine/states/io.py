"""
Lightweight disk-I/O for physics checkpoints.
Writes ONE pickle file instead of many .npz archives.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Union
import numpy as np  # noqa: F401  (imported for type hints)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------


def _gather_arrays(sim) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for solver in sim.active_solvers:
        arrays.update(solver.dump_ckpt_to_numpy())
    return arrays


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------


def save_ckpt(sim, path: PathLike) -> None:
    """Serialize **simulator** state to a single pickle file."""
    state = {"arrays": _gather_arrays(sim)}
    with open(Path(path), "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ckpt(sim, path: PathLike) -> None:
    """Load a pickle produced by `save_ckpt`."""
    with open(Path(path), "rb") as f:
        state = pickle.load(f)

    arrays = state["arrays"]
    for solver in sim.active_solvers:
        solver.load_ckpt_from_numpy(arrays)
