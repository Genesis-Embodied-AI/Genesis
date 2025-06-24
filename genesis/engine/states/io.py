"""
Very-simple SimState pickle helpers
----------------------------------

save_state(state, path)   –>  write a picklable copy (pure Python + NumPy)
load_state(path)          –>  rebuild a SimState so scene.sim.reset() works
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import genesis as gs
from genesis.engine.states.solvers import SimState

# autograd tensor (may be absent)
try:
    from genesis.grad.tensor import Tensor as GradTensor
except ModuleNotFoundError:
    GradTensor = ()


# --------------------------------------------------------------------------
def _make_picklable(obj, seen: set[int]):
    """
    Recursively clone *obj* into a structure that pickle can handle.
    Cycles are broken; exotic objects fall back to their repr().
    """
    oid = id(obj)
    if oid in seen:
        return None  # break cycles
    seen.add(oid)

    # ---------------- gs.Tensor / grad.Tensor -----------------------------
    if isinstance(obj, (gs.Tensor, GradTensor)):
        # best-effort conversion to NumPy
        if hasattr(obj, "to_numpy"):
            arr = obj.to_numpy()
        elif hasattr(obj, "to_torch"):
            arr = obj.to_torch().detach().cpu().numpy()
        else:  # raw torch tensor
            arr = obj.detach().cpu().numpy()
        return ("tensor", arr)

    # ---------------- Taichi objects --------------------------------------
    if obj.__class__.__module__.startswith("taichi."):
        # scalar Expr?
        try:
            return float(obj)
        except Exception:
            pass
        # field or matrix with to_numpy()
        if hasattr(obj, "to_numpy"):
            return obj.to_numpy()
        # iterable matrix/vector
        try:
            return [float(e) for e in obj]
        except Exception:
            return repr(obj)

    # ---------------- containers ------------------------------------------
    if isinstance(obj, dict):
        return {k: _make_picklable(v, seen) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_picklable(x, seen) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_make_picklable(x, seen) for x in obj)
    if isinstance(obj, set):
        return {"__set__": [_make_picklable(x, seen) for x in obj]}

    # ---------------- objects with __dict__ -------------------------------
    if hasattr(obj, "__dict__"):
        return {k: _make_picklable(v, seen) for k, v in obj.__dict__.items()}

    # ---------------- already picklable? ----------------------------------
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _restore(obj):
    """
    Undo _make_picklable.

    A tagged tuple has exactly two elements: (tag : str, payload).
    Anything else is treated as ordinary data.
    """
    # --------------------------- tagged tuples ----------------------------
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str):
        tag, payload = obj

        if tag == "tensor":
            return gs.tensor(payload, requires_grad=False)

        if tag in ("scalar", "matrix", "repr"):
            return payload

        if tag == "dict":
            return {k: _restore(v) for k, v in payload.items()}

        if tag == "list":
            return [_restore(x) for x in payload]

        if tag == "tuple":
            return tuple(_restore(x) for x in payload)

        if tag == "set":  # ← now list, not set
            return [_restore(x) for x in payload]

        if tag == "cycle":
            return None

    # --------------------------- ordinary containers -----------------------
    if isinstance(obj, list):
        return [_restore(x) for x in obj]

    if isinstance(obj, tuple):
        return tuple(_restore(x) for x in obj)

    if isinstance(obj, set):  # restore as list
        return [_restore(x) for x in obj]

    if isinstance(obj, dict):
        # encoded set came in as {"__set__": [...]}
        if "__set__" in obj and len(obj) == 1:
            return [_restore(x) for x in obj["__set__"]]  # list, not set
        return {k: _restore(v) for k, v in obj.items()}

    # --------------------------- primitives --------------------------------
    return obj


# ---------------------------------------------------------------- public API
def save_state(state: SimState, path: str | Path):
    data = _make_picklable(state.__dict__, seen=set())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(path: str | Path) -> SimState:
    path = Path(path)
    with path.open("rb") as f:
        raw = pickle.load(f)
    state = SimState.__new__(SimState)  # bypass __init__
    state.__dict__.update(_restore(raw))
    if getattr(state, "_solvers_state", None) is None:
        state._solvers_state = []
    return state
