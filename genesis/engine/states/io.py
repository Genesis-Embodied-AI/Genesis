from __future__ import annotations
from pathlib import Path
import pickle, numpy as np
import genesis as gs


def save_ckpt(sim: gs.Simulator, path: str | Path, tag: str = "0") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for solver in sim.active_solvers:
        arrays.update(solver.dump_ckpt_to_numpy())

    np.savez_compressed(path.with_suffix(".npz"), **arrays)

    with open(path.with_suffix(".meta"), "wb") as f:
        pickle.dump(dict(dt=sim.dt, substeps=sim.substeps, tag=tag, gravity=sim.gravity), f)

    gs.logger.info(f"[ckpt] wrote {len(arrays)} arrays -> {path}.npz")


def load_ckpt(sim: gs.Simulator, path: str | Path, tag: str = "0") -> None:
    path = Path(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=True) as nz:
        arrays = dict(nz)

    for solver in sim.active_solvers:
        solver.load_ckpt_from_numpy(arrays)

    sim.reset_grad()
    sim._cur_substep_global = 0
    gs.logger.info(f"[ckpt] restored from {path}.npz")


class _StatesFacade:
    save_ckpt = staticmethod(save_ckpt)
    load_ckpt = staticmethod(load_ckpt)


import sys as _sys, genesis as _gs

_sys.modules["genesis.states"] = _StatesFacade()
_gs.states = _StatesFacade()
