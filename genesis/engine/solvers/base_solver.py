from typing import TYPE_CHECKING
import numpy as np
import taichi as ti

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.repr_base import RBC


if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class Solver(RBC):
    def __init__(self, scene: "Scene", sim: "Simulator", options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._dt: float = options.dt
        self._substep_dt: float = options.dt / sim.substeps
        self._init_gravity = getattr(options, "gravity", None)
        self._gravity = None
        self._entities: list[Entity] = gs.List()

        # force fields
        self._ffs = list()

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

    def build(self):
        self._B = self._sim._B
        if self._init_gravity is not None:
            g_np = np.asarray(self._init_gravity, dtype=gs.np_float)
        else:
            g_np = np.asarray(self._sim._gravity, dtype=gs.np_float)
        g_np = np.repeat(g_np[None], self._B, axis=0)

        self._gravity = ti.Vector.field(3, dtype=gs.ti_float, shape=self._B)
        self._gravity.from_numpy(g_np)

    def set_gravity(self, gravity, envs_idx=None):
        if self._gravity is None:
            return
        if envs_idx is None:
            self._gravity.copy_from(gravity)
        else:
            self._gravity[envs_idx] = gravity

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def dt(self):
        return self._dt

    @property
    def is_built(self):
        return self._scene._is_built

    @property
    def substep_dt(self):
        return self._substep_dt

    @property
    def gravity(self):
        return self._gravity.to_numpy() if self._gravity is not None else None

    @property
    def entities(self):
        return self._entities

    @property
    def n_entities(self):
        return len(self._entities)

    def _repr_brief(self):
        repr_str = f"{self._repr_type()}: {self._uid}, n_entities: {self.n_entities}"
        return repr_str
