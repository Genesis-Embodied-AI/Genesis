import numpy as np
import taichi as ti

import genesis as gs
from genesis.repr_base import RBC


class Solver(RBC):
    def __init__(self, scene, sim, options):
        self._uid = gs.UID()
        self._sim = sim
        self._scene = scene
        self._dt = options.dt
        self._substep_dt = options.dt / sim.substeps

        if hasattr(options, "gravity"):
            self._gravity = ti.field(dtype=gs.ti_vec3, shape=())
            self._gravity.from_numpy(np.array(options.gravity, dtype=gs.np_float))
        else:
            self._gravity = None

        self._entities = gs.List()

        # force fields
        self._ffs = list()

    def _add_force_field(self, force_field):
        self._ffs.append(force_field)

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
