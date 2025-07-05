from dataclasses import dataclass
from typing import Callable

import taichi as ti

import genesis as gs

# we will use struct for DofsState and DofsInfo after Hugh adds array_struct feature to taichi
DofsState = ti.template()
DofsInfo = ti.template()


@ti.data_oriented
class RigidGlobalInfo:
    def __init__(self, n_dofs: int, n_entities: int, n_geoms: int, f_batch: Callable):
        self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch())
        self.awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch(n_dofs))
