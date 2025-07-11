import numpy as np
import taichi as ti
import torch

import igl

import genesis as gs
import genesis.utils.element as eu
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.states.cache import QueriedStates
from genesis.engine.states.entities import FEMEntityState
from genesis.utils.misc import to_gs_tensor

from .base_entity import Entity
from genesis.engine.coupler import SAPCoupler


@ti.data_oriented
class StaticEntity(Entity):
    """
    A static entity that has a mesh for visualization only.
    """

    def __init__(self, idx, scene, morph, material, surface):
        super().__init__(idx, scene, morph, None, material, surface)
