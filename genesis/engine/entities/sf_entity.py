import numpy as np
import taichi as ti
from scipy.spatial import KDTree

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.base_entity import Entity
from genesis.engine.entities.particle_entity import ParticleEntity
from genesis.ext import trimesh


@ti.data_oriented
class SFParticleEntity(ParticleEntity):
    """
    PBD-based entity represented solely by particles.
    """

    def __init__(self, scene, solver, material, morph, surface, particle_size, idx, particle_start):
        super().__init__(
            scene, solver, material, morph, surface, particle_size, idx, particle_start, need_skinning=False
        )

    def process_input(self, in_backward=False):
        # TODO: implement this
        pass

    def _add_to_solver_(self):
        # particles in SF is purely for visualization
        # it doesn't make sense to add here
        pass

    def sample(self):
        pass

    def update_particles(self, particles):
        self._particles = particles
        self._n_particles = particles.shape[0]
