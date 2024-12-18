import taichi as ti

from genesis.engine.boundaries import FloorBoundary
from genesis.engine.entities import ToolEntity
from genesis.engine.states.solvers import ToolSolverState
from genesis.utils.misc import *

from .base_solver import Solver


@ti.data_oriented
class ToolSolver(Solver):
    """
    Note
    ----
    !! This class will be removed once we added differntiability to the RigidSolver. This is just a temporary solution to obtain rigid->soft one-way differeitable coupling.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self.floor_height = options.floor_height

        # boundary
        self.setup_boundary()

    def build(self):
        for entity in self._entities:
            entity.build()

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self.floor_height)

    def add_entity(self, idx, material, morph, surface):
        entity = ToolEntity(
            scene=self._scene,
            idx=idx,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
        )
        self._entities.append(entity)
        return entity

    def reset_grad(self):
        for entity in self._entities:
            entity.reset_grad()

    def get_state(self, f):
        if self.is_active():
            state = ToolSolverState(self._scene)
            for entity in self._entities:
                state.entities.append(entity.get_state(f))
        else:
            state = None
        return state

    def set_state(self, f, state):
        if state is not None:
            assert len(state) == len(self._entities)
            for i, entity in enumerate(self._entities):
                entity.set_state(f, state[i])

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities[::-1]:
            entity.process_input_grad()

    def substep_pre_coupling(self, f):
        for entity in self._entities:
            entity.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_pre_coupling_grad(f)

    def substep_post_coupling(self, f):
        for entity in self._entities:
            entity.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for entity in self._entities[::-1]:
            entity.substep_post_coupling_grad(f)

    def add_grad_from_state(self, state):
        # Nothing needed here, since tool_solver state is composed of tool_entity.get_state(), which has already been cached inside each tool_entity.
        pass

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        if self.is_active():
            for entity in self._entities:
                entity.collect_output_grads()

    def save_ckpt(self, ckpt_name):
        for entity in self._entities:
            entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        for entity in self._entities:
            entity.load_ckpt(ckpt_name=ckpt_name)

    def is_active(self):
        return self.n_entities > 0

    @ti.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        for entity in ti.static(self._entities):
            pos_world = entity.pbd_collide(f, pos_world, thickness, dt)
        return pos_world
