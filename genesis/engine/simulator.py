import numpy as np
import taichi as ti

import genesis as gs
from genesis.repr_base import RBC

from .coupler import Coupler
from .entities import HybridEntity
from .solvers import (
    AvatarSolver,
    FEMSolver,
    MPMSolver,
    PBDSolver,
    RigidSolver,
    SFSolver,
    SPHSolver,
    ToolSolver,
)
from .states.cache import QueriedStates
from .states.solvers import SimState


@ti.data_oriented
class Simulator(RBC):
    """
    A simulator is a scene-level simulation manager, which manages all simulation-related operations in the scene, including multiple solvers and the inter-solver coupler.

    Parameters
    ----------
    scene : gs.Scene
        The scene object that the simulator is associated with.
    options : gs.SimOptions
        A SimOptions object that contains all simulator-level options.
    coupler_options : gs.CouplerOptions
        A CouplerOptions object that contains all the options for the coupler.
    tool_options : gs.ToolOptions
        A ToolOptions object that contains all the options for the ToolSolver.
    rigid_options : gs.RigidOptions
        A RigidOptions object that contains all the options for the RigidSolver.
    avatar_options : gs.AvatarOptions
        An AvatarOptions object that contains all the options for the AvatarSolver.
    mpm_options : gs.MPMOptions
        An MPMOptions object that contains all the options for the MPMSolver.
    sph_options : gs.SPHOptions
        An SPHOptions object that contains all the options for the SPHSolver.
    fem_options : gs.FEMOptions
        An FEMOptions object that contains all the options for the FEMSolver.
    sf_options : gs.SFOptions
        An SFOptions object that contains all the options for the SFSolver.
    pbd_options : gs.PBDOptions
        A PBDOptions object that contains all the options for the PBDSolver.
    """

    def __init__(
        self,
        scene,
        options,
        coupler_options,
        tool_options,
        rigid_options,
        avatar_options,
        mpm_options,
        sph_options,
        fem_options,
        sf_options,
        pbd_options,
    ):
        self._scene = scene

        # options
        self.options = options
        self.coupler_options = coupler_options
        self.tool_options = tool_options
        self.rigid_options = rigid_options
        self.avatar_options = avatar_options
        self.mpm_options = mpm_options
        self.sph_options = sph_options
        self.fem_options = fem_options
        self.sf_options = sf_options
        self.pbd_options = pbd_options

        self._dt = options.dt
        self._substep_dt = options.dt / options.substeps
        self._substeps = options.substeps
        self._substeps_local = options.substeps_local
        self._requires_grad = options.requires_grad
        self._steps_local = options._steps_local

        self._cur_substep_global = 0
        self._gravity = np.array(options.gravity)

        # solvers
        self.tool_solver = ToolSolver(self.scene, self, self.tool_options)
        self.rigid_solver = RigidSolver(self.scene, self, self.rigid_options)
        self.avatar_solver = AvatarSolver(self.scene, self, self.avatar_options)
        self.mpm_solver = MPMSolver(self.scene, self, self.mpm_options)
        self.sph_solver = SPHSolver(self.scene, self, self.sph_options)
        self.pbd_solver = PBDSolver(self.scene, self, self.pbd_options)
        self.fem_solver = FEMSolver(self.scene, self, self.fem_options)
        self.sf_solver = SFSolver(self.scene, self, self.sf_options)

        self._solvers = gs.List(
            [
                self.tool_solver,
                self.rigid_solver,
                self.avatar_solver,
                self.mpm_solver,
                self.sph_solver,
                self.pbd_solver,
                self.fem_solver,
                self.sf_solver,
            ]
        )

        self._active_solvers = gs.List()

        # coupler
        self._coupler = Coupler(self, self.coupler_options)

        # states
        self._queried_states = QueriedStates()

        # entities
        self._entities = gs.List()

    def _add_entity(self, morph, material, surface, visualize_contact=False):
        if isinstance(material, gs.materials.Tool):
            entity = self.tool_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.Avatar):
            entity = self.avatar_solver.add_entity(self.n_entities, material, morph, surface, visualize_contact)

        elif isinstance(material, gs.materials.Rigid):
            entity = self.rigid_solver.add_entity(self.n_entities, material, morph, surface, visualize_contact)

        elif isinstance(material, gs.materials.MPM.Base):
            entity = self.mpm_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.SPH.Base):
            entity = self.sph_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.PBD.Base):
            entity = self.pbd_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.FEM.Base):
            entity = self.fem_solver.add_entity(self.n_entities, material, morph, surface)

        elif isinstance(material, gs.materials.Hybrid):
            entity = HybridEntity(
                self.n_entities, self.scene, material, morph, surface
            )  # adding to solver is handled in the hybrid entity

        else:
            gs.raise_exception(f"Material not supported.: {material}")

        self._entities.append(entity)
        return entity

    def _add_force_field(self, force_field):
        for solver in self._solvers:
            solver._add_force_field(force_field)

    def build(self):

        self.n_envs = self.scene.n_envs
        self._B = self.scene._B
        self._para_level = self.scene._para_level

        # solvers
        self._rigid_only = self.rigid_solver.is_active()
        for solver in self._solvers:
            solver.build()
            if solver.is_active():
                self._active_solvers.append(solver)
                if not isinstance(solver, RigidSolver):
                    self._rigid_only = False
        self._coupler.build()

        if self.n_envs > 0 and not self._rigid_only:
            gs.raise_exception("Batching is only supported for rigid-only scenes as of now.")

        # hybrid
        for entity in self._entities:
            if isinstance(entity, HybridEntity):
                entity.build()

    def reset(self, state):
        for solver, solver_state in zip(self._solvers, state):
            solver.set_state(0, solver_state)

        self.coupler.reset()

        self.reset_grad()
        self._cur_substep_global = 0

    def reset_grad(self):
        for solver in self._active_solvers:
            solver.reset_grad()

        # clear up all queried scene states and free up memory
        self._queried_states.clear()

    # ------------------------------------------------------------------------------------
    # -------------------------------- step computation ----------------------------------
    # ------------------------------------------------------------------------------------
    """
    We use f to represent substep, and s to represent step.
    """

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self._substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self._substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self._substeps
        return s_global

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def step(self, in_backward=False):
        if self._rigid_only:  # "Only Advance!" --Thomas Wade :P
            for _ in range(self._substeps):
                self.rigid_solver.substep()
                self._cur_substep_global += 1

        else:
            self.process_input(in_backward=in_backward)
            for _ in range(self._substeps):
                self.substep(self.cur_substep_local)

                self._cur_substep_global += 1
                if self.cur_substep_local == 0 and not in_backward:
                    self.save_ckpt()

    def _step_grad(self):
        for _ in range(self._substeps - 1, -1, -1):

            if self.cur_substep_local == 0:
                self.load_ckpt()
            self._cur_substep_global -= 1

            self.sub_step_grad(self.cur_substep_local)

        self.process_input_grad()

    def process_input(self, in_backward=False):
        """
        setting _tgt state using external commands
        note that external inputs are given at step level, not substep
        """
        for solver in self._active_solvers:
            solver.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for solver in reversed(self._active_solvers):
            solver.process_input_grad()

    def substep(self, f):
        self._coupler.preprocess(f)
        self.substep_pre_coupling(f)
        self._coupler.couple(f)
        self.substep_post_coupling(f)

    def sub_step_grad(self, f):
        self.substep_post_coupling_grad(f)
        self._coupler.couple_grad(f)
        self.substep_pre_coupling_grad(f)

    # -------------- pre coupling --------------
    def substep_pre_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_pre_coupling(f)

    def substep_pre_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_pre_coupling_grad(f)

    # -------------- post coupling --------------
    def substep_post_coupling(self, f):
        for solver in self._active_solvers:
            solver.substep_post_coupling(f)

    def substep_post_coupling_grad(self, f):
        for solver in reversed(self._active_solvers):
            solver.substep_post_coupling_grad(f)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ gradient --------------------------------------
    # ------------------------------------------------------------------------------------

    def add_grad_from_state(self, state):
        for solver, solver_state in zip(self._solvers, state):
            solver.add_grad_from_state(solver_state)

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """

        # simulator-level states
        if self.cur_step_global in self._queried_states:
            # one step could have multiple states
            for state in self._queried_states[self.cur_step_global]:
                self.add_grad_from_state(state)

        # each solver will have their own entities, each of which stores a set of _queried_states
        for solver in self._active_solvers:
            solver.collect_output_grads()

    def save_ckpt(self):
        """
        This function refreshes the gpu memory (copy the last frame to the first frame in the local memory), and then saves the checkpoint.
        This function is called every `substeps_local` steps, which means it's called only once per step when `requires_grad` is True.
        """
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.save_ckpt(ckpt_name)

        if self._requires_grad:
            gs.logger.debug(
                f"Forward: Saved checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {self._cur_substep_global}."
            )

    def load_ckpt(self):
        ckpt_start_substep = self._cur_substep_global - self._substeps_local
        ckpt_end_step = self._cur_substep_global - 1
        ckpt_name = f"{ckpt_start_substep}"

        for solver in self._active_solvers:
            solver.load_ckpt(ckpt_name)

        # now that we loaded the first frame, we do a forward pass to fill up the rest
        self._cur_substep_global = ckpt_start_substep
        for _ in range(self._steps_local):
            self.step(in_backward=True)

        gs.logger.debug(
            f"Backward: Loaded checkpoint for global substep {ckpt_start_substep} to {ckpt_end_step}. Now starts from substep {ckpt_start_substep}."
        )

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        state = SimState(
            scene=self.scene,
            s_global=self.cur_step_global,
            f_local=self.cur_substep_local,
            solvers=self._solvers,
        )

        # store all queried states to track gradient flow
        self._queried_states.append(state)

        return state

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def dt(self):
        """The time duration for each simulation step."""
        return self._dt

    @property
    def substeps(self):
        """The number of substeps per simulation step."""
        return self._substeps

    @property
    def scene(self):
        """The scene object that the simulator is associated with."""
        return self._scene

    @property
    def gravity(self):
        """The gravity vector."""
        return self._gravity

    @property
    def requires_grad(self):
        """Whether the simulator requires gradients."""
        return self._requires_grad

    @property
    def n_entities(self):
        """The number of entities in the simulator."""
        return len(self._entities)

    @property
    def entities(self):
        """The list of entities in the simulator."""
        return self._entities

    @property
    def substeps_local(self):
        """The number of substeps stored in local memory."""
        return self._substeps_local

    @property
    def cur_substep_global(self):
        """The current substep of the simulation."""
        return self._cur_substep_global

    @property
    def cur_substep_local(self):
        """The current substep of the simulation in local memory."""
        return self.f_global_to_f_local(self._cur_substep_global)

    @property
    def cur_step_local(self):
        """The current step of the simulation in local memory."""
        return self.f_global_to_s_local(self._cur_substep_global)

    @property
    def cur_step_global(self):
        """The current step of the simulation."""
        return self.f_global_to_s_global(self._cur_substep_global)

    @property
    def cur_t(self):
        """The current time of the simulation."""
        return self._cur_substep_global * self._substep_dt

    @property
    def coupler(self):
        """The coupler object that manages the inter-solver coupling."""
        return self._coupler

    @property
    def solvers(self):
        """The list of solvers in the simulator."""
        return self._solvers

    @property
    def active_solvers(self):
        """The list of active solvers in the simulator."""
        return self._active_solvers
