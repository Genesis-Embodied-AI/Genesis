import numpy as np
import gstaichi as ti
import genesis as gs
from genesis.engine.entities import AvatarEntity
from genesis.engine.states.solvers import AvatarSolverState

from .base_solver import Solver
from .rigid.rigid_solver_decomp import RigidSolver


@ti.data_oriented
class AvatarSolver(RigidSolver):
    """
    Avatar, similar to Rigid, maintains a kinematic tree but does not consider actual physics.
    """

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        Solver.__init__(self, scene, sim, options)

        # options
        self._enable_collision = options.enable_collision
        self._enable_self_collision = options.enable_self_collision
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._max_collision_pairs = options.max_collision_pairs
        self._options = options

    def _init_mass_mat(self):
        self.entity_max_dofs = max([entity.n_dofs for entity in self._entities])

    def _init_invweight(self):
        pass

    def update_body(self):
        self._kernel_forward_kinematics()
        self._kernel_update_geoms()

    def substep(self):
        self._kernel_step()

    def _init_constraint_solver(self):
        self.constraint_solver = None

    @ti.kernel
    def _kernel_step(self):
        self._func_integrate()

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_forward_kinematics(i_b)
            self._func_update_geoms(i_b)
        if self._enable_collision:
            self._func_detect_collision()

    @ti.kernel
    def _kernel_forward_kinematics_links_geoms(self, envs_idx: ti.types.ndarray()):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            self._func_forward_kinematics(
                i_b,
                self.links_state,
                self.links_info,
                self.joints_state,
                self.joints_info,
                self.dofs_state,
                self.dofs_info,
                self.entities_info,
                self._rigid_global_info,
                self._static_rigid_sim_config,
            )
            self._func_update_geoms(
                i_b,
                self.links_state,
                self.links_info,
                self.joints_state,
                self.joints_info,
                self.dofs_state,
                self.dofs_info,
                self.entities_info,
                self._rigid_global_info,
                self._static_rigid_sim_config,
            )

    @ti.func
    def _func_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def get_state(self, f):
        if self.is_active():
            state = AvatarSolverState(self.scene)
            self._kernel_get_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
        else:
            state = None
        return state

    def print_contact_data(self):
        batch_idx = 0
        n_contacts = self.collider._collider_state.n_contacts[batch_idx]
        print("collision_pairs:")
        if n_contacts > 0:
            contact_data = self.collider._collider_state.contact_data.to_numpy()
            links_a = contact_data["link_a"][:n_contacts, batch_idx]
            links_b = contact_data["link_b"][:n_contacts, batch_idx]
            link_pairs = np.vstack([links_a, links_b]).T
            print(link_pairs)
