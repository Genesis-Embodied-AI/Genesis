from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.states import QueriedStates
from genesis.engine.states.solvers import KinematicSolverState
from genesis.options.solvers import RigidOptions, KinematicOptions
from genesis.utils.misc import (
    qd_to_torch,
    sanitize_indexed_tensor,
    indices_to_mask,
    broadcast_tensor,
    assign_indexed_tensor,
)

from .base_solver import Solver
from .rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_link_fields,
    kernel_init_joint_fields,
    kernel_init_vvert_fields,
    kernel_init_vgeom_fields,
    kernel_init_entity_fields,
    kernel_update_heterogeneous_links_vgeom,
    kernel_update_vgeoms_render_T,
)
from .rigid.abd.forward_kinematics import (
    kernel_forward_kinematics,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics,
    kernel_masked_forward_velocity,
    kernel_update_vgeoms,
)
from .rigid.abd.accessor import (
    kernel_get_kinematic_state,
    kernel_get_state_grad,
    kernel_set_kinematic_state,
    kernel_set_links_pos_grad,
    kernel_set_links_quat_grad,
    kernel_set_dofs_position,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
    kernel_set_dofs_zero_velocity,
    kernel_set_links_pos,
    kernel_set_links_quat,
    kernel_set_qpos,
    kernel_get_links_vel,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


def _balanced_variant_mapping(n_variants, B):
    """Map N variants to B environments using balanced block assignment."""
    if B >= n_variants:
        base = B // n_variants
        extra = B % n_variants
        sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
        return np.repeat(np.arange(n_variants), sizes)
    else:
        return np.arange(B)


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class KinematicSolver(Solver):
    """
    Base solver for articulated kinematic entities (FK, rendering, state get/set).

    Provides the full build pipeline, field init methods, counter properties,
    render methods, and IO sanitization shared by both KinematicSolver and RigidSolver.

    RigidSolver extends this with physics (collision, constraints, dynamics).
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: "KinematicOptions") -> None:
        super().__init__(scene, sim, options)

        self._options = options

        self._enable_collision = False
        self._enable_mujoco_compatibility = False
        self._requires_grad = False
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        self.collider = None
        self.constraint_solver = None

        self.qpos = None

        self._is_forward_pos_updated: bool = False
        self._is_forward_vel_updated: bool = False

        self._queried_states = QueriedStates()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(self, idx, material, morph, surface, visualize_contact=False, name=None):
        morph_heterogeneous = []
        if isinstance(morph, (tuple, list)):
            morph, *morph_heterogeneous = morph
            self._enable_heterogeneous |= bool(morph_heterogeneous)

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = KinematicEntity(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            morph_heterogeneous=morph_heterogeneous,
            name=name,
        )
        self._entities.append(entity)
        return entity

    # ------------------------------------------------------------------------------------
    # ------------------------------------ build -----------------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        super().build()

        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._para_level = self.sim._para_level

        for entity in self._entities:
            entity._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_entities = self.n_entities

        self._vgeoms = self.vgeoms
        self._links = self.links
        self._joints = self.joints

        base_links_idx = []
        for link in self.links:
            if link.parent_idx == -1 and link.is_fixed:
                base_links_idx.append(link.idx)
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                base_links_idx.append(joint.link.idx)
        self._base_links_idx = torch.tensor(base_links_idx, dtype=gs.tc_int, device=gs.device)

        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_entities_ = max(1, self.n_entities)

        # batch_links_info is required when heterogeneous simulation is used.
        # We must update options because get_links_info reads from solver._options.batch_links_info.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True

        self._build_static_config()
        self._create_data_manager()

        self._init_dof_fields()
        self._init_vvert_fields()
        self._init_vgeom_fields()
        self._init_link_fields()
        self._init_entity_fields()

        self._init_envs_offset()

    def _build_static_config(self):
        # Static config with all physics disabled
        self._static_rigid_sim_config = array_class.StructRigidSimStaticConfig(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=False,
            use_hibernation=False,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=False,
            batch_joints_info=False,
            enable_heterogeneous=self._enable_heterogeneous,
            enable_mujoco_compatibility=False,
            enable_multi_contact=False,
            enable_collision=False,
            enable_joint_limit=False,
            box_box_detection=False,
            sparse_solve=False,
            integrator=gs.integrator.approximate_implicitfast,
            solver_type=0,
        )

    def _create_data_manager(self):
        self.data_manager = array_class.DataManager(self, kinematic_only=True)
        self._rigid_global_info = self.data_manager.rigid_global_info
        self._rigid_adjoint_cache = self.data_manager.rigid_adjoint_cache

    # ------------------------------------------------------------------------------------
    # --------------------------------- hook methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_joint_sol_params(self, sol_params):
        """Hook: sanitize joint constraint solver params. No-op in base (no constraints)."""
        return sol_params

    def _sanitize_geom_sol_params(self, sol_params):
        """Hook: sanitize geom constraint solver params. No-op in base (no constraints)."""
        return sol_params

    # ------------------------------------------------------------------------------------
    # --------------------------------- init methods -------------------------------------
    # ------------------------------------------------------------------------------------

    def _init_dof_fields(self):
        self.dofs_info = self.data_manager.dofs_info
        self.dofs_state = self.data_manager.dofs_state

        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:
            kernel_init_dof_fields(
                entity_idx=np.concatenate(
                    [(joint.link._entity_idx_in_solver,) * joint.n_dofs for joint in joints if joint.n_dofs],
                    dtype=gs.np_int,
                ),
                dofs_motion_ang=np.concatenate([joint.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                dofs_motion_vel=np.concatenate([joint.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                dofs_limit=np.concatenate([joint.dofs_limit for joint in joints], dtype=gs.np_float),
                dofs_invweight=np.concatenate([joint.dofs_invweight for joint in joints], dtype=gs.np_float),
                dofs_stiffness=np.concatenate([joint.dofs_stiffness for joint in joints], dtype=gs.np_float),
                dofs_damping=np.concatenate([joint.dofs_damping for joint in joints], dtype=gs.np_float),
                dofs_frictionloss=np.concatenate([joint.dofs_frictionloss for joint in joints], dtype=gs.np_float),
                dofs_armature=np.concatenate([joint.dofs_armature for joint in joints], dtype=gs.np_float),
                dofs_kp=np.concatenate([joint.dofs_kp for joint in joints], dtype=gs.np_float),
                dofs_kv=np.concatenate([joint.dofs_kv for joint in joints], dtype=gs.np_float),
                dofs_force_range=np.concatenate([joint.dofs_force_range for joint in joints], dtype=gs.np_float),
                dofs_info=self.dofs_info,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        self.dofs_state.force.fill(0)

    def _init_link_fields(self):
        self.links_info = self.data_manager.links_info
        self.links_state = self.data_manager.links_state

        if self.links:
            links = self.links
            kernel_init_link_fields(
                links_parent_idx=np.array([link.parent_idx for link in links], dtype=gs.np_int),
                links_root_idx=np.array([link.root_idx for link in links], dtype=gs.np_int),
                links_q_start=np.array([link.q_start for link in links], dtype=gs.np_int),
                links_dof_start=np.array([link.dof_start for link in links], dtype=gs.np_int),
                links_joint_start=np.array([link.joint_start for link in links], dtype=gs.np_int),
                links_q_end=np.array([link.q_end for link in links], dtype=gs.np_int),
                links_dof_end=np.array([link.dof_end for link in links], dtype=gs.np_int),
                links_joint_end=np.array([link.joint_end for link in links], dtype=gs.np_int),
                links_invweight=np.array([link.invweight for link in links], dtype=gs.np_float),
                links_is_fixed=np.array([link.is_fixed for link in links], dtype=gs.np_bool),
                links_pos=np.array([link.pos for link in links], dtype=gs.np_float),
                links_quat=np.array([link.quat for link in links], dtype=gs.np_float),
                links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
                links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
                links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
                links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
                links_entity_idx=np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
                links_geom_start=np.array([link.geom_start for link in links], dtype=gs.np_int),
                links_geom_end=np.array([link.geom_end for link in links], dtype=gs.np_int),
                links_vgeom_start=np.array([link.vgeom_start for link in links], dtype=gs.np_int),
                links_vgeom_end=np.array([link.vgeom_end for link in links], dtype=gs.np_int),
                links_info=self.links_info,
                links_state=self.links_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        self.joints_info = self.data_manager.joints_info
        self.joints_state = self.data_manager.joints_state

        if self.joints:
            joints = self.joints
            joints_sol_params = np.array([joint.sol_params for joint in joints], dtype=gs.np_float)
            joints_sol_params = self._sanitize_joint_sol_params(joints_sol_params)

            kernel_init_joint_fields(
                joints_type=np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params=joints_sol_params,
                joints_q_start=np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                joints_dof_start=np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                joints_q_end=np.array([joint.q_end for joint in joints], dtype=gs.np_int),
                joints_dof_end=np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                joints_pos=np.array([joint.pos for joint in joints], dtype=gs.np_float),
                joints_info=self.joints_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        # Set initial qpos
        self.qpos = self._rigid_global_info.qpos
        self.qpos0 = self._rigid_global_info.qpos0
        if self.n_qs > 0:
            init_qpos = np.tile(np.expand_dims(self.init_qpos, -1), (1, self._B))

            # Dispatch per-variant init_qpos for heterogeneous entities
            for entity in self.entities:
                if entity._variant_init_qpos is None:
                    continue
                n_variants = len(entity._variant_init_qpos)
                variant_idx = _balanced_variant_mapping(n_variants, self._B)
                q_s, q_e = entity.q_start, entity.q_start + entity.n_qs
                for i_b in range(self._B):
                    init_qpos[q_s:q_e, i_b] = entity._variant_init_qpos[variant_idx[i_b]]

            self.qpos0.from_numpy(init_qpos)
            is_init_qpos_out_of_bounds = False
            for joint in self.joints:
                if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    is_init_qpos_out_of_bounds |= (joint.dofs_limit[0, 0] > init_qpos[joint.q_start]).any()
                    is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > joint.dofs_limit[0, 1]).any()
            if is_init_qpos_out_of_bounds:
                gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
            self.qpos.from_numpy(init_qpos)

        self.links_T = self._rigid_global_info.links_T

        # Dispatch heterogeneous variant vgeom ranges per-environment
        self._dispatch_heterogeneous_vgeoms()

    def _dispatch_heterogeneous_vgeoms(self):
        """Override per-link vgeom ranges for heterogeneous variants. RigidSolver extends this."""
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self._B)

            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

            kernel_update_heterogeneous_links_vgeom(link.idx, vgeom_starts, vgeom_ends, self.links_info)

            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_vvert_fields(self):
        self.vverts_info = self.data_manager.vverts_info
        self.vfaces_info = self.data_manager.vfaces_info
        if self.n_vverts > 0:
            vgeoms = self.vgeoms
            kernel_init_vvert_fields(
                vverts=np.concatenate([vgeom.init_vverts for vgeom in vgeoms], dtype=gs.np_float),
                vfaces=np.concatenate([vgeom.init_vfaces + vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vnormals=np.concatenate([vgeom.init_vnormals for vgeom in vgeoms], dtype=gs.np_float),
                vverts_vgeom_idx=np.concatenate(
                    [np.full(vgeom.n_vverts, vgeom.idx) for vgeom in vgeoms], dtype=gs.np_int
                ),
                vverts_info=self.vverts_info,
                vfaces_info=self.vfaces_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_vgeom_fields(self):
        self.vgeoms_info = self.data_manager.vgeoms_info
        self.vgeoms_state = self.data_manager.vgeoms_state
        self._vgeoms_render_T = np.empty((self.n_vgeoms_, self._B, 4, 4), dtype=np.float32)

        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            kernel_init_vgeom_fields(
                vgeoms_pos=np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_quat=np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_link_idx=np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_start=np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_start=np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_end=np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_end=np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_color=np.array([vgeom._color for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_info=self.vgeoms_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_entity_fields(self):
        self.entities_info = self.data_manager.entities_info
        self.entities_state = self.data_manager.entities_state

        if self._entities:
            entities = self._entities
            kernel_init_entity_fields(
                entities_dof_start=np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                entities_dof_end=np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                entities_link_start=np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                entities_link_end=np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                entities_geom_start=np.array([0 for entity in entities], dtype=gs.np_int),
                entities_geom_end=np.array([0 for entity in entities], dtype=gs.np_int),
                entities_gravity_compensation=np.array([0.0 for entity in entities], dtype=gs.np_float),
                entities_is_local_collision_mask=np.array([False for entity in entities], dtype=gs.np_bool),
                entities_info=self.entities_info,
                entities_state=self.entities_state,
                links_info=self.links_info,
                dofs_info=self.dofs_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_envs_offset(self):
        self.envs_offset = self._rigid_global_info.envs_offset
        self.envs_offset.from_numpy(self._scene.envs_offset)

    # ------------------------------------------------------------------------------------
    # -------------------------------- simulation no-ops ----------------------------------
    # ------------------------------------------------------------------------------------

    def substep_pre_coupling(self, f):
        pass

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        if not self._is_forward_pos_updated or not self._is_forward_vel_updated:
            kernel_forward_kinematics(
                self.scene._envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    def substep_post_coupling_grad(self, f):
        pass

    def add_grad_from_state(self, state):
        if self.is_active:
            qpos_grad = gs.zeros_like(state.qpos)
            dofs_vel_grad = gs.zeros_like(state.dofs_vel)
            links_pos_grad = gs.zeros_like(state.links_pos)
            links_quat_grad = gs.zeros_like(state.links_quat)

            if state.qpos.grad is not None:
                qpos_grad = state.qpos.grad
            if state.dofs_vel.grad is not None:
                dofs_vel_grad = state.dofs_vel.grad
            if state.links_pos.grad is not None:
                links_pos_grad = state.links_pos.grad
            if state.links_quat.grad is not None:
                links_quat_grad = state.links_quat.grad

            kernel_get_state_grad(
                qpos_grad=qpos_grad,
                vel_grad=dofs_vel_grad,
                links_pos_grad=links_pos_grad,
                links_quat_grad=links_quat_grad,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            assert len(self._queried_states[self._sim.cur_step_global]) == 1
            state = self._queried_states[self._sim.cur_step_global][0]
            self.add_grad_from_state(state)

    def reset_grad(self):
        for entity in self._entities:
            entity.reset_grad()
        self._queried_states.clear()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- render -----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_vgeoms_render_T(self):
        kernel_update_vgeoms_render_T(
            self._vgeoms_render_T,
            vgeoms_info=self.vgeoms_info,
            vgeoms_state=self.vgeoms_state,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f=None):
        if self.is_active:
            s_global = self.sim.cur_step_global
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = KinematicSolverState(self._scene, s_global)

            kernel_get_kinematic_state(
                qpos=state.qpos,
                vel=state.dofs_vel,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            kernel_set_kinematic_state(
                envs_idx=envs_idx,
                qpos=state.qpos,
                dofs_vel=state.dofs_vel,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            kernel_forward_kinematics(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    # ------------------------------------------------------------------------------------
    # -------------------------------- process_input -------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        """Process input for entities (set qpos from user commands)."""
        for entity in self._entities:
            entity.process_input()

    def process_input_grad(self):
        """No-op: kinematic solver does not support gradients."""
        pass

    def save_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not save checkpoints."""
        pass

    def load_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not load checkpoints."""
        pass

    @property
    def is_active(self):
        return self.n_links > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_io_variables(
        self,
        tensor,
        inputs_idx,
        input_size,
        idx_name,
        envs_idx=None,
        element_shape=(),
        *,
        batched=True,
        skip_allocation=False,
    ):
        envs_idx_ = self._scene._sanitize_envs_idx(envs_idx) if batched else self._scene._envs_idx[:0]

        if self.n_envs == 0 or not batched:
            tensor_, (inputs_idx_,) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (inputs_idx,),
                (-1, *element_shape),
                (input_size, *element_shape),
                (idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )
        else:
            tensor_, (envs_idx_, inputs_idx_) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (envs_idx_, inputs_idx),
                (-1, -1, *element_shape),
                (self.n_envs, input_size, *element_shape),
                ("envs_idx", idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )

        return tensor_, inputs_idx_, envs_idx_

    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        pos, links_idx, envs_idx = self._sanitize_io_variables(
            pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos = pos[None]

        kernel_set_links_pos(
            relative,
            pos,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_kinematics(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def set_base_links_pos_grad(self, links_idx, envs_idx, relative, pos_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        pos_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            pos_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos_grad_ = pos_grad_.unsqueeze(0)
        kernel_set_links_pos_grad(
            relative,
            pos_grad_,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_base_links_quat(self, quat, links_idx=None, envs_idx=None, *, relative=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        quat, links_idx, envs_idx = self._sanitize_io_variables(
            quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat = quat[None]

        kernel_set_links_quat(
            relative,
            quat,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_kinematics(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def set_base_links_quat_grad(self, links_idx, envs_idx, relative, quat_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        quat_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            quat_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        assert relative == False, "Backward pass for relative quaternion is not supported yet."
        kernel_set_links_quat_grad(
            relative,
            quat_grad_,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            data = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2:
                    # Note that it is necessary to create a new temporary view because it will be modified in-place
                    qs_data.masked_scatter_(envs_idx[:, None], qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask)
                assign_indexed_tensor(data, mask, qpos)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]
            kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics
            else:
                fn = kernel_forward_kinematics
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dofs_state.vel, transpose=True, copy=False)
            dofs_mask = indices_to_mask(dofs_idx)
            if (
                (not dofs_mask or isinstance(dofs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and (
                    (velocity is None and (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
                    or (velocity is not None and velocity.ndim == 2 and envs_idx.dtype == torch.bool)
                )
            ):
                dofs_vel = vel[(slice(None), *dofs_mask)]
                if velocity is None:
                    if envs_idx.dtype == torch.bool:
                        dofs_vel.masked_fill_(envs_idx[:, None], 0.0)
                    else:
                        dofs_vel.scatter_(0, envs_idx[:, None].expand((-1, dofs_vel.shape[1])), 0.0)
                else:
                    if velocity.ndim == 2:
                        # Note that it is necessary to create a new temporary view because it will be modified in-place
                        dofs_vel.masked_scatter_(envs_idx[:, None], velocity.view_as(velocity))
                    else:
                        velocity = broadcast_tensor(velocity, gs.tc_float, dofs_vel.shape)
                        torch.where(envs_idx[:, None], velocity, dofs_vel, out=dofs_vel)
            else:
                mask = (0, *dofs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *dofs_mask)
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)
            else:
                if self.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

        if not skip_forward:
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_velocity
            else:
                fn = kernel_forward_velocity
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=False,
            )
            self._is_forward_vel_updated = True
        else:
            self._is_forward_vel_updated = False

    def set_dofs_velocity_grad(self, dofs_idx, envs_idx, velocity_grad):
        velocity_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(
            velocity_grad_, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config
        )

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        kernel_set_dofs_position(
            position,
            dofs_idx,
            envs_idx,
            self.dofs_state,
            self.links_info,
            self.joints_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

        kernel_forward_kinematics(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def get_links_pos(self, links_idx=None, envs_idx=None):
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        tensor = qd_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(self, links_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(links_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.links_state.cd_vel, transpose=True)
            cd_ang = qd_to_torch(self.links_state.cd_ang, transpose=True)
            pos = qd_to_torch(self.links_state.pos, transpose=True)
            root_COM = qd_to_torch(self.links_state.root_COM, transpose=True)
            return cd_vel[mask] + cd_ang[mask].cross(pos[mask] - root_COM[mask], dim=-1)

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        assert _tensor is not None
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_vel(tensor, links_idx, envs_idx, 2, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def _build_dof_to_q_map(self, dofs_idx_t):
        """Build a mapping from DOF indices to qpos indices for revolute/prismatic joints."""
        dof_to_q = torch.zeros(self.n_dofs, dtype=torch.long, device=gs.device)
        for entity in self._entities:
            for joint in entity.joints:
                if joint.n_dofs == 0:
                    continue
                for i in range(joint.n_dofs):
                    dof_to_q[joint.dof_start - entity.dof_start + i] = joint.q_start - entity.q_start + i
        return dof_to_q[dofs_idx_t]

    def get_qpos(self, qs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.qpos, envs_idx, qs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dofs_state.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        """Read current DOF positions."""
        tensor = qd_to_torch(self.dofs_state.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_vgeoms(self):
        kernel_update_vgeoms(self.vgeoms_info, self.vgeoms_state, self.links_state, self._static_rigid_sim_config)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def links(self):
        if self.is_built:
            return self._links
        return gs.List(link for entity in self._entities for link in entity.links)

    @property
    def joints(self):
        if self.is_built:
            return self._joints
        return gs.List(joint for entity in self._entities for joint in entity.joints)

    @property
    def geoms(self):
        if self.is_built:
            return self._geoms
        return gs.List(geom for entity in self._entities for geom in entity.geoms)

    @property
    def vgeoms(self):
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for entity in self._entities for vgeom in entity.vgeoms)

    @property
    def n_links(self):
        if self.is_built:
            return self._n_links
        return len(self.links)

    @property
    def n_joints(self):
        if self.is_built:
            return self._n_joints
        return len(self.joints)

    @property
    def n_vgeoms(self):
        if self.is_built:
            return self._n_vgeoms
        return len(self.vgeoms)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def n_qs(self):
        if self.is_built:
            return self._n_qs
        return sum(entity.n_qs for entity in self._entities)

    @property
    def n_dofs(self):
        if self.is_built:
            return self._n_dofs
        return sum(entity.n_dofs for entity in self._entities)

    @property
    def init_qpos(self):
        if self._entities:
            return np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)
        return np.array([], dtype=gs.np_float)
