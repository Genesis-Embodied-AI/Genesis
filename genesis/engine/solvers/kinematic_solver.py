from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.states import QueriedStates
from genesis.engine.states.solvers import KinematicSolverState
from genesis.options.solvers import RigidOptions, KinematicOptions
from genesis.utils.misc import qd_to_torch, sanitize_indexed_tensor

from .base_solver import Solver
from .rigid.abd.misc import (
    kernel_init_dof_fields,
    kernel_init_link_fields,
    kernel_init_joint_fields,
    kernel_init_vert_fields,
    kernel_init_vvert_fields,
    kernel_init_geom_fields,
    kernel_init_vgeom_fields,
    kernel_init_entity_fields,
    kernel_update_geoms_render_T,
    kernel_update_heterogeneous_link_info,
    kernel_update_vgeoms_render_T,
    kernel_set_zero,
)
from .rigid.abd.forward_kinematics import (
    kernel_forward_kinematics_links_geoms,
    kernel_update_geoms,
    kernel_update_vgeoms,
)
from .rigid.abd.accessor import (
    kernel_get_state,
    kernel_set_state,
    kernel_set_dofs_position,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


class KinematicSolver(Solver):
    """
    Base solver for articulated kinematic entities (FK, rendering, state get/set).

    Provides the full build pipeline, field init methods, counter properties,
    render methods, and IO sanitization shared by both KinematicSolver and RigidSolver.

    RigidSolver extends this with physics (collision, constraints, dynamics).
    """

    def __init__(self, scene: "Scene", sim: "Simulator", options: "KinematicOptions") -> None:
        super().__init__(scene, sim, options)

        if isinstance(options, RigidOptions):
            self._options = options
        elif isinstance(options, KinematicOptions):
            self._options = RigidOptions(
                dt=options.dt,
                enable_collision=False,
                enable_joint_limit=False,
                enable_self_collision=False,
                enable_neutral_collision=False,
                enable_adjacent_collision=False,
                disable_constraint=True,
                max_collision_pairs=0,
                enable_multi_contact=False,
                enable_mujoco_compatibility=False,
                use_contact_island=False,
                use_hibernation=False,
                max_dynamic_constraints=0,
                iterations=0,
            )
        else:
            gs.raise_exception(f"Invalid options type: {type(options)}")

        self._enable_collision = False
        self._enable_mujoco_compatibility = False
        self._requires_grad = False
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs
        # Hibernation parameters (zeroed out, required by DataManager)
        self._hibernation_thresh_vel = 0.0
        self._hibernation_thresh_acc = 0.0

        self.collider = None
        self.constraint_solver = None

        self.qpos = None

        self._fk_dirty = False

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
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            visualize_contact=False,
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
        self._post_build_entities()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_entities = self.n_entities

        self._geoms = self.geoms
        self._vgeoms = self.vgeoms
        self._links = self.links
        self._joints = self.joints

        self._setup_equalities()

        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_geoms_ = max(1, self.n_geoms)
        self.n_cells_ = max(1, self.n_cells)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_entities_ = max(1, self.n_entities)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)

        # batch_links_info is required when heterogeneous simulation is used.
        # We must update options because get_links_info reads from solver._options.batch_links_info.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True

        self._build_static_config()
        self._create_data_manager()

        self._init_dof_fields()
        self._init_vert_fields()
        self._init_vvert_fields()
        self._init_geom_fields()
        self._init_vgeom_fields()
        self._init_link_fields()
        self._process_heterogeneous_link_info()
        self._init_entity_fields()

        self._init_envs_offset()

        self._func_update_geoms(self._scene._envs_idx, force_update_fixed_geoms=True)

    def _post_build_entities(self):
        pass

    def _setup_equalities(self):
        self._n_equalities = 0
        self.n_candidate_equalities_ = 1

    def _build_static_config(self):
        # Static config with all physics disabled
        self._static_rigid_sim_config = array_class.StructRigidSimStaticConfig(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=False,
            use_hibernation=False,
            batch_links_info=False,
            batch_dofs_info=False,
            batch_joints_info=False,
            enable_heterogeneous=False,
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
        self.data_manager = array_class.DataManager(self)
        self._errno = self.data_manager.errno
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

    def _process_heterogeneous_link_info(self):
        """
        Process heterogeneous link info: dispatch geoms per environment and compute per-env inertial properties.
        This method is called after _init_link_fields to update the per-environment inertial properties
        for entities with heterogeneous morphs.
        """
        for entity in self._entities:
            # Skip non-heterogeneous entities
            if not entity._enable_heterogeneous:
                continue

            # Get the number of variants for this entity
            n_variants = len(entity.variants_geom_start)

            # Distribute variants across environments using balanced block assignment:
            # - If B >= n_variants: first B/n_variants environments get variant 0, next get variant 1, etc.
            # - If B < n_variants: each environment gets a different variant (some variants unused)
            if self._B >= n_variants:
                base = self._B // n_variants
                extra = self._B % n_variants  # first `extra` chunks get one more
                sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
                variant_idx = np.repeat(np.arange(n_variants), sizes)
            else:
                # Each environment gets a unique variant; variants beyond B are unused
                variant_idx = np.arange(self._B)

            # Get arrays from entity
            np_geom_start = np.array(entity.variants_geom_start, dtype=gs.np_int)
            np_geom_end = np.array(entity.variants_geom_end, dtype=gs.np_int)
            np_vgeom_start = np.array(entity.variants_vgeom_start, dtype=gs.np_int)
            np_vgeom_end = np.array(entity.variants_vgeom_end, dtype=gs.np_int)

            # Process each link in this heterogeneous entity (currently only single-link supported)
            for link in entity.links:
                i_l = link.idx

                # Build per-env arrays for geom/vgeom ranges
                links_geom_start = np_geom_start[variant_idx]
                links_geom_end = np_geom_end[variant_idx]
                links_vgeom_start = np_vgeom_start[variant_idx]
                links_vgeom_end = np_vgeom_end[variant_idx]

                # Build per-env arrays for inertial properties
                links_inertial_mass = np.array(
                    [entity.variants_inertial_mass[v] for v in variant_idx], dtype=gs.np_float
                )
                links_inertial_pos = np.array([entity.variants_inertial_pos[v] for v in variant_idx], dtype=gs.np_float)
                links_inertial_i = np.array([entity.variants_inertial_i[v] for v in variant_idx], dtype=gs.np_float)

                # Update links_info with per-environment values
                # Note: when batch_links_info is True, the shape is (n_links, B)
                kernel_update_heterogeneous_link_info(
                    i_l,
                    links_geom_start,
                    links_geom_end,
                    links_vgeom_start,
                    links_vgeom_end,
                    links_inertial_mass,
                    links_inertial_pos,
                    links_inertial_i,
                    self.links_info,
                )

                # Update active_envs_idx for geoms and vgeoms - indicates which environments each geom is active in
                for geom in link.geoms:
                    active_envs_mask = (links_geom_start <= geom.idx) & (geom.idx < links_geom_end)
                    geom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                    (geom.active_envs_idx,) = np.where(active_envs_mask)

                for vgeom in link.vgeoms:
                    active_envs_mask = (links_vgeom_start <= vgeom.idx) & (vgeom.idx < links_vgeom_end)
                    vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                    (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_vert_fields(self):
        self.verts_info = self.data_manager.verts_info
        self.faces_info = self.data_manager.faces_info
        self.edges_info = self.data_manager.edges_info
        self.free_verts_state = self.data_manager.free_verts_state
        self.fixed_verts_state = self.data_manager.fixed_verts_state

        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                verts=np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                faces=np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                edges=np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                normals=np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                verts_geom_idx=np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                init_center_pos=np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                verts_state_idx=np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                is_fixed=np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                verts_info=self.verts_info,
                faces_info=self.faces_info,
                edges_info=self.edges_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

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

    def _init_geom_fields(self):
        self.geoms_info = self.data_manager.geoms_info
        self.geoms_state = self.data_manager.geoms_state
        self.geoms_init_AABB = self._rigid_global_info.geoms_init_AABB
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), dtype=np.float32)

        if self.n_geoms > 0:
            geoms = self.geoms
            geoms_sol_params = np.array([geom.sol_params for geom in geoms], dtype=gs.np_float)
            geoms_sol_params = self._sanitize_geom_sol_params(geoms_sol_params)

            geoms_center = []
            for geom in geoms:
                tmesh = geom.mesh.trimesh
                if tmesh.is_watertight:
                    geoms_center.append(tmesh.center_mass)
                else:
                    geoms_center.append(np.mean(tmesh.vertices, axis=0))

            kernel_init_geom_fields(
                geoms_pos=np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
                geoms_center=np.array(geoms_center, dtype=gs.np_float),
                geoms_quat=np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
                geoms_link_idx=np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
                geoms_type=np.array([geom.type for geom in geoms], dtype=gs.np_int),
                geoms_friction=np.array([geom.friction for geom in geoms], dtype=gs.np_float),
                geoms_sol_params=geoms_sol_params,
                geoms_vert_start=np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
                geoms_face_start=np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
                geoms_edge_start=np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_start=np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
                geoms_vert_end=np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
                geoms_face_end=np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
                geoms_edge_end=np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_end=np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
                geoms_data=np.array([geom.data for geom in geoms], dtype=gs.np_float),
                geoms_is_convex=np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
                geoms_needs_coup=np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
                geoms_contype=np.array([geom.contype for geom in geoms], dtype=np.int32),
                geoms_conaffinity=np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
                geoms_coup_softness=np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                geoms_coup_friction=np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                geoms_coup_restitution=np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                geoms_is_fixed=np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
                geoms_is_decomp=np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                verts_info=self.verts_info,
                geoms_init_AABB=self.geoms_init_AABB,
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
                entities_geom_start=np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
                entities_geom_end=np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
                entities_gravity_compensation=np.array(
                    [entity.gravity_compensation for entity in entities], dtype=gs.np_float
                ),
                entities_is_local_collision_mask=np.array(
                    [entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool
                ),
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

    def _func_update_geoms(self, envs_idx, *, force_update_fixed_geoms=False):
        kernel_update_geoms(
            envs_idx,
            self.entities_info,
            self.geoms_info,
            self.geoms_state,
            self.links_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            force_update_fixed_geoms,
        )

    # ------------------------------------------------------------------------------------
    # -------------------------------- sanitize I/O --------------------------------------
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

    # ------------------------------------------------------------------------------------
    # --------------------------------- DOF position & FK --------------------------------
    # ------------------------------------------------------------------------------------

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        """Write joint positions. FK is deferred to update_visual_states."""
        if gs.use_zerocopy and self.n_envs == 0:
            # Fast path: direct tensor writes, no kernel launch or sanitization.

            # Cache stable tensor views (created once, never change).
            if not hasattr(self, "_zerocopy_views"):
                self._zerocopy_views = (
                    qd_to_torch(self.dofs_state.pos, transpose=True, copy=False),
                    qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False),
                )
            pos_view, qpos_view = self._zerocopy_views

            # Cache dof index mapping, keyed on dofs_idx so it invalidates when dofs_idx changes.
            cache_key = tuple(int(x) for x in dofs_idx) if dofs_idx is not None else None
            dof_cache = getattr(self, "_zerocopy_dof_cache", None)
            if dof_cache is None or dof_cache[0] != cache_key:
                dofs_idx_t = (
                    torch.as_tensor(dofs_idx, dtype=torch.long, device=gs.device)
                    if dofs_idx is not None
                    else torch.arange(self.n_dofs, dtype=torch.long, device=gs.device)
                )
                qs_idx_t = self._build_dof_to_q_map(dofs_idx_t)
                self._zerocopy_dof_cache = (cache_key, dofs_idx_t, qs_idx_t)
            _, dofs_idx_t, qs_idx_t = self._zerocopy_dof_cache

            if not isinstance(position, torch.Tensor):
                position = torch.as_tensor(position, dtype=gs.tc_float, device=gs.device)
            pos_view[0, dofs_idx_t] = position
            qpos_view[0, qs_idx_t] = position
        else:
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
        self._fk_dirty = True

    def set_dofs_velocity(self, velocity=None, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        """No-op: kinematic entities have no dynamics, so velocity is always zero."""
        pass

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        """Read current DOF positions."""
        tensor = qd_to_torch(self.dofs_state.pos, envs_idx, dofs_idx, transpose=True, copy=True)
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

    def forward_kinematics(self):
        """Run FK to update link/geom poses from current qpos. Called by the visualizer."""
        if not self._fk_dirty:
            return
        envs_idx = self._scene._sanitize_envs_idx(None)
        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._fk_dirty = False

    # ------------------------------------------------------------------------------------
    # ----------------------------------- render -----------------------------------------
    # ------------------------------------------------------------------------------------

    def update_geoms_render_T(self):
        kernel_update_geoms_render_T(
            self._geoms_render_T,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def update_vgeoms_render_T(self):
        kernel_update_vgeoms_render_T(
            self._vgeoms_render_T,
            vgeoms_info=self.vgeoms_info,
            vgeoms_state=self.vgeoms_state,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def update_vgeoms(self):
        kernel_update_vgeoms(self.vgeoms_info, self.vgeoms_state, self.links_state, self._static_rigid_sim_config)

    # ------------------------------------------------------------------------------------
    # -------------------------------- simulation no-ops ----------------------------------
    # ------------------------------------------------------------------------------------

    def substep(self, f):
        """No-op: kinematic entities are not simulated."""
        pass

    def substep_pre_coupling(self, f):
        """No-op: kinematic entities do not participate in coupling."""
        pass

    def substep_post_coupling(self, f):
        """No-op: kinematic entities do not participate in coupling."""
        pass

    def check_errno(self):
        """No-op: kinematic solver has no error conditions to check."""
        pass

    def clear_external_force(self):
        """No-op: kinematic entities have no external forces."""
        pass

    def reset_grad(self):
        """No-op: kinematic solver does not support gradients."""
        self._queried_states.clear()

    def process_input_grad(self):
        """No-op: kinematic solver does not support gradients."""
        pass

    def substep_pre_coupling_grad(self, f):
        """No-op: kinematic solver does not support gradients."""
        pass

    def substep_post_coupling_grad(self, f):
        """No-op: kinematic solver does not support gradients."""
        pass

    def add_grad_from_state(self, state):
        """No-op: kinematic solver does not support gradients."""
        pass

    def collect_output_grads(self):
        """No-op: kinematic solver does not support gradients."""
        pass

    def save_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not save checkpoints."""
        pass

    def load_ckpt(self, ckpt_name):
        """No-op: kinematic solver does not load checkpoints."""
        pass

    # ------------------------------------------------------------------------------------
    # -------------------------------- process_input -------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self, in_backward=False):
        """Process input for entities (set qpos from user commands)."""
        for entity in self._entities:
            entity.process_input()

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def _make_scratch_physics_tensors(self):
        """Create temporary zero tensors for kernel_get_state/kernel_set_state compatibility.

        The kernels require velocity/acceleration/mass/friction fields, but kinematic
        entities have no dynamics so these are always zero. Allocated per-call since
        get_state/set_state are infrequent.
        """
        _B = self._B
        args = {"dtype": gs.tc_float, "requires_grad": False, "scene": self._scene}
        return (
            gs.zeros((_B, self.n_dofs), **args),  # dofs_vel
            gs.zeros((_B, self.n_dofs), **args),  # dofs_acc
            gs.zeros((_B, self.n_links), **args),  # mass_shift
            gs.ones((_B, self.n_geoms), **args),  # friction_ratio
        )

    def get_state(self, f=None):
        if self.is_active:
            s_global = self.sim.cur_step_global
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = KinematicSolverState(self._scene, s_global)

            _vel, _acc, _mass, _friction = self._make_scratch_physics_tensors()
            kernel_get_state(
                qpos=state.qpos,
                vel=_vel,
                acc=_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=_mass,
                friction_ratio=_friction,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
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

            if gs.use_zerocopy:
                errno = qd_to_torch(self._errno, copy=False)
                errno[envs_idx] = 0
            else:
                kernel_set_zero(envs_idx, self._errno)

            _vel, _acc, _mass, _friction = self._make_scratch_physics_tensors()
            kernel_set_state(
                qpos=state.qpos,
                dofs_vel=_vel,
                dofs_acc=_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=_mass,
                friction_ratio=_friction,
                envs_idx=envs_idx,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            kernel_forward_kinematics_links_geoms(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_active(self):
        return self.n_links > 0

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
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        return sum(entity.n_cells for entity in self._entities)

    @property
    def n_vgeoms(self):
        if self.is_built:
            return self._n_vgeoms
        return len(self.vgeoms)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum(entity.n_vverts for entity in self._entities)

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum(entity.n_faces for entity in self._entities)

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum(entity.n_vfaces for entity in self._entities)

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum(entity.n_edges for entity in self._entities)

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
