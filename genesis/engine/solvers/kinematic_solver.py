from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity import KinematicEntity
from genesis.engine.materials import Kinematic
from genesis.engine.solvers.rigid import kinematic_tools
from genesis.engine.solvers.rigid.kinematic_tools import balanced_variant_mapping as _balanced_variant_mapping
from genesis.engine.solvers.rigid.kinematic_tools import fill_base_link_geom_offsets as _fill_base_link_geom_offsets
from genesis.engine.solvers.rigid.kinematic_tools import offset_world_shift as _offset_world_shift
from genesis.engine.solvers.rigid.kinematic_tools import select_links_offset as _select_links_offset
from genesis.engine.solvers.rigid.rigid_tools import sanitize_io_variables
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    kernel_inverse_kinematics_entity,
    kernel_set_ik_targets,
)
from genesis.engine.states.solvers import KinematicSolverState, SolverCheckpoint
from genesis.options.morphs import Morph
from genesis.options.solvers import KinematicOptions
from genesis.utils.misc import (
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_torch,
    qd_zero_grad,
)

from .base_solver import MutatedLinks, Solver, StateChange, mutates
from .rigid.abd.accessor import (
    kernel_get_kinematic_state,
    kernel_get_links_vel,
    kernel_set_dofs_force_grad,
    kernel_set_dofs_position,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
    kernel_set_dofs_zero_velocity,
    kernel_set_kinematic_state,
    kernel_set_links_pos,
    kernel_set_links_pos_grad,
    kernel_set_links_quat,
    kernel_set_links_quat_grad,
    kernel_set_qpos,
    kernel_set_vverts,
)
from .rigid.abd.forward_kinematics import (
    kernel_forward_kinematics,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics,
    kernel_masked_forward_velocity,
    kernel_update_vgeoms,
    kernel_update_vverts_for_vgeoms,
)
from .rigid.abd.misc import (
    kernel_init_entity_fields,
    kernel_update_heterogeneous_links_vgeom,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class KinematicSolver(Solver):
    """
    Base solver for articulated kinematic entities (FK, rendering, state get/set).

    Provides the full build pipeline, field init methods, counter properties,
    render methods, and IO sanitization shared by both KinematicSolver and RigidSolver.

    RigidSolver extends this with physics (collision, constraints, dynamics).
    """

    material_cls = Kinematic
    # The kinds of entity this solver builds, each with the morph class it is built from, the most specific first
    _entity_classes = ((Morph, KinematicEntity),)

    def __init__(self, scene: "Scene", sim: "Simulator", options: "KinematicOptions") -> None:
        super().__init__(scene, sim, options)

        self._enable_collision = False
        self._enable_mujoco_compatibility = False
        self._requires_grad = False
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        self.collider = None
        self.constraint_solver = None

        self.qpos = None

        self._is_forward_pos_updated: bool = False
        self._is_forward_vel_updated: bool = False

        self._vfaces_raycast_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------------------------
    # ----------------------------------- add_entity -------------------------------------
    # ------------------------------------------------------------------------------------

    def add_entity(
        self, idx, material, morph, surface, visualize_contact=False, name=None, desc=None
    ) -> "KinematicEntity":
        """Create an entity from its description.

        Given no description, one is resolved from the other arguments (see 'KinematicEntityDescription'), the morph
        deciding the kind of entity among those the solver builds ('_entity_classes'). The class of a given description
        says which kind it holds. Creation from a description reads no asset.
        """
        if desc is None:
            morphs = (morph,) if isinstance(morph, Morph) else tuple(morph)
            entity_cls = next(cls for morph_cls, cls in self._entity_classes if isinstance(morphs[0], morph_cls))
            desc = entity_cls._description_cls.resolve(
                morphs, material, surface, self._options, self._enable_mujoco_compatibility, visualize_contact, name
            )
        else:
            for _, entity_cls in self._entity_classes:
                if entity_cls._description_cls is type(desc):
                    break
            else:
                gs.raise_exception(f"{type(self).__name__} simulates no entity described by {type(desc).__name__}.")
        self._enable_heterogeneous |= bool(desc.variants)
        entity = entity_cls(self._scene, self, idx, desc)
        self._entities.append(entity)
        # An attachment names an entity added before this one, which 'attach' requires, so it is restored here
        if desc.attachment is not None:
            entity.attach(self._scene.get_entity(desc.attachment.entity_name), desc.attachment.link_name)
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

        # Resolve the link-reach structures of @mutates notifications; see Solver.__init__.
        self._articulated_links_idx = np.array([link.idx for link in self.links if not link.is_fixed])
        self._links_parent_idx = np.array([link.parent_idx for link in self.links])

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_custom_vverts = self.n_custom_vverts
        self._n_custom_vfaces = self.n_custom_vfaces
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

        # World<-authored pose offset, applied by the relative get/set methods. Links carry the morph offset composed
        # with the base-link inertial alignment (identity for non-base links); geoms and visual geoms carry only the
        # morph offset, since alignment re-expresses them into the aligned frame. Heterogeneous entities vary the
        # base-link offset per environment (one offset per variant), so the link offset is stored per-env when any such
        # entity has divergent variant offsets, and without an environment axis otherwise. The relative getters
        # recompute the inverse on the fly.
        links_offset_per_env = any(
            not all(
                np.allclose(variant.offset_pos, entity._desc.variants[0].offset_pos)
                and np.allclose(variant.offset_quat, entity._desc.variants[0].offset_quat)
                for variant in entity._desc.variants
            )
            for entity in self._entities
        )
        offset_shape = (self._B, self.n_links) if links_offset_per_env else (self.n_links,)
        links_offset_pos = np.zeros((*offset_shape, 3), dtype=gs.np_float)
        links_offset_quat = np.tile(gu.identity_quat(), (*offset_shape, 1))
        vgeoms_offset_pos = np.zeros((self.n_vgeoms, 3), dtype=gs.np_float)
        vgeoms_offset_quat = np.tile(gu.identity_quat(), (self.n_vgeoms, 1))
        for entity in self._entities:
            if links_offset_per_env and entity._desc.variants:
                variant_idx = _balanced_variant_mapping(len(entity._desc.variants), self._B)
                links_offset_pos[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_pos for v in variant_idx]
                )
                links_offset_quat[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_quat for v in variant_idx]
                )
                vgeoms_ranges = entity.base_link._variant_vgeom_ranges
            else:
                # Each root link carries its own offset; children inherit it through the kinematic chain and stay
                # identity. Visual geoms get the morph offset conjugated into their own frame.
                for local_idx, link in enumerate(entity.links):
                    links_offset_pos[..., entity._link_start + local_idx, :] = link.desc.offset_pos
                    links_offset_quat[..., entity._link_start + local_idx, :] = link.desc.offset_quat
                vgeoms_ranges = None
            _fill_base_link_geom_offsets(vgeoms_offset_pos, vgeoms_offset_quat, entity, entity.vgeoms, vgeoms_ranges)
        # Per-link identity masks gate the relative get/set/backward paths: a relative set on a link whose offset is
        # identity is a plain passthrough, while a non-identity offset rewrites the world position or drops the
        # composition jacobian. Kept as host-side numpy (never used in kernels) so the gates avoid a GPU->host sync.
        self._links_offset_quat_is_identity = np.all(
            np.isclose(gu.quat_to_xyz(links_offset_quat), 0.0, atol=gs.EPS), axis=-1
        )
        self._links_offset_pos_is_identity = np.all(np.isclose(links_offset_pos, 0.0, atol=gs.EPS), axis=-1)
        if links_offset_per_env:
            self._links_offset_quat_is_identity = self._links_offset_quat_is_identity.all(axis=0)
            self._links_offset_pos_is_identity = self._links_offset_pos_is_identity.all(axis=0)
        # The link offset tensors are allocated even when every link is identity, so the getter kernels can take them
        # unconditionally.
        self._links_offset_pos = torch.from_numpy(links_offset_pos).to(device=gs.device, dtype=gs.tc_float)
        self._links_offset_quat = torch.from_numpy(links_offset_quat).to(device=gs.device, dtype=gs.tc_float)
        self._vgeoms_offset_pos = self._vgeoms_offset_quat = None
        if not (
            np.allclose(vgeoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(vgeoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._vgeoms_offset_pos = torch.from_numpy(vgeoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._vgeoms_offset_quat = torch.from_numpy(vgeoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_custom_vverts_ = max(1, self.n_custom_vverts)
        self.n_custom_vfaces_ = max(1, self.n_custom_vfaces)
        self.n_entities_ = max(1, self.n_entities)

        # The kinematic roots and trees, see roots_link_idx and trees_root_idx in array_class.py. A link no dof moves,
        # its own or an ancestor's, is static and belongs to no tree. A moving link roots the tree of its parent, or its
        # own when its parent is static or absent. The parents precede the children.
        self._n_roots = len({link.root_idx for link in self.links})
        self.n_roots_ = max(1, self._n_roots)
        links_is_static = np.ones(self.n_links, dtype=bool)
        links_tree_root_idx = np.full(self.n_links, -1, dtype=gs.np_int)
        for link in self.links:
            is_parent_static = link.parent_idx == -1 or links_is_static[link.parent_idx]
            links_is_static[link.idx] = is_parent_static and link.n_dofs == 0
            if links_is_static[link.idx]:
                continue
            links_tree_root_idx[link.idx] = link.idx if is_parent_static else links_tree_root_idx[link.parent_idx]
        self._links_tree_root_idx = links_tree_root_idx
        self._n_trees = np.unique(links_tree_root_idx[links_tree_root_idx >= 0]).size
        self.n_trees_ = max(1, self._n_trees)

        # The level of a link in its root and in its tree is its number of ancestors below the root link of either (see
        # trees_levels_links_idx in array_class.py). Each pass moves every link one ancestor up, so the depth of a link
        # is the number of passes it takes to run out of parents.
        links_depth = np.zeros(self.n_links, dtype=gs.np_int)
        links_ancestor_idx = self._links_parent_idx
        while (links_ancestor_idx >= 0).any():
            has_ancestor = links_ancestor_idx >= 0
            links_depth += has_ancestor
            links_ancestor_idx = np.where(has_ancestor, self._links_parent_idx[links_ancestor_idx], -1)
        self._links_root_idx = np.array([link.root_idx for link in self.links], dtype=gs.np_int)
        self._links_root_level = links_depth - links_depth[self._links_root_idx]
        self._links_tree_level = np.where(links_tree_root_idx >= 0, links_depth - links_depth[links_tree_root_idx], -1)

        # Heterogeneous simulation stores the link, joint and dof info per environment, every variant stating its own
        # (see 'dispatch_heterogeneous_kinematics'). We must update options because the getters read from them.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True
            self._options.batch_joints_info = True
            self._options.batch_dofs_info = True

        self._build_static_config()
        self._create_data_manager()

        kinematic_tools.init_dof_fields(self.joints, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kinematic_tools.init_vvert_fields(self.vgeoms, self._entities, self.n_vverts, self.dyn_info, self.rigid_config)
        kinematic_tools.init_vgeom_fields(self.vgeoms, self.dyn_info, self.rigid_config)
        kinematic_tools.init_link_fields(
            self.links,
            self.joints,
            self._entities,
            self.init_qpos,
            self.n_dofs_,
            self._B,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self._sanitize_joint_sol_params,
        )
        self.qpos = self.rigid_info.qpos
        self.qpos0 = self.rigid_info.qpos0
        self.links_T = self.rigid_info.links_T
        # Dispatch heterogeneous variant vgeom ranges per-environment
        self._dispatch_heterogeneous_vgeoms()
        kinematic_tools.init_tree_fields(
            self.links,
            self.n_dofs,
            self._links_root_idx,
            self._links_parent_idx,
            self._links_tree_root_idx,
            self._links_tree_level,
            self._links_root_level,
            self.rigid_info,
            self.rigid_config,
        )
        self._init_entity_fields()
        self._vfaces_raycast_mask = kinematic_tools.init_vfaces_raycast_mask(
            self.entities, self.n_vgeoms_, self.dyn_info
        )

        self._init_vverts_state()

    def _init_vverts_state(self):
        # Seed vverts_state.pos with one FK pass so opt-in entities show their initial pose from frame 0 even before
        # the user touches the buffer with set_vverts.
        if self.n_custom_vverts == 0:
            return
        opt_in_vgeoms_idx = np.array(
            [vgeom.idx for entity in self._entities if entity._morph.enable_custom_vverts for vgeom in entity.vgeoms],
            dtype=gs.np_int,
        )
        if opt_in_vgeoms_idx.size == 0:
            return
        self.update_vgeoms()
        self.update_vverts_for_vgeoms(opt_in_vgeoms_idx)

    def _build_static_config(self):
        # Static config with all physics disabled
        self.rigid_config = array_class.RigidSimStaticConfig(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=False,
            use_hibernation=False,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=self._options.batch_dofs_info,
            batch_joints_info=self._options.batch_joints_info,
            enable_mujoco_compatibility=False,
            enable_elliptic_friction=False,
            enable_signorini_contact=False,
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
        self.rigid_info = self.data_manager.rigid_info
        self._rigid_adjoint_cache = self.data_manager.rigid_adjoint_cache
        self.dyn_info = self.data_manager.dyn_info
        self.dyn_state = self.data_manager.dyn_state
        self.kinematics_scratch = self.data_manager.kinematics_scratch

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        yield from array_class.iter_data(self, "rigid_config")
        yield from array_class.iter_data(self.data_manager, "errno", array_class.DataKind.STATE)
        structs = ["rigid_info", "dyn_info", "dyn_state", "rigid_adjoint_cache"]
        if self.rigid_config.requires_grad:
            structs.append("dyn_state_adjoint_cache")
        for name in structs:
            yield from array_class.iter_data(self.data_manager, name)
        # The flags describe the derived poses and velocities, so they travel with them
        yield from array_class.iter_data(self, "_is_forward_pos_updated", array_class.DataKind.DERIVED)
        yield from array_class.iter_data(self, "_is_forward_vel_updated", array_class.DataKind.DERIVED)

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

    def _dispatch_heterogeneous_vgeoms(self):
        """Override per-link vgeom ranges for heterogeneous variants. RigidSolver extends this."""
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self._B)

            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

            kernel_update_heterogeneous_links_vgeom(link.idx, vgeom_starts, vgeom_ends, self.dyn_info)

            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_entity_fields(self):
        if self._entities:
            entities = self._entities
            kernel_init_entity_fields(
                np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                np.array([0 for entity in entities], dtype=gs.np_int),
                np.array([0 for entity in entities], dtype=gs.np_int),
                np.array([0.0 for entity in entities], dtype=gs.np_float),
                np.array([False for entity in entities], dtype=gs.np_bool),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

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
                self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

    def substep_post_coupling_grad(self, f):
        pass

    def add_grad_from_state(self, state):
        if self.is_active:
            kinematic_tools.add_grad_from_state(state, self.dyn_state, self.rigid_info, self.rigid_config)

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
        # Zero every Quadrants-side adjoint buffer the solver owns so a subsequent backward pass does not accumulate on
        # top of stale residues from the previous one. The rigid/kinematic `.grad` pipeline writes through
        # `qd.atomic_add` (e.g. `kernel_get_state_grad`, the `substep_pre_coupling_grad` chain), which is the correct
        # reverse-mode accumulation rule for a single backward pass but requires the buffers to start at zero. MPM's
        # `reset_grad` already zeros its fields explicitly; the rigid/kinematic solvers historically relied on users
        # rebuilding the scene per `loss.backward()`, which breaks training loops that call `scene.reset()` between
        # backward passes.
        if self._requires_grad:
            qd_zero_grad(self.dyn_state.links)
            qd_zero_grad(self.dyn_state.dofs)
            qd_zero_grad(self.dyn_state.joints)
            qd_zero_grad(self.rigid_info)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
        for entity in self._entities:
            entity.reset_grad()
        self._queried_states.clear()

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
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                self.dyn_state,
                self.rigid_info,
                self.rigid_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(self, f, state, envs_idx=None, *, partial: bool = False) -> None:
        if not self.is_active:
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        kernel_set_kinematic_state(
            envs_idx,
            state.qpos,
            state.dofs_vel,
            state.links_pos,
            state.links_quat,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        if not partial:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def __setstate__(self, state: SolverCheckpoint) -> None:
        super().__setstate__(state)
        if array_class.DataKind.DERIVED not in state.kinds:
            # Without the derived arrays, forward kinematics runs now so the scene is drawn where the record put it. The
            # next step recomputes the rest of the derived state.
            self._is_forward_pos_updated = self._is_forward_vel_updated = False
            self.update_forward_pos()

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
        return sanitize_io_variables(
            self._scene,
            tensor,
            inputs_idx,
            input_size,
            idx_name,
            envs_idx,
            element_shape,
            batched=batched,
            skip_allocation=skip_allocation,
        )

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        # Without any offset position on the targeted links, the authored and internal link origins coincide, so a
        # relative set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and self._links_offset_pos_is_identity[idx].all():
            relative = False
        pos, links_idx, envs_idx = self._sanitize_io_variables(
            pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos = pos[None]

        if relative:
            # Compose the body-frame offset onto the authored position while keeping the current orientation, then set
            # the resulting world position.
            cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
            relative = False

        kernel_set_links_pos(
            links_idx, envs_idx, pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if not skip_forward:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_base_links_pos_grad(self, links_idx, envs_idx, relative, pos_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        # A relative 'set_pos' adds 'R(authored_quat) @ offset_pos', which reads the current orientation; that
        # dependency is not propagated, so with a non-zero offset position the backward pass is only supported on links
        # without one. 'relative=False' sets the world position directly and is a plain passthrough.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            gs.raise_exception(
                "Backward pass for 'set_pos' with 'relative=True' is only supported on links without an offset "
                "position (no inertial alignment shifting the link origin). Use 'relative=False' to set the world "
                "position."
            )
        pos_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            pos_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos_grad_ = pos_grad_.unsqueeze(0)
        kernel_set_links_pos_grad(
            links_idx, envs_idx, pos_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_quat(self, quat, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        # Without any pose offset on the targeted links, the authored and internal link origins coincide, so a relative
        # set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            relative = False
        relative_pos_passthrough = relative and self._links_offset_pos_is_identity[idx].all()
        quat, links_idx, envs_idx = self._sanitize_io_variables(
            quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat = quat[None]

        if relative:
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            if not relative_pos_passthrough:
                # The offset position rotates with the orientation, so keep the authored-frame position fixed by
                # rewriting the world position from the current authored position and the new authored orientation.
                cur_pos = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)
                cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                authored_pos = cur_pos - _offset_world_shift(offset_pos, offset_quat, cur_quat)
                world_pos = authored_pos + gu.transform_by_quat(offset_pos, quat)
                kernel_set_links_pos(
                    links_idx, envs_idx, world_pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
                )
            # Compose the offset onto the authored orientation, then set the resulting world orientation.
            quat = gu.transform_quat_by_quat(offset_quat, quat)
            relative = False

        kernel_set_links_quat(
            links_idx, envs_idx, quat, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if not skip_forward:
            kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_base_links_quat_grad(self, links_idx, envs_idx, relative, quat_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        # A relative 'set_quat' composes the orientation offset (a constant right-multiplication) and, when the offset
        # position is non-zero, also rewrites the world position from the input orientation. Neither jacobian is
        # propagated, so the backward pass is only supported when the targeted links carry no pose offset. With
        # 'relative=False' the world orientation is set directly, which is a plain passthrough and always differentiable.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            gs.raise_exception(
                "Backward pass for 'set_quat' with 'relative=True' is only supported on links without a pose offset "
                "(no up-axis conversion or inertial alignment). Use 'relative=False' to set the world orientation."
            )
        quat_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            quat_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        kernel_set_links_quat_grad(
            links_idx, envs_idx, quat_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2 and len(qpos) != len(qs_data):
                    # Note that it is necessary to create a new temporary view because it will be reshaped in-place
                    qs_data.masked_scatter_(envs_idx[:, None], qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask, boolean_mask=False)
                assign_indexed_tensor(data, mask, qpos)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]
            kernel_set_qpos(qs_idx, envs_idx, qpos, self.rigid_info, self.rigid_config)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics
            else:
                fn = kernel_forward_kinematics
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    @mutates(StateChange.DYNAMICS, links=MutatedLinks.ARTICULATED)
    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
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
                    if velocity.ndim == 2 and len(dofs_vel) != len(velocity):
                        # Note that it is necessary to create a new temporary view because it will be reshaped in-place
                        dofs_vel.masked_scatter_(envs_idx[:, None], velocity.view_as(velocity))
                    else:
                        velocity = broadcast_tensor(velocity, gs.tc_float, dofs_vel.shape)
                        torch.where(envs_idx[:, None], velocity, dofs_vel, out=dofs_vel)
            else:
                mask = (
                    (0, *dofs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *dofs_mask, boolean_mask=False)
                )
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dyn_state, self.rigid_config)
            else:
                if self.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

        if not skip_forward:
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_velocity
            else:
                fn = kernel_forward_velocity
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=False)
            self._is_forward_vel_updated = True
        else:
            self._is_forward_vel_updated = False

    def set_dofs_velocity_grad(self, dofs_idx, envs_idx, velocity_grad):
        velocity_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad_, self.dyn_state, self.rigid_config)

    def set_dofs_force_grad(self, dofs_idx, envs_idx, force_grad):
        force_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            force_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force_grad_ = force_grad_.unsqueeze(0)
        kernel_set_dofs_force_grad(dofs_idx, envs_idx, force_grad_, self.dyn_state, self.rigid_config)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        kernel_set_dofs_position(
            dofs_idx, envs_idx, position, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        kernel_forward_kinematics(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def get_terrain_height(self, positions, link_idx, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        return kinematic_tools.get_terrain_height(
            self._links[link_idx].entity,
            link_idx,
            positions,
            envs_idx,
            self.n_envs > 0,
            self.dyn_state,
            self.rigid_config,
        )

    def get_links_pos(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        tensor = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_quat_is_identity[idx].all():
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_pos(self, joints_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.joints.xanchor, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_axis(self, joints_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.joints.xaxis, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_pos(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.vgeoms.pos, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_pos is not None:
            quat = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
            offset_pos = self._vgeoms_offset_pos if vgeoms_idx is None else self._vgeoms_offset_pos[vgeoms_idx]
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_quat(self, vgeoms_idx=None, envs_idx=None, *, relative=False):
        tensor = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_quat is not None:
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(self, links_idx=None, envs_idx=None, *, relative=False):
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        is_relative = relative and not self._links_offset_pos_is_identity[idx].all()
        if gs.use_zerocopy:
            mask = indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, transpose=True)
            cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, transpose=True)
            pos = qd_to_torch(self.dyn_state.links.pos, transpose=True)
            root_COM = qd_to_torch(self.dyn_state.links.root_COM, transpose=True)
            cpos = pos[mask] - root_COM[mask]
            if is_relative:
                quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                cpos = cpos - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel[mask] + cd_ang[mask].cross(cpos, dim=-1)
            return tensor[0] if self.n_envs == 0 else tensor

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        # The frame is passed as a plain int by the owning solver's reference-frame sanitizer.
        kernel_get_links_vel(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            ref=int(gs.link_ref_frame.link_origin),
            is_relative=is_relative,
        )
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
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
        tensor = qd_to_torch(self.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        """Read current DOF positions."""
        tensor = qd_to_torch(self.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_vgeoms(self):
        kernel_update_vgeoms(self.dyn_state, self.dyn_info, self.rigid_config)

    def update_forward_pos(self):
        """Run forward kinematics if links_state is not already up to date for the current pose."""
        if self._is_forward_pos_updated:
            return
        kernel_forward_kinematics(
            self.scene._envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        self._is_forward_pos_updated = True

    def update_vverts_for_vgeoms(self, vgeoms_idx):
        """Refresh the vverts_state.pos slice for the requested vgeoms by re-running FK.

        Used by set_vverts(None, ...) and at scene-build time to initialize the custom buffer with a meaningful pose.
        The kernel is a no-op for vgeoms whose vverts have no state slot.
        """
        if self.n_custom_vverts == 0:
            return
        kernel_update_vverts_for_vgeoms(vgeoms_idx, self.dyn_state, self.dyn_info, self.rigid_config)

    @mutates(StateChange.GEOMETRY)
    def set_vverts(self, custom_vvert_start, custom_vvert_end, vgeoms_idx, vverts, envs_idx=None):
        """Write the slice [custom_vvert_start:custom_vvert_end] of vverts_state.pos.

        vverts=None re-populates the slice from FK by running update_vverts_for_vgeoms over the vgeoms owning the slice.
        Otherwise vverts is broadcast to the slice shape and written directly. vverts_state.pos is always batched;
        envs_idx selects which envs to write to.
        """
        if vverts is None:
            self.update_vverts_for_vgeoms(vgeoms_idx)
            return

        if gs.use_zerocopy:
            data = qd_to_torch(self.dyn_state.vverts.pos, transpose=True, copy=False)
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                pos_slice = data[:, custom_vvert_start:custom_vvert_end]
                if vverts.ndim == 3 and len(vverts) != len(pos_slice):
                    pos_slice.masked_scatter_(envs_idx[:, None, None], vverts.view_as(vverts))
                else:
                    vverts_b = broadcast_tensor(vverts, gs.tc_float, pos_slice.shape)
                    torch.where(envs_idx[:, None, None], vverts_b, pos_slice, out=pos_slice)
            else:
                vverts_mask = indices_to_mask(slice(custom_vvert_start, custom_vvert_end))
                pos_mask = (0, *vverts_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *vverts_mask)
                assign_indexed_tensor(data, pos_mask, vverts)
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        target_shape = (envs_idx.shape[0], custom_vvert_end - custom_vvert_start, 3)
        vverts = broadcast_tensor(vverts, gs.tc_float, target_shape, ("envs", "vverts", "xyz")).contiguous()
        kernel_set_vverts(custom_vvert_start, envs_idx, vverts, self.dyn_state, self.rigid_config)

    def get_vverts(self, custom_vvert_start, custom_vvert_end, envs_idx=None):
        """Return a copy of the vverts_state.pos slice for the given custom-vvert range.

        Shape: (len(envs_idx), custom_vvert_end - custom_vvert_start, 3). envs_idx=None returns every env.
        """
        tensor = qd_to_torch(
            self.dyn_state.vverts.pos, envs_idx, slice(custom_vvert_start, custom_vvert_end), transpose=True, copy=True
        )
        return tensor[0] if self.n_envs == 0 else tensor

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_links_jacobian(self, link_idx, dof_start, n_dofs, local_point=None):
        jacobian = self.kinematics_scratch.jacobian
        return kinematic_tools.get_links_jacobian(
            jacobian,
            link_idx,
            dof_start,
            n_dofs,
            local_point,
            self._B,
            self.dyn_state,
            self.dyn_info,
            self.rigid_config,
        )

    def forward_kinematics_query(self, entity, qpos, envs_idx=None):
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        return kinematic_tools.forward_kinematics_query(
            entity,
            qpos,
            envs_idx,
            self.n_envs > 0,
            self.kinematics_scratch.qpos_cache,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

    def inverse_kinematics(
        self,
        entity_idx,
        q_start,
        link_start,
        joint_start,
        links_idx,
        dofs_idx,
        envs_idx,
        poss,
        quats,
        local_points,
        init_qpos,
        pos_mask,
        rot_mask,
        link_pos_mask,
        link_rot_mask,
        n_qs,
        n_dofs,
        custom_init_qpos,
        max_samples,
        max_solver_iters,
        damping,
        pos_tol,
        rot_tol,
        max_step_size,
        seed,
        respect_joint_limit,
    ):
        """Damped-least-squares inverse kinematics of one entity for the given link targets.

        Returns the best configuration per environment, spanning the entity, and the stacked pose residual of every
        target it carries. Callers narrow both to the environments and targets they asked for.
        """
        n_links = len(links_idx)
        n_tgts = self._options.IK_max_targets
        if n_links > n_tgts:
            gs.raise_exception(f"Cannot solve for {n_links} targets. Raise 'IK_max_targets' above {n_tgts}.")

        ik_state, ik_fk, targets = self.data_manager.ik_scratch

        # Only the leading rows each buffer holds for this solve are written, and the kernel reads no further.
        if gs.use_zerocopy:
            for member, data in (
                (targets.links_idx, links_idx),
                (targets.local_point, local_points),
                (targets.link_pos_mask, link_pos_mask),
                (targets.link_rot_mask, link_rot_mask),
            ):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links] = data
            dofs_idx_t = qd_to_torch(targets.dofs_idx, copy=False)
            dofs_idx_t[: len(dofs_idx)] = dofs_idx
            envs_idx_t = qd_to_torch(targets.envs_idx, copy=False)
            envs_idx_t[: len(envs_idx)] = envs_idx
            pos_mask_t = qd_to_torch(targets.pos_mask, copy=False)
            pos_mask_t[:] = pos_mask
            rot_mask_t = qd_to_torch(targets.rot_mask, copy=False)
            rot_mask_t[:] = rot_mask
            for member, data in ((targets.pos, poss), (targets.quat, quats)):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links, envs_idx] = data
            init_qpos_t = qd_to_torch(targets.init_qpos, copy=False)
            init_qpos_t[envs_idx, :n_qs] = init_qpos
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_ik_targets(
                links_idx,
                dofs_idx,
                envs_idx,
                poss,
                quats,
                local_points,
                init_qpos,
                pos_mask,
                rot_mask,
                link_pos_mask,
                link_rot_mask,
                targets,
            )

        kernel_inverse_kinematics_entity(
            entity_idx,
            q_start,
            link_start,
            joint_start,
            self.dyn_state,
            ik_state,
            ik_fk,
            targets,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            n_qs,
            n_dofs,
            n_links,
            len(dofs_idx),
            len(envs_idx),
            custom_init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            max_step_size,
            seed,
            respect_joint_limit,
        )

        qpos = qd_to_torch(ik_state.qpos_best, transpose=True, copy=True)[..., :n_qs]
        err_pose = qd_to_torch(ik_state.err_pose_best, transpose=True, copy=True).reshape((-1, n_tgts, 6))
        return qpos, err_pose[:, :n_links]

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
    def equalities(self):
        """The equality constraints the solver enforces, which a kinematic solver holds none of."""
        return gs.List()

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
    def vfaces_raycast_mask(self) -> torch.Tensor:
        """Per-vface mask, shape (n_vfaces,), selecting the vfaces opted into visual raycasting.

        A vface is opted in iff its owning vgeom belongs to an entity whose material has use_visual_raycasting=True.
        Both the entity materials and the vface-to-vgeom mapping are fixed once the scene is built, so the mask is
        fitted at build time and shared by every visual raycast BVH (raycaster sensors, viewer plugins) to gate which
        vfaces contribute.
        """
        return self._vfaces_raycast_mask

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
    def n_custom_vverts(self):
        if self.is_built:
            return self._n_custom_vverts
        return sum(entity.n_vverts for entity in self._entities if entity._morph.enable_custom_vverts)

    @property
    def n_custom_vfaces(self):
        if self.is_built:
            return self._n_custom_vfaces
        return sum(entity.n_vfaces for entity in self._entities if entity._morph.enable_custom_vverts)

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
