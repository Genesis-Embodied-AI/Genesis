"""System of the contact between rigid geoms, which opens the second graph of every rigid substep."""

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.collider.collider import (
    build_collider_static_config,
    collider_backward,
    collider_backward_narrowphase,
    collider_clear,
    collider_get_contacts,
    collider_reset,
    func_substep_detection,
    initialize_collider_fields,
    iter_collider_data,
)
from genesis.engine.solvers.rigid.collider import gjk, mpr, support_field
from genesis.utils.misc import qd_to_numpy, qd_to_torch, qd_zero_grad

from .rigid_sdf_system import RigidSdfSystem
from .rigid_solver_system import RigidSolverSystem
from ..simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

EnvIndex: TypeAlias = ArrayLike | torch.Tensor | None
ContactArray: TypeAlias = np.ndarray | torch.Tensor
ContactValue: TypeAlias = ContactArray | tuple[ContactArray, ...]
ContactResult: TypeAlias = dict[str, ContactValue]


class RigidRigidContactSystem(System):
    """Own the rigid-rigid collision data and detect contacts when collision is enabled.

    It covers every pair of rigid geoms, terrains and planes included, and hands the contacts to the constraint solve.
    The contacts of other solvers with rigid geoms belong to their couplings.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """All collision state, configuration and host caches owned by rigid-rigid contact.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        collider_config: array_class.ColliderStaticConfig
        collider_info: array_class.ColliderInfo
        collider_state: array_class.ColliderState
        mpr_info: array_class.MPRInfo
        mpr_state: array_class.MPRState
        contact0_mpr_state: array_class.MPRState
        multicontact_mpr_state: array_class.MPRState
        gjk_config: array_class.GJKStaticConfig
        gjk_info: array_class.GJKInfo
        gjk_state: array_class.GJKState
        contact0_gjk_state: array_class.GJKState
        multicontact_gjk_state: array_class.GJKState
        support_field_info: array_class.SupportFieldInfo

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_solver_system = Require(RigidSolverSystem)

    rigid_sdf_system = Require(RigidSdfSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__()
        self._scene = scene
        self.data = self.Data()
        self._delegated_link_idxs = frozenset()
        self._exclusive_link_idxs = frozenset()
        self._mc_perturbation = 1e-3 if scene.options.rigid.enable_mujoco_compatibility else 5e-3
        self._mc_tolerance = 1e-3 if scene.options.rigid.enable_mujoco_compatibility else 1.5e-2
        self._box_max_contacts_per_pair = 16
        self._diff_pos_tolerance = 1e-2
        self._diff_normal_tolerance = 1e-2
        self._prune_deep_penetration_ratio = 3.0
        self._prune_max_contacts_per_link_pair = 32
        self._prune_max_contacts_floor = 512
        self._contact_data_cache: dict[
            tuple[bool, bool, bool],
            dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
        ] = {}
        self._contact_data: dict[str, torch.Tensor] | None = None

    def build(self) -> None:
        from .rigid_island_system import RigidIslandSystem

        self.island_system = self.require(RigidIslandSystem)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def prepare(self):
        return self.clear_contact_data_cache

    @System.action(kind="stage")
    def detect(self):
        rigid = self.rigid_solver_system
        return (
            func_substep_detection,
            rigid.data,
            rigid.rigid_config,
            self.data,
            self.island_system.data,
        )

    @System.action(kind="host")
    def detect_kernel(self):
        rigid = self.rigid_solver_system
        return (
            kernel_substep_detection,
            rigid.data,
            rigid.rigid_config,
            self.data,
            self.island_system.data,
        )

    @System.action(kind="host")
    def prepare_backward(self):
        return self.clear_backward_input_grads

    # -------------------------------- Public get/set API --------------------------------

    def configure_external_collision_links(
        self,
        delegated_link_idxs: Iterable[int],
        exclusive_link_idxs: Iterable[int],
    ) -> None:
        """Configure links whose rigid-rigid collision handling belongs to an external backend."""
        if self.is_building:
            raise RuntimeError("Collision exclusions are configured after Engine.build freezes the System topology.")
        self._delegated_link_idxs = frozenset(delegated_link_idxs)
        self._exclusive_link_idxs = frozenset(exclusive_link_idxs)

    def detect_launching_kernels(self) -> None:
        """Detect the contacts between rigid geoms of a substep through the detection kernel alone."""
        self.clear_contact_data_cache()
        self.detect_kernel.invoke()

    def reset_contacts(self, envs_idx: EnvIndex, *, cache_only: bool = True) -> None:
        """Restart the warm start of the detection in the given environments."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        collider_reset(
            self.data,
            self._contact_data_cache,
            self._contact_data,
            self.rigid_solver_system.rigid_config,
            envs_idx,
            cache_only=cache_only,
        )

    def clear_contacts(self, envs_idx: EnvIndex) -> None:
        """Clear the contacts and the warm start of the detection in the given environments."""
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        collider_clear(
            self.data,
            self._contact_data_cache,
            self._contact_data,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            self.rigid_solver_system._use_hibernation,
            envs_idx,
        )

    def get_contacts(
        self,
        as_tensor: bool = True,
        to_torch: bool = True,
        keep_batch_dim: bool = False,
        is_padded: bool = False,
    ) -> ContactResult:
        return collider_get_contacts(
            self.data,
            self._contact_data_cache,
            self._contact_data,
            self.rigid_solver_system.rigid_config,
            self.rigid_solver_system.n_envs,
            as_tensor,
            to_torch,
            keep_batch_dim,
            is_padded,
        )

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        return iter_collider_data(self.data, self._use_split_narrowphase)

    def backward(
        self,
        dL_dposition: np.ndarray | torch.Tensor,
        dL_dnormal: np.ndarray | torch.Tensor,
        dL_dpenetration: np.ndarray | torch.Tensor,
    ) -> None:
        collider_backward(
            self.data,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
            dL_dposition,
            dL_dnormal,
            dL_dpenetration,
        )

    def backward_narrowphase(self) -> None:
        collider_backward_narrowphase(
            self.data,
            self.rigid_solver_system.data,
            self.rigid_solver_system.rigid_config,
        )

    def clear_backward_input_grads(self) -> None:
        """Clear the Contact-owned adjoints seeded by reversed constraint rows."""
        contact_data = self.data.collider_state.contact_data
        qd_zero_grad(contact_data.pos)
        qd_zero_grad(contact_data.normal)
        qd_zero_grad(contact_data.penetration)
        if gs.use_zerocopy and gs.backend == gs.metal:
            torch.mps.synchronize()

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the collision data and record the configuration the detection graph compiles against."""
        rigid = self.rigid_solver_system
        self._mpr = mpr.MPR(
            batch_size=rigid._B,
            enable_mujoco_compatibility=rigid._enable_mujoco_compatibility,
        )
        self._gjk = gjk.GJK(
            batch_size=rigid._B,
            rigid_config=rigid.rigid_config,
            options=rigid._options,
            enable_mujoco_compatibility=rigid._enable_mujoco_compatibility,
            enable_multi_contact=rigid._enable_multi_contact,
        )
        self._support_field = support_field.SupportField(
            geoms=rigid.geoms,
            dyn_info=rigid.data.dyn_info,
            rigid_config=rigid.rigid_config,
        )

        (
            self.data.collider_config,
            self._n_possible_pairs,
            self._collision_pair_idx,
            self._valid_collision_pairs,
            self._large_contact_pair_mask,
        ) = build_collider_static_config(
            options=rigid._options,
            rigid_config=rigid.rigid_config,
            batch_size=rigid._B,
            n_envs=rigid.n_envs,
            n_geoms=rigid.n_geoms,
            entities=tuple(rigid.entities),
            links=tuple(rigid.links),
            geoms=tuple(rigid.geoms),
            equalities=tuple(rigid.equalities),
            geoms_pos=qd_to_numpy(rigid.data.dyn_state.geoms.pos, transpose=True),
            geoms_quat=qd_to_numpy(rigid.data.dyn_state.geoms.quat, transpose=True),
            enable_mujoco_compatibility=rigid._enable_mujoco_compatibility,
            enable_self_collision=rigid._enable_self_collision,
            enable_adjacent_collision=rigid._enable_adjacent_collision,
            enable_neutral_collision=rigid._enable_neutral_collision,
            requires_grad=rigid._requires_grad,
            delegated_link_idxs=self._delegated_link_idxs,
            exclusive_link_idxs=self._exclusive_link_idxs,
            box_max_contacts_per_pair=self._box_max_contacts_per_pair,
        )

        self._use_split_narrowphase = self.data.collider_config.split_narrowphase
        self._use_coop_dedup = self.data.collider_config.coop_dedup
        if self._use_split_narrowphase:
            gpu_cores = self.data.collider_config.gpu_cores
            n_contact0_threads = rigid._B * ((gpu_cores + rigid._B - 1) // rigid._B)
            self._mpr.activate(n_contact0_threads, gpu_cores)
            self._gjk.activate(n_contact0_threads, gpu_cores)
        elif self.data.collider_config.has_non_box_plane_convex_convex:
            self._gjk.activate()
        if (
            self.data.collider_config.has_terrain
            or self.data.collider_config.has_non_box_plane_convex_convex
        ):
            self._support_field.activate()

        self.data.mpr_info = self._mpr._mpr_info
        self.data.mpr_state = self._mpr.mpr_state
        self.data.contact0_mpr_state = self._mpr.contact0_mpr_state
        self.data.multicontact_mpr_state = self._mpr.multicontact_mpr_state
        self.data.gjk_config = self._gjk.gjk_config
        self.data.gjk_info = self._gjk._gjk_info
        self.data.gjk_state = self._gjk.gjk_state
        self.data.contact0_gjk_state = self._gjk.contact0_gjk_state
        self.data.multicontact_gjk_state = self._gjk.multicontact_gjk_state
        self.data.support_field_info = self._support_field.support_field_info

        initialize_collider_fields(
            self.data,
            rigid_data=rigid.data,
            sdf_info=self.rigid_sdf_system.data.sdf_info,
            mpr_info=self.data.mpr_info,
            gjk_info=self.data.gjk_info,
            support_field_info=self.data.support_field_info,
            options=rigid._options,
            rigid_config=rigid.rigid_config,
            collider_config=self.data.collider_config,
            batch_size=rigid._B,
            n_geoms=rigid.n_geoms,
            n_links=rigid.n_links,
            n_verts=rigid.n_verts,
            max_collision_pairs=rigid.max_collision_pairs,
            entities=tuple(rigid.entities),
            geoms=tuple(rigid.geoms),
            requires_grad=rigid._requires_grad,
            n_possible_pairs=self._n_possible_pairs,
            collision_pair_idx=self._collision_pair_idx,
            valid_collision_pairs=self._valid_collision_pairs,
            large_contact_pair_mask=self._large_contact_pair_mask,
            mc_perturbation=self._mc_perturbation,
            mc_tolerance=self._mc_tolerance,
            diff_pos_tolerance=self._diff_pos_tolerance,
            diff_normal_tolerance=self._diff_normal_tolerance,
            prune_deep_penetration_ratio=self._prune_deep_penetration_ratio,
            prune_max_contacts_per_link_pair=self._prune_max_contacts_per_link_pair,
            prune_max_contacts_floor=self._prune_max_contacts_floor,
        )
        self._initialize_contact_views()
        self.clear_contacts(None)

    def _initialize_contact_views(self) -> None:
        if not gs.use_zerocopy:
            return
        self._contact_data = {}
        try:
            qd_to_torch(self.data.collider_state.n_contacts, copy=False)
            qd_to_torch(self.data.collider_state.contact_sort_idx, transpose=True, copy=False)
            qd_to_torch(self.data.collider_state.first_time, copy=False)
            if self.data.collider_config.has_non_box_plane_convex_convex:
                qd_to_torch(self.data.collider_state.contact_cache.normal, copy=False)
                qd_to_torch(self.data.collider_state.contact_cache.penetration, copy=False)
            for key, name in (
                ("link_a", "link_a"),
                ("link_b", "link_b"),
                ("geom_a", "geom_a"),
                ("geom_b", "geom_b"),
                ("penetration", "penetration"),
                ("position", "pos"),
                ("normal", "normal"),
                ("force", "force"),
            ):
                self._contact_data[key] = qd_to_torch(
                    getattr(self.data.collider_state.contact_data, name),
                    transpose=True,
                    copy=False,
                )
        except ValueError:
            self._contact_data = None

    def clear_contact_data_cache(self) -> None:
        """Drop the contacts read back during the previous substep, which the detection is about to replace."""
        self._contact_data_cache.clear()

@register_system(SimulatorEngine)
def create_rigid_rigid_contact_system(engine: SimulatorEngine) -> RigidRigidContactSystem | None:
    """Create the rigid contact data owner used by the public Collider API."""
    from genesis.engine.entities import RigidEntity

    return (
        RigidRigidContactSystem(engine.scene)
        if engine.scene.options.rigid.enable_collision
        and any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
        else None
    )


@qd.kernel
def kernel_substep_detection(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    contact_data: qd.template(),
    island_data: qd.template(),
):
    func_substep_detection(rigid_data, rigid_config, contact_data, island_data)
