from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.constants import link_ref_frame
from genesis.engine.entities import RigidEntity
from genesis.engine.materials import Rigid
from genesis.engine.states import RigidSolverState
from genesis.options.solvers import RigidOptions

from ..base_solver import MutatedLinks, StateChange, Subscriber, mutates, register_subscriber
from .rigid_tools import sanitize_io_variables

if TYPE_CHECKING:
    from genesis.engine.core import System
    from genesis.engine.solvers.rigid.collider import gjk, mpr
    from genesis.engine.entities.rigid_entity.rigid_equality import RigidEquality
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.force_fields import ForceField
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.engine.simulator_engine import SimulatorEngine
    from genesis.engine.systems.rigid_constraint import RigidConstraint
    from genesis.engine.systems.rigid_api import RigidAPI
    from genesis.engine.systems.rigid_dynamics import RigidDynamics
    from genesis.engine.systems.rigid_equality_constraint import RigidEqualityConstraint
    from genesis.engine.systems.rigid_island import RigidIsland
    from genesis.engine.systems.rigid_rigid_contact import RigidRigidContact

TensorLike: TypeAlias = ArrayLike | torch.Tensor
IndexLike: TypeAlias = ArrayLike | torch.Tensor | None
HostTensor: TypeAlias = NDArray[np.generic] | torch.Tensor
StructuredValue: TypeAlias = HostTensor | tuple[HostTensor, ...]
StructuredResult: TypeAlias = dict[str, StructuredValue]


class Collider:
    """External compatibility surface of the rigid-rigid contact System, which it delegates to."""

    def __init__(self, rigid_rigid_contact: "RigidRigidContact") -> None:
        self._rigid_rigid_contact = rigid_rigid_contact
        self._data = rigid_rigid_contact.data

    @property
    def collider_config(self) -> array_class.ColliderStaticConfig:
        return self._data.collider_config

    @property
    def collider_info(self) -> array_class.ColliderInfo:
        return self._data.collider_info

    @property
    def collider_state(self) -> array_class.ColliderState:
        return self._data.collider_state

    @property
    def mpr(self) -> "mpr.MPR":
        return self._rigid_rigid_contact.mpr

    @property
    def gjk(self) -> "gjk.GJK":
        return self._rigid_rigid_contact.gjk

    @property
    def _mc_perturbation(self) -> float:
        return self._rigid_rigid_contact.mc_perturbation

    @property
    def _mc_tolerance(self) -> float:
        return self._rigid_rigid_contact.mc_tolerance

    @property
    def _collision_pair_idx(self) -> NDArray[np.integer]:
        return self._rigid_rigid_contact.collision_pair_idx

    @property
    def _use_split_narrowphase(self) -> bool:
        return self._data.collider_config.split_narrowphase

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        """Yield every array and static config of the collider (see 'Solver.data')."""
        return self._rigid_rigid_contact.iter_data()

    def reset(self, envs_idx: IndexLike = None, *, cache_only: bool = True) -> None:
        self._rigid_rigid_contact.reset_contacts(envs_idx, cache_only=cache_only)

    def clear(self, envs_idx: IndexLike = None) -> None:
        self._rigid_rigid_contact.clear_contacts(envs_idx)

    def detection(self) -> None:
        self._rigid_rigid_contact.detect_launching_kernels()

    def detect_collision(self, env_idx: int = 0) -> NDArray[np.integer]:
        """Run detection and return the live geometry pairs of one environment."""
        return self._rigid_rigid_contact.get_collision_pairs(env_idx)

    def get_contacts(
        self, as_tensor: bool = True, to_torch: bool = True, keep_batch_dim: bool = False, is_padded: bool = False
    ) -> StructuredResult:
        return self._rigid_rigid_contact.get_contacts(as_tensor, to_torch, keep_batch_dim, is_padded)

    def backward(
        self,
        dL_dposition: NDArray[np.floating] | torch.Tensor,
        dL_dnormal: NDArray[np.floating] | torch.Tensor,
        dL_dpenetration: NDArray[np.floating] | torch.Tensor,
    ) -> None:
        self._rigid_rigid_contact.backward(dL_dposition, dL_dnormal, dL_dpenetration)

    def backward_narrowphase(self) -> None:
        self._rigid_rigid_contact.backward_narrowphase()


def _constraint_state_property(name: str) -> property:
    return property(lambda facade: getattr(facade._rigid_constraint.data.constraint_state, name))


class ConstraintSolver:
    """External compatibility surface of the rigid constraint Systems.

    It reads the state of the constraint manager and delegates to its owner APIs, and to the equality constraints for
    the equality and weld queries and edits.
    """

    def __init__(
        self,
        rigid_constraint: "RigidConstraint",
        rigid_equality_constraint: "RigidEqualityConstraint",
        participants: tuple["System", ...],
    ) -> None:
        self._rigid_constraint = rigid_constraint
        self._rigid_equality_constraint = rigid_equality_constraint
        self._participants = participants

    @property
    def constraint_state(self) -> array_class.ConstraintState:
        return self._rigid_constraint.data.constraint_state

    @property
    def len_constraints(self) -> int:
        return self._rigid_constraint.get_len_constraints()

    @property
    def _n_iterations(self) -> int:
        return self._rigid_constraint.get_n_iterations()

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        yield from self._rigid_constraint.iter_data()
        for participant in self._participants:
            yield from participant.iter_data()

    qd_n_equalities = _constraint_state_property("qd_n_equalities")
    n_constraints = _constraint_state_property("n_constraints")
    n_constraints_equality = _constraint_state_property("n_constraints_equality")
    n_constraints_frictionloss = _constraint_state_property("n_constraints_frictionloss")
    n_constraints_cone = _constraint_state_property("n_constraints_cone")
    active = _constraint_state_property("active")
    jac = _constraint_state_property("jac")
    efc_D = _constraint_state_property("efc_D")
    aref = _constraint_state_property("aref")
    qacc = _constraint_state_property("qacc")
    qacc_ws = _constraint_state_property("qacc_ws")
    qfrc_constraint = _constraint_state_property("qfrc_constraint")
    efc_force = _constraint_state_property("efc_force")
    grad = _constraint_state_property("grad")

    def reset(self, envs_idx: IndexLike = None) -> None:
        return self._rigid_constraint.reset(envs_idx)

    def clear(self, envs_idx: IndexLike = None) -> None:
        return self._rigid_constraint.clear(envs_idx)

    def clear_query_cache(self) -> None:
        return self._rigid_equality_constraint.clear_query_cache()

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True) -> StructuredResult:
        return self._rigid_equality_constraint.get_equality_constraints(as_tensor, to_torch)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True) -> StructuredResult:
        return self._rigid_equality_constraint.get_weld_constraints(as_tensor, to_torch)

    def add_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: IndexLike = None) -> None:
        return self._rigid_equality_constraint.add_weld_constraint(link1_idx, link2_idx, envs_idx)

    def delete_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: IndexLike = None) -> None:
        return self._rigid_equality_constraint.delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    def backward(self) -> None:
        return self._rigid_constraint.backward()


class RigidSolver:
    material_cls = Rigid

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: RigidOptions) -> None:
        self._uid = gs.UID()
        self._scene = scene
        self._sim = sim
        self._options = options
        self._ffs = []
        self._subscribers: set[Subscriber] = set()
        self._is_mutating = False
        self._mutation_changes: set[StateChange] = set()
        self._mutation_links_idx_list: list[np.ndarray] | None = None
        self._articulated_links_idx: np.ndarray | None = None
        self._links_parent_idx: np.ndarray | None = None
        self._substeps = max(1, round(sim.dt / options.dt)) if "dt" in options.model_fields_set else sim.substeps
        self._substep_dt = sim.dt / self._substeps

        self._enable_mujoco_compatibility = options.enable_mujoco_compatibility
        self._requires_grad = sim.options.requires_grad

        # An admitted Rigid backend owns simulation state. Scenes without Rigid entities keep this facade unwired.
        self._rigid_dynamics: "RigidDynamics | None" = None
        self._rigid_api: "RigidAPI | None" = None
        self._rigid_island: "RigidIsland | None" = None
        self._rigid_rigid_contact: "RigidRigidContact | None" = None
        self._rigid_constraint: "RigidConstraint | None" = None

        self.collider = None
        self.constraint_solver = None

    def wire_backend(self, engine: "SimulatorEngine") -> None:
        """Connect the public facade to the owner APIs of the Systems of the Engine after initialization."""
        from genesis.engine.systems import RigidAPI, RigidDynamics, RigidEqualityConstraint

        system = engine.require(RigidDynamics)
        self._rigid_dynamics = system
        self._rigid_api = engine.require(RigidAPI)
        self._rigid_island = engine.rigid_island
        self._rigid_rigid_contact = engine.rigid_rigid_contact
        self._rigid_constraint = engine.rigid_constraint
        self._articulated_links_idx = system.articulated_links_idx
        self._links_parent_idx = system.links_parent_idx
        self.collider = None if engine.rigid_rigid_contact is None else Collider(engine.rigid_rigid_contact)
        self.constraint_solver = None
        if engine.rigid_constraint is not None:
            self.constraint_solver = ConstraintSolver(
                engine.rigid_constraint,
                engine.require(RigidEqualityConstraint),
                engine.rigid_constraint_participants,
            )

    # Explicit compatibility exports of Rigid-owned data: the facade never owns or reimplements the fields, and a new
    # backend field reaches it only through an intentional export.
    @property
    def dyn_state(self) -> array_class.DynState:
        return self._rigid_dynamics.data.dyn_state

    @property
    def dyn_info(self) -> array_class.DynInfo:
        return self._rigid_dynamics.data.dyn_info

    @property
    def rigid_info(self) -> array_class.RigidInfo:
        return self._rigid_dynamics.data.rigid_info

    @property
    def rigid_config(self) -> array_class.RigidSimStaticConfig:
        return self._rigid_dynamics.rigid_config

    @property
    def qpos(self) -> qd.Tensor:
        return self._rigid_dynamics.data.rigid_info.qpos

    @property
    def qpos0(self) -> qd.Tensor:
        return self._rigid_dynamics.data.rigid_info.qpos0

    @property
    def mass_mat(self) -> qd.Tensor:
        return self._rigid_dynamics.data.rigid_info.mass_mat

    @property
    def meaninertia(self) -> qd.Tensor:
        return self._rigid_dynamics.data.rigid_info.meaninertia

    @property
    def weight_scratch(self) -> array_class.WeightScratch:
        return self._rigid_dynamics.data.weight_scratch

    @property
    def geoms_init_AABB(self) -> qd.Tensor:
        return self._rigid_dynamics.data.geoms_init_AABB

    @property
    def _errno(self) -> qd.Tensor:
        return self._rigid_dynamics.data.errno

    @property
    def _sol_min_timeconst(self) -> float:
        return self._rigid_dynamics.sol_min_timeconst

    @property
    def _base_links_idx(self) -> torch.Tensor:
        return self._rigid_dynamics.base_links_idx

    @property
    def _sol_default_timeconst(self) -> float | None:
        return self._rigid_dynamics.sol_default_timeconst

    def build(self) -> None:
        """Scene owns authoring; RigidDynamics collects and builds the backend model."""

    def _add_force_field(self, force_field: "ForceField") -> None:
        self._ffs.append(force_field)

    def subscribe(self, subscriber: Subscriber) -> None:
        register_subscriber(self._subscribers, subscriber, self.n_links)

    @property
    def uid(self) -> gs.UID:
        return self._uid

    @property
    def scene(self) -> "Scene":
        return self._scene

    @property
    def sim(self) -> "Simulator":
        return self._sim

    @property
    def is_built(self) -> bool:
        return self._scene._is_built

    @property
    def entities(self) -> list[RigidEntity]:
        return gs.List(entity for entity in self._scene.entities if isinstance(entity, RigidEntity))

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def force_fields(self) -> list["ForceField"]:
        return self._ffs

    @property
    def _enable_heterogeneous(self) -> bool:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.enable_heterogeneous
        return any(entity.desc.variants for entity in self.entities)

    @property
    def is_active(self) -> bool:
        return bool(self.entities)

    @property
    def dt(self) -> float:
        return self._sim.dt

    @property
    def substep_dt(self) -> float:
        return self._substep_dt

    @property
    def substeps(self) -> int:
        return self._substeps

    @property
    def _B(self) -> int:
        return self._rigid_dynamics._B if self._rigid_dynamics is not None else self._scene._B

    @property
    def n_envs(self) -> int:
        return self._rigid_dynamics.n_envs if self._rigid_dynamics is not None else self._scene.n_envs

    @property
    def _para_level(self) -> int:
        return self._rigid_dynamics.para_level if self._rigid_dynamics is not None else self._scene._para_level

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_gravity(self, gravity: TensorLike, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_gravity(gravity, envs_idx)

    def get_gravity(self, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_gravity(envs_idx)

    @property
    def links(self) -> list["RigidLink"]:
        return (
            self._rigid_dynamics.links
            if self._rigid_dynamics is not None
            else gs.List(link for entity in self.entities for link in entity.links)
        )

    @property
    def joints(self) -> list["RigidJoint"]:
        return (
            self._rigid_dynamics.joints
            if self._rigid_dynamics is not None
            else gs.List(joint for entity in self.entities for joint in entity.joints)
        )

    @property
    def geoms(self) -> list["RigidGeom"]:
        return (
            self._rigid_dynamics.geoms
            if self._rigid_dynamics is not None
            else gs.List(geom for entity in self.entities for geom in entity.geoms)
        )

    @property
    def vgeoms(self) -> list["RigidVisGeom"]:
        return (
            self._rigid_dynamics.vgeoms
            if self._rigid_dynamics is not None
            else gs.List(vgeom for entity in self.entities for vgeom in entity.vgeoms)
        )

    @property
    def n_links(self) -> int:
        return self._rigid_dynamics.n_links if self._rigid_dynamics is not None else len(self.links)

    @property
    def n_joints(self) -> int:
        return self._rigid_dynamics.n_joints if self._rigid_dynamics is not None else len(self.joints)

    @property
    def n_qs(self) -> int:
        return (
            self._rigid_dynamics.n_qs
            if self._rigid_dynamics is not None
            else sum(entity.n_qs for entity in self.entities)
        )

    @property
    def n_dofs(self) -> int:
        return (
            self._rigid_dynamics.n_dofs
            if self._rigid_dynamics is not None
            else sum(entity.n_dofs for entity in self.entities)
        )

    @property
    def n_vgeoms(self) -> int:
        return self._rigid_dynamics.n_vgeoms if self._rigid_dynamics is not None else len(self.vgeoms)

    @property
    def n_vverts(self) -> int:
        return (
            self._rigid_dynamics.n_vverts
            if self._rigid_dynamics is not None
            else sum(entity.n_vverts for entity in self.entities)
        )

    @property
    def n_vfaces(self) -> int:
        return (
            self._rigid_dynamics.n_vfaces
            if self._rigid_dynamics is not None
            else sum(entity.n_vfaces for entity in self.entities)
        )

    @property
    def n_custom_vverts(self) -> int:
        return (
            self._rigid_dynamics.n_custom_vverts
            if self._rigid_dynamics is not None
            else sum(entity.n_vverts for entity in self.entities if entity._morph.enable_custom_vverts)
        )

    @property
    def n_custom_vfaces(self) -> int:
        return (
            self._rigid_dynamics.n_custom_vfaces
            if self._rigid_dynamics is not None
            else sum(entity.n_vfaces for entity in self.entities if entity._morph.enable_custom_vverts)
        )

    @property
    def init_qpos(self) -> NDArray[np.floating]:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.init_qpos
        return (
            np.concatenate([entity.init_qpos for entity in self.entities], dtype=gs.np_float)
            if self.entities
            else np.array([], dtype=gs.np_float)
        )

    @property
    def vfaces_raycast_mask(self) -> torch.Tensor:
        return self._rigid_dynamics.vfaces_raycast_mask

    def _sanitize_io_variables(
        self,
        tensor: TensorLike,
        inputs_idx: IndexLike,
        input_size: int,
        idx_name: str,
        envs_idx: IndexLike = None,
        element_shape: tuple[int, ...] = (),
        *,
        batched: bool = True,
        skip_allocation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def add_grad_from_state(self, state: RigidSolverState) -> None:
        return self._rigid_dynamics.add_grad_from_state(state)

    def collect_output_grads(self) -> None:
        return self._rigid_dynamics.collect_output_grads()

    def set_base_links_pos_grad(
        self,
        links_idx: IndexLike,
        envs_idx: IndexLike,
        relative: bool,
        pos_grad: torch.Tensor,
    ) -> None:
        return self._rigid_dynamics.set_base_links_pos_grad(links_idx, envs_idx, relative, pos_grad)

    def set_base_links_quat_grad(
        self,
        links_idx: IndexLike,
        envs_idx: IndexLike,
        relative: bool,
        quat_grad: torch.Tensor,
    ) -> None:
        return self._rigid_dynamics.set_base_links_quat_grad(links_idx, envs_idx, relative, quat_grad)

    def set_dofs_velocity_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, velocity_grad: torch.Tensor) -> None:
        return self._rigid_dynamics.set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad)

    def set_dofs_force_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, force_grad: torch.Tensor) -> None:
        return self._rigid_dynamics.set_dofs_force_grad(dofs_idx, envs_idx, force_grad)

    def get_terrain_height(self, positions: TensorLike, link_idx: int, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_terrain_height(positions, link_idx, envs_idx)

    def get_links_quat(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_links_quat(links_idx, envs_idx, relative=relative)

    def get_joints_anchor_pos(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_joints_anchor_pos(joints_idx, envs_idx)

    def get_joints_anchor_axis(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_joints_anchor_axis(joints_idx, envs_idx)

    def get_vgeoms_pos(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_vgeoms_pos(vgeoms_idx, envs_idx, relative=relative)

    def get_vgeoms_quat(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_vgeoms_quat(vgeoms_idx, envs_idx, relative=relative)

    def get_links_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_ang(links_idx, envs_idx)

    def get_links_net_contact_force(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_net_contact_force(links_idx, envs_idx)

    def get_qpos(self, qs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_qpos(qs_idx, envs_idx)

    def get_dofs_velocity(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_velocity(dofs_idx, envs_idx)

    def get_dofs_position(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_position(dofs_idx, envs_idx)

    def get_dofs_limit(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.get_dofs_limit(dofs_idx, envs_idx)

    def update_vgeoms(self) -> None:
        return self._rigid_api.update_vgeoms()

    def update_vverts_for_vgeoms(self, vgeoms_idx: IndexLike) -> None:
        return self._rigid_api.update_vverts_for_vgeoms(vgeoms_idx)

    @mutates(StateChange.GEOMETRY)
    def set_vverts(
        self,
        custom_vvert_start: int,
        custom_vvert_end: int,
        vgeoms_idx: IndexLike,
        vverts: TensorLike | None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.set_vverts(custom_vvert_start, custom_vvert_end, vgeoms_idx, vverts, envs_idx)

    def get_vverts(self, custom_vvert_start: int, custom_vvert_end: int, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_vverts(custom_vvert_start, custom_vvert_end, envs_idx)

    def get_fixed_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        return self._rigid_api.get_fixed_collision_verts(verts_idx)

    def get_free_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        return self._rigid_api.get_free_collision_verts(verts_idx)

    def get_vgeom_transforms_batched(self, envs_idx: IndexLike = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.get_vgeom_transforms_batched(envs_idx)

    def get_links_jacobian(
        self,
        link_idx: int,
        dof_start: int,
        n_dofs: int,
        local_point: TensorLike | None = None,
    ) -> torch.Tensor:
        return self._rigid_api.get_links_jacobian(link_idx, dof_start, n_dofs, local_point)

    def forward_kinematics_query(
        self, entity: RigidEntity, qpos: TensorLike, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.forward_kinematics_query(entity, qpos, envs_idx)

    def inverse_kinematics(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.inverse_kinematics(*args, **kwargs)

    def update_forward_pos(self) -> None:
        return self._rigid_dynamics.update_forward_pos()

    def get_error_envs_mask(self) -> torch.Tensor:
        return self._rigid_api.get_error_envs_mask()

    def check_errno(self) -> None:
        return self._sim._engine.check_errno()

    def _kernel_detect_collision(self) -> None:
        if self._rigid_rigid_contact is not None:
            self._rigid_rigid_contact.redetect_contacts()

    def detect_collision(self, env_idx: int = 0) -> NDArray[np.integer]:
        if self._rigid_rigid_contact is None:
            return np.empty((0, 2), dtype=np.int32)
        return self._rigid_rigid_contact.get_collision_pairs(env_idx)

    def _func_constraint_force(self) -> None:
        return self._sim._engine.constraint_force_host()

    def _func_forward_dynamics(self) -> None:
        return self._rigid_dynamics.compute_forward_dynamics()

    def _func_update_acc(self) -> None:
        return self._rigid_dynamics.update_acceleration()

    def apply_links_external_wrench(
        self,
        force: TensorLike | None = None,
        torque: TensorLike | None = None,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        pos: TensorLike | None = None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ) -> None:
        return self._rigid_api.apply_links_external_wrench(
            force, torque, links_idx, envs_idx, pos=pos, ref=ref, local=local
        )

    def reset_grad(self) -> None:
        return self._rigid_dynamics.reset_grad()

    # ------------------------------------------------------------------------------------
    # -------------------------------- state get/set -------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self, f: int | None = None) -> RigidSolverState | None:
        return None if self._rigid_dynamics is None else self._rigid_dynamics.get_state(f)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def set_state(
        self,
        f: int,
        state: RigidSolverState,
        envs_idx: IndexLike = None,
        *,
        partial: bool = False,
    ) -> None:
        # Restarting the contact and constraint warm starts on a Rigid state write is a legacy behavior of this facade,
        # which asks their owners for it directly.
        contact, constraint = self._rigid_rigid_contact, self._rigid_constraint
        if contact is not None:
            if partial:
                contact.reset_contacts(envs_idx)
            else:
                contact.clear_contacts(envs_idx)
        if constraint is not None:
            if partial:
                constraint.reset(envs_idx)
            else:
                constraint.clear(envs_idx)
        self._rigid_dynamics.set_state(f, state, envs_idx, partial=partial)
        self._restart()

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def notify_state_change(self, envs_idx: IndexLike = None) -> None:
        """Notify the frontend subscribers of a Rigid state the Simulator wrote, such as a scene reset or restore."""

    @mutates(StateChange.MODEL)
    def notify_model_change(self, envs_idx: IndexLike = None) -> None:
        """Notify the frontend subscribers of a Rigid model parameter the Simulator wrote, such as the gravity."""

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        if self._rigid_dynamics is None:
            return
        yield from self._sim.iter_rigid_data()

    def _restart(self) -> None:
        self._rigid_dynamics.restart_owner()
        if self._rigid_rigid_contact is not None:
            self._rigid_rigid_contact.clear_contact_data_cache()
        if self._rigid_constraint is not None:
            self._rigid_constraint.clear_readback_caches()

    def _reset_dependents(self, envs_idx: IndexLike = None) -> None:
        if self._rigid_rigid_contact is not None:
            self._rigid_rigid_contact.reset_contacts(envs_idx)
        if self._rigid_constraint is not None:
            self._rigid_constraint.reset(envs_idx)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_links_pos(self, pos: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_links_pos(pos, links_idx, envs_idx)

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_pos(
        self,
        pos: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        relative: bool = False,
        skip_forward: bool = False,
    ) -> None:
        return self._rigid_api.set_base_links_pos(
            pos, links_idx, envs_idx, relative=relative, skip_forward=skip_forward
        )

    def set_links_quat(self, quat: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_links_quat(quat, links_idx, envs_idx)

    @mutates(StateChange.GEOMETRY, links="links_idx")
    def set_base_links_quat(
        self,
        quat: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        relative: bool = False,
        skip_forward: bool = False,
    ) -> None:
        return self._rigid_api.set_base_links_quat(
            quat, links_idx, envs_idx, relative=relative, skip_forward=skip_forward
        )

    @mutates(StateChange.MODEL)
    def set_links_mass(
        self,
        mass: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        scale_inertia: bool = False,
    ) -> None:
        return self._rigid_api.set_links_mass(mass, links_idx, envs_idx, scale_inertia)

    @mutates(StateChange.MODEL)
    def set_links_inertia(self, inertia: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_links_inertia(inertia, links_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_links_COM(self, com: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_links_COM(com, links_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_ratio(
        self, friction_ratio: TensorLike, geoms_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_api.set_geoms_friction_ratio(friction_ratio, geoms_idx, envs_idx)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_qpos(
        self,
        qpos: TensorLike,
        qs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        self._reset_dependents(envs_idx)
        return self._rigid_api.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

    @mutates(StateChange.MODEL)
    def set_global_sol_params(self, sol_params: TensorLike) -> None:
        return self._rigid_api.set_global_sol_params(sol_params)

    @mutates(StateChange.MODEL)
    def set_sol_params(
        self,
        sol_params: TensorLike,
        geoms_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        joints_idx: IndexLike = None,
        eqs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.set_sol_params(sol_params, geoms_idx, envs_idx, joints_idx=joints_idx, eqs_idx=eqs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_kp(self, kp: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_kp(kp, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_kv(self, kv: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_kv(kv, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_act_gain(self, act_gain: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_act_gain(act_gain, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_act_bias(
        self,
        bias0: TensorLike,
        bias1: TensorLike,
        bias2: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.set_dofs_act_bias(bias0, bias1, bias2, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_force_range(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.set_dofs_force_range(lower, upper, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_stiffness(self, stiffness: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_armature(self, armature: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_armature(armature, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_damping(self, damping: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.set_dofs_damping(damping, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_frictionloss(
        self, frictionloss: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_api.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_dofs_limit(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.set_dofs_limit(lower, upper, dofs_idx, envs_idx)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_dofs_position(self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._reset_dependents(envs_idx)
        return self._rigid_api.set_dofs_position(position, dofs_idx, envs_idx)

    @mutates(StateChange.DYNAMICS, links=MutatedLinks.ARTICULATED)
    def set_dofs_velocity(
        self,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        return self._rigid_api.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_api.control_dofs_force(force, dofs_idx, envs_idx)

    def control_dofs_velocity(
        self, velocity: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_api.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    def control_dofs_position(
        self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_api.control_dofs_position(position, dofs_idx, envs_idx)

    def control_dofs_position_velocity(
        self,
        position: TensorLike,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_api.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

    def get_sol_params(
        self,
        geoms_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        joints_idx: IndexLike = None,
        eqs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_api.get_sol_params(geoms_idx, envs_idx, joints_idx=joints_idx, eqs_idx=eqs_idx)

    def get_links_pos(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        return self._rigid_api.get_links_pos(links_idx, envs_idx, ref=ref, relative=relative)

    def get_links_vel(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        return self._rigid_api.get_links_vel(links_idx, envs_idx, ref=ref, relative=relative)

    def get_links_acc(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_links_acc(links_idx, envs_idx, relative=relative)

    def get_links_acc_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_acc_ang(links_idx, envs_idx)

    def get_links_root_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_root_COM(links_idx, envs_idx)

    def get_links_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_COM(links_idx, envs_idx)

    @property
    def is_links_info_batched(self) -> bool:
        return self._rigid_api.is_links_info_batched()

    def get_links_mass(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_mass(links_idx, envs_idx)

    def get_links_inertia(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_inertia(links_idx, envs_idx)

    def get_links_invweight(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_links_invweight(links_idx, envs_idx)

    def get_geoms_friction_ratio(self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_geoms_friction_ratio(geoms_idx, envs_idx)

    def get_geoms_pos(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_geoms_pos(geoms_idx, envs_idx, relative=relative)

    def get_geoms_quat(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_api.get_geoms_quat(geoms_idx, envs_idx, relative=relative)

    def get_dofs_control_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_control_force(dofs_idx, envs_idx)

    def get_dofs_actuator_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_actuator_force(dofs_idx, envs_idx)

    def get_dofs_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_force(dofs_idx, envs_idx)

    def get_dofs_acc(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_acc(dofs_idx, envs_idx)

    def get_dofs_kp(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_kp(dofs_idx, envs_idx)

    def get_dofs_kv(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_kv(dofs_idx, envs_idx)

    def get_dofs_act_gain(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_act_gain(dofs_idx, envs_idx)

    def get_dofs_act_bias(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._rigid_api.get_dofs_act_bias(dofs_idx, envs_idx)

    def get_dofs_force_range(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.get_dofs_force_range(dofs_idx, envs_idx)

    def get_dofs_stiffness(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_stiffness(dofs_idx, envs_idx)

    def get_dofs_invweight(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_invweight(dofs_idx, envs_idx)

    def get_dofs_armature(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_armature(dofs_idx, envs_idx)

    def get_dofs_damping(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_damping(dofs_idx, envs_idx)

    def get_dofs_frictionloss(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_dofs_frictionloss(dofs_idx, envs_idx)

    def get_mass_mat(
        self,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        decompose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_api.get_mass_mat(dofs_idx, envs_idx, decompose)

    def get_kinetic_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_api.get_kinetic_energy(links_idx, dofs_idx, envs_idx)

    def get_potential_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_api.get_potential_energy(links_idx, dofs_idx, envs_idx)

    def get_total_energy(self, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_total_energy(envs_idx)

    def get_geoms_friction(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_geoms_friction(geoms_idx)

    def get_geoms_friction_torsional(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_geoms_friction_torsional(geoms_idx)

    def get_geoms_friction_rolling(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_api.get_geoms_friction_rolling(geoms_idx)

    def get_AABB(self, entities_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self._sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")
        return self._rigid_api.get_AABB(entities_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction(self, friction: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_api.set_geom_friction(friction, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_api.set_geom_friction_torsional(friction_torsional, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_api.set_geom_friction_rolling(friction_rolling, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction(self, friction: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_api.set_geoms_friction(friction, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_api.set_geoms_friction_torsional(friction_torsional, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_api.set_geoms_friction_rolling(friction_rolling, geoms_idx)

    @mutates(StateChange.MODEL)
    def add_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: IndexLike = None) -> None:
        if self.constraint_solver is None:
            gs.raise_exception("Cannot add a weld constraint when the constraint solver is disabled.")
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def delete_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: IndexLike = None) -> None:
        if self.constraint_solver is None:
            gs.raise_exception("Cannot delete a weld constraint when the constraint solver is disabled.")
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    def get_weld_constraints(
        self, as_tensor: bool = True, to_torch: bool = True
    ) -> StructuredResult | list[HostTensor]:
        if self.constraint_solver is None:
            return []
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(
        self, as_tensor: bool = True, to_torch: bool = True
    ) -> StructuredResult | list[HostTensor]:
        if self.constraint_solver is None:
            return []
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self) -> None:
        return self._rigid_dynamics.clear_external_force()

    def update_drone_propeller_vgeoms(
        self,
        propellers_vgeom_idxs: TensorLike,
        propellers_revs: TensorLike,
        propellers_spin: TensorLike,
    ) -> None:
        return self._rigid_api.update_drone_propeller_vgeoms(propellers_vgeom_idxs, propellers_revs, propellers_spin)

    def set_drone_rpm(
        self,
        propellers_link_idx: TensorLike,
        kf: TensorLike,
        km: TensorLike,
        propellers_rpm: TensorLike,
        propellers_spin: TensorLike,
        invert: bool,
    ) -> None:
        return self._rigid_api.set_drone_rpm(propellers_link_idx, kf, km, propellers_rpm, propellers_spin, invert)

    def update_verts_for_geoms(self, geoms_idx: IndexLike) -> None:
        return self._rigid_api.update_verts_for_geoms(geoms_idx)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_geoms(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_geoms
        return len(self.geoms)

    @property
    def n_verts(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_verts
        return sum(entity.n_verts for entity in self.entities)

    @property
    def n_free_verts(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_faces(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_faces
        return sum(entity.n_faces for entity in self.entities)

    @property
    def n_edges(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_edges
        return sum(entity.n_edges for entity in self.entities)

    @property
    def max_collision_pairs(self) -> int:
        return (
            self._rigid_dynamics.max_collision_pairs
            if self._rigid_dynamics is not None
            else self._options.max_collision_pairs
        )

    @property
    def n_equalities(self) -> int:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.n_equalities
        return sum(entity.n_equalities for entity in self.entities)

    @property
    def n_candidate_equalities_(self) -> int:
        return self._rigid_dynamics.get_n_candidate_equalities()

    @property
    def equalities(self) -> list["RigidEquality"]:
        if self._rigid_dynamics is not None:
            return self._rigid_dynamics.equalities
        return gs.List(equality for entity in self.entities for equality in entity.equalities)
