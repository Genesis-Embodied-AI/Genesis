from collections.abc import Iterator, Sequence
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
from genesis.engine.states import RigidSolverState, SolverCheckpoint
from genesis.engine.states.cache import QueriedStates
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import data_to_array, indices_to_mask, qd_to_torch

from ..base_solver import MutatedLinks, StateChange, Subscriber, mutates, register_subscriber
from .collider import Collider
from .rigid_tools import sanitize_sol_params as _sanitize_sol_params

if TYPE_CHECKING:
    from genesis.engine.core import System
    from genesis.engine.entities.rigid_entity.rigid_equality import RigidEquality
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom, RigidVisGeom
    from genesis.engine.entities.rigid_entity.rigid_joint import RigidJoint
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.force_fields import ForceField
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.engine.simulator_engine import SimulatorEngine
    from genesis.engine.systems.rigid_constraint import RigidConstraint
    from genesis.engine.systems.rigid_equality_constraint import RigidEqualityConstraint
    from genesis.engine.systems.rigid_frictionloss_constraint import RigidFrictionLossConstraint
    from genesis.engine.systems.rigid_island import RigidIsland
    from genesis.engine.systems.rigid_dynamics import RigidDynamics

TensorLike: TypeAlias = ArrayLike | torch.Tensor
IndexLike: TypeAlias = ArrayLike | torch.Tensor | None
HostTensor: TypeAlias = NDArray[np.generic] | torch.Tensor
StructuredValue: TypeAlias = HostTensor | tuple[HostTensor, ...]
StructuredResult: TypeAlias = dict[str, StructuredValue]


def _constraint_state_property(name: str) -> property:
    return property(lambda facade: getattr(facade._system.data.constraint_state, name))


class ConstraintSolver:
    """External compatibility surface of the rigid constraint Systems.

    It reads the state of the constraint manager and delegates to its owner APIs, and to the equality constraints for
    the equality and weld queries and edits.
    """

    def __init__(
        self,
        system: "RigidConstraint",
        equality_system: "RigidEqualityConstraint",
        participants: tuple["System", ...],
    ) -> None:
        self._rigid_constraint = system
        self._rigid_equality_constraint = equality_system
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
        self._queried_states = QueriedStates()
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
        self._rigid_island: "RigidIsland | None" = None
        self._rigid_frictionloss_constraint: "RigidFrictionLossConstraint | None" = None

        self.collider = None
        self.constraint_solver = None

        self._ckpt = dict()

    def init_ckpt(self) -> None:
        pass

    def wire_backend(self, engine: "SimulatorEngine") -> None:
        """Connect the public facade to the owner APIs of the Systems of the Engine after initialization."""
        from genesis.engine.systems import (
            RigidContactConstraint,
            RigidEqualityConstraint,
            RigidFrictionLossConstraint,
            RigidJointLimitConstraint,
            RigidDynamics,
        )

        system = engine.require(RigidDynamics)
        self._rigid_dynamics = system
        self._rigid_island = engine.rigid_island
        self._articulated_links_idx = system._articulated_links_idx
        self._links_parent_idx = system._links_parent_idx
        self._rigid_frictionloss_constraint = engine.find(RigidFrictionLossConstraint)
        self.collider = None if engine.rigid_rigid_contact is None else Collider(engine.rigid_rigid_contact)
        self.constraint_solver = None
        if engine.rigid_constraint is not None:
            participant_types = (
                RigidEqualityConstraint,
                RigidFrictionLossConstraint,
                RigidContactConstraint,
                RigidJointLimitConstraint,
            )
            participants = tuple(
                participant for participant in map(engine.find, participant_types) if participant is not None
            )
            self.constraint_solver = ConstraintSolver(
                engine.rigid_constraint, engine.require(RigidEqualityConstraint), participants
            )

    # Explicit compatibility exports of Rigid-owned data. These replace the former inheritance/__getattr__ surface:
    # the facade never owns or reimplements the fields, and adding a new backend field requires an intentional export.
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
        return self._rigid_dynamics.qpos

    @property
    def qpos0(self) -> qd.Tensor:
        return self._rigid_dynamics.qpos0

    @property
    def mass_mat(self) -> qd.Tensor:
        return self._rigid_dynamics.mass_mat

    @property
    def meaninertia(self) -> qd.Tensor:
        return self._rigid_dynamics.rigid_info.meaninertia

    @property
    def weight_scratch(self) -> array_class.WeightScratch:
        return self._rigid_dynamics.weight_scratch

    @property
    def geoms_init_AABB(self) -> qd.Tensor:
        return self._rigid_dynamics.data.geoms_init_AABB

    @property
    def _errno(self) -> qd.Tensor:
        return self._rigid_dynamics.data.errno

    @property
    def _sol_min_timeconst(self) -> float:
        return self._rigid_dynamics._sol_min_timeconst

    @property
    def _sol_default_timeconst(self) -> float | None:
        return self._rigid_dynamics._sol_default_timeconst

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
            return self._rigid_dynamics._enable_heterogeneous
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
        return self._rigid_dynamics._para_level if self._rigid_dynamics is not None else self._scene._para_level

    @gs.assert_built
    @mutates(StateChange.MODEL)
    def set_gravity(self, gravity: TensorLike, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_gravity(gravity, envs_idx)

    def get_gravity(self, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_gravity(envs_idx)

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
        return self._rigid_dynamics.n_qs if self._rigid_dynamics is not None else sum(entity.n_qs for entity in self.entities)

    @property
    def n_dofs(self) -> int:
        return self._rigid_dynamics.n_dofs if self._rigid_dynamics is not None else sum(entity.n_dofs for entity in self.entities)

    @property
    def n_vgeoms(self) -> int:
        return self._rigid_dynamics.n_vgeoms if self._rigid_dynamics is not None else len(self.vgeoms)

    @property
    def n_vverts(self) -> int:
        return self._rigid_dynamics.n_vverts if self._rigid_dynamics is not None else sum(entity.n_vverts for entity in self.entities)

    @property
    def n_vfaces(self) -> int:
        return self._rigid_dynamics.n_vfaces if self._rigid_dynamics is not None else sum(entity.n_vfaces for entity in self.entities)

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
        return self._rigid_dynamics._vfaces_raycast_mask

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
        return self._rigid_dynamics._sanitize_io_variables(
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
        return self._rigid_dynamics.get_terrain_height(positions, link_idx, envs_idx)

    def get_links_quat(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_links_quat(links_idx, envs_idx, relative=relative)

    def get_joints_anchor_pos(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_joints_anchor_pos(joints_idx, envs_idx)

    def get_joints_anchor_axis(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_joints_anchor_axis(joints_idx, envs_idx)

    def get_vgeoms_pos(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_vgeoms_pos(vgeoms_idx, envs_idx, relative=relative)

    def get_vgeoms_quat(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_vgeoms_quat(vgeoms_idx, envs_idx, relative=relative)

    def get_links_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_ang(links_idx, envs_idx)

    def get_links_net_contact_force(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_net_contact_force(links_idx, envs_idx)

    def get_qpos(self, qs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_qpos(qs_idx, envs_idx)

    def get_dofs_velocity(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_velocity(dofs_idx, envs_idx)

    def get_dofs_position(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_position(dofs_idx, envs_idx)

    def get_dofs_limit(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.get_dofs_limit(dofs_idx, envs_idx)

    def update_vgeoms(self) -> None:
        return self._rigid_dynamics.update_vgeoms()

    def update_vverts_for_vgeoms(self, vgeoms_idx: IndexLike) -> None:
        return self._rigid_dynamics.update_vverts_for_vgeoms(vgeoms_idx)

    def set_vverts(
        self,
        custom_vvert_start: int,
        custom_vvert_end: int,
        vgeoms_idx: IndexLike,
        vverts: TensorLike | None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics.set_vverts(custom_vvert_start, custom_vvert_end, vgeoms_idx, vverts, envs_idx)

    def get_vverts(self, custom_vvert_start: int, custom_vvert_end: int, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_vverts(custom_vvert_start, custom_vvert_end, envs_idx)

    def get_fixed_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        return self._rigid_dynamics.get_fixed_collision_verts(verts_idx)

    def get_free_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        return self._rigid_dynamics.get_free_collision_verts(verts_idx)

    def get_vgeom_transforms_batched(self, envs_idx: IndexLike = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.get_vgeom_transforms_batched(envs_idx)

    def get_links_jacobian(
        self,
        link_idx: int,
        dof_start: int,
        n_dofs: int,
        local_point: TensorLike | None = None,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_links_jacobian(link_idx, dof_start, n_dofs, local_point)

    def forward_kinematics_query(
        self, entity: RigidEntity, qpos: TensorLike, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.forward_kinematics_query(entity, qpos, envs_idx)

    def inverse_kinematics(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.inverse_kinematics(*args, **kwargs)

    def update_forward_pos(self) -> None:
        return self._rigid_dynamics.update_forward_pos()

    def get_error_envs_mask(self) -> torch.Tensor:
        return self._rigid_dynamics.get_error_envs_mask()

    def check_errno(self) -> None:
        errno = self._rigid_dynamics.read_errno()
        contact_data = None if self.collider is None else self.collider._data
        if errno & array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS:
            maximum = contact_data.collider_info.max_collision_pairs_broad[None]
            gs.raise_exception(
                f"Exceeding max number of broad phase candidate contact pairs ({maximum}). Please increase "
                "'multiplier_collision_broad_phase'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS:
            maximum = contact_data.collider_info.max_candidate_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of candidate contact points ({maximum}). Please increase 'max_collision_pairs'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_CONTACTS:
            maximum = contact_data.collider_info.max_contacts[None]
            gs.raise_exception(
                f"Exceeding max number of post-pruning contact points ({maximum}). Please increase 'max_contacts'."
            )
        if errno & array_class.ErrorCode.INVALID_CONTACT_NAN:
            gs.raise_exception("Collision detection reported a non-finite contact; please report this internal error.")
        if errno & array_class.ErrorCode.INVALID_FORCE_NAN:
            gs.raise_exception("Invalid constraint forces causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.INVALID_ACC_NAN:
            gs.raise_exception("Invalid accelerations causing 'nan'. Please decrease Rigid simulation timestep.")

    def _kernel_detect_collision(self) -> None:
        if self.collider is not None:
            self.collider.clear()
            self.collider.detection()

    def detect_collision(self, env_idx: int = 0) -> NDArray[np.integer]:
        if self.collider is None:
            return np.empty((0, 2), dtype=np.int32)
        return self.collider.detect_collision(env_idx)

    def _func_constraint_force(self) -> None:
        return self._sim._engine.constraint_force_host()

    def _func_forward_dynamics(self) -> None:
        return self._rigid_dynamics._func_forward_dynamics()

    def _func_update_acc(self) -> None:
        return self._rigid_dynamics._func_update_acc()

    def _func_integrate_dq_entity(
        self,
        dq: qd.template(),
        i_e: int,
        i_b: int,
        respect_joint_limit: bool,
    ) -> None:
        return self._rigid_dynamics._func_integrate_dq_entity(dq, i_e, i_b, respect_joint_limit)

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
        return self._rigid_dynamics.apply_links_external_wrench(
            force, torque, links_idx, envs_idx, pos=pos, ref=ref, local=local
        )

    def reset_grad(self) -> None:
        return self._rigid_dynamics.reset_grad()

    def _update_cartesian_grad(self, envs_idx: torch.Tensor) -> None:
        return self._rigid_dynamics._update_cartesian_grad(envs_idx)

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
        if self.collider is not None:
            if partial:
                self.collider.reset(envs_idx)
            else:
                self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            if partial:
                self.constraint_solver.reset(envs_idx)
            else:
                self.constraint_solver.clear(envs_idx)
        self._rigid_dynamics.set_state(f, state, envs_idx, partial=partial)
        self._restart()

    @property
    def data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        if self._rigid_dynamics is None:
            return
        yield from self._rigid_dynamics.iter_data()
        yield from self._rigid_island.iter_data()
        if self.collider is not None:
            yield from self.collider.data
        if self.constraint_solver is not None:
            yield from self.constraint_solver.data

    def __getstate__(self) -> SolverCheckpoint:
        arrays = {}
        configs = {}
        for item in self.data:
            if isinstance(item, array_class.ConfigItem):
                configs[item.name] = item.config
            elif item.kind in array_class.CHECKPOINT_KINDS:
                if isinstance(item, array_class.ScalarItem):
                    arrays[item.name] = np.array(getattr(item.owner, item.attr))
                else:
                    arrays[item.name] = data_to_array(item.array, to_torch=gs.use_zerocopy, copy=True)
        return SolverCheckpoint(arrays=arrays, configs=configs, kinds=array_class.CHECKPOINT_KINDS)

    def _restart(self) -> None:
        self._rigid_dynamics.restart_owner()
        if self.collider is not None:
            self.collider._rigid_rigid_contact.clear_contact_data_cache()
        if self.constraint_solver is not None:
            self.constraint_solver.clear_query_cache()

    def _reset_dependents(self, envs_idx: IndexLike = None) -> None:
        if self.collider is not None:
            self.collider.reset(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)

    @mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
    def __setstate__(self, state: SolverCheckpoint) -> None:
        items = (
            item for item in self.data if isinstance(item, array_class.DataItem) and item.kind in state.kinds
        )
        array_class.fill_data(items, state.arrays)
        if array_class.DataKind.DERIVED not in state.kinds:
            self._rigid_dynamics.invalidate_forward_state()
            self._rigid_dynamics.update_forward_pos()
        self._restart()

    def process_input(self, in_backward: bool = False) -> None:
        return self._rigid_dynamics.process_input(in_backward)

    def process_input_grad(self) -> None:
        return self._rigid_dynamics.process_input_grad()

    def save_ckpt(self, ckpt_name: str) -> None:
        return self._rigid_dynamics.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name: str) -> None:
        return self._rigid_dynamics.load_ckpt(ckpt_name)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def set_links_pos(self, pos: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_links_pos(pos, links_idx, envs_idx)

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
        return self._rigid_dynamics.set_base_links_pos(pos, links_idx, envs_idx, relative=relative, skip_forward=skip_forward)

    def set_links_quat(self, quat: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_links_quat(quat, links_idx, envs_idx)

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
        return self._rigid_dynamics.set_base_links_quat(quat, links_idx, envs_idx, relative=relative, skip_forward=skip_forward)

    @mutates(StateChange.MODEL)
    def _set_links_info(
        self,
        values: TensorLike,
        links_idx: IndexLike,
        name: str,
        envs_idx: IndexLike = None,
        *,
        scale_inertia: bool = False,
    ) -> None:
        return self._rigid_dynamics._set_links_info(values, links_idx, name, envs_idx, scale_inertia=scale_inertia)

    def set_links_mass(
        self,
        mass: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        scale_inertia: bool = False,
    ) -> None:
        return self._rigid_dynamics.set_links_mass(mass, links_idx, envs_idx, scale_inertia)

    def set_links_inertia(self, inertia: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_links_inertia(inertia, links_idx, envs_idx)

    def set_links_COM(self, com: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_links_COM(com, links_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_ratio(
        self, friction_ratio: TensorLike, geoms_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_dynamics.set_geoms_friction_ratio(friction_ratio, geoms_idx, envs_idx)

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
        return self._rigid_dynamics.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

    @mutates(StateChange.MODEL)
    def set_global_sol_params(self, sol_params: TensorLike) -> None:
        return self._rigid_dynamics.set_global_sol_params(sol_params)

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
        return self._rigid_dynamics.set_sol_params(sol_params, geoms_idx, envs_idx, joints_idx=joints_idx, eqs_idx=eqs_idx)

    @mutates(StateChange.MODEL)
    def _set_dofs_info(
        self,
        tensor_list: Sequence[TensorLike],
        dofs_idx: IndexLike,
        name: str,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics._set_dofs_info(tensor_list, dofs_idx, name, envs_idx)

    def set_dofs_kp(self, kp: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_kp(kp, dofs_idx, envs_idx)

    def set_dofs_kv(self, kv: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_kv(kv, dofs_idx, envs_idx)

    def set_dofs_act_gain(self, act_gain: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_act_gain(act_gain, dofs_idx, envs_idx)

    def set_dofs_act_bias(
        self,
        bias0: TensorLike,
        bias1: TensorLike,
        bias2: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics.set_dofs_act_bias(bias0, bias1, bias2, dofs_idx, envs_idx)

    def set_dofs_force_range(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics.set_dofs_force_range(lower, upper, dofs_idx, envs_idx)

    def set_dofs_stiffness(self, stiffness: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)

    def set_dofs_armature(self, armature: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_armature(armature, dofs_idx, envs_idx)

    def set_dofs_damping(self, damping: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_dofs_damping(damping, dofs_idx, envs_idx)

    def set_dofs_frictionloss(
        self, frictionloss: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_dynamics.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)

    def set_dofs_limit(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics.set_dofs_limit(lower, upper, dofs_idx, envs_idx)

    @mutates(StateChange.GEOMETRY, links=MutatedLinks.ARTICULATED)
    def set_dofs_position(self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._reset_dependents(envs_idx)
        return self._rigid_dynamics.set_dofs_position(position, dofs_idx, envs_idx)

    def _wake_dofs(self, dofs_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        return self._rigid_dynamics._wake_dofs(dofs_idx, envs_idx)

    def set_dofs_velocity(
        self,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        return self._rigid_dynamics.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.control_dofs_force(force, dofs_idx, envs_idx)

    def control_dofs_velocity(
        self, velocity: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_dynamics.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    def control_dofs_position(
        self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        return self._rigid_dynamics.control_dofs_position(position, dofs_idx, envs_idx)

    def control_dofs_position_velocity(
        self,
        position: TensorLike,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        return self._rigid_dynamics.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

    def get_sol_params(
        self,
        geoms_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        joints_idx: IndexLike = None,
        eqs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_sol_params(geoms_idx, envs_idx, joints_idx=joints_idx, eqs_idx=eqs_idx)

    def get_links_pos(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_links_pos(links_idx, envs_idx, ref=ref, relative=relative)

    def get_links_vel(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_links_vel(links_idx, envs_idx, ref=ref, relative=relative)

    def get_links_acc(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_links_acc(links_idx, envs_idx, relative=relative)

    def get_links_acc_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_acc_ang(links_idx, envs_idx)

    def get_links_root_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_root_COM(links_idx, envs_idx)

    def get_links_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_COM(links_idx, envs_idx)

    @property
    def is_links_info_batched(self) -> bool:
        return self._rigid_dynamics.is_links_info_batched()

    def get_links_mass(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_mass(links_idx, envs_idx)

    def get_links_inertia(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_inertia(links_idx, envs_idx)

    def get_links_invweight(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_links_invweight(links_idx, envs_idx)

    def get_geoms_friction_ratio(self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_friction_ratio(geoms_idx, envs_idx)

    def get_geoms_pos(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_pos(geoms_idx, envs_idx, relative=relative)

    def get_geoms_quat(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_quat(geoms_idx, envs_idx, relative=relative)

    def get_dofs_control_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_control_force(dofs_idx, envs_idx)

    def get_dofs_actuator_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        rigid = self._rigid_dynamics
        qf_applied = qd_to_torch(rigid.data.dyn_state.dofs.qf_applied, envs_idx, transpose=True)
        constraint_state = None if self.constraint_solver is None else self.constraint_solver.constraint_state
        qacc_source = rigid.data.dyn_state.dofs.acc if constraint_state is None else constraint_state.qacc
        qacc = qd_to_torch(qacc_source, envs_idx, transpose=True)
        qf_passive = qd_to_torch(rigid.data.dyn_state.dofs.qf_passive, envs_idx, transpose=True)
        if self._options.batch_dofs_info:
            armature = qd_to_torch(rigid.data.dyn_info.dofs.armature, envs_idx, transpose=True)
        else:
            armature = qd_to_torch(rigid.data.dyn_info.dofs.armature, transpose=True)

        if self._rigid_frictionloss_constraint is None:
            qf_frictionloss = torch.zeros_like(qf_applied)
        else:
            qf_frictionloss = self._rigid_frictionloss_constraint.get_dofs_frictionloss_force(envs_idx)

        actuator_force = qf_applied - armature * qacc + qf_frictionloss + qf_passive
        if dofs_idx is not None:
            actuator_force = actuator_force[indices_to_mask(None, dofs_idx)]
        return actuator_force[0] if self.n_envs == 0 else actuator_force

    def get_dofs_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_force(dofs_idx, envs_idx)

    def get_dofs_acc(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_acc(dofs_idx, envs_idx)

    def get_dofs_kp(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_kp(dofs_idx, envs_idx)

    def get_dofs_kv(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_kv(dofs_idx, envs_idx)

    def get_dofs_act_gain(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_act_gain(dofs_idx, envs_idx)

    def get_dofs_act_bias(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.get_dofs_act_bias(dofs_idx, envs_idx)

    def get_dofs_force_range(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.get_dofs_force_range(dofs_idx, envs_idx)

    def get_dofs_stiffness(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_stiffness(dofs_idx, envs_idx)

    def get_dofs_invweight(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_invweight(dofs_idx, envs_idx)

    def get_dofs_armature(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_armature(dofs_idx, envs_idx)

    def get_dofs_damping(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_damping(dofs_idx, envs_idx)

    def get_dofs_frictionloss(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_dofs_frictionloss(dofs_idx, envs_idx)

    def get_mass_mat(
        self,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        decompose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._rigid_dynamics.get_mass_mat(dofs_idx, envs_idx, decompose)

    def get_kinetic_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_kinetic_energy(links_idx, dofs_idx, envs_idx)

    def get_potential_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        return self._rigid_dynamics.get_potential_energy(links_idx, dofs_idx, envs_idx)

    def get_total_energy(self, envs_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_total_energy(envs_idx)

    def get_geoms_friction(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_friction(geoms_idx)

    def get_geoms_friction_torsional(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_friction_torsional(geoms_idx)

    def get_geoms_friction_rolling(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return self._rigid_dynamics.get_geoms_friction_rolling(geoms_idx)

    def get_AABB(self, entities_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self._sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")
        return self._rigid_dynamics.get_AABB(entities_idx, envs_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction(self, friction: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_dynamics.set_geom_friction(friction, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_dynamics.set_geom_friction_torsional(friction_torsional, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geom_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike) -> None:
        return self._rigid_dynamics.set_geom_friction_rolling(friction_rolling, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction(self, friction: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_geoms_friction(friction, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_geoms_friction_torsional(friction_torsional, geoms_idx)

    @mutates(StateChange.MODEL)
    def set_geoms_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike = None) -> None:
        return self._rigid_dynamics.set_geoms_friction_rolling(friction_rolling, geoms_idx)

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
        return self._rigid_dynamics.update_drone_propeller_vgeoms(propellers_vgeom_idxs, propellers_revs, propellers_spin)

    def set_drone_rpm(
        self,
        propellers_link_idx: TensorLike,
        kf: TensorLike,
        km: TensorLike,
        propellers_rpm: TensorLike,
        propellers_spin: TensorLike,
        invert: bool,
    ) -> None:
        return self._rigid_dynamics.set_drone_rpm(propellers_link_idx, kf, km, propellers_rpm, propellers_spin, invert)

    def update_verts_for_geoms(self, geoms_idx: IndexLike) -> None:
        return self._rigid_dynamics.update_verts_for_geoms(geoms_idx)

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
        return self._rigid_dynamics.max_collision_pairs if self._rigid_dynamics is not None else self._options.max_collision_pairs

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
