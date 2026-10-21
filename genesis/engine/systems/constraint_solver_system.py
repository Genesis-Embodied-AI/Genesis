"""System owning the constraint solve Actions composed across the Rigid graph phases."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint import backward as backward_constraint_solver
from genesis.engine.solvers.rigid.constraint.solver import (
    constraint_solver_kernel_clear,
    constraint_solver_kernel_masked_clear,
    constraint_solver_kernel_reset,
    func_substep_add_equality_constraints,
    func_substep_add_inequality_constraints,
    func_substep_add_inequality_constraints_no_contact,
    func_substep_resolve_post,
    func_substep_resolve_post_no_contact,
    func_solve_body,
    kernel_add_weld_constraint,
    kernel_delete_weld_constraint,
    kernel_get_equality_constraints,
)
from genesis.utils.misc import assign_indexed_tensor, indices_to_mask, qd_to_torch

from .rigid_solver_system import RigidSolverSystem
from ..simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

EnvIndex: TypeAlias = ArrayLike | torch.Tensor | None
ConstraintArray: TypeAlias = np.ndarray | torch.Tensor
ConstraintValue: TypeAlias = ConstraintArray | tuple[ConstraintArray, ...]
ConstraintResult: TypeAlias = dict[str, ConstraintValue]

IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class ConstraintSolverSystem(System):
    """Own the rigid constraint data and run the solve when constraints are enabled.

    It assembles the equality constraints after the forward dynamics and the inequality constraints after collision
    detection, solves them on the host, then turns the solved forces into accelerations and contact forces. The System
    is absent entirely when constraints are disabled.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Device state owned by the rigid constraint solve.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        constraint_state: array_class.ConstraintState
        noslip: bool

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_solver_system = Require(RigidSolverSystem)

    def __init__(self, scene: "Scene") -> None:
        super().__init__()
        self._scene = scene
        self.data = self.Data()

    def build(self) -> None:
        from .rigid_island_system import RigidIslandSystem
        from .rigid_rigid_contact_system import RigidRigidContactSystem

        self.island_system = self.require(RigidIslandSystem)
        self.contact_system = self.find(RigidRigidContactSystem)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def prepare(self):
        return self.clear_equality_info_cache

    @System.action(kind="stage")
    def assemble_equalities(self):
        rigid = self.rigid_solver_system
        return (
            func_substep_add_equality_constraints,
            rigid.data,
            rigid.rigid_config,
            self.data,
        )

    @System.action(kind="host")
    def assemble_equalities_kernel(self):
        rigid = self.rigid_solver_system
        return (
            kernel_substep_add_equality_constraints,
            rigid.data,
            rigid.rigid_config,
            self.data,
        )

    @System.action(kind="stage")
    def assemble_inequalities(self):
        rigid = self.rigid_solver_system
        if self.contact_system is None:
            return (
                func_substep_add_inequality_constraints_no_contact,
                rigid.data,
                rigid.rigid_config,
                self.data,
            )
        return (
            func_substep_add_inequality_constraints,
            rigid.data,
            rigid.rigid_config,
            self.contact_system.data,
            self.data,
        )

    @System.action(kind="host")
    def assemble_inequalities_kernel(self):
        rigid = self.rigid_solver_system
        if self.contact_system is None:
            return (
                kernel_substep_add_inequality_constraints_no_contact,
                rigid.data,
                rigid.rigid_config,
                self.data,
            )
        return (
            kernel_substep_add_inequality_constraints,
            rigid.data,
            rigid.rigid_config,
            self.contact_system.data,
            self.data,
        )

    @System.action(kind="host")
    def solve(self):
        return self.solve_constraints

    @System.action(kind="stage")
    def update_forces(self):
        rigid = self.rigid_solver_system
        if self.contact_system is None:
            return (
                func_substep_resolve_post_no_contact,
                rigid.data,
                rigid.rigid_config,
                self.data,
            )
        return (
            func_substep_resolve_post,
            rigid.data,
            rigid.rigid_config,
            self.contact_system.data,
            self.data,
        )

    @System.action(kind="host")
    def update_forces_kernel(self):
        rigid = self.rigid_solver_system
        if self.contact_system is None:
            return (
                kernel_substep_resolve_post_no_contact,
                rigid.data,
                rigid.rigid_config,
                self.data,
            )
        return (
            kernel_substep_resolve_post,
            rigid.data,
            rigid.rigid_config,
            self.contact_system.data,
            self.data,
        )

    @System.action(kind="host")
    def reverse_solve(self):
        return self.reverse_constraint_solve

    @System.action(kind="host")
    def reverse_collision_rows(self):
        return self.reverse_collision_constraints

    @System.action(kind="host")
    def reverse_joint_limit_rows(self):
        return self.reverse_joint_limit_constraints

    # -------------------------------- Public get/set API --------------------------------

    def get_len_constraints(self) -> int:
        return self.len_constraints

    def get_n_iterations(self) -> int:
        return self.n_iterations

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for item in array_class.iter_data(self.data, "constraint_state"):
            if not item.name.startswith(("constraint_state.island.", "constraint_state.qd_n_equalities")):
                yield item

    def reset(self, envs_idx: EnvIndex = None) -> None:
        self._eq_const_info_cache.clear()

        if gs.use_zerocopy:
            envs_mask = indices_to_mask(envs_idx)
            dofs_mask = (slice(None), *envs_mask)
            is_warmstart = qd_to_torch(self.data.constraint_state.is_warmstart, copy=False)
            qacc_ws = qd_to_torch(self.data.constraint_state.qacc_ws, copy=False)
            assign_indexed_tensor(is_warmstart, envs_mask, False)
            assign_indexed_tensor(qacc_ws, dofs_mask, 0.0)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self.rigid_solver_system._scene._sanitize_envs_idx(envs_idx)
        constraint_solver_kernel_reset(envs_idx, self.data.constraint_state, self.rigid_solver_system.rigid_config)

    def clear(self, envs_idx: EnvIndex = None) -> None:
        self.reset(envs_idx)

        if gs.use_zerocopy and (
            not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool)
        ):
            n_constraints = qd_to_torch(self.data.constraint_state.n_constraints, copy=False)
            n_constraints_equality = qd_to_torch(self.data.constraint_state.n_constraints_equality, copy=False)
            n_constraints_frictionloss = qd_to_torch(self.data.constraint_state.n_constraints_frictionloss, copy=False)
            n_constraints_cone = qd_to_torch(self.data.constraint_state.n_constraints_cone, copy=False)
            qd_n_equalities = qd_to_torch(self.data.constraint_state.qd_n_equalities, copy=False)
            n_eq = self.rigid_solver_system._n_equalities
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                n_constraints.masked_fill_(envs_idx, 0)
                n_constraints_equality.masked_fill_(envs_idx, 0)
                n_constraints_frictionloss.masked_fill_(envs_idx, 0)
                n_constraints_cone.masked_fill_(envs_idx, 0)
                qd_n_equalities.masked_fill_(envs_idx, n_eq)
            elif isinstance(envs_idx, torch.Tensor):
                n_constraints.scatter_(0, envs_idx, 0)
                n_constraints_equality.scatter_(0, envs_idx, 0)
                n_constraints_frictionloss.scatter_(0, envs_idx, 0)
                n_constraints_cone.scatter_(0, envs_idx, 0)
                qd_n_equalities.scatter_(0, envs_idx, n_eq)
            else:
                env_mask = indices_to_mask(envs_idx)
                assign_indexed_tensor(n_constraints, env_mask, 0)
                assign_indexed_tensor(n_constraints_equality, env_mask, 0)
                assign_indexed_tensor(n_constraints_frictionloss, env_mask, 0)
                assign_indexed_tensor(n_constraints_cone, env_mask, 0)
                assign_indexed_tensor(qd_n_equalities, env_mask, n_eq)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self.rigid_solver_system._scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = constraint_solver_kernel_masked_clear
        else:
            fn = constraint_solver_kernel_clear
        fn(
            envs_idx,
            self.data.constraint_state,
            self.rigid_solver_system.rigid_info,
            self.rigid_solver_system.rigid_config,
        )

    def add_equality_constraints(self) -> None:
        self._eq_const_info_cache.clear()
        self.assemble_equalities_kernel.invoke()

    def add_inequality_constraints(self) -> None:
        self.assemble_inequalities_kernel.invoke()

    def resolve(self) -> None:
        """Solve the constraint forces, then update the accelerations and the contact forces from them."""
        self.solve.invoke()
        self.update_forces_kernel.invoke()

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True) -> ConstraintResult:
        # Early return if already pre-computed
        eq_const_info = self._eq_const_info_cache.get((as_tensor, to_torch))
        if eq_const_info is not None:
            return eq_const_info.copy()

        n_eqs = tuple(self.data.constraint_state.qd_n_equalities.to_numpy())
        n_envs = len(n_eqs)
        n_eqs_max = max(n_eqs)

        if as_tensor:
            out_size = n_envs * n_eqs_max
        else:
            *n_eqs_starts, out_size = np.cumsum(n_eqs)

        if to_torch:
            iout = torch.full((out_size, 3), -1, dtype=gs.tc_int, device=gs.device)
            fout = torch.zeros((out_size, 6), dtype=gs.tc_float, device=gs.device)
        else:
            iout = np.full((out_size, 3), -1, dtype=gs.np_int)
            fout = np.zeros((out_size, 6), dtype=gs.np_float)

        if n_eqs_max > 0:
            kernel_get_equality_constraints(
                iout,
                fout,
                self.data.constraint_state,
                self.rigid_solver_system.dyn_info,
                self.rigid_solver_system.rigid_config,
                as_tensor,
            )

        if as_tensor:
            iout = iout.reshape((n_envs, n_eqs_max, 3))
            eq_type, obj_a, obj_b = (iout[..., i] for i in range(3))
            efc_force = fout.reshape((n_envs, n_eqs_max, 6))
            values = (eq_type, obj_a, obj_b, fout)
        else:
            if to_torch:
                iout_chunks = torch.split(iout, n_eqs)
                efc_force = torch.split(fout, n_eqs)
            else:
                iout_chunks = np.split(iout, n_eqs_starts)
                efc_force = np.split(fout, n_eqs_starts)
            eq_type, obj_a, obj_b = tuple(zip(*([data[..., i] for i in range(3)] for data in iout_chunks)))

        values = (eq_type, obj_a, obj_b, efc_force)
        eq_const_info = dict(zip(("type", "obj_a", "obj_b", "force"), values))

        # Cache equality constraint information before returning
        self._eq_const_info_cache[(as_tensor, to_torch)] = eq_const_info

        return eq_const_info.copy()

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True) -> ConstraintResult:
        eq_const_info = self.get_equality_constraints(as_tensor, to_torch)
        eq_type = eq_const_info.pop("type")

        weld_const_info = {}
        if as_tensor:
            weld_mask = eq_type == gs.EQUALITY_TYPE.WELD
            n_envs = len(weld_mask)
            n_welds = weld_mask.sum(dim=-1) if to_torch else np.sum(weld_mask, axis=-1)
            n_welds_max = max(n_welds)
            for key, value in eq_const_info.items():
                shape = (n_envs, n_welds_max, *value.shape[2:])
                if to_torch:
                    if torch.is_floating_point(value):
                        weld_const_info[key] = torch.zeros(shape, dtype=value.dtype, device=value.device)
                    else:
                        weld_const_info[key] = torch.full(shape, -1, dtype=value.dtype, device=value.device)
                else:
                    if np.issubdtype(value.dtype, np.floating):
                        weld_const_info[key] = np.zeros(shape, dtype=value.dtype)
                    else:
                        weld_const_info[key] = np.full(shape, -1, dtype=value.dtype)
            for i_b, (n_welds_i, weld_mask_i) in enumerate(zip(n_welds, weld_mask)):
                for eq_value, weld_value in zip(eq_const_info.values(), weld_const_info.values()):
                    weld_value[i_b, :n_welds_i] = eq_value[i_b, weld_mask_i]
        else:
            weld_mask_chunks = tuple(eq_type_i == gs.EQUALITY_TYPE.WELD for eq_type_i in eq_type)
            for key, value in eq_const_info.items():
                weld_const_info[key] = tuple(data[weld_mask] for weld_mask, data in zip(weld_mask_chunks, value))

        weld_const_info["link_a"] = weld_const_info.pop("obj_a")
        weld_const_info["link_b"] = weld_const_info.pop("obj_b")

        return weld_const_info

    def add_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: EnvIndex = None) -> None:
        envs_idx = self.rigid_solver_system._scene._sanitize_envs_idx(envs_idx)
        link1_idx, link2_idx = int(link1_idx), int(link2_idx)

        assert link1_idx >= 0 and link2_idx >= 0
        weld_const_info = self.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        assert not (
            ((link_a == link1_idx) | (link_b == link1_idx)) & ((link_a == link2_idx) | (link_b == link2_idx))
        ).any()

        self._eq_const_info_cache.clear()
        overflow = kernel_add_weld_constraint(
            link1_idx,
            link2_idx,
            envs_idx,
            self.rigid_solver_system.dyn_state,
            self.data.constraint_state,
            self.rigid_solver_system.dyn_info,
            self.rigid_solver_system.rigid_info,
            self.rigid_solver_system.rigid_config,
        )
        if overflow:
            gs.logger.warning(
                "Ignoring dynamically registered weld constraint to avoid exceeding max number of equality constraints"
                f"({self.rigid_info.n_candidate_equalities.to_numpy()}). Please increase the value of "
                "RigidSolver's option 'max_dynamic_constraints'."
            )

    def delete_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: EnvIndex = None) -> None:
        envs_idx = self.rigid_solver_system._scene._sanitize_envs_idx(envs_idx)
        self._eq_const_info_cache.clear()
        kernel_delete_weld_constraint(
            int(link1_idx),
            int(link2_idx),
            envs_idx,
            self.data.constraint_state,
            self.rigid_solver_system.dyn_info,
            self.rigid_solver_system.rigid_info,
            self.rigid_solver_system.rigid_config,
        )

    def backward(self) -> None:
        """Adjoint solve of the constraint force computation.

        The caller must pre-populate the upstream gradient constraint_state.dL_dqacc (see
        RigidSolver._constraint_force_grad)."""
        if not self.rigid_solver_system._requires_grad:
            gs.raise_exception("Please set `requires_grad` to True in SimOptions to enable differentiable mode.")

        # 1. We first need to find a solution to A^T * u = g system.
        backward_constraint_solver.kernel_solve_adjoint_u(
            self.data.constraint_state,
            self.rigid_solver_system.dyn_info,
            self.rigid_solver_system.rigid_info,
            self.rigid_solver_system.rigid_config,
        )

        # 2. Using the solution u, we can compute the gradients of the input variables.
        backward_constraint_solver.kernel_compute_gradients(
            self.data.constraint_state, self.rigid_solver_system.rigid_info
        )

    def reverse_constraint_solve(self) -> None:
        """Reverse the host constraint solve and non-contact row assembly."""
        rigid = self.rigid_solver_system
        constraint_state = self.data.constraint_state
        if gs.use_zerocopy:
            acc_grad = qd_to_torch(rigid.dyn_state.dofs.acc.grad, copy=False)
            dL_dqacc = qd_to_torch(constraint_state.dL_dqacc, copy=False)
            dL_dqacc.copy_(acc_grad)
            acc_grad.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            backward_constraint_solver.kernel_load_dL_dqacc_from_acc_grad(
                rigid.dyn_state,
                constraint_state,
                rigid.rigid_config,
            )

        self.backward()

        if gs.use_zerocopy:
            force_grad = qd_to_torch(rigid.dyn_state.dofs.force.grad, copy=False)
            dL_dforce = qd_to_torch(constraint_state.dL_dforce, copy=False)
            force_grad.add_(dL_dforce)
            mass_mat_grad = qd_to_torch(rigid.rigid_info.mass_mat.grad, copy=False)
            dL_dM = qd_to_torch(constraint_state.dL_dM, copy=False)
            mass_mat_grad.add_(dL_dM)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            backward_constraint_solver.kernel_accumulate_constraint_solver_grads(
                rigid.dyn_state,
                constraint_state,
                rigid.rigid_info,
                rigid.rigid_config,
            )

        backward_constraint_solver.kernel_manual_add_equality_constraints_bw(
            rigid.dyn_state,
            constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )
        backward_constraint_solver.kernel_manual_add_frictionloss_constraints_bw(
            rigid.dyn_state,
            constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )

    def reverse_collision_constraints(self) -> None:
        """Reverse contact-row assembly when the optional Contact owner exists."""
        if self.contact_system is None:
            return
        rigid = self.rigid_solver_system
        backward_constraint_solver.kernel_manual_add_collision_constraints_bw(
            rigid.dyn_state,
            self.contact_system.data.collider_state,
            self.data.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
        )

    def reverse_joint_limit_constraints(self) -> None:
        """Reverse joint-limit row assembly independently of Contact admission."""
        rigid = self.rigid_solver_system
        if not rigid._enable_joint_limit:
            return
        backward_constraint_solver.kernel_manual_add_joint_limit_constraints_bw(
            rigid.dyn_state,
            None if self.contact_system is None else self.contact_system.data.collider_state,
            self.data.constraint_state,
            rigid.dyn_info,
            rigid.rigid_info,
            rigid.rigid_config,
            enable_collision=self.contact_system is not None,
        )

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate constraint data from the rigid model and the contact layout."""
        rigid = self.rigid_solver_system
        max_contacts = 0
        if self.contact_system is not None:
            collider_data = self.contact_system.data
            max_contacts = rigid._options.max_contacts or collider_data.collider_info.max_contacts[None]
        rows_per_contact = rigid.rigid_config.rows_per_contact
        self.len_constraints = (
            rows_per_contact * max_contacts
            + sum(joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC) for joint in rigid.joints)
            + rigid.n_dofs
            + rigid.n_candidate_equalities_ * 6
        )
        self.len_constraints_ = max(1, self.len_constraints)
        self.n_cone_constraints_ = max(1, rows_per_contact * max_contacts)
        self.sparse_solve = rigid.rigid_config.sparse_solve
        self.n_iterations = int(rigid._options.iterations)
        noslip = self.contact_system is not None and rigid._options.noslip_iterations > 0
        self._eq_const_info_cache = {}
        state = array_class.get_constraint_state(
            batch_size=rigid._B,
            n_dofs=rigid.n_dofs_,
            n_trees=rigid.n_trees_,
            links=rigid.links,
            requires_grad=rigid._requires_grad,
            noslip_iterations=rigid._options.noslip_iterations,
            rigid_config=rigid.rigid_config,
            rigid_info=rigid.rigid_info,
            len_constraints=self.len_constraints_,
            n_cone_constraints=self.n_cone_constraints_,
            sparse_solve=self.sparse_solve,
            island_state=self.island_system.data.island,
            qd_n_equalities=self.island_system.data.qd_n_equalities,
        )
        self.data.constraint_state = state
        self.data.noslip = noslip
        self.reset()

    def clear_equality_info_cache(self) -> None:
        """Drop the equality constraints read back during the previous substep, which the assembly is about to
        replace."""
        self._eq_const_info_cache.clear()

    def solve_constraints(self) -> None:
        """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
        rigid_data = self.rigid_solver_system.data
        func_solve_body(
            rigid_data.dyn_state,
            self.data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            self.rigid_solver_system.rigid_config,
            self.n_iterations,
        )


@register_system(SimulatorEngine)
def create_constraint_solver_system(engine: SimulatorEngine) -> ConstraintSolverSystem | None:
    """Create the rigid constraint solver unless constraints are disabled."""
    from genesis.engine.entities import RigidEntity

    has_rigid = any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
    return (
        ConstraintSolverSystem(engine.scene)
        if has_rigid and not engine.scene.options.rigid.disable_constraint
        else None
    )


@qd.kernel
def kernel_substep_add_equality_constraints(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    constraint_data: qd.template(),
):
    func_substep_add_equality_constraints(rigid_data, rigid_config, constraint_data)


@qd.kernel
def kernel_substep_add_inequality_constraints(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    contact_data: qd.template(),
    constraint_data: qd.template(),
):
    func_substep_add_inequality_constraints(rigid_data, rigid_config, contact_data, constraint_data)


@qd.kernel
def kernel_substep_add_inequality_constraints_no_contact(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    constraint_data: qd.template(),
):
    func_substep_add_inequality_constraints_no_contact(rigid_data, rigid_config, constraint_data)


@qd.kernel
def kernel_substep_resolve_post(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    contact_data: qd.template(),
    constraint_data: qd.template(),
):
    func_substep_resolve_post(rigid_data, rigid_config, contact_data, constraint_data)


@qd.kernel
def kernel_substep_resolve_post_no_contact(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    constraint_data: qd.template(),
):
    func_substep_resolve_post_no_contact(rigid_data, rigid_config, constraint_data)
