"""Manager System of the rigid constraint rows, which the constraint kinds append and the rigid solve solves."""

from collections.abc import Iterator
from enum import IntEnum
from typing import TypeAlias

from numpy.typing import ArrayLike
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, HostAction, Require, StageAction, System
from genesis.engine.solvers.rigid.constraint import backward as backward_constraint_solver
from genesis.engine.solvers.rigid.constraint.solver import (
    constraint_solver_kernel_clear,
    constraint_solver_kernel_masked_clear,
    constraint_solver_kernel_reset,
    func_apply_forces,
    func_assemble_equality_rows,
    func_assemble_inequality_rows,
    func_solve_body,
)
from genesis.utils.misc import assign_indexed_tensor, indices_to_mask, qd_to_torch

from .rigid_island import RigidIsland
from .rigid_dynamics import RigidDynamics

EnvIndex: TypeAlias = ArrayLike | torch.Tensor | None

IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class RowCategory(IntEnum):
    """The categories the solve classifies the constraint rows by, in the order of their segments in every env.

    The equality rows are bilateral, the frictionloss rows Huber, the contact rows elliptic cone groups or pyramid
    rows, and the limit rows unilateral (see the category counters of ConstraintState).
    """

    EQUALITY = 0
    FRICTIONLOSS = 1
    CONTACT = 2
    LIMIT = 3


class RigidConstraint(System):
    """Own the constraint rows of the rigid solve and solve them.

    The constraint kinds join as participants (see on_rows): each appends its rows to the category the solve treats
    them as, may read its solved forces back or cache what it reads back from them, and reverses its rows. Every
    substep the manager begins the rows with the equality category in the dynamics phase, appends the other categories
    over the island partition after the collision detection, solves them on the host, then applies the solved forces.
    Only the Require of a participant admits it, so a scene whose constraints are disabled has neither.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """The rows and the solver state every constraint kind shares, and whether the noslip pass runs.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        constraint_state: array_class.ConstraintState
        noslip: bool

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)

    rigid_island = Require(RigidIsland)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._row_capacity = dict.fromkeys(RowCategory, 0)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def clear_readbacks(self):
        return self.clear_readback_caches

    @System.action(kind="stage")
    def assemble_equality_rows(self):
        rigid = self.rigid_dynamics
        return func_assemble_equality_rows, self.data, rigid.data, rigid.rigid_config, self.equality_rows

    @System.action(kind="stage")
    def assemble_inequality_rows(self):
        rigid = self.rigid_dynamics
        return (
            func_assemble_inequality_rows,
            self.data,
            rigid.data,
            rigid.rigid_config,
            self.rigid_island.partition,
            self.frictionloss_rows,
            self.contact_rows,
            self.limit_rows,
        )

    @System.action(kind="host")
    def solve(self):
        return self.solve_constraints

    @System.action(kind="stage")
    def apply_forces(self):
        rigid = self.rigid_dynamics
        return (
            func_apply_forces,
            self.data,
            rigid.data,
            rigid.rigid_config,
            self.force_readbacks,
            self.rigid_island.settle,
        )

    @System.action(kind="host")
    def reverse(self):
        return self.reverse_constraints

    # -------------------------------- Action collections --------------------------------

    @System.action_collection(kind="host")
    def initializations(self):
        """The initialize Actions of the participants, see on_rows."""

    @System.action_collection(kind="host")
    def readback_clears(self):
        """The clear_readback Actions of the participants, see on_rows."""

    @System.action_collection(kind="stage")
    def equality_rows(self):
        """The assemble Actions of the equality rows, see on_rows."""

    @System.action_collection(kind="stage")
    def frictionloss_rows(self):
        """The assemble Actions of the frictionloss rows, see on_rows."""

    @System.action_collection(kind="stage")
    def contact_rows(self):
        """The assemble Actions of the contact rows, see on_rows."""

    @System.action_collection(kind="stage")
    def limit_rows(self):
        """The assemble Actions of the limit rows, see on_rows."""

    @System.action_collection(kind="stage")
    def force_readbacks(self):
        """The read_back_forces Actions of the participants, see on_rows."""

    @System.action_collection(kind="host")
    def row_reverses(self):
        """The reverse Actions of the participants, in category order, see on_rows."""

    # -------------------------------- Protocols --------------------------------

    @System.protocol
    def on_rows(
        self,
        category: RowCategory,
        *,
        initialize: HostAction,
        assemble: StageAction,
        reverse: HostAction,
        read_back_forces: StageAction | None = None,
        clear_readback: HostAction | None = None,
    ) -> None:
        """Join the constraint rows with rows of the given category.

        Signature  initialize(), assemble(), read_back_forces(), clear_readback() and reverse(), every argument bound.
        Invoked    initialize once, before the manager allocates the rows. assemble every substep: the equality
                   category in the dynamics phase, the others after the island partition, once the categories before
                   this one have appended their rows; the assemble Actions of one category are peers. read_back_forces
                   every substep, once the accelerations and the noslip pass have taken the solved forces, before the
                   settled islands fall asleep. clear_readback whenever the rows or their forces are recomputed or
                   restored: at the start of every substep, and on reset and clear. reverse in the backward pass, once
                   the adjoint solve has written the adjoints of the rows, after the reverses of the categories before.
        Must       initialize reserves the most rows assemble appends per env (see reserve_rows). assemble appends one
                   contiguous segment per env at the row cursor n_constraints, records where it starts in the Data of
                   its participant and adds its rows to the counter of its category, reserving the whole segment
                   atomically if the category has peers. clear_readback drops what the participant read back from its
                   rows. reverse accumulates into the shared gradients.
        Must not   derive its rows from the counters of another category, or read the rows of another participant.
        """
        category_rows = {
            RowCategory.EQUALITY: self.equality_rows,
            RowCategory.FRICTIONLOSS: self.frictionloss_rows,
            RowCategory.CONTACT: self.contact_rows,
            RowCategory.LIMIT: self.limit_rows,
        }
        self.initializations += initialize
        category_rows[category] += assemble
        if read_back_forces is not None:
            self.force_readbacks += read_back_forces
        if clear_readback is not None:
            self.readback_clears += clear_readback
        # The reverses accumulate into shared gradients: ranking them by category fixes their rounding alone
        self.row_reverses.add(reverse, rank=category)

    # -------------------------------- Public get/set API --------------------------------

    def reserve_rows(self, category: RowCategory, n_rows: int) -> None:
        """Reserve n_rows more rows per env for the given category, from the initialize Action of a participant."""
        self._row_capacity[category] += n_rows

    def get_len_constraints(self) -> int:
        return self.len_constraints

    def get_n_iterations(self) -> int:
        return self.n_iterations

    def get_constrained_acc(self, envs_idx: EnvIndex = None) -> torch.Tensor:
        """The dof accelerations of the last solve, batch first."""
        return qd_to_torch(self.data.constraint_state.qacc, envs_idx, transpose=True)

    def clear_readback_caches(self) -> None:
        """Drop what the participants read back from the rows, which the next query reads again."""
        self.readback_clears.invoke()

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for item in array_class.iter_data(self.data, "constraint_state"):
            if not item.name.startswith(("constraint_state.island.", "constraint_state.qd_n_equalities")):
                yield item

    def reset(self, envs_idx: EnvIndex = None) -> None:
        self.clear_readback_caches()
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

        envs_idx = self.rigid_dynamics.scene._sanitize_envs_idx(envs_idx)
        constraint_solver_kernel_reset(envs_idx, self.data.constraint_state, self.rigid_dynamics.rigid_config)

    def clear(self, envs_idx: EnvIndex = None) -> None:
        self.reset(envs_idx)

        if gs.use_zerocopy and (
            not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool)
        ):
            constraint_state = self.data.constraint_state
            counters = tuple(
                qd_to_torch(counter, copy=False)
                for counter in (
                    constraint_state.n_constraints,
                    constraint_state.n_constraints_equality,
                    constraint_state.n_constraints_frictionloss,
                    constraint_state.n_constraints_cone,
                    constraint_state.n_constraints_contact,
                )
            )
            qd_n_equalities = qd_to_torch(constraint_state.qd_n_equalities, copy=False)
            n_eq = self.rigid_dynamics.n_equalities
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                for counter in counters:
                    counter.masked_fill_(envs_idx, 0)
                qd_n_equalities.masked_fill_(envs_idx, n_eq)
            elif isinstance(envs_idx, torch.Tensor):
                for counter in counters:
                    counter.scatter_(0, envs_idx, 0)
                qd_n_equalities.scatter_(0, envs_idx, n_eq)
            else:
                env_mask = indices_to_mask(envs_idx)
                for counter in counters:
                    assign_indexed_tensor(counter, env_mask, 0)
                assign_indexed_tensor(qd_n_equalities, env_mask, n_eq)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self.rigid_dynamics.scene._sanitize_envs_idx(envs_idx)
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            fn = constraint_solver_kernel_masked_clear
        else:
            fn = constraint_solver_kernel_clear
        fn(
            envs_idx,
            self.data.constraint_state,
            self.rigid_dynamics.data.rigid_info,
            self.rigid_dynamics.rigid_config,
        )

    def backward(self) -> None:
        """Adjoint solve of the constraint force computation.

        The caller must pre-populate the upstream gradient constraint_state.dL_dqacc (see
        RigidSolver._constraint_force_grad)."""
        if not self.rigid_dynamics.requires_grad:
            gs.raise_exception("Please set `requires_grad` to True in SimOptions to enable differentiable mode.")

        # 1. We first need to find a solution to A^T * u = g system.
        backward_constraint_solver.kernel_solve_adjoint_u(
            self.data.constraint_state,
            self.rigid_dynamics.data.dyn_info,
            self.rigid_dynamics.data.rigid_info,
            self.rigid_dynamics.rigid_config,
        )

        # 2. Using the solution u, we can compute the gradients of the input variables.
        backward_constraint_solver.kernel_compute_gradients(
            self.data.constraint_state, self.rigid_dynamics.data.rigid_info
        )

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the rows the participants reserve, once they have allocated their own Data, and the solver state."""
        rigid = self.rigid_dynamics
        self._row_capacity = dict.fromkeys(RowCategory, 0)
        self.initializations.invoke()
        self.len_constraints = sum(self._row_capacity.values())
        self.len_constraints_ = max(1, self.len_constraints)
        # The elliptic cone groups are the contact rows, which size its previous-residual cache
        self.n_cone_constraints_ = max(1, self._row_capacity[RowCategory.CONTACT])
        self.sparse_solve = rigid.rigid_config.sparse_solve
        self.n_iterations = int(rigid.options.iterations)
        self.data.constraint_state = array_class.get_constraint_state(
            batch_size=rigid.batch_size,
            n_dofs=rigid.n_dofs_,
            n_trees=rigid.n_trees_,
            links=rigid.links,
            requires_grad=rigid.requires_grad,
            noslip_iterations=rigid.options.noslip_iterations,
            rigid_config=rigid.rigid_config,
            rigid_info=rigid.data.rigid_info,
            len_constraints=self.len_constraints_,
            n_cone_constraints=self.n_cone_constraints_,
            sparse_solve=self.sparse_solve,
            island_state=self.rigid_island.data.island,
            qd_n_equalities=self.rigid_island.data.qd_n_equalities,
        )
        self.data.noslip = rigid.options.noslip_iterations > 0
        self.reset()

    def solve_constraints(self) -> None:
        """Solve the constraints of a substep, through the solver arm func_solve_body selects at run time."""
        rigid_data = self.rigid_dynamics.data
        func_solve_body(
            rigid_data.dyn_state,
            self.data.constraint_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            self.rigid_dynamics.rigid_config,
            self.n_iterations,
        )

    def reverse_constraints(self) -> None:
        """Reverse the host constraint solve, then every row category through the reverses of its participants."""
        rigid = self.rigid_dynamics
        constraint_state = self.data.constraint_state
        if gs.use_zerocopy:
            acc_grad = qd_to_torch(rigid.data.dyn_state.dofs.acc.grad, copy=False)
            dL_dqacc = qd_to_torch(constraint_state.dL_dqacc, copy=False)
            dL_dqacc.copy_(acc_grad)
            acc_grad.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            backward_constraint_solver.kernel_load_dL_dqacc_from_acc_grad(
                rigid.data.dyn_state,
                constraint_state,
                rigid.rigid_config,
            )

        self.backward()

        if gs.use_zerocopy:
            force_grad = qd_to_torch(rigid.data.dyn_state.dofs.force.grad, copy=False)
            dL_dforce = qd_to_torch(constraint_state.dL_dforce, copy=False)
            force_grad.add_(dL_dforce)
            mass_mat_grad = qd_to_torch(rigid.data.rigid_info.mass_mat.grad, copy=False)
            dL_dM = qd_to_torch(constraint_state.dL_dM, copy=False)
            mass_mat_grad.add_(dL_dM)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            backward_constraint_solver.kernel_accumulate_constraint_solver_grads(
                rigid.data.dyn_state,
                constraint_state,
                rigid.data.rigid_info,
                rigid.rigid_config,
            )

        self.row_reverses.invoke()
