"""Participant System of the rigid equality constraints: connect, weld and joint equalities, dynamic welds included."""

from collections.abc import Iterator
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.core import Data, Require, System, register_system
from genesis.engine.solvers.rigid.constraint.backward import kernel_manual_add_equality_constraints_bw
from genesis.engine.solvers.rigid.constraint.solver import (
    func_substep_add_equality_rows,
    kernel_add_weld_constraint,
    kernel_delete_weld_constraint,
    kernel_get_equality_constraints,
)

from .rigid_constraint import RigidConstraint, RowCategory
from .rigid_dynamics import RigidDynamics
from ..simulator_engine import SimulatorEngine

EnvIndex: TypeAlias = ArrayLike | torch.Tensor | None
ConstraintArray: TypeAlias = np.ndarray | torch.Tensor
ConstraintValue: TypeAlias = ConstraintArray | tuple[ConstraintArray, ...]
ConstraintResult: TypeAlias = dict[str, ConstraintValue]


class RigidEqualityConstraint(System):
    """Append the equality rows of every env, and own the queries and edits of the equality constraints.

    The rows are bilateral: up to 3 for a connect, 6 for a weld and 1 for a joint equality, over the equalities of the
    model and the welds registered at run time (see add_weld_constraint).
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """Where the equality rows start in every env.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        segment: array_class.ConstraintRowSegment

    data: Data

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)

    rigid_constraint = Require(RigidConstraint)

    def __init__(self) -> None:
        super().__init__()
        self.data = self.Data()
        self._eq_const_info_cache: dict[tuple[bool, bool], ConstraintResult] = {}

    def build(self) -> None:
        self.rigid_constraint.on_rows(
            RowCategory.EQUALITY,
            initialize=self.initialize,
            assemble=self.assemble,
            reverse=self.reverse,
            clear_readback=self.clear_readback,
        )

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def clear_readback(self):
        return self.clear_query_cache

    @System.action(kind="stage")
    def assemble(self):
        rigid = self.rigid_dynamics
        return func_substep_add_equality_rows, self.data, self.rigid_constraint.data, rigid.data, rigid.rigid_config

    @System.action(kind="host")
    def reverse(self):
        rigid = self.rigid_dynamics
        return (
            kernel_manual_add_equality_constraints_bw,
            self.data.segment,
            rigid.data.dyn_state,
            self.rigid_constraint.data.constraint_state,
            rigid.data.dyn_info,
            rigid.data.rigid_info,
            rigid.rigid_config,
        )

    # -------------------------------- Public get/set API --------------------------------

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        for item in array_class.iter_data(self.data, "segment"):
            yield item._replace(name=f"equality_rows.{item.name}")

    def clear_query_cache(self) -> None:
        """Drop the equality constraints read back, once the forces or the equalities they were read from changed."""
        self._eq_const_info_cache.clear()

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True) -> ConstraintResult:
        # Early return if already pre-computed
        eq_const_info = self._eq_const_info_cache.get((as_tensor, to_torch))
        if eq_const_info is not None:
            return eq_const_info.copy()

        constraint_state = self.rigid_constraint.data.constraint_state
        n_eqs = tuple(constraint_state.qd_n_equalities.to_numpy())
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
                self.data.segment,
                constraint_state,
                self.rigid_dynamics.data.dyn_info,
                self.rigid_dynamics.rigid_config,
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
        rigid = self.rigid_dynamics
        envs_idx = rigid.scene._sanitize_envs_idx(envs_idx)
        link1_idx, link2_idx = int(link1_idx), int(link2_idx)

        assert link1_idx >= 0 and link2_idx >= 0
        weld_const_info = self.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        assert not (
            ((link_a == link1_idx) | (link_b == link1_idx)) & ((link_a == link2_idx) | (link_b == link2_idx))
        ).any()

        self.clear_query_cache()
        overflow = kernel_add_weld_constraint(
            link1_idx,
            link2_idx,
            envs_idx,
            rigid.data.dyn_state,
            self.rigid_constraint.data.constraint_state,
            rigid.data.dyn_info,
            rigid.data.rigid_info,
            rigid.rigid_config,
        )
        if overflow:
            gs.logger.warning(
                "Ignoring dynamically registered weld constraint to avoid exceeding max number of equality constraints"
                f"({rigid.data.rigid_info.n_candidate_equalities.to_numpy()}). Please increase the value of "
                "RigidSolver's option 'max_dynamic_constraints'."
            )

    def delete_weld_constraint(self, link1_idx: int, link2_idx: int, envs_idx: EnvIndex = None) -> None:
        rigid = self.rigid_dynamics
        envs_idx = rigid.scene._sanitize_envs_idx(envs_idx)
        self.clear_query_cache()
        kernel_delete_weld_constraint(
            int(link1_idx),
            int(link2_idx),
            envs_idx,
            self.rigid_constraint.data.constraint_state,
            rigid.data.dyn_info,
            rigid.data.rigid_info,
            rigid.rigid_config,
        )

    # -------------------------------- Implementation --------------------------------

    def initialize_data(self) -> None:
        """Allocate the segment record and reserve up to 6 rows per candidate equality, a weld taking 6."""
        rigid = self.rigid_dynamics
        self.data.segment = array_class.get_constraint_row_segment(batch_size=rigid.batch_size)
        self.rigid_constraint.reserve_rows(RowCategory.EQUALITY, 6 * rigid.n_candidate_equalities_)
        self.clear_query_cache()


@register_system(SimulatorEngine)
def create_rigid_equality_constraint(engine: SimulatorEngine) -> RigidEqualityConstraint | None:
    """Create the equality rows whenever a Rigid scene solves constraints, since welds may be registered at run time."""
    from genesis.engine.entities import RigidEntity

    has_rigid = any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
    return RigidEqualityConstraint() if has_rigid and not engine.scene.options.rigid.disable_constraint else None
