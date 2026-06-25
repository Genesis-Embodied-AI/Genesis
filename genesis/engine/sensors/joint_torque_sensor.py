from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.options.sensors import JointTorqueSensor as JointTorqueSensorOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_torch

from .base_sensor import (
    SimpleSensor,
    SimpleSensorMetadata,
)

if TYPE_CHECKING:
    from .sensor_manager import SensorManager


@qd.kernel
def _kernel_get_dofs_frictionloss_force(
    fl_rank: qd.types.ndarray(),
    constraint_state: array_class.ConstraintState,
    output: qd.types.ndarray(),
):
    """
    Gather the frictionloss constraint force for each sensor DOF.

    Frictionloss constraints occupy indices [ne, nef) in the constraint list (equality first,
    then frictionloss, then contact).  Their Jacobians are identity so efc_force[ne + rank, i_b]
    is exactly the DOF-space frictionloss force.  DOFs without frictionloss (rank == -1) get zero.
    """
    for i_b, i_s in qd.ndrange(output.shape[0], fl_rank.shape[0]):
        rank = fl_rank[i_s]
        if rank >= 0:
            ne = constraint_state.n_constraints_equality[i_b]
            output[i_b, i_s] = constraint_state.efc_force[ne + rank, i_b]
        else:
            output[i_b, i_s] = gs.qd_float(0.0)


@dataclass
class JointTorqueSensorMetadata(SimpleSensorMetadata):
    """Shared state for all JointTorqueSensor instances in a scene."""

    solver: "gs.RigidSolver | None" = None
    # Global DOF indices concatenated across all sensor instances in scene order.
    dofs_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


class JointTorqueSensorData(NamedTuple):
    torque: torch.Tensor  # (n_dofs,) or (n_envs, n_dofs)


class JointTorqueSensor(
    SimpleSensor[JointTorqueSensorOptions, None, JointTorqueSensorMetadata, JointTorqueSensorData],
):
    """
    Measures the torque transmitted from each actuator to its joint output shaft.
    Formula (derived from Newton's 3rd law at the gearbox interface):
        tau_sensor = tau_control − I_arm · qacc + tau_frictionloss
    """

    def __init__(
        self,
        options: JointTorqueSensorOptions,
        idx: int,
        shared_context: None,
        shared_metadata: JointTorqueSensorMetadata,
        manager: "SensorManager",
    ):
        # Resolve dofs_idx_local=None before super().__init__() because _get_return_format()
        # is called inside the base constructor and needs the final DOF count.
        if options.dofs_idx_local is None:
            entity = manager._sim.entities[options.entity_idx]
            options.dofs_idx_local = tuple(range(entity.n_dofs))
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    # ─────────────────────────── interface ───────────────────────────

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (len(self._options.dofs_idx_local),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    def build(self):
        super().build()

        solver = self._manager._sim.rigid_solver
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = solver

        entity = self._manager._sim.entities[self._options.entity_idx]

        global_dofs_idx = torch.tensor(
            [entity._dof_start + i for i in self._options.dofs_idx_local],
            dtype=gs.tc_int,
            device=gs.device,
        )

        self._shared_metadata.dofs_idx = concat_with_tensor(self._shared_metadata.dofs_idx, global_dofs_idx, dim=0)

    # ─────────────────────── update hook ─────────────────────────────

    @classmethod
    def _update_raw_data(
        cls,
        shared_context: None,
        shared_metadata: JointTorqueSensorMetadata,
        raw_data_T: torch.Tensor,  # shape (n_total_sensor_dofs, B)
    ):
        solver = shared_metadata.solver
        dofs = shared_metadata.dofs_idx
        B = solver._B
        n_total = dofs.shape[0]

        # tau_control: reconstructed PD/actuator output force per DOF
        qf_control = solver.get_dofs_control_force(dofs)  # (n_dofs,) or (B, n_dofs)
        if solver.n_envs == 0:
            qf_control = qf_control.unsqueeze(0)  # (1, n_dofs)

        # qacc: constraint-solved acceleration (NOT the pre-constraint dofs_state.acc)
        # qd field (n_scene_dofs, B); qd_to_torch with transpose=True → (B, n_dofs)
        qacc = qd_to_torch(solver.constraint_solver.qacc, None, dofs, transpose=True, copy=True)

        # tau_frictionloss: Coulomb friction constraint forces (identity Jacobian rows).
        # Recompute ranks each step so they reflect runtime set_dofs_frictionloss() values.
        all_fl = solver.get_dofs_frictionloss()  # (n_scene_dofs,) or (B, n_scene_dofs)
        if all_fl.ndim == 2:
            all_fl = all_fl[0]
        has_fl = all_fl > gs.EPS
        rank_all = torch.cumsum(has_fl.long(), dim=0) - 1
        rank_all[~has_fl] = -1
        fl_rank = rank_all[dofs].to(dtype=gs.tc_int)

        qfrc_fl = torch.empty((B, n_total), dtype=gs.tc_float, device=gs.device)
        _kernel_get_dofs_frictionloss_force(
            fl_rank.contiguous(),
            solver.constraint_solver.constraint_state,
            qfrc_fl,
        )

        armature = solver.get_dofs_armature(dofs)

        # raw_data_T layout is (n_total_sensor_dofs, B); write the transposed result
        raw_data_T.copy_((qf_control - armature * qacc + qfrc_fl).T)
