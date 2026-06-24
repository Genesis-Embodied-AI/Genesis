from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.options.sensors import JointTorque as JointTorqueSensorOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.solvers import RigidSolver

    from .sensor_manager import SensorManager


@dataclass
class JointTorqueSensorMetadata(SimpleSensorMetadata):
    """
    Shared metadata for all joint torque sensors.
    """

    solver: "RigidSolver | None" = None
    # Global dof indices measured across all joint torque sensors, concatenated in sensor order. Each
    # sensor owns a contiguous slice (mirrors `ContactSensorMetadata.expanded_links_idx`).
    dofs_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


class JointTorqueSensor(SimpleSensor[JointTorqueSensorOptions, None, JointTorqueSensorMetadata]):
    """
    Sensor that returns the torque transmitted through the joints of the associated RigidEntity.

    Reports the actuation-side generalized force ``qf_applied + qf_passive`` (see
    ``RigidEntity.get_dofs_actuation_force``), i.e. the quantity a torque sensor mounted in series with the
    actuator would measure.
    """

    def _local_dofs_idx(self) -> torch.Tensor:
        """Local dof indices this sensor measures (all of the entity's dofs when none were specified)."""
        if len(self._options.dofs_idx_local) > 0:
            return torch.as_tensor(self._options.dofs_idx_local, dtype=gs.tc_int, device=gs.device)
        entity = self._manager._sim.entities[self._options.entity_idx]
        return torch.arange(entity.n_dofs, dtype=gs.tc_int, device=gs.device)

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        entity = self._manager._sim.entities[self._options.entity_idx]
        dofs_idx_global = self._local_dofs_idx() + entity.dof_start
        self._shared_metadata.dofs_idx = concat_with_tensor(self._shared_metadata.dofs_idx, dofs_idx_global, dim=0)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._local_dofs_idx().numel(),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: JointTorqueSensorMetadata, raw_data_T: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        force = shared_metadata.solver.get_dofs_actuation_force(dofs_idx=shared_metadata.dofs_idx)
        if shared_metadata.solver.n_envs == 0:
            force = force[None]
        # `force` is (B, total_dofs); raw buffer is (total_dofs, B).
        raw_data_T[:] = force.T
