from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import gstaichi as ti
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import (
    euler_to_quat,
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import Sensor, SensorOptions, SharedSensorMetadata
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


class IMUOptions(SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Note
    ----
    Accelerometers return the so-called classical linear acceleration in local frame minus gravity.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this IMU sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this IMU sensor is attached.
    pos_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink.
    euler_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink in euler angles.
    accelerometer_bias : tuple[float, float, float]
        The bias of the accelerometer.
    gyroscope_bias : tuple[float, float, float]
        The bias of the gyroscope.
    """

    entity_idx: int
    link_idx_local: int = 0
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    accelerometer_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyroscope_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self, scene):
        assert self.entity_idx >= 0 and self.entity_idx < len(scene.entities), "Invalid RigidEntity index."
        entity = scene.entities[self.entity_idx]
        assert isinstance(entity, RigidEntity), "Entity at given index is not a RigidEntity."
        assert (
            self.link_idx_local >= 0 and self.link_idx_local < scene.entities[self.entity_idx].n_links
        ), "Invalid RigidLink index."


@dataclass
class IMUSharedMetadata(SharedSensorMetadata):
    """
    Shared metadata between all IMU sensors.
    """

    solver: RigidSolver | None = None
    links_idx: list[int] = field(default_factory=list)
    offsets_pos: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    offsets_quat: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    acc_bias: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    ang_bias: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))


@register_sensor(IMUOptions, IMUSharedMetadata)
@ti.data_oriented
class IMU(Sensor):

    def build(self):
        """
        Initialize all shared metadata needed to update all IMU sensors.
        """
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.links_idx.append(self._options.entity_idx + self._options.link_idx_local)
        self._shared_metadata.offsets_pos = torch.cat(
            [
                self._shared_metadata.offsets_pos,
                torch.tensor([self._options.pos_offset], dtype=gs.tc_float, device=gs.device),
            ]
        )

        quat_tensor = torch.tensor(euler_to_quat([self._options.euler_offset]), dtype=gs.tc_float, device=gs.device)
        if self._shared_metadata.solver.n_envs > 0:
            quat_tensor = quat_tensor.unsqueeze(0).expand((self._manager._sim._B, 1, 4))
        self._shared_metadata.offsets_quat = torch.cat([self._shared_metadata.offsets_quat, quat_tensor], dim=-2)

        self._shared_metadata.acc_bias = torch.cat(
            [
                self._shared_metadata.acc_bias,
                torch.tensor([self._options.accelerometer_bias], dtype=gs.tc_float, device=gs.device),
            ]
        )
        self._shared_metadata.ang_bias = torch.cat(
            [
                self._shared_metadata.ang_bias,
                torch.tensor([self._options.gyroscope_bias], dtype=gs.tc_float, device=gs.device),
            ]
        )

    def _get_return_format(self) -> dict[str, tuple[int, ...]]:
        return {
            "lin_acc": (3,),
            "ang_vel": (3,),
        }

    def _get_cache_length(self) -> int:
        return 1

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: IMUSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the current ground truth values for all IMU sensors.
        """
        gravity = shared_metadata.solver.get_gravity()
        quats = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        acc = shared_metadata.solver.get_links_acc(links_idx=shared_metadata.links_idx)
        ang = shared_metadata.solver.get_links_ang(links_idx=shared_metadata.links_idx)

        offset_quats = transform_quat_by_quat(quats, shared_metadata.offsets_quat)

        # acc/ang shape: (B, n_imus, 3)
        local_acc = inv_transform_by_trans_quat(acc, shared_metadata.offsets_pos, offset_quats)
        local_ang = inv_transform_by_trans_quat(ang, shared_metadata.offsets_pos, offset_quats)

        *batch_size, n_imus, _ = local_acc.shape
        local_acc = local_acc - gravity.unsqueeze(-2).expand((*batch_size, n_imus, -1))

        # cache shape: (B, n_imus * 6)
        strided_ground_truth_cache = shared_ground_truth_cache.reshape((*batch_size, n_imus, 2, 3))
        strided_ground_truth_cache[..., 0, :].copy_(local_acc)
        strided_ground_truth_cache[..., 1, :].copy_(local_ang)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: dict[str, Any],
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the current measured sensor data for all IMU sensors.

        Note
        ----
        `buffered_data` contains the history of ground truth cache, and noise/bias is only applied to the current
        sensor readout `shared_cache`, not the whole buffer.
        """
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

        # add bias to the shared_cache
        *batch_size, n_imus, _ = shared_metadata.offsets_quat.shape
        strided_shared_cache = shared_cache.reshape((*batch_size, n_imus, 2, 3))
        strided_shared_cache[..., 0, :] += shared_metadata.acc_bias
        strided_shared_cache[..., 1, :] += shared_metadata.ang_bias

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
