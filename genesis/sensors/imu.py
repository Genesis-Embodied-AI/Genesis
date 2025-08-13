from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import taichi as ti
import torch

import genesis as gs
from genesis.engine.solvers import RigidSolver
from genesis.options.sensors import SensorOptions
from genesis.utils.geom import (
    euler_to_quat,
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import Sensor, SharedSensorMetadata
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


class IMUOptions(SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

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
    """

    entity_idx: int
    link_idx_local: int = 0
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    accelerometer_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyroscope_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    return_accelerometer: bool = True
    return_gyroscope: bool = True

    def validate(self, scene):
        assert self.entity_idx >= 0 and self.entity_idx < len(scene.entities), "Invalid RigidEntity index."
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
    offsets_pos: torch.Tensor = torch.tensor([])
    offsets_quat: torch.Tensor = torch.tensor([])


@register_sensor(IMUOptions, IMUSharedMetadata)
@ti.data_oriented
class IMU(Sensor):

    def build(self):
        self._solver = self._manager._sim.rigid_solver
        self._link_idx = self._options.entity_idx + self._options.link_idx_local

        quat_offset = euler_to_quat(self._options.euler_offset)

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._solver

        self._shared_metadata.links_idx.append(self._link_idx)
        self._shared_metadata.offsets_pos = torch.cat(
            [self._shared_metadata.offsets_pos, torch.tensor([self._options.pos_offset], dtype=gs.tc_float)]
        )

        quat_tensor = torch.tensor(quat_offset, dtype=gs.tc_float)
        quat_tensor = quat_tensor.view(1, 1, 4).expand(self._manager._sim._B, 1, 4)
        self._shared_metadata.offsets_quat = torch.cat([self._shared_metadata.offsets_quat, quat_tensor], dim=1)

    def _get_return_format(self) -> dict[str, tuple[int, ...]]:
        return_format = {}
        if self._options.return_accelerometer:
            return_format["lin_acc"] = (3,)
        if self._options.return_gyroscope:
            return_format["ang_vel"] = (3,)
        return return_format

    def _get_cache_length(self) -> int:
        return 1

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: IMUSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        gravity = shared_metadata.solver.get_gravity()
        quats = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)
        acc = shared_metadata.solver.get_links_acc(links_idx=shared_metadata.links_idx)
        ang = shared_metadata.solver.get_links_ang(links_idx=shared_metadata.links_idx)
        if shared_metadata.solver.n_envs == 0:
            gravity = gravity.unsqueeze(0)
            quats = quats.unsqueeze(0)
            acc = acc.unsqueeze(0)
            ang = ang.unsqueeze(0)

        offset_quats = transform_quat_by_quat(quats, shared_metadata.offsets_quat)

        # acc/ang shape: (B, n_links, 3)
        local_acc = inv_transform_by_trans_quat(acc, shared_metadata.offsets_pos, offset_quats)
        local_ang = inv_transform_by_trans_quat(ang, shared_metadata.offsets_pos, offset_quats)

        local_acc = local_acc - gravity.unsqueeze(1).expand(-1, local_acc.shape[1], -1)

        # cache shape: (B, n_links * 6)
        batch_size, n_links, n_xyz = local_acc.shape
        interleaved_cache_view = torch.as_strided(
            shared_ground_truth_cache,
            size=(batch_size, n_links, 2, n_xyz),
            stride=(n_links * 2 * n_xyz, 2 * n_xyz, n_xyz, 1),
        )
        interleaved_cache_view[:, :, 0, :].copy_(local_acc)
        interleaved_cache_view[:, :, 1, :].copy_(local_ang)

    @classmethod
    def _update_shared_cache(
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor, shared_cache: "TensorRingBuffer"
    ):
        shared_cache.append(shared_ground_truth_cache)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
