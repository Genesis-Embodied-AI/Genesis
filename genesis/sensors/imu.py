from typing import TYPE_CHECKING, Any

import taichi as ti
import torch

import genesis as gs
from genesis.options.sensors import SensorOptions
from genesis.utils.geom import (
    euler_to_quat,
    inv_transform_by_trans_quat,
    transform_quat_by_quat,
)

from .base_sensor import Sensor
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


@ti.data_oriented
class IMU(Sensor):

    def build(self):
        self._solver = self._manager._sim.rigid_solver
        assert self._options.link_idx >= 0 and self._options.link_idx < self._solver.n_links, "Invalid RigidLink index."
        self._link_idx = self._options.link_idx

        quat_offset = euler_to_quat(self._options.euler_offset)

        if len(self._shared_metadata) == 0:
            self._shared_metadata["solver"] = self._solver
            self._shared_metadata["links_idx"] = []
            self._shared_metadata["offsets_pos"] = torch.tensor([], dtype=torch.float32)
            self._shared_metadata["offsets_quat"] = torch.tensor([], dtype=torch.float32)

        self._shared_metadata["links_idx"].append(self._link_idx)
        self._shared_metadata["offsets_pos"] = torch.cat(
            [self._shared_metadata["offsets_pos"], torch.tensor([self._options.pos_offset], dtype=torch.float32)]
        )
        self._shared_metadata["offsets_quat"] = torch.cat(
            [self._shared_metadata["offsets_quat"], torch.tensor([quat_offset], dtype=torch.float32)]
        )

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
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata["solver"]
        links_idx = shared_metadata["links_idx"]
        offsets_pos = shared_metadata["offsets_pos"]
        offsets_quat = shared_metadata["offsets_quat"]

        gravity = solver.get_gravity()
        quats = solver.get_links_quat(links_idx=links_idx)
        acc = solver.get_links_acc(links_idx=links_idx)
        ang = solver.get_links_ang(links_idx=links_idx)
        if solver.n_envs == 0:
            gravity = gravity.unsqueeze(0)
            quats = quats.unsqueeze(0)
            acc = acc.unsqueeze(0)
            ang = ang.unsqueeze(0)

        offset_quats = transform_quat_by_quat(quats, offsets_quat.unsqueeze(0).repeat(quats.shape[0], 1, 1))
        # acc/ang shape: (B, n_links, 3)
        local_acc = inv_transform_by_trans_quat(acc, offsets_pos, offset_quats)
        local_ang = inv_transform_by_trans_quat(ang, offsets_pos, offset_quats)

        local_acc = local_acc - gravity.unsqueeze(1).repeat(1, local_acc.shape[1], 1)

        # cache shape: (B, n_links * 6)
        shared_ground_truth_cache.copy_(torch.cat([local_acc, local_ang], dim=2).flatten(1))

    @classmethod
    def _update_shared_cache(
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor, shared_cache: "TensorRingBuffer"
    ):
        shared_cache.append(shared_ground_truth_cache)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float


@register_sensor(IMU)
class IMUOptions(SensorOptions):
    """
    IMU sensor returns the linear acceleration (accelerometer) and angular velocity (gyroscope)
    of the associated entity link.

    Parameters
    ----------
    link_idx : int
        The global index of the RigidLink to which this IMU sensor is attached.
    pos_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink.
    euler_offset : tuple[float, float, float]
        The offset of the IMU sensor from the RigidLink in euler angles.
    """

    link_idx: int
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    return_accelerometer: bool = True
    return_gyroscope: bool = True
