from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.options.sensors import Magnetometer as MagnetometerOptions
from genesis.utils.geom import (
    inv_transform_by_quat,
    transform_by_quat,
    transform_quat_by_quat,
)
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


@dataclass
class MagnetometerSharedMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata between all Magnetometer sensors.
    """

    magnetic_field_vector: torch.Tensor = make_tensor_field((0, 3))


class MagnetometerData(NamedTuple):
    magnetic_field: torch.Tensor


@register_sensor(MagnetometerOptions, MagnetometerSharedMetadata, MagnetometerData)
@ti.data_oriented
class MagnetometerSensor(
    RigidSensorMixin[MagnetometerSharedMetadata],
    NoisySensorMixin[MagnetometerSharedMetadata],
    Sensor[MagnetometerSharedMetadata],
):
    def __init__(
        self,
        options: MagnetometerOptions,
        shared_metadata: MagnetometerSharedMetadata,
        data_cls: Type[MagnetometerData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, shared_metadata, data_cls, manager)

        self.debug_object: "Mesh | None" = None
        self.quat_offset: torch.Tensor
        self.pos_offset: torch.Tensor

    # ================================ internal methods ================================

    def build(self):
        """
        Initialize all shared metadata needed to update all Magnetometer sensors.
        """
        super().build()  # Initialize RigidSensorMixin and NoisySensorMixin

        default_field = self._options.magnetic_field if self._options.magnetic_field is not None else (0.0, 0.0, 0.5)

        if not isinstance(default_field, torch.Tensor):
            default_field = torch.tensor(default_field, device=gs.device, dtype=gs.tc_float)

        self._shared_metadata.magnetic_field_vector = concat_with_tensor(
            self._shared_metadata.magnetic_field_vector,
            default_field,
            expand=(self._manager._sim._B, 3),
            dim=0,
        )

        if self._options.draw_debug:
            self.quat_offset = self._shared_metadata.offsets_quat[0, self._idx]
            self.pos_offset = self._shared_metadata.offsets_pos[0, self._idx]

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return ((3,),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: MagnetometerSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the current ground truth values for all Magnetometer sensors.
        """
        assert shared_metadata.solver is not None

        quats = shared_metadata.solver.get_links_quat(links_idx=shared_metadata.links_idx)

        if quats.ndim == 2:
            quats = quats[None]

        offset_quats = transform_quat_by_quat(quats, shared_metadata.offsets_quat)

        B_world = shared_metadata.magnetic_field_vector
        B_local = inv_transform_by_quat(B_world[:, None, :], offset_quats)

        shared_ground_truth_cache.copy_(B_local.reshape(shared_ground_truth_cache.shape))

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: MagnetometerSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the current measured sensor data for all Magnetometer sensors.
        """
        buffered_data.set(shared_ground_truth_cache)

        # Jitter handling
        if torch.any(shared_metadata.jitter_ts > 0):
            torch.normal(0.0, shared_metadata.jitter_ts, out=shared_metadata.cur_jitter_ts)
            cls._apply_delay_to_shared_cache(
                shared_metadata,
                shared_cache,
                buffered_data,
                shared_metadata.cur_jitter_ts,
                shared_metadata.interpolate,
            )
        else:
            cls._apply_delay_to_shared_cache(
                shared_metadata,
                shared_cache,
                buffered_data,
                None,
                shared_metadata.interpolate,
            )

        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug arrow representing the magnetic north vector in the sensor's frame.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        quat = self._link.get_quat(env_idx).reshape((4,))
        pos = self._link.get_pos(env_idx).reshape((3,)) + transform_by_quat(self.pos_offset, quat)

        # Read the magnetic field (local frame)
        data = self.read(env_idx)
        if isinstance(data, torch.Tensor):
            mag_vec = data.reshape((3,))
        else:
            mag_vec = data.magnetic_field.reshape((3,))

        # Transform to world frame for visualization
        offset_quat = transform_quat_by_quat(self.quat_offset, quat)
        mag_vec_world = transform_by_quat(mag_vec * self._options.debug_scale, offset_quat)

        pos_np = pos.detach().cpu().numpy()
        mag_vec_world_np = mag_vec_world.detach().cpu().numpy()

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        # Use the NumPy versions here
        self.debug_object = context.draw_debug_arrow(pos=pos_np, vec=mag_vec_world_np, color=self._options.debug_color)
