from dataclasses import dataclass
from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import RigidSolver

from .contact_force import ForceSensor, ForceSensorMetadata, ForceSensorOptions
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


class TactileArrayOptions(ForceSensorOptions):
    """
    Sensor that returns estimated force based on penetration depth on a tactile array.

    Parameters
    ----------
    array_resolution : tuple[int, int]
        The number of taxels (tactile pixels) in the x and y directions of the tactile array.
    array_top_left: tuple[float, float, float]
        The position of the top left taxel of the tactile array in the local frame of the RigidLink.
    array_bottom_right: tuple[float, float, float]
        The position of the bottom right taxel of the tactile array in the local frame of the RigidLink.
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    min_force : float | tuple[float, float, float], optional
        The minimum detectable force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | tuple[float, float, float], optional
        The maximum detectable force per each axis. Values above this will be clipped. Default is infinity.
    noise_std : float | tuple[float, float, float], optional
        The standard deviation of the noise.
    bias : float | tuple[float, float, float], optional
        The bias of the sensor.
    bias_drift_std : float | tuple[float, float, float], optional
        The standard deviation of the bias drift.
    delay : float, optional
        The delay in seconds before the sensor data is read.
    jitter : float, optional
        The time jitter standard deviation in seconds before the sensor data is read.
    delay : float, optional
        The read delay time in seconds. Data read will be outdated by this amount.
    interpolate_for_delay : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    """

    array_resolution: tuple[int, int]
    array_top_left: tuple[float, float, float]
    array_bottom_right: tuple[float, float, float]


@dataclass
class TactileArrayMetadata(ForceSensorMetadata):
    """
    Metadata for all rigid tactile array sensors.
    """

    solver: RigidSolver | None = None
    links_idx: torch.Tensor = torch.tensor([])
    min_max_force: torch.Tensor = torch.tensor([])
    # TODO: add metadata needed for getting taxel forces


@register_sensor(TactileArrayOptions, TactileArrayMetadata)
@ti.data_oriented
class TactileArray(ForceSensor):
    """
    Sensor that returns the contact force in the associated RigidLink's local frame.
    """

    def build(self):
        super().build()

    def get_return_format(self) -> dict[str, tuple[int, ...]]:
        return (3,)

    def get_cache_length(self) -> int:
        return np.prod(self._options.array_resolution)

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def update_shared_ground_truth_cache(
        cls, shared_metadata: TactileArrayMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        # TODO: implement ghost collider points
        pass

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: TactileArrayMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0, shared_metadata.jitter_std_in_steps, out=shared_metadata.jitter_in_steps)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.jitter_in_steps,
            shared_metadata.interpolate_for_delay,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
