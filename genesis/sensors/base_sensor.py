from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.options import Options
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


class SensorOptions(Options):
    """
    Base class for all sensor options.
    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.

    Parameters
    ----------
    read_delay : float
        The delay in seconds before the sensor data is read.
    """

    read_delay: float = 0.0

    def validate(self, scene):
        """
        Validate the sensor options values before the sensor is added to the scene.
        """
        read_delay_hz = self.read_delay / scene._sim.dt
        if not np.isclose(read_delay_hz, round(read_delay_hz), atol=1e-6):
            gs.logger.warn(
                f"Read delay should be a multiple of the simulation time step. Got {self.read_delay}"
                f" and {scene._sim.dt}. Actual read delay will be {1/round(read_delay_hz)}."
            )


@dataclass
class SharedSensorMetadata:
    """
    Shared metadata between all sensors of the same class.
    """

    cache_sizes: list[int] = field(default_factory=list)
    read_delay_steps: list[int] = field(default_factory=list)


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    """

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadata = sensor_manager._sensors_metadata[type(self)]

        self._read_delay_steps = round(self._options.read_delay / self._manager._sim.dt)
        self._shared_metadata.read_delay_steps.append(self._read_delay_steps)

        self._shape_indices: list[tuple[int, int]] = []
        return_format = self._get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        tensor_size = 0
        for shape in return_shapes:
            data_size = np.prod(shape)
            self._shape_indices.append((tensor_size, tensor_size + data_size))
            tensor_size += data_size

        self._cache_size = self._get_cache_length() * tensor_size
        self._shared_metadata.cache_sizes.append(self._cache_size)

        self._cache_idx: int = -1  # initialized by SensorManager during build

    # =============================== implementable methods ===============================

    def build(self):
        """
        This method is called by the SensorManager during the scene build phase to initialize the sensor.
        This is where any shared metadata should be initialized.
        """
        raise NotImplementedError("Sensors must implement `build()`.")

    def _get_return_format(self) -> dict[str, tuple[int, ...]] | tuple[int, ...]:
        """
        Data format of the read() return value.

        Returns
        -------
        return_format : dict | tuple
            - If tuple, the final shape of the read() return value.
                e.g. (2, 3) means read() will return a tensor of shape (2, 3).
            - If dict a dictionary with string keys and tensor values will be returned.
                e.g. {"pos": (3,), "quat": (4,)} returns a dict of tensors [0:3] and [3:7] from the cache.
        """
        raise NotImplementedError("Sensors must implement `return_format()`.")

    def _get_cache_length(self) -> int:
        """
        The length of the cache for this sensor instance, e.g. number of points for a Lidar point cloud.
        """
        raise NotImplementedError("Sensors must implement `cache_length()`.")

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the shared sensor ground truth cache for all sensors of this class using metadata in SensorManager.
        """
        raise NotImplementedError("Sensors must implement `update_shared_ground_truth_cache()`.")

    @classmethod
    def _update_shared_cache_with_noise(
        cls,
        shared_metadata: dict[str, Any],
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the shared sensor cache for all sensors of this class using metadata in SensorManager.
        This is where noise should be applied to the sensor data, if applicable.
        """
        raise NotImplementedError("Sensors must implement `update_shared_cache_with_noise()`.")

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        """
        The dtype of the cache for this sensor.
        """
        raise NotImplementedError("Sensors must implement `get_cache_dtype()`.")

    # =============================== shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx: List[int] | None = None):
        """
        Read the sensor data (with noise applied if applicable).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx: List[int] | None = None):
        """
        Read the ground truth sensor data (without noise).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self, is_ground_truth=True), envs_idx)

    def _get_formatted_data(
        self, tensor: torch.Tensor, envs_idx: list[int] | None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Note: This method does not clone the data tensor, it should have been cloned by the caller.

        if envs_idx is None:
            envs_idx = self._manager._sim._scene._envs_idx

        return_format = self._get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        return_values = []

        for i, shape in enumerate(return_shapes):
            start_idx, end_idx = self._shape_indices[i]
            value = tensor[envs_idx, start_idx:end_idx].reshape(len(envs_idx), *shape).squeeze()
            if self._manager._sim.n_envs == 0:
                value = value.squeeze(0)
            return_values.append(value)

        if isinstance(return_format, dict):
            return dict(zip(return_format.keys(), return_values))
        else:
            return return_values[0]

    @property
    def is_built(self) -> bool:
        return self._manager._sim._scene._is_built
