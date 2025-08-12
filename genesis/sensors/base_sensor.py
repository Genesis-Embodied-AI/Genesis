from typing import TYPE_CHECKING, Any, List, Type

import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    A sensor must have a read() method that returns the sensor data.
    """

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._class: Type[Sensor] = self.__class__

        # initialized during build
        self._cache_idx: int = -1  # set by the SensorManager during build
        self._read_delay_steps: int = 0
        # initialized by SceneManager during build
        self._shape_indices: list[tuple[int, int]] = []
        self._cache_size: int = 0
        self._cache: torch.Tensor | None = None
        self._ground_truth_cache: torch.Tensor | None = None
        self._shared_metadata: dict[str, Any] = {}

    # =============================== implementable methods ===============================

    @gs.assert_unbuilt
    def build(self):
        """
        This method is called by the SensorManager during the scene build phase to initialize the sensor.
        This is where any shared metadata should be initialized.
        """
        delay_steps_float = self._options.read_delay / self._manager._sim.dt
        self._read_delay_steps = round(delay_steps_float)
        if not np.isclose(delay_steps_float, self._read_delay_steps):
            gs.logger.warn(
                f"Read delay should be a multiple of the simulation time step. Got {self._options.read_delay} and "
                f"{self._manager._sim.dt}. Actual read delay will be {1/self._read_delay_steps}."
            )

        return_format = self._get_return_format()
        return_length = 0
        if isinstance(return_format, dict):
            return_length = sum(sum(shape) for shape in return_format.values())
        elif isinstance(return_format, tuple):
            return_length = sum(return_format)
        else:
            raise ValueError(f"Invalid sensor return format: {return_format}, should be tuple or dict")

        self._cache_size = self._get_cache_length() * return_length
        self._shared_metadata = self._manager._sensors_metadata[self._class]

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
    def _update_shared_cache(
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor, shared_cache: "TensorRingBuffer"
    ):
        """
        Update the shared sensor cache for all sensors of this class using metadata in SensorManager.
        """
        raise NotImplementedError("Sensors must implement `update_shared_cache()`.")

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
        return self._get_return_data_from_tensor(self._cache.get(self._read_delay_steps), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx: List[int] | None = None):
        """
        Read the ground truth sensor data (without noise).
        """
        return self._get_return_data_from_tensor(self._ground_truth_cache, envs_idx)

    def _get_return_data_from_tensor(self, tensor: torch.Tensor, envs_idx: List[int] | None) -> torch.Tensor:
        if envs_idx is None:
            envs_idx = [0] if self._manager._sim.n_envs == 0 else np.arange(tensor.shape[0])

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
            raise ValueError(f"Invalid sensor return format: {return_format}")

    @property
    def is_built(self) -> bool:
        return self._manager._sim._scene._is_built
