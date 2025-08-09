from typing import Any, List, Optional, TYPE_CHECKING

import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.options.sensors import SensorOptions
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.options.sensors import SensorOptions
    from .sensor_manager import SensorManager


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    A sensor must have a read() method that returns the sensor data.
    """

    def __init__(self, sensor_options: SensorOptions, sensor_idx: int, sensor_manager: SensorManager):
        self._options: SensorOptions = sensor_options
        self._idx: int = sensor_idx
        self._manager: SensorManager = sensor_manager
        self._cache_idx: int = -1  # cache_idx is set by the SensorManager during the scene build phase
        self._read_delay_steps: int = 0
        self._cache_buffer_length: int = 0

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
        self._cache_buffer_length = self._read_delay_steps + 1

    def _get_return_format(self) -> dict[str, tuple[int, int]] | None:
        """
        Data format of the read() return value.
        dict: a dictionary with string keys and tensor values will be returned.
              The tuple (start_idx, end_idx) will be used to get a slice of the values from the cache.
        None: the entire tensor (B, cache_length, cache_shape) will be returned.
        """
        raise NotImplementedError("Sensors must implement `return_format()`.")

    @gs.assert_built
    def _update_shared_gt_cache(self):
        """
        Update the shared sensor ground truth cache for all sensors of this class.
        Shared information is stored in SensorManager
        """
        raise NotImplementedError("Sensors must implement `update_shared_gt_cache()`.")

    @gs.assert_built
    def _update_shared_cache(self):
        """
        Update the shared sensor cache for all sensors of this class using information stored in shared metadata.
        """
        raise NotImplementedError("Sensors must implement `update_shared_cache()`.")

    def _get_cache_length(self) -> int:
        """
        The length (first dimension of cache shape) of the cache for this sensor instance.
        The sum of cache_length for all sensors of this type determines the length of the cache.
        """
        raise NotImplementedError("Sensors must implement `cache_length()`.")

    @classmethod
    def _get_cache_size(cls) -> int:
        """
        The length (second dimension of cache shape) of the cache for this sensor type.
        """
        raise NotImplementedError("Sensors must implement `get_cache_size()`.")

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        """
        The dtype of the cache for this sensor type.
        """
        raise NotImplementedError("Sensors must implement `get_cache_dtype()`.")

    # =============================== shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx: Optional[List[int]] = None):
        """
        Read the sensor data (with noise applied if applicable) from SensorManager._cache.
        """
        if envs_idx is None:
            envs_idx = 0 if self._manager._sim.n_envs == 0 else np.arange(self._cache.shape[0])

        return_format = self._get_return_format()
        if return_format is None:
            return self._cache[envs_idx, self._cache_idx, :].squeeze()
        else:
            cache_length = self._get_cache_length()
            return {
                key: self._cache.get(self._read_delay_steps)[
                    envs_idx, self._cache_idx : self._cache_idx + cache_length, start_idx:end_idx
                ].squeeze()
                for key, (start_idx, end_idx) in return_format.items()
            }

    @gs.assert_built
    def read_ground_truth(self, envs_idx: Optional[List[int]] = None):
        """
        Read the ground truth sensor data (without noise) from SensorManager._gt_cache.
        """
        if envs_idx is None:
            envs_idx = 0 if self._manager._sim.n_envs == 0 else np.arange(self._gt_cache.shape[0])

        return_format = self._get_return_format()
        if return_format is None:
            return self._gt_cache[envs_idx, self._cache_idx, :].squeeze()
        else:
            cache_length = self._get_cache_length()
            return {
                key: self._gt_cache[
                    envs_idx, self._cache_idx : self._cache_idx + cache_length, start_idx:end_idx
                ].squeeze()
                for key, (start_idx, end_idx) in return_format.items()
            }

    @property
    def is_built(self) -> bool:
        return self._manager._sim._scene._is_built

    @property
    def _cache(self) -> "TensorRingBuffer":
        return self._manager._cache[self.__class__]

    @property
    def _gt_cache(self) -> torch.Tensor:
        return self._manager._gt_cache[self.__class__]

    @property
    def _shared_metadata(self) -> dict[str, Any]:
        return self._manager._sensors_metadata[self.__class__]
