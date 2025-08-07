from typing import Any, List, Optional

import taichi as ti
import torch

import genesis as gs
from genesis.options.sensors import SensorOptions
from genesis.repr_base import RBC

from .sensor_manager import SensorManager


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    A sensor must have a read() method that returns the sensor data.
    """

    # These class variables are used by SensorManager to determine the cache metadata for the sensor.
    # Sensor implementations should override these class variable values.
    CACHE_DTYPE: torch.dtype = torch.float32
    CACHE_SHAPE: tuple[int, ...] = (1,)

    def __init__(self, sensor_options: SensorOptions, sensor_idx: int, sensor_manager: SensorManager):
        self._options: SensorOptions = sensor_options
        self._idx: int = sensor_idx
        self._manager: SensorManager = sensor_manager
        self._cache_idx: int = -1  # cache_idx is set by the SensorManager during the scene build phase

    # =============================== implementable methods ===============================

    @gs.assert_unbuilt
    def build(self):
        """
        This method is called by the SensorManager during the scene build phase to initialize the sensor.
        """
        pass

    @gs.assert_built
    def read(self, envs_idx: Optional[List[int]] = None):
        """
        Read the sensor data.
        """
        raise NotImplementedError("Sensors must implement `read()`.")

    @property
    def cache_length(self) -> int:
        """
        The length (first dimension of cache size) of the cache for this sensor.
        """
        return 1

    # =============================== shared methods ===============================

    @property
    def is_built(self) -> bool:
        return self._manager._sim._scene._is_built

    @gs.assert_built
    def _get_cache(self) -> torch.Tensor:
        return self._manager.get_sensor_cache(self.__class__, self._cache_idx)

    @gs.assert_built
    def _is_cache_updated(self) -> bool:
        return self._manager.is_cache_updated(self.__class__)

    @gs.assert_built
    def _set_cache_updated(self):
        self._manager.set_cache_updated(self.__class__)

    @property
    def _cache(self) -> torch.Tensor:
        return self._manager._cache[self.__class__]

    @property
    def _shared_metadata(self) -> dict[str, Any]:
        return self._manager._sensors_metadata[self.__class__]
