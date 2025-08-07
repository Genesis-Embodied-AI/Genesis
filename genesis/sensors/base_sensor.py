from typing import List, Optional

import taichi as ti

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

    def __init__(self, sensor_options: SensorOptions, sensor_idx: int, sensor_manager: SensorManager):
        self._options: SensorOptions = sensor_options
        self._idx: int = sensor_idx
        self._manager: SensorManager = sensor_manager
        self._cache_idx: int = -1  # cache_idx is set by the SensorManager during the scene build phase

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
        Sensor implementations should make use of the caching system located in SensorManager when possible.
        """
        raise NotImplementedError("Sensors must implement `read()`.")

    @property
    def cache_length(self) -> int:
        """
        The length (first dimension of cache size) of the cache for this sensor.
        """
        raise NotImplementedError("Sensors must implement `cache_length()`.")
