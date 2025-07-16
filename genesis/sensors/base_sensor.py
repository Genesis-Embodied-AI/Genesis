import taichi as ti
import genesis as gs

from typing import List, Optional
from genesis.repr_base import RBC


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    A sensor must have a read() method that returns the sensor data.
    """

    @gs.assert_built
    def read(self, envs_idx: Optional[List[int]] = None):
        """
        Read the sensor data.
        Sensor implementations should ideally cache the data to avoid unnecessary computations.
        """
        raise NotImplementedError("The Sensor subclass must implement `read()`.")
