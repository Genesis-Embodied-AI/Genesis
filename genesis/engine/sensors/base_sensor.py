import taichi as ti

from genesis.repr_base import RBC
from genesis.engine.entities.base_entity import Entity


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    """
    _compatible_entity_types = ()

    def __init__(self, entity: Entity):
        self.entity = entity
        assert isinstance(entity, self._compatible_entity_types), \
            f"{type(entity)} is not compatible with sensor {self._repr_type()}."

    def read(self):
        raise NotImplementedError()