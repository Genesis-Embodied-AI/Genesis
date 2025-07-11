import taichi as ti
import genesis as gs

from genesis.repr_base import RBC
from genesis.engine.entities.base_entity import Entity
from .data_collector import DataCollector, OutputMode


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    """
    @staticmethod
    def get_valid_entity_types():
        raise NotImplementedError()

    def __init__(self, entity: Entity):
        assert isinstance(entity, self.get_valid_entity_types()), \
            f"{type(self)} can only be added to entities of type {self.get_valid_entity_types()}, got {type(entity)}."
        self._entity = entity
        self._sim = entity._sim
        self._data_collector: DataCollector = None
        self._is_built = False
    
    @gs.assert_unbuilt
    def build(self):
        """
        Initializes the sensor. Called during scene.build().
        """
        pass

    @gs.assert_built
    def read(self, envs_idx=None):
        """
        Read the sensor's internal buffer and return the latest data.
        """
        raise NotImplementedError()
    
    @gs.assert_built
    def step(self):
        """
        Step the sensor to update its internal state.
        This is called during the simulation step.
        """
        if self._data_collector:
            self._data_collector.step(self._sim.cur_step_global)
    
    @property
    def n_envs(self):
        return self._sim.n_envs

    @property
    def entity(self):
        return self._entity

    @property
    def is_built(self):
        return self._entity.is_built


    # ------------------------------------------------------------------------------------
    # --------------------------------- data collection ----------------------------------
    # ------------------------------------------------------------------------------------
    
    @gs.assert_built
    def start_recording(self, filename, hz=None):
        """
        Start recording data from the sensor.
        """
        self._data_collector = DataCollector(self,
                                mode=OutputMode.CSV,
                                hz=hz, filename=filename
                              )
        self._data_collector.start_recording()

    @gs.assert_built
    def pause_recording(self):
        """
        Pause data recording from the sensor.
        """
        self._data_collector.pause_recording()

    @gs.assert_built
    def stop_recording(self):
        """
        Stop data recording from the sensor.
        """
        self._data_collector.stop_recording()