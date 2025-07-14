import taichi as ti
import genesis as gs

from typing import List, Optional
from genesis.repr_base import RBC
from genesis.engine.entities.base_entity import Entity
from .data_collector import DataCollector, DataOutType, DataStreamConfig


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.
    """

    @staticmethod
    def get_valid_entity_types():
        raise NotImplementedError()

    def __init__(self, entity: Entity):
        assert isinstance(
            entity, self.get_valid_entity_types()
        ), f"{type(self)} can only be added to entities of type {self.get_valid_entity_types()}, got {type(entity)}."
        self._entity = entity
        self._sim = entity._sim
        self._data_collector = None

    def build(self):
        """
        Initializes the sensor. Called during scene.build().
        """
        pass

    @gs.assert_built
    def read(self, envs_idx: Optional[List[int]] = None):
        """
        Read the sensor's internal buffer and return the latest data.
        """
        self._check_envs_idx(envs_idx)
        raise NotImplementedError("The Sensor subclass must implement `read()`.")

    @gs.assert_built
    def step(self):
        """
        This is called by the Simulator during after physics steps.
        Generally, sensor state should only be updated during `read()`.
        """
        if self._data_collector:
            self._data_collector.step(self._sim.cur_step_global)

    def _check_envs_idx(self, envs_idx: Optional[List[int]]):
        if self.n_envs == 0 and envs_idx is not None:
            gs.logger.warning("envs_idx is ignored when n_envs=0, as there is only one environment.")

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_envs(self):
        return self._sim.n_envs

    @property
    def _B(self):
        return self._sim._B

    @property
    def entity(self):
        return self._entity

    @property
    def is_built(self):
        return self._entity.is_built

    @property
    def is_recording(self):
        return self._data_collector is not None and self._data_collector.is_recording

    # ------------------------------------------------------------------------------------
    # --------------------------------- data collection ----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def start_recording(self, filename):
        """
        Start recording data from the sensor.
        Default to CSV data handler.
        """
        config = DataStreamConfig(
            out_type=DataOutType.CSV,
            handler_kwargs=dict(filename=filename),
        )
        self._data_collector = DataCollector(self, config)
        self._data_collector.start_recording()

    @gs.assert_built
    def pause_recording(self):
        """
        Pause data recording from the sensor.
        """
        if self._data_collector:
            self._data_collector.pause_recording()
        else:
            gs.logger.warning("Sensor: start_recording() should have been called before pause_recording().")

    @gs.assert_built
    def stop_recording(self):
        """
        Stop data recording from the sensor.
        """
        if self._data_collector:
            self._data_collector.stop_recording()
            self._data_collector = None
        else:
            gs.logger.warning("Sensor: start_recording() should have been called before stop_recording().")
