from typing import Any, Type

import torch

from genesis.options.sensors import SensorOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_sensor import Sensor


class SensorManager:
    SENSOR_TYPES_MAP: dict[Type[SensorOptions], Type["Sensor"]] = {}
    SENSOR_CACHE_METADATA_MAP: dict[Type["Sensor"], (torch.dtype, tuple[int, ...])] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[Type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[Type["Sensor"], dict[str, Any]] = {}
        self._cache: dict[Type["Sensor"], torch.Tensor] = {}

    def create_sensor(self, sensor_options: SensorOptions):
        sensor_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        if sensor_cls not in self._sensors_by_type:
            self._sensors_by_type[sensor_cls] = []
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        for sensor_cls, sensors in self._sensors_by_type.items():
            total_cache_length = 0
            self._sensors_metadata[sensor_cls] = {}
            for sensor in sensors:
                sensor.build()
                sensor._cache_idx = total_cache_length
                total_cache_length += sensor._get_cache_length()

            self._cache[sensor_cls] = torch.zeros(
                (self._sim._B, total_cache_length, sensor_cls._get_cache_size()), dtype=sensor_cls._get_cache_dtype()
            )

    def step(self):
        for sensor_cls, sensors in self._sensors_by_type.items():
            sensors[0]._update_shared_cache()

    @property
    def sensors(self):
        return [sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list]


def register_sensor(sensor_cls: Type["Sensor"]):
    def _impl(options_cls: Type[SensorOptions]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls
        return options_cls

    return _impl
