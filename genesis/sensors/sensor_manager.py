from typing import Type

import torch

from genesis.options.sensors import SensorOptions

from .base_sensor import Sensor


class SensorManager:
    SENSOR_TYPES_MAP: dict[Type[SensorOptions], Type[Sensor]] = {}
    SENSOR_CACHE_METADATA_MAP: dict[Type[Sensor], (torch.dtype, tuple[int, ...])] = {}

    def __init__(self):
        self.sensors_by_type: dict[Type[Sensor], list[Sensor]] = {}
        self.cache: dict[Type[Sensor], torch.Tensor] = {}
        self.cache_size_map: dict[Type[Sensor], int] = {}

    def create_sensor(self, sensor_options: SensorOptions):
        sensor_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        sensor = sensor_cls(sensor_options, len(self.sensors_by_type[sensor_cls]), self)
        if sensor_cls not in self.sensors_by_type:
            self.sensors_by_type[sensor_cls] = []
        self.sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        for sensor_cls, sensors in self.sensors_by_type.items():
            total_cache_length = 0
            for sensor in sensors:
                sensor.build()
                sensor._cache_idx = total_cache_length
                total_cache_length += sensor.cache_length

            cache_dtype, cache_shape = SensorManager.SENSOR_CACHE_METADATA_MAP[sensor_cls]
            self.cache[sensor_cls] = torch.zeros((total_cache_length, *cache_shape), dtype=cache_dtype)

            for sensor in sensors:
                sensor.build()

    def get_sensor_cache(self, sensor_cls: Type[Sensor], sensor_idx: int | None = None) -> torch.Tensor:
        cache_size = SensorManager.SENSOR_CACHE_SIZE_MAP[sensor_cls]
        if sensor_idx is None:
            return self.cache[sensor_cls]
        return self.cache[sensor_cls][sensor_idx * cache_size : (sensor_idx + 1) * cache_size]

    def set_sensor_cache(self, new_values: torch.Tensor, sensor_cls: Type[Sensor], sensor_idx: int | None = None):
        cache_size = SensorManager.SENSOR_CACHE_SIZE_MAP[sensor_cls]
        if sensor_idx is None:
            self.cache[sensor_cls] = new_values
        else:
            self.cache[sensor_cls][sensor_idx * cache_size : (sensor_idx + 1) * cache_size] = new_values

    @property
    def sensors(self):
        return [sensor for sensor_list in self.sensors_by_type.values() for sensor in sensor_list]


def register_sensor(sensor_cls: Type[Sensor], cache_dtype: torch.dtype, cache_shape: tuple[int, ...]):
    def _impl(sensor_options: SensorOptions):
        SensorManager.SENSOR_TYPES_MAP[type(sensor_options)] = sensor_cls
        SensorManager.SENSOR_CACHE_METADATA_MAP[sensor_cls] = (cache_dtype, cache_shape)
        return sensor_options

    return _impl
