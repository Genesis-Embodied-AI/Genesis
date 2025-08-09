from typing import Any, Type, TYPE_CHECKING

import torch

from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from .base_sensor import Sensor


class SensorManager:
    SENSOR_TYPES_MAP: dict[Type["SensorOptions"], Type["Sensor"]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[Type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[Type["Sensor"], dict[str, Any]] = {}
        self._gt_cache: dict[Type["Sensor"], torch.Tensor] = {}
        self._cache: dict[Type["Sensor"], TensorRingBuffer] = {}

    def create_sensor(self, sensor_options: "SensorOptions"):
        sensor_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        if sensor_cls not in self._sensors_by_type:
            self._sensors_by_type[sensor_cls] = []
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        for sensor_cls, sensors in self._sensors_by_type.items():
            total_cache_length = 0
            max_cache_buf_len = 0
            self._sensors_metadata[sensor_cls] = {}
            for sensor in sensors:
                sensor.build()
                sensor._cache_idx = total_cache_length
                total_cache_length += sensor._get_cache_length()

                max_cache_buf_len = max(max_cache_buf_len, sensor._cache_buffer_length)

            cache_shape = (self._sim._B, total_cache_length, sensor_cls._get_cache_size())
            cache_dtype = sensor_cls._get_cache_dtype()

            self._gt_cache[sensor_cls] = torch.zeros(cache_shape, dtype=cache_dtype)
            self._cache[sensor_cls] = TensorRingBuffer(max_cache_buf_len, cache_shape, dtype=cache_dtype)

    def step(self):
        for sensor_cls, sensors in self._sensors_by_type.items():
            sensors[0]._update_shared_gt_cache()
            sensors[0]._update_shared_cache()

    @property
    def sensors(self):
        return [sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list]


def register_sensor(sensor_cls: Type["Sensor"]):
    def _impl(options_cls: Type["SensorOptions"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls
        return options_cls

    return _impl
