from typing import TYPE_CHECKING, Any, Type

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
        self._ground_truth_cache: dict[Type[torch.dtype], torch.Tensor] = {}
        self._cache: dict[Type[torch.dtype], TensorRingBuffer] = {}

    def create_sensor(self, sensor_options: "SensorOptions"):
        sensor_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        self._sensors_by_type.setdefault(sensor_cls, [])
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        max_cache_buf_len = 0
        cache_size_per_dtype = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            self._sensors_metadata[sensor_cls] = {}

            for sensor in sensors:
                sensor.build()

                cache_size_per_dtype.setdefault(sensor._dtype, 0)

                sensor._cache_start_idx = cache_size_per_dtype[sensor._dtype]
                sensor._cache_end_idx = sensor._cache_start_idx + sensor._cache_size
                cache_size_per_dtype[sensor._dtype] += sensor._cache_size

                max_cache_buf_len = max(max_cache_buf_len, sensor._read_delay_steps + 1)

            for dtype in cache_size_per_dtype.keys():
                cache_shape = (self._sim._B, cache_size_per_dtype[dtype])
                self._ground_truth_cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
                self._cache[dtype] = TensorRingBuffer(max_cache_buf_len, cache_shape, dtype=dtype)

    def step(self):
        for sensor_cls, sensors in self._sensors_by_type.items():
            sensors[0]._update_shared_ground_truth_cache()
            sensors[0]._update_shared_cache()

    @property
    def sensors(self):
        return tuple([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])


def register_sensor(sensor_cls: Type["Sensor"]):
    def _impl(options_cls: Type["SensorOptions"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls
        return options_cls

    return _impl
