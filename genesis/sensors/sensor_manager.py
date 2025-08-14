from typing import TYPE_CHECKING, Type

import torch

from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions

    from .base_sensor import Sensor, SharedSensorMetadata


class SensorManager:
    SENSOR_TYPES_MAP: dict[Type["SensorOptions"], tuple[Type["Sensor"], Type["SharedSensorMetadata"]]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[Type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[Type["Sensor"], SharedSensorMetadata | None] = {}
        self._ground_truth_cache: dict[Type[torch.dtype], torch.Tensor] = {}
        self._cache: dict[Type[torch.dtype], torch.Tensor] = {}
        self._buffered_data: dict[Type[torch.dtype], TensorRingBuffer] = {}
        self._cache_slices_by_type: dict[Type["Sensor"], slice] = {}

        self._last_cache_cloned_step: dict[tuple[bool, Type[torch.dtype]], int] = {}
        self._cloned_cache: dict[tuple[bool, Type[torch.dtype]], torch.Tensor] = {}

    def create_sensor(self, sensor_options: "SensorOptions"):
        sensor_options.validate(self._sim.scene)
        sensor_cls, metadata_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = metadata_cls()
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        max_buffer_len = 0
        cache_size_per_dtype = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()

            for is_ground_truth in [False, True]:
                self._last_cache_cloned_step.setdefault((is_ground_truth, dtype), -1)
                self._cloned_cache.setdefault((is_ground_truth, dtype), torch.zeros(0, dtype=dtype))

            cache_size_per_dtype.setdefault(dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[dtype]

            for sensor in sensors:
                sensor._cache_idx = cache_size_per_dtype[dtype]
                cache_size_per_dtype[dtype] += sensor._cache_size
                max_buffer_len = max(max_buffer_len, sensor._read_delay_steps + 1)

            cls_cache_end_idx = cache_size_per_dtype[dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)

        for dtype in cache_size_per_dtype.keys():
            cache_shape = (self._sim._B, cache_size_per_dtype[dtype])
            self._ground_truth_cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
            self._cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
            self._buffered_data[dtype] = TensorRingBuffer(max_buffer_len, cache_shape, dtype=dtype)

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()
            for sensor in sensors:
                sensor.build()

    def step(self):
        for sensor_cls in self._sensors_by_type.keys():
            dtype = sensor_cls._get_cache_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            sensor_cls._update_shared_ground_truth_cache(
                self._sensors_metadata[sensor_cls], self._ground_truth_cache[dtype][cache_slice]
            )
            sensor_cls._update_shared_cache(
                self._sensors_metadata[sensor_cls],
                self._ground_truth_cache[dtype][cache_slice],
                self._cache[dtype][cache_slice],
                self._buffered_data[dtype][cache_slice],
            )

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        dtype = sensor._get_cache_dtype()
        key = (is_ground_truth, dtype)
        if self._last_cache_cloned_step[key] != self._sim.cur_step_global:
            self._last_cache_cloned_step[key] = self._sim.cur_step_global
            if is_ground_truth:
                self._cloned_cache[key] = self._ground_truth_cache[dtype].clone()
            else:
                self._cloned_cache[key] = self._cache[dtype].clone()
        return self._cloned_cache[key][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]

    @property
    def sensors(self):
        return tuple([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])


def register_sensor(options_cls: Type["SensorOptions"], metadata_cls: Type["SharedSensorMetadata"]):
    def _impl(sensor_cls: Type["Sensor"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls, metadata_cls
        return sensor_cls

    return _impl
