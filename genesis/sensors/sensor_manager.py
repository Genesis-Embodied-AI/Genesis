from typing import TYPE_CHECKING, Any, Type

import numpy as np
import torch

import genesis as gs
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
        self._cache_slices_by_type: dict[Type["Sensor"], slice] = {}

        self._last_ground_truth_cache_cloned_step: dict[Type[torch.dtype], int] = {}
        self._cloned_ground_truth_cache: dict[Type[torch.dtype], torch.Tensor] = {}

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
            dtype = sensor_cls._get_cache_dtype()
            cache_size_per_dtype.setdefault(dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[dtype]

            for sensor in sensors:
                return_format = sensor._get_return_format()
                return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)

                tensor_size = 0
                for shape in return_shapes:
                    data_size = np.prod(shape)
                    sensor._shape_indices.append((tensor_size, tensor_size + data_size))
                    tensor_size += data_size

                delay_steps_float = sensor._options.read_delay / self._sim.dt
                sensor._read_delay_steps = round(delay_steps_float)
                if not np.isclose(delay_steps_float, sensor._read_delay_steps, atol=1e-6):
                    gs.logger.warn(
                        f"Read delay should be a multiple of the simulation time step. Got {sensor._options.read_delay}"
                        f" and {self._sim.dt}. Actual read delay will be {1/sensor._read_delay_steps}."
                    )

                sensor._cache_size = sensor._get_cache_length() * tensor_size
                sensor._cache_idx = cache_size_per_dtype[dtype]
                cache_size_per_dtype[dtype] += sensor._cache_size

                max_cache_buf_len = max(max_cache_buf_len, sensor._read_delay_steps + 1)

            cls_cache_end_idx = cache_size_per_dtype[dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)

        for dtype in cache_size_per_dtype.keys():
            cache_shape = (self._sim._B, cache_size_per_dtype[dtype])
            self._ground_truth_cache[dtype] = torch.zeros(cache_shape, dtype=dtype)
            self._cache[dtype] = TensorRingBuffer(max_cache_buf_len, cache_shape, dtype=dtype)

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()
            for sensor in sensors:
                sensor._shared_metadata = self._sensors_metadata[sensor_cls]
                sensor._cache = self._cache[dtype][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]
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
            )

    def get_cloned_from_ground_truth_cache(self, sensor: "Sensor") -> torch.Tensor:
        dtype = sensor._get_cache_dtype()
        if self._last_ground_truth_cache_cloned_step[dtype] != self._sim.cur_step_global:
            self._last_ground_truth_cache_cloned_step[dtype] = self._sim.cur_step_global
            self._cloned_ground_truth_cache[dtype] = self._ground_truth_cache[dtype].clone()
        return self._cloned_ground_truth_cache[dtype][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]

    @property
    def sensors(self):
        return tuple([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])


def register_sensor(sensor_cls: Type["Sensor"]):
    def _impl(options_cls: Type["SensorOptions"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls
        return options_cls

    return _impl
