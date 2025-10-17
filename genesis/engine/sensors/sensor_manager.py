from typing import TYPE_CHECKING, Type

import numpy as np
import torch

import genesis as gs
from genesis.utils.ring_buffer import TensorRingBuffer

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext

    from .base_sensor import Sensor, SensorOptions, SharedSensorMetadata


class SensorManager:
    SENSOR_TYPES_MAP: dict[Type["SensorOptions"], tuple[Type["Sensor"], Type["SharedSensorMetadata"], Type[tuple]]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[Type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[Type["Sensor"], SharedSensorMetadata | None] = {}
        self._ground_truth_cache: dict[Type[torch.dtype], torch.Tensor] = {}
        self._cache: dict[Type[torch.dtype], torch.Tensor] = {}
        self._buffered_data: dict[Type[torch.dtype], TensorRingBuffer] = {}
        self._cache_slices_by_type: dict[Type["Sensor"], slice] = {}
        self._should_update_cache_by_type: dict[Type["Sensor"], bool] = {}
        self._is_last_cache_cloned: dict[tuple[bool, Type[torch.dtype]], bool] = {}
        self._cloned_cache: dict[tuple[bool, Type[torch.dtype]], torch.Tensor] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        sensor_options.validate(self._sim.scene)
        sensor_cls, metadata_cls, data_cls = SensorManager.SENSOR_TYPES_MAP[type(sensor_options)]
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = metadata_cls()
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), data_cls, self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    def build(self):
        max_buffer_len = 0
        cache_size_per_dtype = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()

            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

            cache_size_per_dtype.setdefault(dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[dtype]

            update_ground_truth_only = True
            for sensor in sensors:
                update_ground_truth_only &= sensor._options.update_ground_truth_only
                sensor._cache_idx = cache_size_per_dtype[dtype]
                cache_size_per_dtype[dtype] += sensor._cache_size
                max_buffer_len = max(max_buffer_len, sensor._delay_ts + 1)
            self._should_update_cache_by_type[sensor_cls] = not update_ground_truth_only

            cls_cache_end_idx = cache_size_per_dtype[dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)

        for dtype in cache_size_per_dtype.keys():
            cache_shape = (self._sim._B, cache_size_per_dtype[dtype])
            self._ground_truth_cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            self._cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            self._buffered_data[dtype] = TensorRingBuffer(max_buffer_len, cache_shape, dtype=dtype)

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_cache_dtype()
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    def reset(self, envs_idx=None):
        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)

        for dtype in self._buffered_data.keys():
            self._ground_truth_cache[dtype][envs_idx] = 0.0
            self._cache[dtype][envs_idx] = 0.0
            self._buffered_data[dtype].buffer[:, envs_idx] = 0.0
            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

        for sensor_cls in self._sensors_by_type.keys():
            sensor_cls.reset(self._sensors_metadata[sensor_cls], envs_idx)

    def step(self):
        for sensor_cls in self._sensors_by_type.keys():
            dtype = sensor_cls._get_cache_dtype()
            cache_slice = self._cache_slices_by_type[sensor_cls]
            sensor_cls._update_shared_ground_truth_cache(
                self._sensors_metadata[sensor_cls], self._ground_truth_cache[dtype][:, cache_slice]
            )
            if self._should_update_cache_by_type[sensor_cls]:
                sensor_cls._update_shared_cache(
                    self._sensors_metadata[sensor_cls],
                    self._ground_truth_cache[dtype][:, cache_slice],
                    self._cache[dtype][:, cache_slice],
                    self._buffered_data[dtype][:, cache_slice],
                )
            for is_ground_truth in (False, True):
                key = (is_ground_truth, dtype)
                self._is_last_cache_cloned[key] = False
                self._cloned_cache[key] = torch.tensor([], dtype=dtype, device=gs.device)

    def draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context, buffer_updates)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        dtype = sensor._get_cache_dtype()
        key = (is_ground_truth, dtype)
        if not self._is_last_cache_cloned[key]:
            self._is_last_cache_cloned[key] = True
            if is_ground_truth:
                self._cloned_cache[key] = self._ground_truth_cache[dtype].clone()
            else:
                self._cloned_cache[key] = self._cache[dtype].clone()
        return self._cloned_cache[key][:, sensor._cache_idx : sensor._cache_idx + sensor._cache_size]

    @property
    def sensors(self):
        return tuple([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])


def register_sensor(
    options_cls: Type["SensorOptions"], metadata_cls: Type["SharedSensorMetadata"], data_cls: Type[tuple]
):
    def _impl(sensor_cls: Type["Sensor"]):
        SensorManager.SENSOR_TYPES_MAP[options_cls] = sensor_cls, metadata_cls, data_cls
        return sensor_cls

    return _impl
