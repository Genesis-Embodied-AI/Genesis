import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, List

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.solvers import RigidSolver
from genesis.options import Options
from genesis.repr_base import RBC
from genesis.utils.geom import euler_to_quat

from .data_recorder import DataRecorder

if TYPE_CHECKING:
    from genesis.options.recording import RecordingOptions
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


class SensorOptions(Options):
    """
    Base class for all sensor options.
    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.

    Parameters
    ----------
    delay : float
        The read delay time in seconds. Data read will be outdated by this amount.
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth data, and not the measured data.
    """

    delay: float = 0.0
    update_ground_truth_only: bool = False

    def validate(self, scene):
        """
        Validate the sensor options values before the sensor is added to the scene.
        """
        delay_hz = self.delay / scene._sim.dt
        if not np.isclose(delay_hz, round(delay_hz), atol=1e-6):
            gs.logger.warn(
                f"{type(self).__name__}: Read delay should be a multiple of the simulation time step. Got {self.delay}"
                f" and {scene._sim.dt}. Actual read delay will be {1 / round(delay_hz)}."
            )


@dataclass
class SharedSensorMetadata:
    """
    Shared metadata between all sensors of the same class.
    """

    cache_sizes: list[int] = field(default_factory=list)
    delay_steps: list[int] = field(default_factory=list)


@ti.data_oriented
class Sensor(RBC):
    """
    Base class for all types of sensors.

    Use scene.add_sensor(sensor_options) to instantiate a sensor.

    NOTE: The Sensor system is designed to be performant.  All sensors of the same type are updated at once and stored
    in a cache in SensorManager. Cache size is inferred from the return format and cache length of each sensor.
    `read()` and `read_ground_truth()`, the public-facing methods of every Sensor, automatically handles indexing into
    the shared cache to return the correct data.
    """

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadata = sensor_manager._sensors_metadata[type(self)]
        self._data_recorder: DataRecorder | None = None

        self._delay_steps = round(self._options.delay / self._manager._sim.dt)
        self._shared_metadata.delay_steps.append(self._delay_steps)

        self._shape_indices: list[tuple[int, int]] = []
        return_format = self._get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        tensor_size = 0
        for shape in return_shapes:
            data_size = np.prod(shape)
            self._shape_indices.append((tensor_size, tensor_size + data_size))
            tensor_size += data_size

        self._cache_size = self._get_cache_length() * tensor_size
        self._shared_metadata.cache_sizes.append(self._cache_size)

        self._cache_idx: int = -1  # initialized by SensorManager during build

    # =============================== methods to implement ===============================

    def build(self):
        """
        This method is called by the SensorManager during the scene build phase to initialize the sensor.
        This is where any shared metadata should be initialized.
        """
        pass

    def _get_return_format(self) -> dict[str, tuple[int, ...]] | tuple[int, ...]:
        """
        Data format of the read() return value.

        Returns
        -------
        return_format : dict | tuple
            - If tuple, the final shape of the read() return value.
                e.g. (2, 3) means read() will return a tensor of shape (2, 3).
            - If dict a dictionary with string keys and tensor values will be returned.
                e.g. {"pos": (3,), "quat": (4,)} returns a dict of tensors [0:3] and [3:7] from the cache.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `get_return_format()`.")

    def _get_cache_length(self) -> int:
        """
        The length of the cache for this sensor instance, e.g. number of points for a Lidar point cloud.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `get_cache_length()`.")

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the shared sensor ground truth cache for all sensors of this class using metadata in SensorManager.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `update_shared_ground_truth_cache()`.")

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the shared sensor cache for all sensors of this class using metadata in SensorManager.

        The information in shared_cache should be the final measured sensor data after all noise and post-processing.
        NOTE: The implementation should include applying the delay using the `_apply_delay_to_shared_cache()` method.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `update_shared_cache()`.")

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        """
        The dtype of the cache for this sensor.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `get_cache_dtype()`.")

    # =============================== public shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx: List[int] | None = None):
        """
        Read the sensor data (with noise applied if applicable).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx: List[int] | None = None):
        """
        Read the ground truth sensor data (without noise).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self, is_ground_truth=True), envs_idx)

    @gs.assert_unbuilt
    def add_recording(self, recording_options: "RecordingOptions", read_ground_truth: bool = False):
        """
        Automatically process data from this sensor.

        When `recording_options.data_func` is not specified, the sensor's `read()` method is used.
        """
        if recording_options.data_func is None:
            recording_options.data_func = self.read_ground_truth if read_ground_truth else self.read
        self._manager._sim._data_recorder.add_recording(recording_options)

    @property
    def is_built(self) -> bool:
        return self._manager._sim._scene._is_built

    # =============================== private shared methods ===============================

    @classmethod
    def _apply_delay_to_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadata,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
        jitter_in_steps: torch.Tensor | None = None,
        interpolate: list[bool] | None = None,
    ):
        """
        Applies the read delay to the shared cache tensor by copying the buffered data at the appropriate index.

        This is a helper method for `update_shared_cache()` to apply the delay to the shared cache tensor.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor.
        shared_cache : torch.Tensor
            The shared cache tensor.
        buffered_data : TensorRingBuffer
            The buffered data tensor.
        jitter_in_steps : torch.Tensor | None
            The jitter in steps before the sensor data is read.
        interpolate : list[bool] | None
            Whether to interpolate the sensor data for the read delay + jitter.
        """
        tensor_idx = 0

        for sensor_idx, (tensor_size, delay_step, interpolate) in enumerate(
            zip(
                shared_metadata.cache_sizes,
                shared_metadata.delay_steps,
                interpolate or itertools.repeat(False),
            )
        ):
            actual_delay_in_steps = torch.full_like(shared_cache[:, sensor_idx], delay_step, dtype=gs.tc_float)
            if jitter_in_steps is not None:
                actual_delay_in_steps = actual_delay_in_steps + jitter_in_steps[:, sensor_idx]
            actual_delay_in_steps_int = actual_delay_in_steps.floor().int()
            if interpolate:
                lerp_weight = actual_delay_in_steps - actual_delay_in_steps_int
                shared_cache[:, tensor_idx : tensor_idx + tensor_size] = torch.lerp(
                    buffered_data.at(actual_delay_in_steps_int)[:, tensor_idx : tensor_idx + tensor_size],
                    buffered_data.at(actual_delay_in_steps_int + 1)[:, tensor_idx : tensor_idx + tensor_size],
                    lerp_weight.unsqueeze(-1).expand(-1, tensor_size),
                )
            else:
                shared_cache[:, tensor_idx : tensor_idx + tensor_size] = buffered_data.at(actual_delay_in_steps_int)[
                    :, tensor_idx : tensor_idx + tensor_size
                ]
            tensor_idx += tensor_size

    def _get_formatted_data(
        self, tensor: torch.Tensor, envs_idx: list[int] | None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Formats the flattened cache tensor into a dict of tensors using the format specified in `get_return_format()`.

        NOTE: This method does not clone the data tensor, it should have been cloned by the caller.
        """

        envs_idx = self._sanitize_envs_idx(envs_idx)

        return_format = self._get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        cache_length = self._get_cache_length()
        return_values = []

        if cache_length == 1:
            work_tensor = tensor[envs_idx]
        else:
            total_data_per_item = sum(np.prod(shape) for shape in return_shapes)
            work_tensor = tensor[envs_idx].reshape(len(envs_idx), cache_length, total_data_per_item)

        for i, shape in enumerate(return_shapes):
            start_idx, end_idx = self._shape_indices[i]

            if cache_length == 1:
                field_data = work_tensor[:, start_idx:end_idx]
                final_shape = (len(envs_idx), *shape)
            else:
                field_data = work_tensor[:, :, start_idx:end_idx]
                final_shape = (len(envs_idx), cache_length, *shape)

            value = field_data.reshape(final_shape).squeeze()
            if self._manager._sim.n_envs == 0:
                value = value.squeeze(0)
            return_values.append(value)

        if isinstance(return_format, dict):
            return dict(zip(return_format.keys(), return_values))
        else:
            return return_values[0]

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor:
        return self._manager._sim._scene._sanitize_envs_idx(envs_idx)


class RigidSensorOptionsMixin:
    """
    Base options class for sensors that are attached to a RigidEntity.

        Parameters
        ----------
        entity_idx : int
            The global entity index of the RigidEntity to which this sensor is attached.
        link_idx_local : int, optional
            The local index of the RigidLink of the RigidEntity to which this sensor is attached.
        pos_offset : tuple[float, float, float]
            The positional offset of the sensor from the RigidLink.
        euler_offset : tuple[float, float, float]
            The rotational offset of the sensor from the RigidLink in degrees.
    =======
        Utility base class for sensors that are attached to a RigidEntity.
    >>>>>>> a99bdb0 (add lidar wip)
    """

    entity_idx: int
    link_idx_local: int = 0
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self, scene):
        super().validate(scene)
        if self.entity_idx < 0 or self.entity_idx >= len(scene.entities):
            gs.raise_exception(f"Invalid RigidEntity index {self.entity_idx}.")
        entity = scene.entities[self.entity_idx]
        if not isinstance(entity, RigidEntity):
            gs.raise_exception(f"Entity at index {self.entity_idx} is not a RigidEntity.")
        if self.link_idx_local < 0 or self.link_idx_local >= entity.n_links:
            gs.raise_exception(f"Invalid RigidLink index {self.link_idx_local} for entity {self.entity_idx}.")


@dataclass
class RigidSensorMetadataMixin:
    """
    Base shared metadata class for sensors that are attached to a RigidEntity.
    """

    solver: RigidSolver | None = None
    links_idx: list[int] = field(default_factory=list)
    offsets_pos: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    offsets_quat: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))


class RigidSensorMixin:
    """
    Base sensor class for sensors that are attached to a RigidEntity.
    """

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.links_idx.append(self._options.link_idx_local + self._options.entity_idx)
        self._shared_metadata.offsets_pos = torch.cat(
            [
                self._shared_metadata.offsets_pos,
                torch.tensor([self._options.pos_offset], dtype=gs.tc_float, device=gs.device),
            ]
        )
        quat_tensor = torch.tensor(euler_to_quat([self._options.euler_offset]), dtype=gs.tc_float, device=gs.device)
        if self._shared_metadata.solver.n_envs > 0:
            quat_tensor = quat_tensor.unsqueeze(0).expand((self._manager._sim._B, 1, 4))
        self._shared_metadata.offsets_quat = torch.cat([self._shared_metadata.offsets_quat, quat_tensor], dim=-2)


class NoisySensorOptionsMixin:
    """
    Base options class for analog sensors that are attached to a RigidEntity.

    Parameters
    ----------
    resolution : float | tuple[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is None, which means no quantization is applied.
    bias : float | tuple[float, ...], optional
        The bias of the sensor.
    random_walk_std : float | tuple[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    noise_std : float | tuple[float, ...], optional
        The standard deviation of the noise.
    delay : float, optional
        The delay in seconds before the sensor data is read.
    jitter : float, optional
        The jitter in seconds modeled as a normal distribution. Like delay, this will affect how outdated the sensor
        data is when it is read. Jitter should be less than delay.
    interpolate : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth data, and not the measured data.
    """

    resolution: float | tuple[float, ...] | None = None
    bias: float | tuple[float, ...] = 0.0
    random_walk_std: float | tuple[float, ...] = 0.0
    noise_std: float | tuple[float, ...] = 0.0
    jitter: float = 0.0
    interpolate: bool = False

    def validate(self, scene):
        super().validate(scene)
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")


@dataclass
class NoisySensorMetadataMixin:
    """
    Base shared metadata class for analog sensors that are attached to a RigidEntity.
    """

    resolution: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    bias: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    cur_random_walk: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    random_walk_std: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    cur_noise: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    noise_std: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    jitter_std_in_steps: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device)
    )
    cur_jitter_in_steps: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device)
    )
    delay_in_steps: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))
    interpolate: list[bool] = field(default_factory=list)


class NoisySensorMixin:
    """
    Base sensor class for analog sensors that are attached to a RigidEntity.
    """

    def _set_metadata_field(self, input, field, field_size, envs_idx):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        idx = self._idx * field_size
        field[envs_idx, idx : idx + field_size] = self._sanitize_for_metadata_tensor(
            input, shape=(len(envs_idx), field_size), dtype=field.dtype
        )

    def _sanitize_for_metadata_tensor(self, input, shape, dtype) -> torch.Tensor:
        if not isinstance(input, Iterable):
            input = [input]
        tensor_input = torch.tensor(input, dtype=dtype, device=gs.device)
        if tensor_input.ndim < len(shape):
            tensor_input = tensor_input.unsqueeze(0)  # add batch dim
        if tensor_input.shape[0] == 1 and shape[0] > 1:
            tensor_input = tensor_input.expand(shape[0], *tensor_input.shape[1:])  # repeat batch dim
        assert (
            tensor_input.shape == shape
        ), f"Input shape {tensor_input.shape} for setting sensor metadata does not match shape {shape}"
        return tensor_input

    @gs.assert_built
    def set_resolution(self, resolution, envs_idx=None):
        self._set_metadata_field(resolution, self._shared_metadata.resolution, self._cache_size, envs_idx)

    @gs.assert_built
    def set_bias(self, bias, envs_idx=None):
        self._set_metadata_field(bias, self._shared_metadata.bias, self._cache_size, envs_idx)

    @gs.assert_built
    def set_random_walk_std(self, random_walk_std, envs_idx=None):
        self._set_metadata_field(random_walk_std, self._shared_metadata.random_walk_std, self._cache_size, envs_idx)

    @gs.assert_built
    def set_noise_std(self, noise_std, envs_idx=None):
        self._set_metadata_field(noise_std, self._shared_metadata.noise_std, self._cache_size, envs_idx)

    @gs.assert_built
    def set_jitter(self, jitter, envs_idx=None):
        jitter_in_steps = np.asarray(jitter, dtype=gs.np_float) / self._manager._sim.dt
        self._set_metadata_field(
            jitter_in_steps, self._shared_metadata.jitter_std_in_steps, field_size=1, envs_idx=envs_idx
        )

    @gs.assert_built
    def set_delay(self, delay, envs_idx=None):
        self._set_metadata_field(delay, self._shared_metadata.delay_in_steps, field_size=1, envs_idx=envs_idx)

    def build(self):
        """
        Initialize all shared metadata needed to update all noisy sensors.
        """
        super().build()

        batch_size = self._manager._sim._B

        if isinstance(self._options.resolution, tuple):
            self._options.resolution = tuple([-1 if r is None else r for r in self._options.resolution])
        self._shared_metadata.resolution = torch.cat(
            [
                self._shared_metadata.resolution,
                torch.tensor([self._options.resolution or -1], dtype=gs.tc_float, device=gs.device).expand(
                    batch_size, -1
                ),
            ],
            dim=-1,
        )
        self._shared_metadata.bias = torch.cat(
            [
                self._shared_metadata.bias,
                torch.tensor([self._options.bias], dtype=gs.tc_float, device=gs.device).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.random_walk_std = torch.cat(
            [
                self._shared_metadata.random_walk_std,
                torch.tensor([self._options.random_walk_std], dtype=gs.tc_float, device=gs.device).expand(
                    batch_size, -1
                ),
            ],
            dim=-1,
        )
        self._shared_metadata.cur_random_walk = torch.zeros_like(self._shared_metadata.random_walk_std)
        self._shared_metadata.noise_std = torch.cat(
            [
                self._shared_metadata.noise_std,
                torch.tensor([self._options.noise_std], dtype=gs.tc_float, device=gs.device).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.cur_noise = torch.zeros_like(self._shared_metadata.noise_std)
        self._shared_metadata.jitter_std_in_steps = torch.cat(
            [
                self._shared_metadata.jitter_std_in_steps,
                torch.tensor(
                    [self._options.jitter / self._manager._sim.dt], dtype=gs.tc_float, device=gs.device
                ).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.cur_jitter_in_steps = torch.zeros_like(
            self._shared_metadata.jitter_std_in_steps, device=gs.device
        )
        self._shared_metadata.interpolate.append(self._options.interpolate)

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: NoisySensorMetadataMixin,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the shared sensor ground truth cache for all sensors of this class using metadata in SensorManager.

        Note
        ----
        `buffered_data` contains the history of ground truth cache, and noise/bias is only applied to the current
        sensor readout `shared_cache`, not the whole buffer.
        """
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0, shared_metadata.jitter_std_in_steps, out=shared_metadata.cur_jitter_in_steps)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.cur_jitter_in_steps,
            shared_metadata.interpolate,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    @classmethod
    def _add_noise_drift_bias(cls, shared_metadata: NoisySensorMetadataMixin, output: torch.Tensor):
        shared_metadata.cur_random_walk += torch.normal(0, shared_metadata.random_walk_std)
        torch.normal(0, shared_metadata.noise_std, out=shared_metadata.cur_noise)
        output += shared_metadata.bias + shared_metadata.cur_noise + shared_metadata.cur_random_walk

    @classmethod
    def _quantize_to_resolution(cls, resolution: torch.Tensor, output: torch.Tensor):
        mask = resolution > 0
        output[mask] = torch.round(output[mask] / resolution[mask]) * resolution[mask]

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
