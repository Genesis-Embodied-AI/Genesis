from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Generic, Sequence, Type, TypeVar

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field

if TYPE_CHECKING:
    from genesis.engine.solvers import RigidSolver
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.recorders.base_recorder import Recorder, RecorderOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


NumericType = int | float | bool
NumericSequenceType = NumericType | Sequence[NumericType]


def _to_tuple(*values: NumericType | torch.Tensor, length_per_value: int = 3) -> tuple[NumericType, ...]:
    """
    Convert all input values to one flattened tuple, where each value is ensured to be a tuple of length_per_value.
    """
    full_tuple = ()
    for value in values:
        if isinstance(value, (int, float)):
            value = (value,) * length_per_value
        elif isinstance(value, torch.Tensor):
            value = value.reshape((-1,))
        full_tuple += tuple(value)
    return full_tuple


# Note: dataclass is used as opposed to pydantic.BaseModel since torch.Tensors are not supported by default
@dataclass
class SharedSensorMetadata:
    """
    Shared metadata between all sensors of the same class.
    """

    cache_sizes: list[int] = field(default_factory=list)
    delays_ts: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)


SharedSensorMetadataT = TypeVar("SharedSensorMetadataT", bound=SharedSensorMetadata)


@ti.data_oriented
class Sensor(RBC, Generic[SharedSensorMetadataT]):
    """
    Base class for all types of sensors.

    To create a sensor, prefer using `scene.add_sensor(sensor_options)` instead of instantiating this class directly.

    Note
    -----
    The Sensor system is designed to be performant. All sensors of the same type are updated at once and stored
    in a cache in SensorManager. Cache size is inferred from the return format and cache length of each sensor.
    `read()` and `read_ground_truth()`, the public-facing methods of every Sensor, automatically handles indexing into
    the shared cache to return the correct data.
    """

    def __init__(
        self, sensor_options: "SensorOptions", sensor_idx: int, data_cls: Type[tuple], sensor_manager: "SensorManager"
    ):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadataT = sensor_manager._sensors_metadata[type(self)]
        self._is_built = False

        self._dt = self._manager._sim.dt
        self._delay_ts = round(self._options.delay / self._dt)

        self._cache_slices: list[slice] = []
        self._return_data_class = data_cls
        return_format = self._get_return_format()
        assert len(return_format) > 0
        if isinstance(return_format[0], int):
            return_format = (return_format,)
        self._return_shapes: tuple[tuple[int, ...], ...] = return_format

        self._cache_size = 0
        for shape in self._return_shapes:
            data_size = np.prod(shape)
            self._cache_slices.append(slice(self._cache_size, self._cache_size + data_size))
            self._cache_size += data_size

        self._cache_idx: int = -1  # initialized by SensorManager during build

    # =============================== methods to implement ===============================

    def build(self):
        """
        Build the sensor.

        This method is called by SensorManager during the scene build phase.
        This is where any shared metadata should be initialized.
        """
        self._shared_metadata.delays_ts = concat_with_tensor(
            self._shared_metadata.delays_ts,
            self._delay_ts,
            expand=(self._manager._sim._B, 1),
            dim=1,
        )
        self._shared_metadata.cache_sizes.append(self._cache_size)

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadataT, envs_idx):
        """
        Reset the sensor.

        This method is called by SensorManager when the scene is reset by `scene.reset()`.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor class.
        envs_idx: array_like
            The indices of the environments to reset. The envs_idx should already be sanitized by SensorManager.
        """
        pass

    def _get_return_format(self) -> tuple[int | tuple[int, ...], ...]:
        """
        Get the data format of the read() return value.

        Returns
        -------
        return_format : tuple[tuple[int, ...], ...]
            The output shape(s) of the tensor data returned by read(), e.g. (2, 3) means read() will return a single
            tensor of shape (2, 3) and ((3,), (3,)) would return two tensors of shape (3,).
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `get_return_format()`.")

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: SharedSensorMetadataT, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Update the shared sensor ground truth cache for all sensors of this class using metadata in SensorManager.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `update_shared_ground_truth_cache()`.")

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadataT,
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

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug shapes for the sensor in the scene.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `draw_debug()`.")

    # =============================== public shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx=None):
        """
        Read the sensor data (with noise applied if applicable).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None):
        """
        Read the ground truth sensor data (without noise).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self, is_ground_truth=True), envs_idx)

    @gs.assert_unbuilt
    def start_recording(self, rec_options: "RecorderOptions") -> "Recorder":
        """
        Automatically read and process sensor data. See RecorderOptions for more details.

        Data from `sensor.read()` is used. If the sensor data needs to be preprocessed before passing to the recorder,
        consider using `scene.start_recording()` instead with a custom data function.

        Parameters
        ----------
        rec_options : RecorderOptions
            The options for the recording.
        """
        return self._manager._sim._scene._recorder_manager.add_recorder(self.read, rec_options)

    @property
    def is_built(self) -> bool:
        return self._is_built

    # =============================== private shared methods ===============================

    @classmethod
    def _apply_delay_to_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadataT,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
        cur_jitter_ts: torch.Tensor | None = None,
        interpolate: Sequence[bool] | None = None,
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
        cur_jitter_ts : torch.Tensor | None
            The current jitter in timesteps (divided by simulation dt) before the sensor data is read.
        interpolate : Sequence[bool] | None
            Whether to interpolate the sensor data for the read delay + jitter. Defaults to False.
        """
        if interpolate is None:
            interpolate = [False for _ in shared_metadata.cache_sizes]

        tensor_start = 0
        for sensor_idx, (tensor_size, interp) in enumerate(zip(shared_metadata.cache_sizes, interpolate)):
            # Compute the current delay of the sensor, taking into account jitter if any
            cur_delay_ts = shared_metadata.delays_ts[:, sensor_idx]
            if cur_jitter_ts is not None:
                cur_delay_ts = cur_delay_ts + cur_jitter_ts[:, sensor_idx]

            # Get int for indexing into ring buffer (0 = most recent, 1 = delayed by one timestep, etc.)
            cur_delay_ts_int = cur_delay_ts.to(dtype=torch.int64)

            # Update shared cached with left data (Zero Order Hold) or linearly interpolated data (First Order)
            envs_idx = torch.arange(len(cur_delay_ts), device=gs.device)
            tensor_slice = slice(tensor_start, tensor_start + tensor_size)
            sensor_cache = shared_cache[:, tensor_slice]
            data_left = buffered_data.at(cur_delay_ts_int, envs_idx, tensor_slice)
            if interp:
                ratio = torch.frac(cur_delay_ts)
                data_right = buffered_data.at(cur_delay_ts_int + 1, envs_idx, tensor_slice)
                torch.lerp(data_left, data_right, ratio.unsqueeze(1), out=sensor_cache)
            else:
                sensor_cache.copy_(data_left)

            tensor_start += tensor_size

    def _get_formatted_data(self, tensor: torch.Tensor, envs_idx=None) -> torch.Tensor:
        """
        Returns tensor(s) matching the return format.

        Note that this method does not clone the data tensor, it should have been cloned by the caller.
        """
        envs_idx = self._sanitize_envs_idx(envs_idx)

        return_values = []
        tensor_chunk = tensor[envs_idx].reshape((len(envs_idx), -1))

        for i, shape in enumerate(self._return_shapes):
            field_data = tensor_chunk[..., self._cache_slices[i]].reshape((len(envs_idx), *shape))
            if self._manager._sim.n_envs == 0:
                field_data = field_data.squeeze(0)
            return_values.append(field_data)

        if len(return_values) == 1:
            return return_values[0]
        return self._return_data_class(*return_values)

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor:
        return self._manager._sim._scene._sanitize_envs_idx(envs_idx)


@dataclass
class RigidSensorMetadataMixin:
    """
    Base shared metadata class for sensors that are attached to a RigidEntity.
    """

    solver: "RigidSolver | None" = None
    links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    offsets_pos: torch.Tensor = make_tensor_field((0, 0, 3))
    offsets_quat: torch.Tensor = make_tensor_field((0, 0, 4))


RigidSensorMetadataMixinT = TypeVar("RigidSensorMetadataMixinT", bound=RigidSensorMetadataMixin)


class RigidSensorMixin(Generic[RigidSensorMetadataMixinT]):
    """
    Base sensor class for sensors that are attached to a RigidEntity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._link: "RigidLink" | None = None

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        batch_size = self._manager._sim._B

        entity = self._shared_metadata.solver.entities[self._options.entity_idx]
        self._link = entity.links[self._options.link_idx_local]
        self._shared_metadata.links_idx = concat_with_tensor(
            self._shared_metadata.links_idx, self._options.link_idx_local + entity.link_start
        )
        self._shared_metadata.offsets_pos = concat_with_tensor(
            self._shared_metadata.offsets_pos,
            self._options.pos_offset,
            expand=(batch_size, 1, 3),
            dim=1,
        )
        self._shared_metadata.offsets_quat = concat_with_tensor(
            self._shared_metadata.offsets_quat,
            euler_to_quat([self._options.euler_offset]),
            expand=(batch_size, 1, 4),
            dim=1,
        )


@dataclass
class NoisySensorMetadataMixin:
    """
    Base shared metadata class for analog sensors that are attached to a RigidEntity.
    """

    resolution: torch.Tensor = make_tensor_field((0, 0))
    bias: torch.Tensor = make_tensor_field((0, 0))
    cur_random_walk: torch.Tensor = make_tensor_field((0, 0))
    random_walk: torch.Tensor = make_tensor_field((0, 0))
    cur_noise: torch.Tensor = make_tensor_field((0, 0))
    noise: torch.Tensor = make_tensor_field((0, 0))
    jitter_ts: torch.Tensor = make_tensor_field((0, 0))
    cur_jitter_ts: torch.Tensor = make_tensor_field((0, 0))
    delay_in_steps: torch.Tensor = make_tensor_field((0, 0))
    interpolate: list[bool] = field(default_factory=list)


NoisySensorMetadataMixinT = TypeVar("NoisySensorMetadataMixinT", bound=NoisySensorMetadataMixin)


class NoisySensorMixin(Generic[NoisySensorMetadataMixinT]):
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
        if not isinstance(input, Sequence):
            input = [input]
        tensor_input = torch.tensor(input, dtype=dtype, device=gs.device)
        if tensor_input.ndim == len(shape) - 1:
            # Batch dimension is missing
            tensor_input = tensor_input.unsqueeze(0).expand((shape[0], *tensor_input.shape))
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
    def set_random_walk(self, random_walk, envs_idx=None):
        self._set_metadata_field(random_walk, self._shared_metadata.random_walk, self._cache_size, envs_idx)

    @gs.assert_built
    def set_noise(self, noise, envs_idx=None):
        self._set_metadata_field(noise, self._shared_metadata.noise, self._cache_size, envs_idx)

    @gs.assert_built
    def set_jitter(self, jitter, envs_idx=None):
        jitter_ts = np.asarray(jitter, dtype=gs.np_float) / self._dt
        self._set_metadata_field(jitter_ts, self._shared_metadata.jitter_ts, field_size=1, envs_idx=envs_idx)

    @gs.assert_built
    def set_delay(self, delay, envs_idx=None):
        self._set_metadata_field(delay, self._shared_metadata.delay_in_steps, field_size=1, envs_idx=envs_idx)

    def build(self):
        """
        Initialize all shared metadata needed to update all noisy sensors.
        """
        super().build()
        to_tuple = partial(_to_tuple, length_per_value=self._cache_size)

        batch_size = self._manager._sim._B

        self._shared_metadata.resolution = concat_with_tensor(
            self._shared_metadata.resolution, to_tuple(self._options.resolution), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.bias = concat_with_tensor(
            self._shared_metadata.bias, to_tuple(self._options.bias), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.random_walk = concat_with_tensor(
            self._shared_metadata.random_walk, to_tuple(self._options.random_walk), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.cur_random_walk = torch.zeros_like(self._shared_metadata.random_walk)
        self._shared_metadata.noise = concat_with_tensor(
            self._shared_metadata.noise, to_tuple(self._options.noise), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.cur_noise = torch.zeros_like(self._shared_metadata.noise)
        self._shared_metadata.jitter_ts = concat_with_tensor(
            self._shared_metadata.jitter_ts, to_tuple(self._options.jitter / self._dt), expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.cur_jitter_ts = torch.zeros_like(self._shared_metadata.jitter_ts, device=gs.device)
        self._shared_metadata.interpolate.append(self._options.interpolate)

    @classmethod
    def reset(cls, shared_metadata: NoisySensorMetadataMixin, envs_idx):
        shared_metadata.cur_random_walk[envs_idx, ...].fill_(0.0)

    @classmethod
    def _add_noise_drift_bias(cls, shared_metadata: NoisySensorMetadataMixin, output: torch.Tensor):
        if torch.any(shared_metadata.random_walk > gs.EPS):
            shared_metadata.cur_random_walk += torch.normal(0.0, shared_metadata.random_walk)
            output += shared_metadata.cur_random_walk
        if torch.any(shared_metadata.noise > gs.EPS):
            torch.normal(0.0, shared_metadata.noise, out=shared_metadata.cur_noise)
            output += shared_metadata.cur_noise
        output += shared_metadata.bias

    @classmethod
    def _quantize_to_resolution(cls, resolution: torch.Tensor, output: torch.Tensor):
        mask = resolution > gs.EPS
        output[mask] = torch.round(output[mask] / resolution[mask]) * resolution[mask]

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
