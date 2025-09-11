import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.solvers import RigidSolver
from genesis.options import Options
from genesis.repr_base import RBC
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field

if TYPE_CHECKING:
    from genesis.recorders.base_recorder import RecorderOptions
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager

NumericType = int | float | bool
NumericSequenceType = NumericType | Sequence[NumericType]
Tuple3FType = tuple[float, float, float]
MaybeTuple3FType = float | Tuple3FType


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
        if not np.isclose(delay_hz, round(delay_hz), atol=gs.EPS):
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
    delays_ts: list[int] = field(default_factory=list)


@ti.data_oriented
class Sensor(RBC):
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

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadata = sensor_manager._sensors_metadata[type(self)]
        self._is_built = False

        self._dt = self._manager._sim.dt
        self._delays_ts = round(self._options.delay / self._dt)
        self._shared_metadata.delays_ts.append(self._delays_ts)

        self._cache_slices: list[slice] = []
        self._return_format = self._get_return_format()
        is_return_dict = isinstance(self._return_format, dict)
        if is_return_dict:
            self._return_shapes = self._return_format.values()
            self._get_formatted_data = self._get_formatted_data_dict
        else:
            self._return_shapes = (self._return_format,)
            self._get_formatted_data = self._get_formatted_data_tuple
        self._cache_size = 0
        for shape in self._return_shapes:
            data_size = np.prod(shape)
            self._cache_slices.append(slice(self._cache_size, self._cache_size + data_size))
            self._cache_size += data_size

        self._shared_metadata.cache_sizes.append(self._cache_size)

        self._cache_idx: int = -1  # initialized by SensorManager during build

    # =============================== methods to implement ===============================

    def build(self):
        """
        Build the sensor.

        This method is called by SensorManager during the scene build phase.
        This is where any shared metadata should be initialized.
        """
        pass

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadata, envs_idx):
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

    def _get_return_format(self) -> dict[str, tuple[int, ...]] | tuple[int, ...]:
        """
        Get the data format of the read() return value.

        Returns
        -------
        return_format : dict | tuple
            - If tuple, the final shape of the read() return value.
                e.g. (2, 3) means read() will return a tensor of shape (2, 3).
            - If dict a dictionary with string keys and tensor values will be returned.
                e.g. {"pos": (3,), "quat": (4,)} returns a dict of tensors [0:3] and [3:7] from the cache.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `get_return_format()`.")

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache: torch.Tensor
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
    def start_recording(self, rec_options: "RecorderOptions"):
        """
        Automatically read and process sensor data. See RecorderOptions for more details.

        Data from `sensor.read()` is used. If the sensor data needs to be preprocessed before passing to the recorder,
        consider using `scene.start_recording()` instead with a custom data function.

        Parameters
        ----------
        rec_options : RecorderOptions
            The options for the recording.
        """
        self._manager._sim._scene._recorder_manager.add_recorder(self.read, rec_options)

    @property
    def is_built(self) -> bool:
        return self._is_built

    # =============================== private shared methods ===============================

    @classmethod
    def _apply_delay_to_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadata,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
        cur_jitter_ts: torch.Tensor | None = None,
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
        cur_jitter_ts : torch.Tensor | None
            The current jitter in timesteps (divided by simulation dt) before the sensor data is read.
        interpolate : list[bool] | None
            Whether to interpolate the sensor data for the read delay + jitter.
        """
        tensor_idx = 0

        for sensor_idx, (tensor_size, delay_ts, interpolate) in enumerate(
            zip(
                shared_metadata.cache_sizes,
                shared_metadata.delays_ts,
                interpolate or itertools.repeat(False),
            )
        ):
            if cur_jitter_ts is not None:
                cur_delay_ts = delay_ts + cur_jitter_ts[:, sensor_idx]
            else:
                cur_delay_ts = torch.tensor(delay_ts, dtype=gs.tc_float, device=gs.device)
            # get int for indexing into ring buffer (0 = most recent, 1 = delayed by one timestep, etc.)
            cur_delay_ts_int = cur_delay_ts.to(dtype=gs.tc_int)
            idx_slices = (slice(None), slice(tensor_idx, tensor_idx + tensor_size))
            cache_length = len(shared_cache)
            if interpolate:
                ratio = torch.frac(cur_delay_ts).unsqueeze(-1)
                data_left = buffered_data.at(cur_delay_ts_int).reshape(cache_length, -1)[idx_slices]
                data_right = buffered_data.at(cur_delay_ts_int + 1).reshape(cache_length, -1)[idx_slices]
                shared_cache[idx_slices] = data_left + ratio * (data_right - data_left)
            else:
                buffered_tensor = buffered_data.at(cur_delay_ts_int).reshape(cache_length, -1)
                shared_cache[idx_slices] = buffered_tensor[idx_slices]
            tensor_idx += tensor_size

    def _get_return_values(self, tensor: torch.Tensor, envs_idx=None) -> list[torch.Tensor]:
        """
        Preprares the given tensor into multiple tensors matching `self._return_shapes`.

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

        return return_values

    def _get_formatted_data_dict(self, tensor: torch.Tensor, envs_idx=None) -> dict[str, torch.Tensor]:
        """Returns a dictionary of tensors matching the return format."""
        return dict(zip(self._return_format.keys(), self._get_return_values(tensor, envs_idx)))

    def _get_formatted_data_tuple(self, tensor: torch.Tensor, envs_idx=None) -> torch.Tensor:
        """Returns a tensor matching the return format."""
        return self._get_return_values(tensor, envs_idx)[0]

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
    """

    entity_idx: int
    link_idx_local: int = 0
    pos_offset: Tuple3FType = (0.0, 0.0, 0.0)
    euler_offset: Tuple3FType = (0.0, 0.0, 0.0)

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
    links_idx: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    offsets_pos: torch.Tensor = make_tensor_field((0, 0, 3))
    offsets_quat: torch.Tensor = make_tensor_field((0, 0, 4))


class RigidSensorMixin:
    """
    Base sensor class for sensors that are attached to a RigidEntity.
    """

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.links_idx = concat_with_tensor(
            self._shared_metadata.links_idx, self._options.link_idx_local + self._options.entity_idx
        )
        self._shared_metadata.offsets_pos = concat_with_tensor(
            self._shared_metadata.offsets_pos, self._options.pos_offset
        )
        quat_tensor = torch.tensor(euler_to_quat([self._options.euler_offset]), dtype=gs.tc_float, device=gs.device)
        if self._shared_metadata.solver.n_envs > 0:
            quat_tensor = quat_tensor.unsqueeze(0).expand((self._manager._sim._B, 1, 4))
        self._shared_metadata.offsets_quat = concat_with_tensor(self._shared_metadata.offsets_quat, quat_tensor, dim=-2)


class NoisySensorOptionsMixin:
    """
    Base options class for analog sensors that are attached to a RigidEntity.

    Parameters
    ----------
    resolution : float | tuple[float, ...], optional
        The measurement resolution of the sensor (smallest increment of change in the sensor reading).
        Default is None, which means no quantization is applied.
    bias : float | tuple[float, ...], optional
        The constant additive bias of the sensor.
    noise : float | tuple[float, ...], optional
        The standard deviation of the additive white noise.
    random_walk : float | tuple[float, ...], optional
        The standard deviation of the random walk, which acts as accumulated bias drift.
    delay : float, optional
        The delay in seconds, affecting how outdated the sensor data is when it is read.
    jitter : float, optional
        The jitter in seconds modeled as a a random additive delay sampled from a normal distribution.
        Jitter cannot be greater than delay. `interpolate` should be True when `jitter` is greater than 0.
    interpolate : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used. Default is False.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth data, and not the measured data.
    """

    resolution: float | tuple[float, ...] | None = None
    bias: float | tuple[float, ...] = 0.0
    noise: float | tuple[float, ...] = 0.0
    random_walk: float | tuple[float, ...] = 0.0
    jitter: float = 0.0
    interpolate: bool = False

    def validate(self, scene):
        super().validate(scene)
        if self.jitter > 0 and not self.interpolate:
            gs.raise_exception(f"{type(self).__name__}: `interpolate` should be True when `jitter` is greater than 0.")
        if self.jitter > self.delay:
            gs.raise_exception(f"{type(self).__name__}: Jitter must be less than or equal to read delay.")


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

        batch_size = self._manager._sim._B

        if isinstance(self._options.resolution, tuple):
            self._options.resolution = tuple([-1 if r is None else r for r in self._options.resolution])
        self._shared_metadata.resolution = concat_with_tensor(
            self._shared_metadata.resolution, self._options.resolution or -1, expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.bias = concat_with_tensor(
            self._shared_metadata.bias, self._options.bias, expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.random_walk = concat_with_tensor(
            self._shared_metadata.random_walk, self._options.random_walk, expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.cur_random_walk = torch.zeros_like(self._shared_metadata.random_walk)
        self._shared_metadata.noise = concat_with_tensor(
            self._shared_metadata.noise, self._options.noise, expand=(batch_size, -1), dim=-1
        )
        self._shared_metadata.cur_noise = torch.zeros_like(self._shared_metadata.noise)
        self._shared_metadata.jitter_ts = concat_with_tensor(
            self._shared_metadata.jitter_ts, self._options.jitter / self._dt, expand=(batch_size, -1), dim=-1
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
        mask = resolution > 0
        output[mask] = torch.round(output[mask] / resolution[mask]) * resolution[mask]

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
