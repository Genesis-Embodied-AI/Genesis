import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.options import Options
from genesis.options.recording import RecordingOptions
from genesis.repr_base import RBC

from .data_handlers import DataHandler
from .data_recorder import SensorDataRecorder

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer

    from .sensor_manager import SensorManager


class SensorOptions(Options):
    """
    Base class for all sensor options.
    Each sensor should have their own options class that inherits from this class.
    The options class should be registered with the SensorManager using the @register_sensor decorator.

    Parameters
    ----------
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    delay : float
        The delay in seconds before the sensor data is read.
    """

    update_ground_truth_only: bool = False
    delay: float = 0.0

    def validate(self, scene):
        """
        Validate the sensor options values before the sensor is added to the scene.
        """
        delay_hz = self.delay / scene._sim.dt
        if not np.isclose(delay_hz, round(delay_hz), atol=1e-6):
            gs.logger.warn(
                f"Read delay should be a multiple of the simulation time step. Got {self.delay}"
                f" and {scene._sim.dt}. Actual read delay will be {1/round(delay_hz)}."
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

    Do not instantiate any Sensor directly! Use scene.add_sensor(), which calls SensorManager.create_sensor().

    NOTE: The Sensor system is designed to be performant.  All sensors of the same type are updated at once and stored
    in a cache in SensorManager. Cache size is inferred from the return format and cache length of each sensor.
    `read()` and `read_ground_truth()`, the public-facing methods of every Sensor, automatically handles indexing into
    the shared cache to return the correct data.
    """

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._sensor_idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadata = sensor_manager._sensors_metadata[type(self)]
        self._data_recorder: SensorDataRecorder | None = None

        self._delay_steps = round(self._options.delay / self._manager._sim.dt)
        self._shared_metadata.delay_steps.append(self._delay_steps)

        self._shape_indices: list[tuple[int, int]] = []
        return_format = self.get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        tensor_size = 0
        for shape in return_shapes:
            data_size = np.prod(shape)
            self._shape_indices.append((tensor_size, tensor_size + data_size))
            tensor_size += data_size

        self._cache_size = self.get_cache_length() * tensor_size
        self._shared_metadata.cache_sizes.append(self._cache_size)

        self._cache_idx: int = -1  # initialized by SensorManager during build

    # =============================== implementable methods ===============================

    def build(self):
        """
        This method is called by the SensorManager during the scene build phase to initialize the sensor.
        This is where any shared metadata should be initialized.
        """
        raise NotImplementedError("Sensors must implement `build()`.")

    def get_return_format(self) -> dict[str, tuple[int, ...]] | tuple[int, ...]:
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
        raise NotImplementedError("Sensors must implement `return_format()`.")

    def get_cache_length(self) -> int:
        """
        The length of the cache for this sensor instance, e.g. number of points for a Lidar point cloud.
        """
        raise NotImplementedError("Sensors must implement `cache_length()`.")

    @classmethod
    def update_shared_ground_truth_cache(cls, shared_metadata: dict[str, Any], shared_ground_truth_cache: torch.Tensor):
        """
        Update the shared sensor ground truth cache for all sensors of this class using metadata in SensorManager.
        """
        raise NotImplementedError("Sensors must implement `update_shared_ground_truth_cache()`.")

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: dict[str, Any],
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the shared sensor cache for all sensors of this class using metadata in SensorManager.

        The information in shared_cache should be the final measured sensor data after all noise and post-processing.
        NOTE: The implementation should include applying the delay using the `_apply_delay_to_shared_cache()` method.
        """
        raise NotImplementedError("Sensors must implement `update_shared_cache_with_noise()`.")

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        """
        The dtype of the cache for this sensor.
        """
        raise NotImplementedError("Sensors must implement `get_cache_dtype()`.")

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

    def add_recorder(self, handler: "DataHandler", rec_options: RecordingOptions | None = None):
        """
        Add a sensor data recorder that processes data from this sensor.
        """
        if self._data_recorder is None:
            self._data_recorder = self._manager.create_data_recorder(self)
        self._data_recorder.add_handler(handler, rec_options)

    def start_recording(self):
        """
        Start recording the sensor data.
        """
        if self._data_recorder is None:
            gs.raise_exception("Sensor data recorder not found. Call `add_recorder()` first.")
        self._data_recorder.start()

    def pause_recording(self):
        """
        Pause recording the sensor data.
        """
        if self._data_recorder is None:
            gs.raise_exception("Sensor data recorder not found. Call `add_recorder()` first.")
        self._data_recorder.pause()

    def stop_recording(self):
        """
        Stop recording the sensor data.
        """
        if self._data_recorder is None:
            gs.raise_exception("Sensor data recorder not found. Call `add_recorder()` first.")
        self._data_recorder.stop()

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
        jitter: torch.Tensor | None = None,
        interpolate_for_delay: list[bool] | None = None,
    ):
        """
        Applies the read delay to the shared cache tensor by copying the buffered data at the appropriate index.

        NOTE: This is a helper method for `update_shared_cache()` to apply the delay to the shared cache tensor.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor.
        shared_cache : torch.Tensor
            The shared cache tensor.
        buffered_data : TensorRingBuffer
            The buffered data tensor.
        jitter : torch.Tensor | None
            The jitter in steps before the sensor data is read.
        interpolate_for_delay : list[bool] | None
            Whether to interpolate the sensor data for the read delay.
        """
        tensor_idx = 0

        for sensor_idx, (tensor_size, delay_step, interpolate) in enumerate(
            zip(
                shared_metadata.cache_sizes,
                shared_metadata.delay_steps,
                interpolate_for_delay or itertools.repeat(False),
            )
        ):
            if interpolate and jitter is not None:
                shared_cache[:, tensor_idx : tensor_idx + tensor_size] = torch.lerp(
                    buffered_data.at(delay_step + 1)[:, tensor_idx : tensor_idx + tensor_size],
                    buffered_data.at(delay_step)[:, tensor_idx : tensor_idx + tensor_size],
                    jitter[:, sensor_idx].unsqueeze(-1).expand(-1, tensor_size),
                )
            else:
                shared_cache[:, tensor_idx : tensor_idx + tensor_size] = buffered_data.at(delay_step)[
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

        return_format = self.get_return_format()
        return_shapes = return_format.values() if isinstance(return_format, dict) else (return_format,)
        return_values = []

        for i, shape in enumerate(return_shapes):
            start_idx, end_idx = self._shape_indices[i]
            value = tensor[envs_idx, start_idx:end_idx].reshape(len(envs_idx), *shape).squeeze()
            if self._manager._sim.n_envs == 0:
                value = value.squeeze(0)
            return_values.append(value)

        if isinstance(return_format, dict):
            return dict(zip(return_format.keys(), return_values))
        else:
            return return_values[0]

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor:
        return self._manager._sim._scene._sanitize_envs_idx(envs_idx)


class NoisySensorOptions(SensorOptions):
    """
    Base class for noisy sensor options.

    Parameters
    ----------
    noise_std : tuple[float, ...]
        The standard deviation of the noise.
    bias : tuple[float, ...]
        The bias of the sensor.
    bias_drift_std : tuple[float, ...]
        The standard deviation of the bias drift.
    delay : float
        The delay in seconds before the sensor data is read.
    jitter : float
        The time jitter standard deviation in seconds before the sensor data is read.
    interpolate_for_delay : bool
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used.
    """

    bias: tuple[float, ...] = field(default_factory=tuple)
    bias_drift_std: tuple[float, ...] = field(default_factory=tuple)
    noise_std: tuple[float, ...] = field(default_factory=tuple)
    jitter: float = 0.0
    interpolate_for_delay: bool = False

    def validate(self, scene):
        super().validate(scene)
        assert len(self.bias_drift_std) == len(self.bias), "Bias drift std must be the same length as bias."
        assert len(self.noise_std) == len(self.bias), "Noise std must be the same length as bias."
        assert self.jitter <= self.delay, "Jitter must be less than or equal to read delay."


@dataclass
class NoisySensorMetadata(SharedSensorMetadata):
    """
    Base class for all common sensor metadata.
    """

    bias: torch.Tensor = torch.tensor([])
    bias_drift: torch.Tensor = torch.tensor([])
    bias_drift_std: torch.Tensor = torch.tensor([])
    noise_std: torch.Tensor = torch.tensor([])
    jitter_std_in_steps: torch.Tensor = torch.tensor([])
    delay_in_steps: torch.Tensor = torch.tensor([])
    interpolate_for_delay: list[bool] = field(default_factory=list)


class NoisySensor(Sensor):
    """
    Base class for sensors with noise, bias, drift, and jitter.
    """

    @gs.assert_built
    def set_bias(self, bias, envs_idx=None):
        self._set_metadata_tensor(bias, self._shared_metadata.bias, envs_idx, self._cache_size)

    @gs.assert_built
    def set_bias_drift_std(self, bias_drift_std, envs_idx=None):
        self._set_metadata_tensor(bias_drift_std, self._shared_metadata.bias_drift_std, envs_idx, self._cache_size)

    @gs.assert_built
    def set_noise_std(self, noise_std, envs_idx=None):
        self._set_metadata_tensor(noise_std, self._shared_metadata.noise_std, envs_idx, self._cache_size)

    @gs.assert_built
    def set_jitter(self, jitter, envs_idx=None):
        if isinstance(jitter, list):
            jitter_in_steps = np.array(jitter) / self._manager._sim.dt
        else:  # scalar or torch tensor or np array
            jitter_in_steps = jitter / self._manager._sim.dt
        self._set_metadata_tensor(jitter_in_steps, self._shared_metadata.jitter_std_in_steps, envs_idx, 1)

    @gs.assert_built
    def set_delay(self, delay, envs_idx=None):
        self._set_metadata_tensor(delay, self._shared_metadata.delay_in_steps, envs_idx, 1)

    def _set_metadata_tensor(self, input, metadata_tensor, envs_idx=None, n_per_sensor=1):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        idx = self._sensor_idx * n_per_sensor
        metadata_tensor[envs_idx, idx : idx + n_per_sensor] = self._sanitize_for_metadata_tensor(
            input, shape=(len(envs_idx), n_per_sensor)
        )

    def _sanitize_for_metadata_tensor(self, input, shape) -> torch.Tensor:
        if np.isscalar(input):
            input = [input]
        tensor_input = torch.tensor(input, dtype=gs.tc_float, device=gs.device)
        if tensor_input.ndim < len(shape):
            tensor_input = tensor_input.unsqueeze(0)  # add batch dim
        if tensor_input.shape[0] == 1 and shape[0] > 1:
            tensor_input = tensor_input.expand(shape[0], *tensor_input.shape[1:])  # repeat batch dim
        assert (
            tensor_input.shape == shape
        ), f"Input shape {tensor_input.shape} for setting sensor metadata does not match shape {shape}"
        return tensor_input

    def build(self):
        """
        Initialize all shared metadata needed to update all noisy sensors.
        """
        batch_size = self._manager._sim._B

        self._shared_metadata.bias = torch.cat(
            [
                self._shared_metadata.bias.to(gs.device),
                torch.tensor([self._options.bias], dtype=gs.tc_float, device=gs.device).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.bias_drift_std = torch.cat(
            [
                self._shared_metadata.bias_drift_std.to(gs.device),
                torch.tensor([self._options.bias_drift_std], dtype=gs.tc_float, device=gs.device).expand(
                    batch_size, -1
                ),
            ],
            dim=-1,
        )
        self._shared_metadata.bias_drift = torch.zeros_like(self._shared_metadata.bias_drift_std)
        self._shared_metadata.noise_std = torch.cat(
            [
                self._shared_metadata.noise_std.to(gs.device),
                torch.tensor([self._options.noise_std], dtype=gs.tc_float, device=gs.device).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.jitter_std_in_steps = torch.cat(
            [
                self._shared_metadata.jitter_std_in_steps.to(gs.device),
                torch.tensor(
                    [self._options.jitter / self._manager._sim.dt], dtype=gs.tc_float, device=gs.device
                ).expand(batch_size, -1),
            ],
            dim=-1,
        )
        self._shared_metadata.jitter_in_steps = torch.zeros_like(
            self._shared_metadata.jitter_std_in_steps, device=gs.device
        )
        self._shared_metadata.interpolate_for_delay.append(self._options.interpolate_for_delay)

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: NoisySensorMetadata,
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
        torch.normal(0, shared_metadata.jitter_std_in_steps, out=shared_metadata.jitter_in_steps)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.jitter_in_steps,
            shared_metadata.interpolate_for_delay,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)

    @classmethod
    def _add_noise_drift_bias(cls, shared_metadata: NoisySensorMetadata, shared_cache: torch.Tensor):
        shared_metadata.bias_drift += torch.normal(0, shared_metadata.bias_drift_std)
        shared_cache += torch.normal(shared_metadata.bias, shared_metadata.noise_std) + shared_metadata.bias_drift

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
