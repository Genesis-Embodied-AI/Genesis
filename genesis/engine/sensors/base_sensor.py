from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Generic, NamedTuple, Sequence, TypeVar, get_args, get_origin

import numpy as np
import torch
from typing_extensions import TypeVar as TypeVarWithDefault

import genesis as gs
from genesis.repr_base import RBC
from genesis.typing import NumArrayType, NumericType
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import broadcast_tensor, concat_with_tensor, make_tensor_field

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.engine.solvers.kinematic_solver import KinematicSolver
    from genesis.options.sensors.options import SensorOptions
    from genesis.recorders.base_recorder import Recorder, RecorderOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


def _to_tuple(*values: NumArrayType, length_per_value: int = 3) -> tuple[NumericType, ...]:
    """
    Convert all input values to one flattened tuple, where each value is ensured to be a tuple of length_per_value.
    """
    full_tuple = ()
    for value in values:
        if isinstance(value, NumericType):
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
    history_lengths: list[int] = field(default_factory=list)
    # Per-sensor: interpolate delayed reads (aligned with cache_sizes). Appended in Sensor.build.
    interpolate: list[bool] = field(default_factory=list)
    # True iff at least one sensor in the class has a nonzero read delay. Precomputed at build (and latched True by
    # `set_delay`) so the per-step fast path in `_apply_delay_to_shared_cache` can avoid a GPU-syncing reduction.
    has_any_delay: bool = False
    # True iff at least one sensor in the class has a nonzero jitter option. Stays False for non-noisy sensor classes
    # (Contact, Raycaster) since they never sample jitter. Same precompute-and-latch contract as `has_any_delay`.
    has_any_jitter: bool = False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self):
        """
        Destroy shared metadata.

        This method is called by SensorManager when the scene is destroyed. his should remove any references to the
        sensors from the shared metadata, and clean up any resources associated with the sensors.
        """


SharedSensorMetadataT = TypeVar("SharedSensorMetadataT", bound=SharedSensorMetadata)
OptionsT = TypeVar("OptionsT", bound="SensorOptions")
DataT = TypeVarWithDefault("DataT", default=tuple, covariant=True)


class Sensor(RBC, Generic[OptionsT, SharedSensorMetadataT, DataT]):
    """
    Base class for all types of sensors.

    To create a sensor, prefer using `scene.add_sensor(sensor_options)` instead of instantiating this class directly.

    Each concrete sensor class declares its associated options, metadata, and data types via Generic type parameters::

        class MySensor(Sensor[MyOptions, MyMetadata, MyData]):
            ...  # DataT defaults to tuple; specify explicitly for NamedTuple returns

    Note
    -----
    The Sensor system is designed to be performant. All sensors of the same type are updated at once and stored
    in a cache in SensorManager. Cache size is inferred from the return format and cache length of each sensor.
    `read()` and `read_ground_truth()`, the public-facing methods of every Sensor, automatically handles indexing into
    the shared cache to return the correct data.
    """

    _options_cls: ClassVar[type]
    _metadata_cls: ClassVar[type]
    _return_data_class: ClassVar[type] = tuple

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            origin = get_origin(base)
            if origin is not None and issubclass(origin, Sensor):
                args = get_args(base)
                if len(args) >= 1 and not isinstance(args[0], TypeVar):
                    cls._options_cls = args[0]
                if len(args) >= 2 and not isinstance(args[1], TypeVar):
                    cls._metadata_cls = args[1]
                if len(args) >= 3 and not isinstance(args[2], TypeVar):
                    cls._return_data_class = args[2]
                break
        # Auto-register if this class defines its own options (not inherited).
        # Enforce that concrete sensor classes also specify the metadata type parameter.
        if "_options_cls" in cls.__dict__:
            if "_metadata_cls" not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must specify Sensor[OptionsT, MetadataT, DataT=tuple].")
            from .sensor_manager import SensorManager

            SensorManager.SENSOR_TYPES_MAP[cls._options_cls] = cls

    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        self._options: "SensorOptions" = sensor_options
        self._idx: int = sensor_idx
        self._manager: "SensorManager" = sensor_manager
        self._shared_metadata: SharedSensorMetadataT = sensor_manager._sensors_metadata[type(self)]
        self._is_built = False

        self._dt = self._manager._sim.dt
        self._delay_ts = round(self._options.delay / self._dt)

        self._cache_slices: list[slice] = []
        return_format = self._get_return_format()
        assert len(return_format) > 0
        intrinsic_shapes: tuple[tuple[int, ...], ...] = (
            (return_format,) if isinstance(return_format[0], int) else return_format
        )

        history_length = self._options.history_length
        self._cache_size = 0
        self._read_flat_slices: list[slice] = []
        read_off = 0
        for shape in intrinsic_shapes:
            data_size = np.prod(shape)
            self._cache_slices.append(slice(self._cache_size, self._cache_size + data_size))
            self._cache_size += data_size

            span = data_size * history_length if history_length > 0 else data_size
            self._read_flat_slices.append(slice(read_off, read_off + span))
            read_off += span

        if history_length > 0:
            self._return_shapes = tuple((history_length, *s) for s in intrinsic_shapes)
        else:
            self._return_shapes = intrinsic_shapes

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
        self._shared_metadata.history_lengths.append(self._options.history_length)
        self._shared_metadata.interpolate.append(getattr(self._options, "interpolate", False))
        if self._delay_ts > 0:
            self._shared_metadata.has_any_delay = True

    @classmethod
    def reset(cls, shared_metadata: SharedSensorMetadataT, shared_ground_truth_cache: torch.Tensor, envs_idx):
        """
        Reset the sensor.

        This method is called by SensorManager when the scene is reset by `scene.reset()`.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor class.
        shared_ground_truth_cache : torch.Tensor
            The shared ground truth cache for the sensor class.
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
    def _update_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadataT,
        current_ground_truth_data_T: torch.Tensor,
        measured_data_timeline: "TensorRingBuffer",
    ):
        """
        Update the shared ground-truth cache slice (shape ``(cols, B)``, C-contiguous rows) and write this physics
        step's pre-delay measured values into ``measured_data_timeline``. The current write slot is
        ``measured_data_timeline.at(0, copy=False)`` (shape ``(B, cols)``); sensors may read older slots via
        ``at(k, ...)`` (for example first-order filtering). SensorManager applies read delay/jitter from this timeline
        into ``read()``.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `update_shared_cache()`.")

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        """
        The dtype of the cache for this sensor.
        """
        raise NotImplementedError(f"{cls.__name__} has not implemented `get_cache_dtype()`.")

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug shapes for the sensor in the scene.
        """
        raise NotImplementedError(f"{type(self).__name__} has not implemented `draw_debug()`.")

    # =============================== public shared methods ===============================

    @gs.assert_built
    def read(self, envs_idx=None) -> DataT:
        """
        Read the sensor data (with noise applied if applicable).
        """
        return self._get_formatted_data(self._manager.get_cloned_from_cache(self), envs_idx)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None) -> DataT:
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
        measured_data_timeline: "TensorRingBuffer",
        interpolate: Sequence[bool] | None = None,
    ):
        """
        Applies the read delay (and jitter, if any) to the shared cache tensor by copying the buffered data at the
        appropriate index. When the class has neither delay nor jitter, takes a fast path that writes the most
        recent ring frame to the cache class-wide.

        Parameters
        ----------
        shared_metadata : SharedSensorMetadata
            The shared metadata for the sensor.
        shared_cache : torch.Tensor
            The shared cache tensor.
        measured_data_timeline : TensorRingBuffer
            Pre-delay measured timeline ring for this sensor class slice (current step already written by
            ``_update_shared_cache``).
        interpolate : Sequence[bool] | None
            Whether to interpolate the sensor data for the read delay + jitter. Defaults to False.
        """
        # Fast path: when no sensor in the class has any delay or jitter, this is equivalent to copying the most
        # recent ring frame to the measured cache class-wide. Avoids a Python loop over every sensor. Both flags
        # are precomputed Python bools (no GPU sync) latched at build / by setters.
        if not shared_metadata.has_any_delay and not shared_metadata.has_any_jitter:
            shared_cache.copy_(measured_data_timeline.at(0))
            return

        # Sample uniform jitter in [0, jitter_ts) per env per sensor. One-sided so samples are non-negative;
        # combined with the `jitter < dt` option constraint, the effective per-step shift is strictly bounded
        # within [0, 1) steps and cannot wrap the ring.
        if shared_metadata.has_any_jitter:
            shared_metadata.cur_jitter_ts.uniform_(0.0, 1.0).mul_(shared_metadata.jitter_ts)
            cur_jitter_ts = shared_metadata.cur_jitter_ts
        else:
            cur_jitter_ts = None

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
            tensor_slice = slice(tensor_start, tensor_start + tensor_size)
            sensor_cache = shared_cache[:, tensor_slice]
            data_left = measured_data_timeline.at(cur_delay_ts_int, tensor_slice, per_row=True)
            if interp:
                ratio = torch.frac(cur_delay_ts)
                data_right = measured_data_timeline.at(cur_delay_ts_int + 1, tensor_slice, per_row=True)
                torch.lerp(data_left, data_right, ratio[:, None], out=sensor_cache)
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
            sl = self._read_flat_slices[i]
            field_data = tensor_chunk[..., sl].reshape((len(envs_idx), *shape))
            if self._manager._sim.n_envs == 0:
                field_data = field_data[0]
            return_values.append(field_data)

        if len(return_values) == 1:
            return return_values[0]
        return self._return_data_class(*return_values)

    def _sanitize_envs_idx(self, envs_idx) -> torch.Tensor:
        return self._manager._sim._scene._sanitize_envs_idx(envs_idx)

    def _set_metadata_field(self, value, field, field_size, envs_idx=None):
        envs_idx = self._sanitize_envs_idx(envs_idx)
        if field.ndim == 2:
            # flat field structure
            idx = self._idx * field_size
            index_slice = slice(idx, idx + field_size)
        else:
            # per sensor field structure
            index_slice = self._idx

        field[:, index_slice] = broadcast_tensor(value, field.dtype, (len(envs_idx), field_size), ("envs_idx", ""))


class _SolverLinkGroup(NamedTuple):
    """Per-solver bucket: (solver, in-solver link indices, sensor columns)."""

    solver: "KinematicSolver"
    links_idx: torch.Tensor  # solver-local link indices, one per sensor in this group
    sensor_cols: torch.Tensor  # which per-class sensor column each link pose lands in


@dataclass
class KinematicSensorMetadataMixin:
    """
    Shared metadata for sensors attached to a KinematicEntity (or any subclass, including RigidEntity). Sensors are
    bucketed at build time into per-solver _SolverLinkGroup entries so the per-step gather is one bulk read per solver.
    Static sensors (entity_idx<0) are not bucketed and keep an identity link pose, leaving the kernel to apply
    pos_offset / euler_offset in world frame.
    """

    offsets_pos: torch.Tensor = make_tensor_field((0, 0, 3))
    offsets_quat: torch.Tensor = make_tensor_field((0, 0, 4))
    solver_groups: list[_SolverLinkGroup] = field(default_factory=list)

    @property
    def n_sensors(self) -> int:
        return self.offsets_pos.shape[1]


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
KinematicSensorMetadataMixinT = TypeVar("KinematicSensorMetadataMixinT", bound=KinematicSensorMetadataMixin)


class _LinkAttachedSensorMixin:
    """
    Common boilerplate for sensors attached to a link: holds the python-side `_link` reference, concatenates
    per-sensor pos/euler offsets into shared metadata at build time, and exposes `set_{pos,quat}_offset`. Subclasses
    implement `_register_link` to record the link mapping in solver-specific shared-metadata shape (single tensor
    for RigidSensorMixin, per-solver buckets for KinematicSensorMixin).
    """

    _link: "RigidLink | None" = None

    def build(self):
        super().build()

        batch_size = self._manager._sim._B
        if self._options.entity_idx >= 0:
            entity = self._manager._sim.entities[self._options.entity_idx]
            self._link = entity.links[self._options.link_idx_local]
            link_idx = self._options.link_idx_local + entity.link_start
            self._register_link(entity, link_idx)

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

    def _register_link(self, entity, link_idx: int):
        raise NotImplementedError

    @gs.assert_built
    def set_pos_offset(self, pos_offset, envs_idx=None):
        self._set_metadata_field(pos_offset, self._shared_metadata.offsets_pos, 3, envs_idx)

    @gs.assert_built
    def set_quat_offset(self, quat_offset, envs_idx=None):
        self._set_metadata_field(quat_offset, self._shared_metadata.offsets_quat, 4, envs_idx)


class RigidSensorMixin(_LinkAttachedSensorMixin, Generic[RigidSensorMetadataMixinT]):
    """Base sensor class for sensors that are attached to a RigidEntity."""

    def build(self):
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver
        super().build()

    def _register_link(self, entity, link_idx: int):
        self._shared_metadata.links_idx = concat_with_tensor(self._shared_metadata.links_idx, link_idx)


class KinematicSensorMixin(_LinkAttachedSensorMixin, Generic[KinematicSensorMetadataMixinT]):
    """
    Base sensor class for sensors that may attach to entities across solvers (rigid or kinematic). Bucketing into
    shared_metadata.solver_groups happens at build time so the per-step gather is one bulk read per solver.
    """

    def _register_link(self, entity, link_idx: int):
        sensor_col = self._shared_metadata.n_sensors
        groups = self._shared_metadata.solver_groups
        existing = next((i for i, g in enumerate(groups) if g.solver is entity.solver), None)
        if existing is None:
            groups.append(
                _SolverLinkGroup(
                    solver=entity.solver,
                    links_idx=concat_with_tensor(torch.empty(0, device=gs.device, dtype=gs.tc_int), link_idx),
                    sensor_cols=concat_with_tensor(torch.empty(0, device=gs.device, dtype=gs.tc_int), sensor_col),
                )
            )
        else:
            group = groups[existing]
            groups[existing] = _SolverLinkGroup(
                solver=group.solver,
                links_idx=concat_with_tensor(group.links_idx, link_idx),
                sensor_cols=concat_with_tensor(group.sensor_cols, sensor_col),
            )


@dataclass
class ImperfectSensorMetadataMixin:
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
    # Precomputed Python bool flags gate the per-step noise/bias/quantize work without GPU sync. Set at build
    # from options and refreshed by the corresponding setters. Conservatively True once any sensor has nonzero
    # value; never flipped back to False (avoids tracking per-sensor state).
    has_any_noise: bool = False
    has_any_random_walk: bool = False
    has_any_bias: bool = False
    has_any_resolution: bool = False


ImperfectSensorMetadataMixinT = TypeVar("ImperfectSensorMetadataMixinT", bound=ImperfectSensorMetadataMixin)


class ImperfectSensorMixin(Generic[ImperfectSensorMetadataMixinT]):
    """
    Base sensor class for analog sensors that are attached to a RigidEntity.
    """

    @gs.assert_built
    def set_resolution(self, resolution, envs_idx=None):
        self._set_metadata_field(resolution, self._shared_metadata.resolution, self._cache_size, envs_idx)
        self._shared_metadata.has_any_resolution = bool((self._shared_metadata.resolution > gs.EPS).any().item())

    @gs.assert_built
    def set_bias(self, bias, envs_idx=None):
        self._set_metadata_field(bias, self._shared_metadata.bias, self._cache_size, envs_idx)
        self._shared_metadata.has_any_bias = bool((self._shared_metadata.bias != 0).any().item())

    @gs.assert_built
    def set_random_walk(self, random_walk, envs_idx=None):
        self._set_metadata_field(random_walk, self._shared_metadata.random_walk, self._cache_size, envs_idx)
        self._shared_metadata.has_any_random_walk = bool((self._shared_metadata.random_walk > gs.EPS).any().item())

    @gs.assert_built
    def set_noise(self, noise, envs_idx=None):
        self._set_metadata_field(noise, self._shared_metadata.noise, self._cache_size, envs_idx)
        self._shared_metadata.has_any_noise = bool((self._shared_metadata.noise > gs.EPS).any().item())

    @gs.assert_built
    def set_jitter(self, jitter, envs_idx=None):
        jitter_np = np.asarray(jitter, dtype=gs.np_float)
        if np.any(jitter_np < 0):
            gs.raise_exception(f"Sensor jitter must be non-negative; got jitter={tuple(jitter_np.ravel())}.")
        if np.any(jitter_np >= self._dt + gs.EPS):
            gs.raise_exception(
                f"Sensor jitter must not exceed the simulation step dt={self._dt}; got "
                f"jitter={tuple(jitter_np.ravel())}."
            )
        self._set_metadata_field(jitter_np / self._dt, self._shared_metadata.jitter_ts, 1, envs_idx)
        # Recompute the slow-path flag from the freshly-written class metadata. One GPU->CPU sync at setter
        # call time; setters are not hot path. The check covers partial envs_idx writes and other sensors.
        self._shared_metadata.has_any_jitter = bool((self._shared_metadata.jitter_ts > gs.EPS).any().item())

    def build(self):
        """
        Initialize all shared metadata needed to update all noisy sensors.
        """
        super().build()
        to_tuple = partial(_to_tuple, length_per_value=self._cache_size)

        batch_size = self._manager._sim._B

        # Jitter must not exceed the simulation step so it can only shift a read by at most one extra ring slot.
        # The shared GT ring is sized at build to accommodate `max_delay + 1` slots; a larger jitter would wrap
        # modulo the ring depth and silently return wrong-frame data. An EPS slack lets jitter == dt pass cleanly
        # despite float quantization.
        jitter_np = np.asarray(self._options.jitter, dtype=gs.np_float)
        if np.any(jitter_np >= self._dt + gs.EPS):
            gs.raise_exception(
                f"Sensor jitter must not exceed the simulation step dt={self._dt}; got "
                f"jitter={tuple(jitter_np.ravel())}."
            )

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
        if np.any(jitter_np > gs.EPS):
            self._shared_metadata.has_any_jitter = True
        if np.any(np.asarray(self._options.noise, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_noise = True
        if np.any(np.asarray(self._options.random_walk, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_random_walk = True
        if np.any(np.asarray(self._options.bias, dtype=gs.np_float) != 0):
            self._shared_metadata.has_any_bias = True
        if np.any(np.asarray(self._options.resolution, dtype=gs.np_float) > gs.EPS):
            self._shared_metadata.has_any_resolution = True

    @classmethod
    def reset(cls, shared_metadata: ImperfectSensorMetadataMixin, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        shared_metadata.cur_random_walk[envs_idx, ...].fill_(0.0)

    @classmethod
    def _apply_imperfections(cls, shared_metadata: "ImperfectSensorMetadataMixin", output: torch.Tensor):
        """Transform ground truth into realistic measured data in-place: apply random_walk drift, Gaussian noise,
        bias, then quantize to the configured resolution. Each contribution is gated by a precomputed Python bool
        flag (`has_any_*`) so sensor classes with all-zero values pay no GPU work."""
        if shared_metadata.has_any_random_walk:
            shared_metadata.cur_random_walk += torch.normal(0.0, shared_metadata.random_walk)
            output += shared_metadata.cur_random_walk
        if shared_metadata.has_any_noise:
            torch.normal(0.0, shared_metadata.noise, out=shared_metadata.cur_noise)
            output += shared_metadata.cur_noise
        if shared_metadata.has_any_bias:
            output += shared_metadata.bias
        if shared_metadata.has_any_resolution:
            resolution = shared_metadata.resolution
            mask = resolution > gs.EPS
            output[mask] = torch.round(output[mask] / resolution[mask]) * resolution[mask]

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float
