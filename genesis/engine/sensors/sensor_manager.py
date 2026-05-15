import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors import types as _sensor_types_namespace
from genesis.options.sensors.options import SensorOptions
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext

    from .base_sensor import Sensor


class SensorManager:
    # Maps sensor options class -> sensor class for runtime dispatch.
    SENSOR_TYPES_MAP: dict[type[SensorOptions], type["Sensor"]] = {}

    def __init__(self, sim):
        self._sim = sim
        self._sensors_by_type: dict[type["Sensor"], list["Sensor"]] = {}
        self._sensors_metadata: dict[type["Sensor"], SharedSensorMetadata | None] = {}
        # Per-dtype intermediate caches: pre-`_post_process` storage in intermediate space. The transposed GT cache
        # is `(cols, B)` for C-contiguous per-class row slices required by kernel writes.
        self._ground_truth_intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        self._intermediate_cache: dict[type[torch.dtype], torch.Tensor] = {}
        # Per-class return caches: post-`_post_process` storage in return space. When `_post_process` is identity,
        # alias-views into the per-dtype intermediate cache; when overridden, separate buffers in return shape/dtype.
        self._return_cache: dict[type["Sensor"], torch.Tensor] = {}
        self._ground_truth_return_cache: dict[type["Sensor"], torch.Tensor] = {}
        self._ground_truth_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        # Measured timeline: post-imperfection, pre-delay snapshots. Shares ring idx with _ground_truth_timeline_ring
        # per dtype. v17 has no separate measured-history ring (history is served from this ring's slots or from the
        # per-class linearized history buffer).
        self._measured_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        # Linearized per-class history of shape (B, max_history_for_class, class_cache_size). Refreshed every step.
        # Lets read_sensors and per-sensor history reads return views instead of re-gathering from the ring.
        self._linearized_ground_truth_history: dict[type["Sensor"], torch.Tensor] = {}
        self._linearized_measured_history: dict[type["Sensor"], torch.Tensor] = {}
        # Per-class precomputed history index tensor [0, 1, ..., max_history-1]. Used to fancy-index the rings each
        # step.
        self._hist_idx_by_class: dict[type["Sensor"], torch.Tensor] = {}
        self._cache_slices_by_type: dict[type["Sensor"], slice] = {}
        # (sensor class, entity_idx) -> slice within the class cache. entity_idx == -1 means static sensors.
        self._entity_slice_in_class: dict[type["Sensor"], dict[int, slice]] = {}
        self._max_history_by_class: dict[type["Sensor"], int] = {}

    def create_sensor(self, sensor_options: "SensorOptions") -> "Sensor":
        sensor_options.validate_scene(self._sim.scene)
        sensor_cls = SensorManager._resolve_sensor_cls(type(sensor_options))
        self._sensors_by_type.setdefault(sensor_cls, [])
        if sensor_cls not in self._sensors_metadata:
            self._sensors_metadata[sensor_cls] = sensor_cls._metadata_cls()
        sensor = sensor_cls(sensor_options, len(self._sensors_by_type[sensor_cls]), self)
        self._sensors_by_type[sensor_cls].append(sensor)
        return sensor

    @staticmethod
    def _resolve_sensor_cls(options_cls: type) -> type["Sensor"]:
        """Resolve the sensor class for the given options class, triggering lazy discovery if needed."""
        sensor_cls = SensorManager.SENSOR_TYPES_MAP.get(options_cls)
        if sensor_cls is not None:
            return sensor_cls

        # Not registered yet — check that the options class specifies its sensor type, then try to discover it.
        # The sensor class name is extracted from the generic metadata on the options class bases.
        is_parameterized = False
        for base in options_cls.__bases__:
            meta = base.__pydantic_generic_metadata__
            if meta["origin"] is not None and issubclass(meta["origin"], SensorOptions):
                is_parameterized = bool(meta["args"]) and isinstance(meta["args"][0], str)
                break
        # Fallback: typing introspection on __orig_bases__ (for pydantic versions that flatten bases)
        if not is_parameterized:
            for base in options_cls.__orig_bases__:
                origin = get_origin(base)
                if origin is not None and issubclass(origin, SensorOptions):
                    args = get_args(base)
                    is_parameterized = bool(args) and isinstance(args[0], (str, ForwardRef))
                    break

        if not is_parameterized:
            gs.raise_exception(
                f"{options_cls.__name__} must parameterize its SensorOptions base with a sensor class, "
                f"e.g. `class {options_cls.__name__}(SensorOptions['MySensor']): ...`"
            )

        # Try to discover the sensor module from sibling modules of the options package.
        options_module = options_cls.__module__
        if "." in options_module:
            pkg_name = options_module.rsplit(".", 1)[0]
            pkg = sys.modules.get(pkg_name)
            if pkg is not None:
                pkg_path = pkg.__dict__.get("__path__")
                if pkg_path is not None:
                    for _, modname, _ in pkgutil.iter_modules(pkg_path, pkg.__name__ + "."):
                        if modname not in sys.modules:
                            try:
                                importlib.import_module(modname)
                            except Exception:
                                continue
                        if options_cls in SensorManager.SENSOR_TYPES_MAP:
                            return SensorManager.SENSOR_TYPES_MAP[options_cls]

        gs.raise_exception(
            f"No sensor class registered for {options_cls.__name__}. Ensure the sensor module is in the same "
            "package as the options module, or import the sensor class manually before calling add_sensor()."
        )

    def build(self):
        # Sort each class by entity_idx so sensors attached to the same entity occupy a contiguous slice of the
        # class cache. Static sensors have entity_idx=-1 and group together. Python's sort is stable, so
        # registration order is preserved within each entity bucket.
        for sensors in self._sensors_by_type.values():
            sensors.sort(key=lambda s: s._options.entity_idx)
            for new_idx, sensor in enumerate(sensors):
                sensor._idx = new_idx

        # Per-class intermediate dtype is the dtype of `sensor.intermediate_spec`. Per-class return dtype is the
        # dtype of `sensor.return_spec`. All instances of a sensor class share the same intermediate / return
        # dtypes (specs may vary by shape per-instance but dtype is class-uniform); use any-instance's spec.
        cache_size_per_dtype: dict[torch.dtype, int] = {}
        delay_depth_per_dtype: dict[torch.dtype, int] = {}
        max_history_per_dtype: dict[torch.dtype, int] = {}
        intermediate_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        return_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            intermediate_dtype = sensors[0].intermediate_spec.dtype
            return_dtype = sensors[0].return_spec.dtype
            intermediate_dtype_by_class[sensor_cls] = intermediate_dtype
            return_dtype_by_class[sensor_cls] = return_dtype

            cache_size_per_dtype.setdefault(intermediate_dtype, 0)
            cls_cache_start_idx = cache_size_per_dtype[intermediate_dtype]
            entity_offsets: dict[int, list[int]] = {}
            cls_offset = 0
            cls_max_history = 0
            for sensor in sensors:
                sensor._cache_idx = cache_size_per_dtype[intermediate_dtype]
                cache_size_per_dtype[intermediate_dtype] += sensor._cache_size
                delay_depth_per_dtype[intermediate_dtype] = max(
                    delay_depth_per_dtype.get(intermediate_dtype, 0), sensor._delay_ts + 1
                )
                hist = sensor._options.history_length
                if hist > 0:
                    max_history_per_dtype[intermediate_dtype] = max(
                        max_history_per_dtype.get(intermediate_dtype, 0), hist
                    )
                    cls_max_history = max(cls_max_history, hist)
                eid = sensor._options.entity_idx
                if eid in entity_offsets:
                    entity_offsets[eid][1] = cls_offset + sensor._cache_size
                else:
                    entity_offsets[eid] = [cls_offset, cls_offset + sensor._cache_size]
                cls_offset += sensor._cache_size

            cls_cache_end_idx = cache_size_per_dtype[intermediate_dtype]
            self._cache_slices_by_type[sensor_cls] = slice(cls_cache_start_idx, cls_cache_end_idx)
            self._entity_slice_in_class[sensor_cls] = {
                eid: slice(start, stop) for eid, (start, stop) in entity_offsets.items()
            }
            self._max_history_by_class[sensor_cls] = cls_max_history

        self._ground_truth_timeline_ring.clear()
        self._measured_timeline_ring.clear()
        self._return_cache.clear()
        self._ground_truth_return_cache.clear()
        self._linearized_ground_truth_history.clear()
        self._linearized_measured_history.clear()
        self._hist_idx_by_class.clear()

        dtype_uses_measured: dict[torch.dtype, bool] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = intermediate_dtype_by_class[sensor_cls]
            cur = dtype_uses_measured.get(dtype, False)
            dtype_uses_measured[dtype] = cur or any(sensor.uses_measured_pipeline for sensor in sensors)

        for dtype, total_cols in cache_size_per_dtype.items():
            cache_shape = (self._sim._B, total_cols)
            # Ground truth cache is stored transposed (cols, B) so that per-class row slices are C-contiguous,
            # which is required for kernel writes. The cache and ring buffer stay (B, cols) since they only
            # receive data via .copy_() / torch.lerp which handle non-contiguous targets.
            gt_cache_shape = (total_cols, self._sim._B)
            self._ground_truth_intermediate_cache[dtype] = torch.zeros(gt_cache_shape, dtype=dtype, device=gs.device)
            self._intermediate_cache[dtype] = torch.zeros(cache_shape, dtype=dtype, device=gs.device)
            delay_n = max(delay_depth_per_dtype.get(dtype, 1), 1)
            hist_n = max_history_per_dtype.get(dtype, 0)
            ring_n = max(delay_n, hist_n)
            self._ground_truth_timeline_ring[dtype] = TensorRingBuffer(ring_n, cache_shape, dtype=dtype)
            if dtype_uses_measured[dtype]:
                self._measured_timeline_ring[dtype] = TensorRingBuffer(
                    ring_n, cache_shape, dtype=dtype, idx=self._ground_truth_timeline_ring[dtype]._idx
                )

        # Per-class return caches. View alias into intermediate when `_post_process` is identity; separate buffer
        # when overridden.
        for sensor_cls, sensors in self._sensors_by_type.items():
            intermediate_dtype = intermediate_dtype_by_class[sensor_cls]
            return_dtype = return_dtype_by_class[sensor_cls]
            cls_slice = self._cache_slices_by_type[sensor_cls]
            if self._post_process_is_overridden(sensor_cls):
                # Separate return buffer in return dtype.
                cls_size = cls_slice.stop - cls_slice.start
                self._return_cache[sensor_cls] = torch.zeros(
                    (self._sim._B, cls_size), dtype=return_dtype, device=gs.device
                )
                self._ground_truth_return_cache[sensor_cls] = torch.zeros(
                    (self._sim._B, cls_size), dtype=return_dtype, device=gs.device
                )
            else:
                # Alias view of the intermediate cache's class slice. Same dtype/shape; no extra allocation.
                self._return_cache[sensor_cls] = self._intermediate_cache[intermediate_dtype][:, cls_slice]
                self._ground_truth_return_cache[sensor_cls] = self._ground_truth_intermediate_cache[intermediate_dtype][
                    cls_slice, :
                ].T

        for sensor_cls in self._sensors_by_type.keys():
            self._sensors_metadata[sensor_cls].envs_idx = self._sim._scene._envs_idx

        # Per-class linearized history buffers. Refreshed every step from the timeline ring.
        for sensor_cls, cls_max_history in self._max_history_by_class.items():
            if cls_max_history == 0:
                continue
            dtype = intermediate_dtype_by_class[sensor_cls]
            cache_slice = self._cache_slices_by_type[sensor_cls]
            cls_size = cache_slice.stop - cache_slice.start
            shape = (self._sim._B, cls_max_history, cls_size)
            self._linearized_ground_truth_history[sensor_cls] = torch.zeros(shape, dtype=dtype, device=gs.device)
            self._linearized_measured_history[sensor_cls] = torch.zeros(shape, dtype=dtype, device=gs.device)
            self._hist_idx_by_class[sensor_cls] = torch.arange(cls_max_history, device=gs.device, dtype=torch.int32)

        for sensor_cls, sensors in self._sensors_by_type.items():
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

    @staticmethod
    def _post_process_is_overridden(sensor_cls: type["Sensor"]) -> bool:
        from .base_sensor import Sensor as _Sensor

        return sensor_cls._post_process.__func__ is not _Sensor._post_process.__func__

    def destroy(self):
        for sensors_metadata in self._sensors_metadata.values():
            if sensors_metadata is not None:
                sensors_metadata.destroy()
        self._sensors_metadata.clear()
        self._sensors_by_type.clear()

    def reset(self, envs_idx=None):
        if not self._sensors_by_type:
            return

        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)

        for dtype in self._ground_truth_intermediate_cache.keys():
            self._ground_truth_intermediate_cache[dtype][:, envs_idx] = 0.0
            self._intermediate_cache[dtype][envs_idx] = 0.0
            self._ground_truth_timeline_ring[dtype].buffer[:, envs_idx] = 0.0
            if dtype in self._measured_timeline_ring:
                self._measured_timeline_ring[dtype].buffer[:, envs_idx] = 0.0

        # Reset per-class return caches that are distinct buffers (overridden `_post_process`); alias views are
        # already cleared via the intermediate-cache zero above.
        for sensor_cls, return_cache in self._return_cache.items():
            if self._post_process_is_overridden(sensor_cls):
                return_cache[envs_idx] = 0
                self._ground_truth_return_cache[sensor_cls][envs_idx] = 0

        for linearized in self._linearized_ground_truth_history.values():
            linearized[envs_idx] = 0.0
        for linearized in self._linearized_measured_history.values():
            linearized[envs_idx] = 0.0

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensors[0].intermediate_spec.dtype
            cache_slice = self._cache_slices_by_type[sensor_cls]
            sensor_cls.reset(
                self._sensors_metadata[sensor_cls],
                self._ground_truth_intermediate_cache[dtype][cache_slice],
                envs_idx,
            )

    def step(self):
        for ring in self._ground_truth_timeline_ring.values():
            ring.rotate()

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensors[0].intermediate_spec.dtype
            cache_slice = self._cache_slices_by_type[sensor_cls]
            ground_truth_slice = self._ground_truth_intermediate_cache[dtype][cache_slice]
            if dtype in self._measured_timeline_ring:
                measured_data_timeline = self._measured_timeline_ring[dtype][:, cache_slice]
            else:
                measured_data_timeline = None
            sensor_cls._update_shared_cache(
                self._sensors_metadata[sensor_cls],
                ground_truth_slice,
                measured_data_timeline,
                self._intermediate_cache[dtype][:, cache_slice],
                self._return_cache[sensor_cls],
            )
            # GT timeline ring write is required: history reads access older slots even at delay=0, so the slot for
            # the current step must be populated independent of the delay sampling done inside `_update_shared_cache`.
            self._ground_truth_timeline_ring[dtype][:, cache_slice].set(ground_truth_slice.T)
            # Mirror eager `_post_process` for the GT path. The orchestrator handles the measured path; here we
            # populate the GT return cache from the GT intermediate slice. No-op when buffers alias.
            if self._post_process_is_overridden(sensor_cls):
                gt_return = self._ground_truth_return_cache[sensor_cls]
                gt_return.copy_(sensor_cls._post_process(self._sensors_metadata[sensor_cls], ground_truth_slice.T))

        # Linearize per-class history once per step so that per-sensor and bulk reads are pure views. Stored in
        # intermediate space; sensors with overridden `_post_process` apply it per-slot on retrieval (rare path).
        for sensor_cls, cls_max_history in self._max_history_by_class.items():
            if cls_max_history == 0:
                continue
            dtype = self._sensors_by_type[sensor_cls][0].intermediate_spec.dtype
            cache_slice = self._cache_slices_by_type[sensor_cls]
            hist_idx = self._hist_idx_by_class[sensor_cls]
            ground_truth_view = self._ground_truth_timeline_ring[dtype].at(hist_idx, slice(None), cache_slice)
            self._linearized_ground_truth_history[sensor_cls].copy_(ground_truth_view.transpose(0, 1))
            if dtype in self._measured_timeline_ring:
                meas_view = self._measured_timeline_ring[dtype].at(hist_idx, slice(None), cache_slice)
                self._linearized_measured_history[sensor_cls].copy_(meas_view.transpose(0, 1))

    def draw_debug(self, context: "RasterizerContext"):
        for sensor in self.sensors:
            if sensor._options.draw_debug:
                sensor._draw_debug(context)

    def get_cloned_from_cache(self, sensor: "Sensor", is_ground_truth: bool = False) -> torch.Tensor:
        sensor_cls = type(sensor)
        cls_slice = self._cache_slices_by_type[sensor_cls]
        rel_start = sensor._cache_idx - cls_slice.start
        history_length = sensor._options.history_length

        if history_length > 0:
            linearized = (
                self._linearized_ground_truth_history[sensor_cls]
                if is_ground_truth
                else self._linearized_measured_history[sensor_cls]
            )
            sensor_hist = linearized[:, :history_length, rel_start : rel_start + sensor._cache_size]
            # When `_post_process` is overridden, the linearized buffer is in intermediate space; apply per-slot.
            if self._post_process_is_overridden(sensor_cls):
                metadata = self._sensors_metadata[sensor_cls]
                # sensor_hist: (B, H, n) — flatten H into B for per-call _post_process, then unflatten.
                B, H, n = sensor_hist.shape
                projected = sensor_cls._post_process(metadata, sensor_hist.reshape(B * H, n)).reshape(B, H, -1)
                sensor_hist = projected
            blocks = [sensor_hist[..., rel_slice].flatten(1, 2) for rel_slice in sensor._cache_slices]
            if len(blocks) == 1:
                return blocks[0]
            return torch.cat(blocks, dim=1)

        # Pure view into the per-class return cache. Eager `_post_process` already populated it during step().
        return_cache = (
            self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
        )
        return return_cache[:, rel_start : rel_start + sensor._cache_size]

    def read_sensors(
        self,
        entity_idx: int | None = None,
        envs_idx=None,
        copy: bool = False,
        is_ground_truth: bool = False,
    ) -> dict[int, torch.Tensor]:
        """
        Read the latest data of every sensor class in scope as a single tensor per class.

        Parameters
        ----------
        entity_idx : int | None
            - None (default): include every sensor in the scene.
            - k >= 0: include only sensors whose `entity_idx == k`.
            - -1: include only static sensors (those not attached to any entity).
        envs_idx : array-like | int | slice | None
            Environment selection. Integer or slice indexing produces a view along the batch axis; list/tensor
            (fancy) indexing produces a copy along the batch axis.
        copy : bool
            When True, returned tensors are cloned. When False (default), returned tensors are views into the
            per-class return cache. Honest for every sensor class — including ContactSensor and ContactForceSensor
            whose `_post_process` overrides write into real per-class storage at step end.
        is_ground_truth : bool
            When True, return ground-truth tensors instead of measured tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from sensor-type tag (`gs.sensors.types.<Name>`) to a tensor of shape
            (B, [history,] class_or_entity_cache_size). For sensors without history, the history
            dimension is omitted.
        """
        # Use basic indexing (slice/int) whenever possible to preserve view semantics. Fancy indexing
        # (list/tensor) on the env axis is only triggered when the user explicitly passes a list/tensor,
        # and is documented to copy.
        if envs_idx is None or isinstance(envs_idx, slice):
            env_index = slice(None) if envs_idx is None else envs_idx
        elif isinstance(envs_idx, int):
            env_index = envs_idx
        else:
            env_index = self._sim._scene._sanitize_envs_idx(envs_idx)

        result: dict[int, torch.Tensor] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            entity_slice_map = self._entity_slice_in_class.get(sensor_cls, {})
            if entity_idx is None:
                cls_slice = self._cache_slices_by_type[sensor_cls]
                within_cls_slice = slice(0, cls_slice.stop - cls_slice.start)
            else:
                eid = -1 if entity_idx < 0 else entity_idx
                if eid not in entity_slice_map:
                    continue
                within_cls_slice = entity_slice_map[eid]

            cls_max_history = self._max_history_by_class[sensor_cls]
            if cls_max_history > 0:
                linearized = (
                    self._linearized_ground_truth_history[sensor_cls]
                    if is_ground_truth
                    else self._linearized_measured_history[sensor_cls]
                )
                tensor = linearized[env_index, :, within_cls_slice]
                # When `_post_process` is overridden, the linearized buffer is in intermediate space; apply per-step.
                if self._post_process_is_overridden(sensor_cls):
                    metadata = self._sensors_metadata[sensor_cls]
                    B, H, n = tensor.shape
                    tensor = sensor_cls._post_process(metadata, tensor.reshape(B * H, n)).reshape(B, H, -1)
            else:
                return_cache = (
                    self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
                )
                tensor = return_cache[env_index, within_cls_slice]

            if copy:
                tensor = tensor.clone()
            if self._sim.n_envs == 0:
                tensor = tensor[0]
            options_cls = type(sensors[0]._options)
            type_id = getattr(_sensor_types_namespace, options_cls.__name__)
            result[type_id] = tensor
        return result

    def get_sensors_by_entity(self, entity_idx: int) -> "gs.List[Sensor]":
        """List of all sensors attached to the given entity (or static sensors for entity_idx == -1)."""
        target_eid = -1 if entity_idx < 0 else entity_idx
        return gs.List(
            sensor
            for sensor_list in self._sensors_by_type.values()
            for sensor in sensor_list
            if sensor._options.entity_idx == target_eid
        )

    @property
    def sensors(self):
        return gs.List([sensor for sensor_list in self._sensors_by_type.values() for sensor in sensor_list])
