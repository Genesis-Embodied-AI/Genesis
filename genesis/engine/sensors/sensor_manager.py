import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING, ForwardRef, get_args, get_origin

import torch

import genesis as gs
from genesis.options.sensors import types as _sensor_types_namespace
from genesis.options.sensors.options import SensorOptions
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import Sensor, SharedSensorMetadata

if TYPE_CHECKING:
    from genesis.vis.rasterizer_context import RasterizerContext


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
        # per dtype. No separate measured-history ring - history reads are served from this ring's slots or from the
        # per-class linearized history buffer.
        self._measured_timeline_ring: dict[type[torch.dtype], TensorRingBuffer] = {}
        # Per-class return-space rings, only allocated when `_post_process` is overridden AND any sensor of the class
        # has `history_length > 0`. Each step the eager `_post_process` result is written to slot 0; history reads
        # gather from here so the projection is computed once at write time (correct for stateful `_post_process`)
        # and history reads avoid re-projecting older slots. For identity `_post_process` the intermediate ring is
        # already in return space, so no extra ring is needed.
        self._ground_truth_return_timeline_ring: dict[type["Sensor"], TensorRingBuffer] = {}
        self._measured_return_timeline_ring: dict[type["Sensor"], TensorRingBuffer] = {}
        # Per-class precomputed history index tensor [0, 1, ..., max_history-1]. Used to fancy-index the rings on
        # history reads.
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

        # Per-class intermediate / return dtypes come from `_get_intermediate_dtype` / `_get_cache_dtype`. Dtype is
        # class-uniform by design (the per-class slice into the per-dtype intermediate buffer must be contiguous, so
        # all instances of a class share one dtype). Shape is per-instance via `_get_intermediate_format` /
        # `_get_return_format` and contributes to the class slice size below.
        cache_size_per_dtype: dict[torch.dtype, int] = {}
        delay_depth_per_dtype: dict[torch.dtype, int] = {}
        max_history_per_dtype: dict[torch.dtype, int] = {}
        intermediate_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        return_dtype_by_class: dict[type["Sensor"], torch.dtype] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            intermediate_dtype = sensor_cls._get_intermediate_dtype()
            return_dtype = sensor_cls._get_cache_dtype()
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
        self._ground_truth_return_timeline_ring.clear()
        self._measured_return_timeline_ring.clear()
        self._hist_idx_by_class.clear()

        dtype_uses_measured: dict[torch.dtype, bool] = {}
        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = intermediate_dtype_by_class[sensor_cls]
            cur = dtype_uses_measured.get(dtype, False)
            dtype_uses_measured[dtype] = cur or sensor_cls.uses_measured_pipeline

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
            if sensor_cls._post_process.__func__ is not Sensor._post_process.__func__:
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

        # Per-class precomputed history index + (overridden `_post_process` only) per-class return-space rings. The
        # return rings record each step's eager `_post_process` snapshot so history reads gather post-processed
        # values directly, instead of re-projecting older slots from the intermediate ring (which would be wrong for
        # stateful `_post_process`).
        for sensor_cls, cls_max_history in self._max_history_by_class.items():
            if cls_max_history == 0:
                continue
            intermediate_dtype = intermediate_dtype_by_class[sensor_cls]
            self._hist_idx_by_class[sensor_cls] = torch.arange(cls_max_history, device=gs.device, dtype=torch.int32)
            if sensor_cls._post_process.__func__ is not Sensor._post_process.__func__:
                cache_slice = self._cache_slices_by_type[sensor_cls]
                cls_size = cache_slice.stop - cache_slice.start
                return_dtype = return_dtype_by_class[sensor_cls]
                ring_n = max(delay_depth_per_dtype.get(intermediate_dtype, 1), cls_max_history)
                ring_shape = (self._sim._B, cls_size)
                shared_idx = self._ground_truth_timeline_ring[intermediate_dtype]._idx
                self._ground_truth_return_timeline_ring[sensor_cls] = TensorRingBuffer(
                    ring_n, ring_shape, dtype=return_dtype, idx=shared_idx
                )
                self._measured_return_timeline_ring[sensor_cls] = TensorRingBuffer(
                    ring_n, ring_shape, dtype=return_dtype, idx=shared_idx
                )

        for sensor_cls, sensors in self._sensors_by_type.items():
            for sensor in sensors:
                sensor.build()
                sensor._is_built = True

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
        # already cleared via the intermediate-cache zero above. Same logic for the per-class return-space rings.
        for sensor_cls, return_cache in self._return_cache.items():
            if sensor_cls._post_process.__func__ is not Sensor._post_process.__func__:
                return_cache[envs_idx] = 0
                self._ground_truth_return_cache[sensor_cls][envs_idx] = 0
        for ring in self._ground_truth_return_timeline_ring.values():
            ring.buffer[:, envs_idx] = 0
        for ring in self._measured_return_timeline_ring.values():
            ring.buffer[:, envs_idx] = 0

        for sensor_cls, sensors in self._sensors_by_type.items():
            dtype = sensor_cls._get_intermediate_dtype()
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
            dtype = sensor_cls._get_intermediate_dtype()
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
            if sensor_cls._post_process.__func__ is not Sensor._post_process.__func__:
                gt_return = self._ground_truth_return_cache[sensor_cls]
                gt_return.copy_(sensor_cls._post_process(self._sensors_metadata[sensor_cls], ground_truth_slice.T))
                # Record both eager projections in the per-class return-space rings so history reads pull
                # post-processed snapshots directly (correct for stateful `_post_process`). Identity-projection
                # classes don't allocate these rings; their history reads gather from the intermediate ring instead.
                if sensor_cls in self._ground_truth_return_timeline_ring:
                    self._ground_truth_return_timeline_ring[sensor_cls].set(gt_return)
                    self._measured_return_timeline_ring[sensor_cls].set(self._return_cache[sensor_cls])

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
            sensor_hist = self._gather_history(sensor_cls, history_length, is_ground_truth)
            sensor_slice = slice(rel_start, rel_start + sensor._cache_size)
            sensor_hist = sensor_hist[:, :, sensor_slice]
            blocks = [sensor_hist[..., rel_slice].flatten(1, 2) for rel_slice in sensor._cache_slices]
            if len(blocks) == 1:
                return blocks[0]
            return torch.cat(blocks, dim=1)

        # Pure view into the per-class return cache. Eager `_post_process` already populated it during step().
        return_cache = (
            self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
        )
        return return_cache[:, rel_start : rel_start + sensor._cache_size]

    def _gather_history(self, sensor_cls: type["Sensor"], history_length: int, is_ground_truth: bool) -> torch.Tensor:
        # Gather the last `history_length` snapshots for the whole class into a fresh `(B, H, cls_size)` tensor.
        # For overridden `_post_process` we read from the per-class return-space ring (snapshots are already
        # post-processed at write time). For identity `_post_process` we read from the per-dtype intermediate ring,
        # since intermediate == return there.
        hist_idx = self._hist_idx_by_class[sensor_cls][:history_length]
        if sensor_cls in self._ground_truth_return_timeline_ring:
            ring = (
                self._ground_truth_return_timeline_ring[sensor_cls]
                if is_ground_truth
                else self._measured_return_timeline_ring[sensor_cls]
            )
            return ring.at(hist_idx).transpose(0, 1)
        dtype = sensor_cls._get_intermediate_dtype()
        cache_slice = self._cache_slices_by_type[sensor_cls]
        ring = self._ground_truth_timeline_ring[dtype] if is_ground_truth else self._measured_timeline_ring[dtype]
        return ring.at(hist_idx, slice(None), cache_slice).transpose(0, 1)

    def read_sensors(
        self,
        entity_idx: int | None = None,
        envs_idx=None,
        is_ground_truth: bool = False,
    ) -> dict[int, torch.Tensor]:
        """
        Read the latest data of every sensor class in scope as a single tensor per class.

        Always returns a fresh tensor per class, independent of the internal sensor storage; the caller is free to
        mutate the result.

        Parameters
        ----------
        entity_idx : int | None
            - None (default): include every sensor in the scene.
            - k >= 0: include only sensors whose `entity_idx == k`.
            - -1: include only static sensors (those not attached to any entity).
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.
        is_ground_truth : bool
            When True, return ground-truth tensors instead of measured tensors.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from sensor-type tag (`gs.sensors.types.<Name>`) to a tensor of shape
            (B, [history,] class_or_entity_cache_size). For sensors without history, the history
            dimension is omitted.
        """
        # Sanitize envs_idx to a 1D tensor so fancy-indexing the batch axis always allocates a fresh tensor; this
        # is what gives the function its mutation-safe contract.
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
                sensor_hist = self._gather_history(sensor_cls, cls_max_history, is_ground_truth)
                tensor = sensor_hist[env_index, :, within_cls_slice]
            else:
                return_cache = (
                    self._ground_truth_return_cache[sensor_cls] if is_ground_truth else self._return_cache[sensor_cls]
                )
                tensor = return_cache[env_index, within_cls_slice]

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
