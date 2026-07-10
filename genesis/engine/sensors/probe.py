from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

if TYPE_CHECKING:
    from genesis.options.sensors.options import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def func_noised_probe_radius(probe_radius: float, probe_radius_noise: float) -> float:
    radius = probe_radius
    if probe_radius_noise > gs.EPS:
        radius = qd.max(
            gs.qd_float(0.0),
            probe_radius + (qd.random(gs.qd_float) * gs.qd_float(2.0) - gs.qd_float(1.0)) * probe_radius_noise,
        )
    return radius


@dataclass
class ProbeSensorMetadataMixin:
    """Shared metadata for sensors that register multiple probes in a fused layout."""

    total_n_probes: int = 0
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_radii: torch.Tensor = make_tensor_field((0,))
    # No-op scaffolding kept for the kernels' dual-radius / gain machinery: always zeros / ones so the
    # measured branch equals ground truth. ``has_any_*`` stay False.
    probe_radii_noise: torch.Tensor = make_tensor_field((0,))
    has_any_probe_radius_noise: bool = False
    has_any_probe_gain: bool = False
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)

    measured_scratch_T: torch.Tensor = make_tensor_field((0, 0))

    probe_gains: torch.Tensor = make_tensor_field((0, 0))


ProbeSensorSharedMetadataT = TypeVar("ProbeSensorSharedMetadataT", bound=ProbeSensorMetadataMixin)


def get_measured_bufs(
    shared_metadata: "ProbeSensorMetadataMixin",
    current_ground_truth_data_T: torch.Tensor,
    measured_data_timeline: "TensorRingBuffer",
) -> tuple[torch.Tensor, torch.Tensor]:
    current_ground_truth_data_T.zero_()
    measured_slot = measured_data_timeline.at(0, copy=False)
    measured_slot.zero_()
    if shared_metadata.measured_scratch_T.shape != current_ground_truth_data_T.shape:
        shared_metadata.measured_scratch_T = torch.empty_like(current_ground_truth_data_T)
    return measured_slot, shared_metadata.measured_scratch_T


class ProbeSensorMixin(Generic[ProbeSensorSharedMetadataT]):
    """Shared logic for registering this sensor's probes in ``ProbeSensorMetadataMixin`` fields."""

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        # `_get_return_format` runs inside `super().__init__`, so the probe layout fields must be set first.
        raw_pos = torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._probe_layout_shape = raw_pos.shape[:-1]
        self._n_probes = int(np.prod(self._probe_layout_shape))
        self._probe_local_pos = raw_pos.reshape(self._n_probes, 3).contiguous()
        self._debug_objects: list = []
        super().__init__(options, idx, shared_context, shared_metadata, manager)

    def build(self) -> None:
        super().build()
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start, self._shared_metadata.total_n_probes, expand=(1,)
        )
        self._shared_metadata.total_n_probes += self._n_probes
        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor, self._n_probes, expand=(1,)
        )
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0,
            expand=(1,),
        )
        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((self._n_probes,), self._idx, dtype=gs.tc_int, device=gs.device),
            expand=(self._n_probes,),
        )
        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(self._n_probes, 3)
        )
        if isinstance(self._options.probe_radius, float):
            probe_radii = torch.full((self._n_probes,), self._options.probe_radius, dtype=gs.tc_float, device=gs.device)
        else:
            probe_radii = torch.tensor(self._options.probe_radius, dtype=gs.tc_float, device=gs.device).reshape(
                self._n_probes
            )
        self._shared_metadata.probe_radii = concat_with_tensor(
            self._shared_metadata.probe_radii, probe_radii, expand=(self._n_probes,)
        )
        # No-op scaffolding (always zeros): the kernels' dual-radius machinery reads this; zero noise means the
        # measured radius equals the nominal radius (measured branch == ground truth).
        self._shared_metadata.probe_radii_noise = concat_with_tensor(
            self._shared_metadata.probe_radii_noise,
            torch.zeros((self._n_probes,), dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
        )
        # No-op scaffolding (always ones): the kernels multiply the measured-branch signal by this per-probe gain.
        B = self._manager._sim._B
        self._shared_metadata.probe_gains = concat_with_tensor(
            self._shared_metadata.probe_gains,
            torch.ones((B, self._n_probes), dtype=gs.tc_float, device=gs.device),
            expand=(B, self._n_probes),
            dim=1,
        )

    @property
    def probe_local_pos(self) -> torch.Tensor:
        return self._probe_local_pos

    @property
    def n_probes(self) -> int:
        return self._n_probes

    def _compute_probes_world_pos(self, context: "RasterizerContext"):
        """
        Transform probe positions from link-local to world frame for debug drawing.

        Returns ``(envs_idx, n_debug_envs, env_offsets, probe_world_flat)``. ``probe_world_flat`` is ``(n_debug_envs *
        n_probes, 3)`` with env-offset already added. Assumes ``self._link`` is set (consumer inherits
        ``RigidSensorMixin``).
        """
        if self._manager._sim.n_envs > 0:
            envs_idx = list(context.rendered_envs_idx)
            n_debug_envs = len(envs_idx)
            env_offsets = context.scene.envs_offset[np.asarray(envs_idx, dtype=gs.np_int)]
            link_pos = self._link.get_pos(envs_idx, relative=False)[:, None, :]
            link_quat = self._link.get_quat(envs_idx, relative=False)[:, None, :]
            probe_world = gu.transform_by_trans_quat(
                self._probe_local_pos.reshape(-1, 3)[None, :, :], link_pos, link_quat
            )
            probe_world = tensor_to_array(probe_world) + env_offsets[:, None, :]
        else:
            envs_idx = None
            n_debug_envs = 1
            env_offsets = None
            link_pos = self._link.get_pos(envs_idx, relative=False).reshape(3)
            link_quat = self._link.get_quat(envs_idx, relative=False).reshape(4)
            probe_world = tensor_to_array(
                gu.transform_by_trans_quat(self._probe_local_pos.reshape(-1, 3), link_pos, link_quat)
            )
        return envs_idx, n_debug_envs, env_offsets, probe_world.reshape(-1, 3)

    def _draw_probe_spheres(
        self,
        context: "RasterizerContext",
        probe_world: np.ndarray,
        rgb,
        probe_radii: np.ndarray | None = None,
        probe_radii_noise: np.ndarray | None = None,
    ) -> list:
        """
        Draw a small opaque center sphere and a translucent outer sensing sphere at each ``probe_world`` position.

        ``probe_world`` is ``(N, 3)`` (already tiled over rendered envs). ``probe_radii`` and ``probe_radii_noise``
        are the matching ``(N,)`` per-position nominal sensing radius and additive uniform noise; both default to
        the per-probe values from shared metadata, tiled to match ``probe_world``. When noise is positive, each
        outer sphere is drawn at a fresh sample ``clip(r + U(-noise, +noise), 0, inf)`` rounded to the nearest
        ``noise`` magnitude so the unique-radius batches stay small. Returns the created debug objects.
        """
        options = self._options
        rgb = tuple(float(c) for c in rgb)
        center_color = (*rgb, 1.0)
        objs = [
            context.draw_debug_spheres(
                poss=probe_world,
                radius=float(options.debug_probe_center_radius),
                color=center_color,
            )
        ]
        if options.debug_probe_sphere_opacity <= 0.0:
            return objs
        outer_color = (*rgb, float(options.debug_probe_sphere_opacity))
        probe_start = int(self._shared_metadata.sensor_probe_start[self._idx].item())
        probe_slice = slice(probe_start, probe_start + self._n_probes)
        n_tile = probe_world.shape[0] // self._n_probes if self._n_probes > 0 else 0
        n_tile = max(n_tile, 1)
        if probe_radii is None:
            per_probe = tensor_to_array(self._shared_metadata.probe_radii[probe_slice]).reshape(-1)
            probe_radii = np.tile(per_probe, n_tile)
        if probe_radii_noise is None:
            per_probe_noise = tensor_to_array(self._shared_metadata.probe_radii_noise[probe_slice]).reshape(-1)
            probe_radii_noise = np.tile(per_probe_noise, n_tile)
        nz = probe_radii_noise > 0.0
        if nz.any():
            jitter = np.random.uniform(-1.0, 1.0, size=probe_radii.shape) * probe_radii_noise
            noisy = np.maximum(0.0, probe_radii + jitter)
            rounded = probe_radii.astype(float, copy=True)
            rounded[nz] = np.round(noisy[nz] / probe_radii_noise[nz]) * probe_radii_noise[nz]
            probe_radii = rounded
        for r in np.unique(probe_radii):
            if r <= 0.0:
                continue
            mask = probe_radii == r
            objs.append(
                context.draw_debug_spheres(
                    poss=probe_world[mask],
                    radius=float(r),
                    color=outer_color,
                )
            )
        return objs

    def _draw_debug_probes(
        self,
        context: "RasterizerContext",
        color_groups_fn: Callable[[list[int] | None], list[tuple]] | None = None,
    ) -> tuple[list[int] | None, int, np.ndarray | None]:
        """
        Generic per-probe debug renderer.

        Clears prior debug objects, then for each provided color group draws the two-sphere marker (small opaque
        center + translucent outer sensing sphere) on the selected probe positions.

        ``color_groups_fn(envs_idx)`` returns a list of ``(rgb, mask)`` pairs, where ``rgb`` is a length-3 sequence
        and ``mask`` is a flat ``(n_debug_envs * n_probes,)`` bool array (or tensor castable to bool) selecting
        which probe positions take that color. Passing ``None`` falls back to a single group covering every probe
        in the sensor's ``debug_probe_color`` (no contact-state assumption -- usable by any probe sensor).

        Returns ``(envs_idx, n_debug_envs, env_offsets)`` so subclasses can extend the drawing with additional
        debug geometry without recomputing the env layout.
        """
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        envs_idx, n_debug_envs, env_offsets, probe_world = self._compute_probes_world_pos(context)
        probe_start = int(self._shared_metadata.sensor_probe_start[self._idx].item())
        probe_slice = slice(probe_start, probe_start + self._n_probes)
        n_tile = max(n_debug_envs, 1)
        radii_tiled = np.tile(tensor_to_array(self._shared_metadata.probe_radii[probe_slice]).reshape(-1), n_tile)
        noise_tiled = np.tile(tensor_to_array(self._shared_metadata.probe_radii_noise[probe_slice]).reshape(-1), n_tile)
        if color_groups_fn is None:
            groups = [(self._options.debug_probe_color, np.ones(probe_world.shape[0], dtype=bool))]
        else:
            groups = color_groups_fn(envs_idx)
        for rgb, mask in groups:
            mask_arr = tensor_to_array(mask, dtype=bool).reshape(-1)
            (probes_idx,) = np.nonzero(mask_arr)
            if probes_idx.size == 0:
                continue
            self._debug_objects.extend(
                self._draw_probe_spheres(
                    context, probe_world[probes_idx], rgb, radii_tiled[probes_idx], noise_tiled[probes_idx]
                )
            )
        return envs_idx, n_debug_envs, env_offsets

    def _tactile_color_groups_fn(
        self, get_is_contact_flat: Callable[[list[int] | None], object]
    ) -> Callable[[list[int] | None], list[tuple]]:
        """
        Build a ``color_groups_fn`` for the common tactile split: not-in-contact probes get ``debug_probe_color``
        and in-contact probes get ``debug_contact_color``.

        The sensor's options must expose ``debug_contact_color`` (i.e. inherit ``TactileProbeSensorOptionsMixin``).
        """

        def fn(envs_idx):
            is_contact = tensor_to_array(get_is_contact_flat(envs_idx), dtype=bool).reshape(-1)
            return [
                (self._options.debug_probe_color, ~is_contact),
                (self._options.debug_contact_color, is_contact),
            ]

        return fn


@dataclass
class ProbesWithNormalSensorMetadataMixin(ProbeSensorMetadataMixin):
    """Shared metadata for probe sensors that also carry a per-probe outward normal."""

    probe_local_normal: torch.Tensor = make_tensor_field((0, 3))


ProbesWithNormalSensorSharedMetadataT = TypeVar(
    "ProbesWithNormalSensorSharedMetadataT", bound=ProbesWithNormalSensorMetadataMixin
)


class ProbesWithNormalSensorMixin(ProbeSensorMixin[ProbesWithNormalSensorSharedMetadataT]):
    """Probe sensor whose probes carry a per-probe outward normal in link-local frame."""

    def __init__(
        self,
        options: "SensorOptions",
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        raw_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        if raw_normal.ndim == 1:
            self._probe_local_normal = raw_normal.expand(self._n_probes, 3).contiguous()
        else:
            self._probe_local_normal = raw_normal.reshape(self._n_probes, 3).contiguous()

    def build(self) -> None:
        super().build()
        self._shared_metadata.probe_local_normal = concat_with_tensor(
            self._shared_metadata.probe_local_normal, self._probe_local_normal, expand=(self._n_probes, 3)
        )

    @property
    def probe_local_normal(self) -> torch.Tensor:
        return self._probe_local_normal
