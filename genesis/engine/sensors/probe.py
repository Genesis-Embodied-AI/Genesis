from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

if TYPE_CHECKING:
    from genesis.options.sensors.options import SensorOptions
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
    probe_radii_noise: torch.Tensor = make_tensor_field((0,))
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Class-level scratch for the kernel-writes-(cols, B) -> ring-slot-(B, cols) transpose-copy pattern. Lazy-allocated
    # on first hot-path call to avoid per-step `torch.empty_like` allocations.
    measured_scratch_T: torch.Tensor = make_tensor_field((0, 0))


ProbeSensorSharedMetadataT = TypeVar("ProbeSensorSharedMetadataT", bound=ProbeSensorMetadataMixin)


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
        # `_get_return_format` runs inside `super().__init__`, so `_probe_local_pos` / `_n_probes` must already be set.
        self._probe_local_pos = torch.tensor(options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._n_probes = int(np.prod(self._probe_local_pos.shape[:-1]))
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
            probe_radii = torch.tensor(self._options.probe_radius, dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.probe_radii = concat_with_tensor(
            self._shared_metadata.probe_radii, probe_radii, expand=(self._n_probes,)
        )
        self._shared_metadata.probe_radii_noise = concat_with_tensor(
            self._shared_metadata.probe_radii_noise,
            torch.full((self._n_probes,), self._options.probe_radius_noise, dtype=gs.tc_float, device=gs.device),
            expand=(self._n_probes,),
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
        self._probe_local_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        if self._probe_local_normal.ndim == 1:
            self._probe_local_normal = self._probe_local_normal.expand(self._n_probes, 3).contiguous()

    def build(self) -> None:
        super().build()
        self._shared_metadata.probe_local_normal = concat_with_tensor(
            self._shared_metadata.probe_local_normal, self._probe_local_normal, expand=(self._n_probes, 3)
        )

    @property
    def probe_local_normal(self) -> torch.Tensor:
        return self._probe_local_normal
