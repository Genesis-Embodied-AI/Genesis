from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.options.sensors import TemperatureGrid as TemperatureGridOptions
from genesis.options.sensors import TemperatureProperties
from genesis.utils import mesh as mu
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m²·K⁴)
KELVIN_OFFSET = 273.15

TEMP_MIN_CLAMP = -50.0
TEMP_MAX_CLAMP = 500.0


def _compute_K2_rfft3(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> torch.Tensor:
    """Squared wave numbers for 3D real FFT: K2[i,j,k] = (2*pi*kx)^2 + (2*pi*ky)^2 + (2*pi*kz)^2 with rfft layout."""
    kx = torch.fft.fftfreq(nx, d=dx, device=gs.device).to(gs.tc_float)
    ky = torch.fft.fftfreq(ny, d=dy, device=gs.device).to(gs.tc_float)
    kz = torch.fft.rfftfreq(nz, d=dz, device=gs.device).to(gs.tc_float)
    two_pi = 2.0 * np.pi
    K2 = (two_pi * kx).reshape(-1, 1, 1) ** 2
    K2 = K2 + (two_pi * ky).reshape(1, -1, 1) ** 2
    K2 = K2 + (two_pi * kz).reshape(1, 1, -1) ** 2
    # Avoid division by zero at DC (rfftn DC is at [0,0,0])
    K2[0, 0, 0] = max(K2[0, 0, 0].item(), 1e-12)
    return K2


def _compute_surface_mask(nx: int, ny: int, nz: int) -> torch.Tensor:
    """Boolean mask of boundary voxels (at least one face on grid boundary). Shape (nx, ny, nz)."""
    ix = torch.arange(nx, device=gs.device)
    iy = torch.arange(ny, device=gs.device)
    iz = torch.arange(nz, device=gs.device)
    on_boundary_x = (ix == 0) | (ix == nx - 1)
    on_boundary_y = (iy == 0) | (iy == ny - 1)
    on_boundary_z = (iz == 0) | (iz == nz - 1)
    sx = on_boundary_x.reshape(-1, 1, 1).expand(nx, ny, nz)
    sy = on_boundary_y.reshape(1, -1, 1).expand(nx, ny, nz)
    sz = on_boundary_z.reshape(1, 1, -1).expand(nx, ny, nz)
    return (sx | sy | sz).to(gs.tc_float)


def _apply_diffusion_and_heat_generation(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_grid_size: torch.Tensor,
    sensor_rho_cp: torch.Tensor,
    sensor_heat_generation: list[torch.Tensor | None],
    sensor_voxel_size: torch.Tensor,
    expanded_links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_material_properties: torch.Tensor,
    sensor_K2: list[torch.Tensor],
    dt: float,
    output: torch.Tensor,
) -> None:
    """Batched FFT semi-implicit diffusion with mirror padding (Neumann BC, no wrap-around)."""
    n_sensors = sensor_cache_start.shape[0]
    for i_s in range(n_sensors):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        nx, ny, nz = sensor_grid_size[i_s].tolist()
        rho_cp = sensor_rho_cp[i_s].item()
        mat_idx = link_to_material_idx[expanded_links_idx[i_s].item()].item()
        k = link_material_properties[mat_idx, 1].item()
        alpha = k / rho_cp
        T = output[:, start : start + size].reshape(-1, nx, ny, nz)
        # Mirror-pad to (2*nx, 2*ny, 2*nz) for zero-flux (Neumann) boundaries; avoids FFT wrap-around.
        T_x = torch.cat([T, torch.flip(T, dims=(1,))], dim=1)
        T_xy = torch.cat([T_x, torch.flip(T_x, dims=(2,))], dim=2)
        T_pad = torch.cat([T_xy, torch.flip(T_xy, dims=(3,))], dim=3)
        T_hat = torch.fft.rfftn(T_pad, dim=(-3, -2, -1))
        T_hat = T_hat / (1.0 + dt * alpha * sensor_K2[i_s])
        T_pad = torch.fft.irfftn(T_hat, s=(2 * nx, 2 * ny, 2 * nz), dim=(-3, -2, -1)).real
        T = T_pad[:, :nx, :ny, :nz]
        output[:, start : start + size] = T.reshape(output.shape[0], -1)

        # Add internal heat generation (W/m² -> Q_vol = Q_surface / dz).
        q = sensor_heat_generation[i_s]
        if q is not None:
            dz = max(sensor_voxel_size[i_s, 2].item(), gs.EPS)
            Q_vol = q.reshape(-1) / dz
            delta_T = dt * Q_vol / rho_cp
            output[:, start : start + size] += delta_T.unsqueeze(0).expand(output.shape[0], -1)


@qd.kernel
def _kernel_contact_heat(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    sensor_links_idx: qd.types.ndarray(),
    sensor_aabb_min: qd.types.ndarray(),
    sensor_voxel_size: qd.types.ndarray(),
    sensor_grid_size: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    voxel_volume: qd.types.ndarray(),
    rho_cp: qd.types.ndarray(),
    depth_weight: qd.types.ndarray(),
    link_to_material_idx: qd.types.ndarray(),
    link_material_properties: qd.types.ndarray(),
    dt: gs.qd_float,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    n_batches = output.shape[0]
    n_sensors = sensor_links_idx.shape[0]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        sensor_link_idx = sensor_links_idx[i_s]
        start = sensor_cache_start[i_s]
        nx = sensor_grid_size[i_s, 0]
        ny = sensor_grid_size[i_s, 1]
        nz = sensor_grid_size[i_s, 2]
        vol = voxel_volume[i_s] + eps
        rcp = rho_cp[i_s] + eps
        dw = depth_weight[i_s]
        mat_idx_sensor = link_to_material_idx[sensor_link_idx]
        if mat_idx_sensor < 0:
            continue

        k_sensor = link_material_properties[mat_idx_sensor, 1]
        amin = qd.math.vec3(sensor_aabb_min[i_s, 0], sensor_aabb_min[i_s, 1], sensor_aabb_min[i_s, 2])
        vs = qd.math.vec3(
            sensor_voxel_size[i_s, 0] + eps,
            sensor_voxel_size[i_s, 1] + eps,
            sensor_voxel_size[i_s, 2] + eps,
        )
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            if la != sensor_link_idx and lb != sensor_link_idx:
                continue
            other_link = lb if la == sensor_link_idx else la
            mat_other = link_to_material_idx[other_link]
            if mat_other < 0:
                mat_other = 0
            T_other = link_material_properties[mat_other, 0]
            k_other = link_material_properties[mat_other, 1]
            k_eff = gs.qd_float(2.0) * k_sensor * k_other / (k_sensor + k_other + eps)
            depth = collider_state.contact_data.penetration[i_c, i_b]
            depth_factor = gs.qd_float(1.0) + dw * depth
            p_world = collider_state.contact_data.pos[i_c, i_b]
            link_pos = links_state.pos[sensor_link_idx, i_b]
            link_quat = links_state.quat[sensor_link_idx, i_b]
            p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
            u_x = (p_local.x - amin.x) / vs.x
            u_y = (p_local.y - amin.y) / vs.y
            u_z = (p_local.z - amin.z) / vs.z
            ix = min(max(0, int(u_x)), nx - 1)
            iy = min(max(0, int(u_y)), ny - 1)
            iz = min(max(0, int(u_z)), nz - 1)
            cell_idx = ix * (ny * nz) + iy * nz + iz
            T_cell = output[i_b, start + cell_idx]
            area = max(depth * depth, eps)
            flux = k_eff * depth_factor * (T_other - T_cell)
            Q_vol = flux * area / vol
            delta_T = dt * Q_vol / rcp
            output[i_b, start + cell_idx] = T_cell + delta_T


def _apply_radiation_convection(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_surface_mask: list[torch.Tensor],
    sensor_rho_cp: torch.Tensor,
    sensor_voxel_volume: torch.Tensor,
    sensor_ambient_temperature: torch.Tensor,
    sensor_convection_coefficient: torch.Tensor,
    expanded_links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_material_properties: torch.Tensor,
    dt: float,
    output: torch.Tensor,
) -> None:
    """Radiation + convection on surface voxels. Batched over envs."""
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        mask = sensor_surface_mask[i_s].reshape(-1)
        rho_cp = sensor_rho_cp[i_s].item()
        vol = max(sensor_voxel_volume[i_s].item(), gs.EPS)
        ambient = sensor_ambient_temperature[i_s].item()
        mat_idx = link_to_material_idx[expanded_links_idx[i_s].item()].item()
        emiss = link_material_properties[mat_idx, 4].item()
        h = sensor_convection_coefficient[i_s].item()
        T_amb_K = ambient + KELVIN_OFFSET
        denom = rho_cp * vol
        T_flat = output[:, start : start + size]
        T_K = T_flat + KELVIN_OFFSET
        q_rad = emiss * STEFAN_BOLTZMANN * (T_K**4 - T_amb_K**4)
        q_conv = h * (T_flat - ambient)
        output[:, start : start + size] -= dt * (q_rad + q_conv) / denom * mask


def _apply_T_measured_filter(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_tau: torch.Tensor,
    T_measured: torch.Tensor,
    dt: float,
    output: torch.Tensor,
) -> None:
    """T_measured += (dt/tau)*(T - T_measured); if tau<=0 then T_measured = T. Batched over envs."""
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        tau = sensor_tau[i_s].item()
        T_slice = output[:, start : start + size]
        T_meas_slice = T_measured[:, start : start + size]
        if tau > 0:
            alpha = dt / tau
            T_measured[:, start : start + size] = T_meas_slice + alpha * (T_slice - T_meas_slice)
        else:
            T_measured[:, start : start + size] = T_slice


@dataclass
class TemperatureGridSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all temperature grid sensors."""

    expanded_links_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_aabb_min: torch.Tensor = make_tensor_field((0, 3))
    sensor_aabb_extent: torch.Tensor = make_tensor_field((0, 3))
    sensor_grid_size: torch.Tensor = make_tensor_field((0, 3), dtype=gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_base_temperature: torch.Tensor = make_tensor_field((0,))
    sensor_voxel_size: torch.Tensor = make_tensor_field((0, 3))
    sensor_voxel_volume: torch.Tensor = make_tensor_field((0,))
    sensor_rho_cp: torch.Tensor = make_tensor_field((0,))
    sensor_ambient_temperature: torch.Tensor = make_tensor_field((0,))
    sensor_convection_coefficient: torch.Tensor = make_tensor_field((0,))
    sensor_tau: torch.Tensor = make_tensor_field((0,))
    sensor_contact_depth_weight: torch.Tensor = make_tensor_field((0,))
    sensor_T_measured: torch.Tensor = make_tensor_field((0, 0))
    sensor_K2: list[torch.Tensor] = field(default_factory=list)
    sensor_surface_mask: list[torch.Tensor] = field(default_factory=list)
    sensor_heat_generation: list[torch.Tensor | None] = field(default_factory=list)

    link_to_material_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    link_material_properties: torch.Tensor = make_tensor_field((0, 5), dtype=gs.tc_float)
    properties_dict: dict[int, TemperatureProperties] = field(default_factory=dict)


@register_sensor(TemperatureGridOptions, TemperatureGridSensorMetadata, tuple)
class TemperatureGridSensor(
    RigidSensorMixin[TemperatureGridSensorMetadata],
    NoisySensorMixin[TemperatureGridSensorMetadata],
    Sensor[TemperatureGridSensorMetadata],
):
    def __init__(
        self,
        sensor_options: TemperatureGridOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self._link: "RigidLink | None" = None
        self._debug_objects: list = []
        self._debug_t_min: float = self._options.debug_temperature_range[0]
        self._debug_t_range: float = self._options.debug_temperature_range[1] - self._debug_t_min
        # Precomputed for draw_debug: grid shape and cell-center offsets in grid coords (i+0.5, j+0.5, k+0.5)
        nx, ny, nz = self._options.grid_size
        self._debug_nx, self._debug_ny, self._debug_nz = nx, ny, nz
        xs = np.arange(nx, dtype=gs.np_float) + 0.5
        ys = np.arange(ny, dtype=gs.np_float) + 0.5
        zs = np.arange(nz, dtype=gs.np_float) + 0.5
        self._debug_cell_offsets: np.ndarray = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    def build(self):
        super().build()

        solver = self._shared_metadata.solver
        entity = solver.entities[self._options.entity_idx]
        link_idx_global = self._options.link_idx_local + entity.link_start
        self._link = entity.links[self._options.link_idx_local]

        # Expanded links idx (one per sensor) for compatibility
        self._shared_metadata.expanded_links_idx = concat_with_tensor(
            self._shared_metadata.expanded_links_idx, link_idx_global, expand=(1,), dim=0
        )

        if self._shared_metadata.link_to_material_idx.shape[0] == 0:
            self._shared_metadata.link_to_material_idx = torch.full(
                (solver.n_links,), -1, dtype=gs.tc_int, device=gs.device
            )

        self._shared_metadata.properties_dict.update(self._options.properties_dict)
        if len(self._shared_metadata.properties_dict) > self._shared_metadata.link_material_properties.shape[0]:
            self._shared_metadata.link_material_properties = torch.empty(
                (len(self._shared_metadata.properties_dict), 5), dtype=gs.tc_float, device=gs.device
            )
            # -1 in link_to_material_idx means invalid, 0 uses the default properties
            self._shared_metadata.link_to_material_idx[:] = 0 if -1 in self._shared_metadata.properties_dict else -1
            # sort properties_dict by link index to ensure default properties are at index 0
            for i, (prop_idx, props) in enumerate(
                sorted(self._shared_metadata.properties_dict.items(), key=lambda x: x[0])
            ):
                self._shared_metadata.link_material_properties[i] = torch.tensor(
                    props, dtype=gs.tc_float, device=gs.device
                )
                if prop_idx >= 0:
                    self._shared_metadata.link_to_material_idx[prop_idx] = i

        # Per-sensor: AABB in link-local frame, grid size, base temp, cache start
        aabb_world = self._link.get_AABB()
        aabb_min_w = aabb_world[..., 0, :]  # (3,) or (n_envs, 3)
        aabb_max_w = aabb_world[..., 1, :]
        link_pos = self._link.get_pos()
        link_quat = self._link.get_quat()
        # Flatten to (N, 3) / (N, 4) and take first env so shapes match for inv_transform
        if aabb_min_w.ndim == 1:
            aabb_min_w = aabb_min_w.unsqueeze(0)
            aabb_max_w = aabb_max_w.unsqueeze(0)
        if link_pos.ndim == 1:
            link_pos = link_pos.unsqueeze(0)
            link_quat = link_quat.unsqueeze(0)
        aabb_min_local = gu.inv_transform_by_trans_quat(aabb_min_w[0], link_pos[0], link_quat[0])
        aabb_max_local = gu.inv_transform_by_trans_quat(aabb_max_w[0], link_pos[0], link_quat[0])
        aabb_extent = aabb_max_local - aabb_min_local

        self._shared_metadata.sensor_aabb_min = concat_with_tensor(
            self._shared_metadata.sensor_aabb_min,
            torch.tensor(aabb_min_local, dtype=gs.tc_float, device=gs.device),
            expand=(1, 3),
            dim=0,
        )
        self._shared_metadata.sensor_aabb_extent = concat_with_tensor(
            self._shared_metadata.sensor_aabb_extent,
            torch.tensor(aabb_extent, dtype=gs.tc_float, device=gs.device),
            expand=(1, 3),
            dim=0,
        )

        self._shared_metadata.sensor_grid_size = concat_with_tensor(
            self._shared_metadata.sensor_grid_size,
            torch.tensor(self._options.grid_size, dtype=gs.tc_int, device=gs.device),
            expand=(1, 3),
            dim=0,
        )

        props = self._options.properties_dict.get(link_idx_global) or self._options.properties_dict.get(-1)
        self._shared_metadata.sensor_base_temperature = concat_with_tensor(
            self._shared_metadata.sensor_base_temperature,
            torch.tensor([props.base_temperature], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )

        nx, ny, nz = self._options.grid_size
        voxel_size = aabb_extent / torch.tensor([nx, ny, nz], dtype=gs.tc_float, device=gs.device)
        voxel_volume = voxel_size.prod()
        rho_cp = props.density * props.specific_heat
        self._shared_metadata.sensor_voxel_size = concat_with_tensor(
            self._shared_metadata.sensor_voxel_size,
            torch.tensor(voxel_size, dtype=gs.tc_float, device=gs.device),
            expand=(1, 3),
            dim=0,
        )
        self._shared_metadata.sensor_voxel_volume = concat_with_tensor(
            self._shared_metadata.sensor_voxel_volume,
            torch.tensor([voxel_volume], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.sensor_rho_cp = concat_with_tensor(
            self._shared_metadata.sensor_rho_cp,
            torch.tensor([rho_cp], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.sensor_ambient_temperature = concat_with_tensor(
            self._shared_metadata.sensor_ambient_temperature,
            torch.tensor([self._options.ambient_temperature], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.sensor_convection_coefficient = concat_with_tensor(
            self._shared_metadata.sensor_convection_coefficient,
            torch.tensor([self._options.convection_coefficient], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.sensor_tau = concat_with_tensor(
            self._shared_metadata.sensor_tau,
            torch.tensor([self._options.sensor_time_constant], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.sensor_contact_depth_weight = concat_with_tensor(
            self._shared_metadata.sensor_contact_depth_weight,
            torch.tensor([self._options.contact_depth_weight], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )

        dx, dy, dz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])
        K2_padded = _compute_K2_rfft3(2 * nx, 2 * ny, 2 * nz, dx, dy, dz)
        self._shared_metadata.sensor_K2.append(K2_padded)
        surface_mask = _compute_surface_mask(nx, ny, nz)
        self._shared_metadata.sensor_surface_mask.append(surface_mask)

        if self._options.heat_generation is not None:
            arr = np.asarray(self._options.heat_generation, dtype=np.float64).squeeze()
            if arr.ndim == 3:
                self._shared_metadata.sensor_heat_generation.append(
                    torch.tensor(arr, dtype=gs.tc_float, device=gs.device)
                )
            else:
                self._shared_metadata.sensor_heat_generation.append(None)
        else:
            self._shared_metadata.sensor_heat_generation.append(None)

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            torch.tensor([current_cache_start], dtype=gs.tc_int, device=gs.device),
            expand=(1,),
            dim=0,
        )

        total_cache = sum(self._shared_metadata.cache_sizes)
        if (
            self._shared_metadata.sensor_T_measured.shape[0] != self._manager._sim._B
            or self._shared_metadata.sensor_T_measured.shape[1] != total_cache
        ):
            self._shared_metadata.sensor_T_measured = torch.zeros(
                (self._manager._sim._B, total_cache), dtype=gs.tc_float, device=gs.device
            )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (self._options.grid_size,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: TemperatureGridSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        for i_s in range(shared_metadata.sensor_cache_start.shape[0]):
            base_T = shared_metadata.sensor_base_temperature[i_s].item()
            sl = slice(
                shared_metadata.sensor_cache_start[i_s].item(),
                shared_metadata.sensor_cache_start[i_s].item() + shared_metadata.cache_sizes[i_s],
            )
            shared_ground_truth_cache[envs_idx, sl] = base_T
            if shared_metadata.sensor_T_measured.shape[1] > 0:
                shared_metadata.sensor_T_measured[envs_idx, sl] = base_T

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: TemperatureGridSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        solver = shared_metadata.solver
        dt = solver._sim.dt

        # 1) Batched FFT semi-implicit diffusion + 2) heat generation
        _apply_diffusion_and_heat_generation(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_grid_size,
            shared_metadata.sensor_rho_cp,
            shared_metadata.sensor_heat_generation,
            shared_metadata.sensor_voxel_size,
            shared_metadata.expanded_links_idx,
            shared_metadata.link_to_material_idx,
            shared_metadata.link_material_properties,
            shared_metadata.sensor_K2,
            dt,
            shared_ground_truth_cache,
        )
        # 3) Contact heat transfer
        collider_state = solver.collider._collider_state
        if collider_state.contact_data.link_a.shape[0] > 0:
            output = shared_ground_truth_cache.contiguous()
            _kernel_contact_heat(
                solver.links_state,
                collider_state,
                shared_metadata.expanded_links_idx,
                shared_metadata.sensor_aabb_min,
                shared_metadata.sensor_voxel_size,
                shared_metadata.sensor_grid_size,
                shared_metadata.sensor_cache_start,
                shared_metadata.sensor_voxel_volume,
                shared_metadata.sensor_rho_cp,
                shared_metadata.sensor_contact_depth_weight,
                shared_metadata.link_to_material_idx,
                shared_metadata.link_material_properties,
                dt,
                gs.EPS,
                output,
            )
            if not shared_ground_truth_cache.is_contiguous():
                shared_ground_truth_cache.copy_(output)
        # 4) Radiation and convection
        _apply_radiation_convection(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_surface_mask,
            shared_metadata.sensor_rho_cp,
            shared_metadata.sensor_voxel_volume,
            shared_metadata.sensor_ambient_temperature,
            shared_metadata.sensor_convection_coefficient,
            shared_metadata.expanded_links_idx,
            shared_metadata.link_to_material_idx,
            shared_metadata.link_material_properties,
            dt,
            shared_ground_truth_cache,
        )
        # 5) Clamp for stability
        shared_ground_truth_cache.clamp_(min=TEMP_MIN_CLAMP, max=TEMP_MAX_CLAMP)
        # 6) Sensor time-constant filter -> T_measured
        _apply_T_measured_filter(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_tau,
            shared_metadata.sensor_T_measured,
            dt,
            shared_ground_truth_cache,
        )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: TemperatureGridSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        # Buffer and delay T_measured (filtered output), then apply noise
        buffered_data.set(shared_metadata.sensor_T_measured)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw each grid cell as a box colored by temperature (cool=blue, hot=red).
        Only draws for the first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        if self._link is None:
            return

        for obj in self._debug_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_objects = []

        link_pos = self._link.get_pos(env_idx)
        link_quat = self._link.get_quat(env_idx)
        link_pos = tensor_to_array(link_pos).reshape(3)
        link_quat = tensor_to_array(link_quat).reshape(4)
        link_T = gu.trans_quat_to_T(link_pos, link_quat)

        i_s = self._idx
        aabb_min = tensor_to_array(self._shared_metadata.sensor_aabb_min[i_s]).reshape(3)
        aabb_extent = tensor_to_array(self._shared_metadata.sensor_aabb_extent[i_s]).reshape(3)
        grid_cell_size = aabb_extent / np.array([self._debug_nx, self._debug_ny, self._debug_nz], dtype=np.float64)

        # All cell centers in link frame (nx*ny*nz, 3); cell_offsets precomputed in __init__
        local_positions = aabb_min + self._debug_cell_offsets * grid_cell_size
        # World poses: same rotation as link, translation = link_T @ local_pos
        world_trans = (link_T[:3, :3] @ local_positions.T).T + link_T[:3, 3]
        poses = np.tile(link_T[np.newaxis], (len(world_trans), 1, 1)).astype(np.float64)
        poses[:, :3, 3] = world_trans

        # Per-cell color from temperature (blue=cool, red=hot)
        temps = self.read_ground_truth(env_idx)
        temps = tensor_to_array(temps).reshape(-1)
        t_min, t_range = self._debug_t_min, self._debug_t_range
        if t_range <= 0:
            t_range = 1.0
        norm = np.clip((temps - t_min) / t_range, 0.0, 1.0)
        # Blue (0,0,1) -> Red (1,0,0)
        r = norm
        g = 0.0
        b = 1.0 - norm
        n_cells = len(poses)
        for i in range(n_cells):
            color = (float(r[i]), float(g), float(b[i]), 0.5)
            mesh = mu.create_box(extents=grid_cell_size, color=color)
            self._debug_objects.append(context.draw_debug_mesh(mesh, T=poses[i]))
