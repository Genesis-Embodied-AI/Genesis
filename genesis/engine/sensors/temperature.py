import math
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
PI = math.pi
TWO_PI = 2.0 * PI

PROP_IDX_MAP = {"base_temperature": 0, "conductivity": 1, "emissivity": 2}


def _compute_K2_rfft3(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> torch.Tensor:
    """Squared wave numbers for 3D real FFT: K2[i,j,k] = (2*pi*kx)^2 + (2*pi*ky)^2 + (2*pi*kz)^2 with rfft layout."""
    kx = torch.fft.fftfreq(nx, d=dx, device=gs.device).to(gs.tc_float)
    ky = torch.fft.fftfreq(ny, d=dy, device=gs.device).to(gs.tc_float)
    kz = torch.fft.rfftfreq(nz, d=dz, device=gs.device).to(gs.tc_float)
    K2 = (TWO_PI * kx).reshape(-1, 1, 1) ** 2
    K2 = K2 + (TWO_PI * ky).reshape(1, -1, 1) ** 2
    K2 = K2 + (TWO_PI * kz).reshape(1, 1, -1) ** 2
    K2[0, 0, 0] = max(K2[0, 0, 0].item(), gs.EPS)
    return K2


def _compute_surface_mask(nx: int, ny: int, nz: int) -> torch.Tensor:
    """Boolean mask of boundary voxels (at least one face on grid boundary). Shape (nx, ny, nz)."""
    ix, iy, iz = torch.meshgrid(
        torch.arange(nx, device=gs.device),
        torch.arange(ny, device=gs.device),
        torch.arange(nz, device=gs.device),
        indexing="ij",
    )
    return ((ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1) | (iz == 0) | (iz == nz - 1)).to(gs.tc_float)


def _apply_diffusion_and_heat_generation(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    grid_size: torch.Tensor,
    rho_cp: torch.Tensor,
    heat_generation: list[torch.Tensor | None],
    voxel_size: torch.Tensor,
    links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_conductivity: torch.Tensor,
    K2_spectral: list[torch.Tensor],
    dt: float,
    output: torch.Tensor,
) -> None:
    """Batched FFT semi-implicit diffusion with mirror padding (Neumann BC, no wrap-around)."""
    n_sensors = sensor_cache_start.shape[0]
    n_batches = output.shape[0]
    for i_s in range(n_sensors):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        nx, ny, nz = grid_size[i_s].tolist()
        rcp = rho_cp[i_s].item()
        mat_idx = link_to_material_idx[links_idx[i_s].item()].item()
        k = link_conductivity[mat_idx].item()
        alpha = k / rcp
        T = output[:, start : start + size].reshape(-1, nx, ny, nz)
        # Mirror-pad to (2*nx, 2*ny, 2*nz) for zero-flux (Neumann) boundaries; avoids FFT wrap-around.
        T_x = torch.cat([T, torch.flip(T, dims=(1,))], dim=1)
        T_xy = torch.cat([T_x, torch.flip(T_x, dims=(2,))], dim=2)
        T_pad = torch.cat([T_xy, torch.flip(T_xy, dims=(3,))], dim=3)
        T_hat = torch.fft.rfftn(T_pad, dim=(-3, -2, -1))
        T_hat = T_hat / (1.0 + dt * alpha * K2_spectral[i_s])
        T_pad = torch.fft.irfftn(T_hat, s=(2 * nx, 2 * ny, 2 * nz), dim=(-3, -2, -1)).real
        T = T_pad[:, :nx, :ny, :nz]
        output[:, start : start + size] = T.reshape(n_batches, -1)

        # Add internal heat generation (W/m² -> Q_vol = Q_surface / dz).
        q = heat_generation[i_s]
        if q is not None:
            dz = max(voxel_size[i_s, 2].item(), gs.EPS)
            Q_vol = q.reshape(-1) / dz
            delta_T = dt * Q_vol / rcp
            output[:, start : start + size] += delta_T.unsqueeze(0).expand(n_batches, -1)


@qd.func
def _qd_polygon_area_from_points(
    n: gs.qd_int,
    scratch: qd.types.ndarray(),
    eps: gs.qd_float,
) -> gs.qd_float:
    """Area of polygon from scratch[:, 10:13] (pts), scratch[:, 13:16] (nms), xy in scratch[:, 17:19]."""
    area = gs.qd_float(0.0)
    if n >= 3:
        cx = gs.qd_float(0.0)
        cy = gs.qd_float(0.0)
        cz = gs.qd_float(0.0)
        nx = gs.qd_float(0.0)
        ny = gs.qd_float(0.0)
        nz = gs.qd_float(0.0)
        for i in range(n):
            cx = cx + scratch[i, 10]
            cy = cy + scratch[i, 11]
            cz = cz + scratch[i, 12]
            nx = nx + scratch[i, 13]
            ny = ny + scratch[i, 14]
            nz = nz + scratch[i, 15]
        n_inv = gs.qd_float(1.0) / gs.qd_float(n)
        cx, cy, cz = cx * n_inv, cy * n_inv, cz * n_inv
        nx, ny, nz = nx * n_inv, ny * n_inv, nz * n_inv
        n_norm = qd.sqrt(nx * nx + ny * ny + nz * nz) + eps
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm
        ax = 0 if qd.abs(nx) < gs.qd_float(0.9) else 1
        ux = gs.qd_float(0.0)
        uy = gs.qd_float(0.0)
        uz = gs.qd_float(0.0)
        if ax == 0:
            ux = gs.qd_float(1.0)
        else:
            uy = gs.qd_float(1.0)
        dot = ux * nx + uy * ny + uz * nz
        ux, uy, uz = ux - dot * nx, uy - dot * ny, uz - dot * nz
        u_norm = qd.sqrt(ux * ux + uy * uy + uz * uz) + eps
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux
        v_norm = qd.sqrt(vx * vx + vy * vy + vz * vz) + eps
        vx, vy, vz = vx / v_norm, vy / v_norm, vz / v_norm
        for i in range(n):
            rx = scratch[i, 10] - cx
            ry = scratch[i, 11] - cy
            rz = scratch[i, 12] - cz
            scratch[i, 17] = rx * ux + ry * uy + rz * uz
            scratch[i, 18] = rx * vx + ry * vy + rz * vz
        for i in range(1, n):
            key_x = scratch[i, 17]
            key_y = scratch[i, 18]
            j = i - 1
            key_angle = qd.atan2(key_y, key_x)
            while j >= 0 and qd.atan2(scratch[j, 18], scratch[j, 17]) > key_angle:
                scratch[j + 1, 17] = scratch[j, 17]
                scratch[j + 1, 18] = scratch[j, 18]
                j = j - 1
            scratch[j + 1, 17] = key_x
            scratch[j + 1, 18] = key_y
        for i in range(n):
            i_next = (i + 1) % n
            area = area + scratch[i, 17] * scratch[i_next, 18] - scratch[i_next, 17] * scratch[i, 18]
        area = qd.abs(area) * gs.qd_float(0.5)

    return area


@qd.kernel
def _kernel_compute_contact_areas(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    links_idx: qd.types.ndarray(),
    contact_area: qd.types.ndarray(),
    scratch: qd.types.ndarray(),
    eps: gs.qd_float,
):
    # scratch layout:
    # 0=other_link_idx, 1=idx, 2=depth, 3:6=pos, 6:9=normal, 9=group_idx,
    # 10:13=group_pos, 13:16=group_normal, 16=group_depth, 17:19=xy
    n_batches = contact_area.shape[1]
    n_sensors = links_idx.shape[0]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        sensor_link_idx = links_idx[i_s]
        n_c = collider_state.n_contacts[i_b]
        n_valid = 0
        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            if la != sensor_link_idx and lb != sensor_link_idx:
                continue
            other = lb if la == sensor_link_idx else la
            scratch[n_valid, 0] = gs.qd_float(other)
            scratch[n_valid, 1] = gs.qd_float(i_c)
            scratch[n_valid, 2] = collider_state.contact_data.penetration[i_c, i_b]
            p_world = collider_state.contact_data.pos[i_c, i_b]
            link_pos = links_state.pos[sensor_link_idx, i_b]
            link_quat = links_state.quat[sensor_link_idx, i_b]
            p_local = gu.qd_inv_transform_by_trans_quat(p_world, link_pos, link_quat)
            scratch[n_valid, 3] = p_local.x
            scratch[n_valid, 4] = p_local.y
            scratch[n_valid, 5] = p_local.z
            n_w = collider_state.contact_data.normal[i_c, i_b]
            scratch[n_valid, 6] = n_w.x
            scratch[n_valid, 7] = n_w.y
            scratch[n_valid, 8] = n_w.z
            n_valid = n_valid + 1

        for i in range(n_valid):
            other = gs.qd_int(scratch[i, 0])
            is_first = True
            for k in range(i):
                if gs.qd_int(scratch[k, 0]) == other:
                    is_first = False
            if not is_first:
                continue

            count = 0
            for j in range(n_valid):
                if gs.qd_int(scratch[j, 0]) == other:
                    scratch[count, 9] = scratch[j, 1]
                    scratch[count, 10] = scratch[j, 3]
                    scratch[count, 11] = scratch[j, 4]
                    scratch[count, 12] = scratch[j, 5]
                    scratch[count, 13] = scratch[j, 6]
                    scratch[count, 14] = scratch[j, 7]
                    scratch[count, 15] = scratch[j, 8]
                    scratch[count, 16] = scratch[j, 2]
                    count = count + 1

            group_area = eps
            if count >= 3:
                group_area = _qd_polygon_area_from_points(count, scratch, eps)
            else:
                for k in range(count):
                    d = scratch[k, 16]
                    group_area = group_area + d * d

            total_d2 = eps
            for k in range(count):
                d = scratch[k, 16]
                total_d2 = total_d2 + d * d
            for k in range(count):
                d = scratch[k, 16]
                contact_area[gs.qd_int(scratch[k, 9]), i_b, i_s] = group_area * (d * d / total_d2)


@qd.kernel
def _kernel_contact_heat(
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    links_idx: qd.types.ndarray(),
    aabb_min: qd.types.ndarray(),
    grid_size: qd.types.ndarray(),
    voxel_size: qd.types.ndarray(),
    voxel_volume: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    rho_cp: qd.types.ndarray(),
    depth_weight: qd.types.ndarray(),
    link_to_material_idx: qd.types.ndarray(),
    link_conductivity: qd.types.ndarray(),
    link_base_temperature: qd.types.ndarray(),
    contact_area: qd.types.ndarray(),
    dt: gs.qd_float,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    n_batches = output.shape[0]
    n_sensors = links_idx.shape[0]
    for i_b, i_s in qd.ndrange(n_batches, n_sensors):
        sensor_link_idx = links_idx[i_s]
        start = sensor_cache_start[i_s]
        nx = grid_size[i_s, 0]
        ny = grid_size[i_s, 1]
        nz = grid_size[i_s, 2]
        vol = voxel_volume[i_s] + eps
        rcp = rho_cp[i_s] + eps
        dw = depth_weight[i_s]
        mat_idx_sensor = link_to_material_idx[sensor_link_idx]
        if mat_idx_sensor < 0:
            continue

        k_sensor = link_conductivity[mat_idx_sensor]
        amin = qd.math.vec3(aabb_min[i_s, 0], aabb_min[i_s, 1], aabb_min[i_s, 2])
        vs = qd.math.vec3(
            voxel_size[i_s, 0] + eps,
            voxel_size[i_s, 1] + eps,
            voxel_size[i_s, 2] + eps,
        )
        n_c = collider_state.n_contacts[i_b]
        for i_c in range(n_c):
            la = collider_state.contact_data.link_a[i_c, i_b]
            lb = collider_state.contact_data.link_b[i_c, i_b]
            if la != sensor_link_idx and lb != sensor_link_idx:
                continue
            other_link = lb if la == sensor_link_idx else la
            mat_other = link_to_material_idx[other_link]
            if mat_other >= 0:
                T_other = link_base_temperature[mat_other]
                k_other = link_conductivity[mat_other]
                k_eff = gs.qd_float(2.0) * k_sensor * k_other / (k_sensor + k_other + eps)
                depth = collider_state.contact_data.penetration[i_c, i_b]
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
                area = max(contact_area[i_c, i_b, i_s], PI * dw * depth, eps)
                flux = k_eff * (T_other - T_cell) / (vol / area + eps)
                Q_vol = flux * area / vol
                delta_T = dt * Q_vol / rcp
                output[i_b, start + cell_idx] = T_cell + delta_T


def _apply_radiation_convection(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_surface_mask: list[torch.Tensor],
    rho_cp: torch.Tensor,
    voxel_volume: torch.Tensor,
    convection_coeff: torch.Tensor,
    links_idx: torch.Tensor,
    link_to_material_idx: torch.Tensor,
    link_emissivity: torch.Tensor,
    ambient_temp: float,
    dt: float,
    output: torch.Tensor,
) -> None:
    """Radiation + convection on surface voxels."""
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        mask = sensor_surface_mask[i_s].reshape(-1)
        rcp = rho_cp[i_s].item()
        vol = max(voxel_volume[i_s].item(), gs.EPS)
        mat_idx = link_to_material_idx[links_idx[i_s]]
        emiss = link_emissivity[mat_idx].item()
        h = convection_coeff[i_s].item()
        T_amb_K = ambient_temp + KELVIN_OFFSET
        denom = rcp * vol
        T_flat = output[:, start : start + size]
        T_K = T_flat + KELVIN_OFFSET
        q_rad = emiss * STEFAN_BOLTZMANN * (T_K**4 - T_amb_K**4)
        q_conv = h * (T_flat - ambient_temp)
        output[:, start : start + size] -= dt * (q_rad + q_conv) / denom * mask


def _apply_T_measured_filter(
    sensor_cache_start: torch.Tensor,
    cache_sizes: list[int],
    sensor_time_const: torch.Tensor,
    dt: float,
    T_actual: torch.Tensor,
    T_measured: torch.Tensor,
) -> None:
    """T_measured += (dt/tau)*(T - T_measured); if tau<=0 then T_measured = T. Batched over envs."""
    for i_s in range(sensor_cache_start.shape[0]):
        start = sensor_cache_start[i_s].item()
        size = cache_sizes[i_s]
        tau = sensor_time_const[i_s].item()
        T_slice = T_actual[:, start : start + size]
        T_meas_slice = T_measured[:, start : start + size]
        if tau > 0:
            alpha = dt / tau
            T_measured[:, start : start + size] = T_meas_slice + alpha * (T_slice - T_meas_slice)
        else:
            T_measured[:, start : start + size] = T_slice


@dataclass
class TemperatureGridSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all temperature grid sensors."""

    ambient_temperature: float = 21.0
    link_to_material_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    link_material_properties: torch.Tensor = make_tensor_field((0, 5), dtype=gs.tc_float)
    properties_dict: dict[int, TemperatureProperties] = field(default_factory=dict)
    link_temps: torch.Tensor = make_tensor_field((0, 0))

    aabb_min: torch.Tensor = make_tensor_field((0, 3))
    aabb_extent: torch.Tensor = make_tensor_field((0, 3))
    grid_size: torch.Tensor = make_tensor_field((0, 3), dtype=gs.tc_int)
    voxel_size: torch.Tensor = make_tensor_field((0, 3))
    voxel_volume: torch.Tensor = make_tensor_field((0,))
    rho_cp: torch.Tensor = make_tensor_field((0,))
    convection_coeff: torch.Tensor = make_tensor_field((0,))
    sensor_time_const: torch.Tensor = make_tensor_field((0,))
    contact_depth_weight: torch.Tensor = make_tensor_field((0,))
    K2_spectral: list[torch.Tensor] = field(default_factory=list)
    sensor_surface_mask: list[torch.Tensor] = field(default_factory=list)
    heat_generation: list[torch.Tensor | None] = field(default_factory=list)
    contact_area_scratch: torch.Tensor = make_tensor_field((0, 0, 19))

    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)


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

        # Same for all sensors
        self._shared_metadata.ambient_temperature = self._options.ambient_temperature
        if self._shared_metadata.link_to_material_idx.shape[0] == 0:
            self._shared_metadata.link_to_material_idx = torch.full(
                (self._shared_metadata.solver.n_links,), -1, dtype=gs.tc_int, device=gs.device
            )
        self._shared_metadata.properties_dict.update(self._options.properties_dict)
        if len(self._shared_metadata.properties_dict) > self._shared_metadata.link_material_properties.shape[0]:
            self._shared_metadata.link_material_properties = torch.empty(
                (len(PROP_IDX_MAP), len(self._shared_metadata.properties_dict)), dtype=gs.tc_float, device=gs.device
            )
            # -1 in link_to_material_idx means invalid, 0 uses the default properties
            self._shared_metadata.link_to_material_idx[:] = 0 if -1 in self._shared_metadata.properties_dict else -1
            # sort properties_dict by link index to ensure default properties are at index 0
            for i, (prop_idx, props) in enumerate(
                sorted(self._shared_metadata.properties_dict.items(), key=lambda x: x[0])
            ):
                self._shared_metadata.link_material_properties[:, i] = torch.tensor(
                    [getattr(props, name) for name in PROP_IDX_MAP.keys()], dtype=gs.tc_float, device=gs.device
                )
                if prop_idx >= 0:
                    self._shared_metadata.link_to_material_idx[prop_idx] = i
        self._shared_metadata.contact_area_scratch = torch.empty(
            self._shared_metadata.solver.collider._collider_info.max_contact_pairs[None],
            19,
            device=gs.device,
            dtype=gs.tc_float,
        )

        # Per-sensor
        aabb_world = self._link.get_AABB()
        if aabb_world.ndim == 2:
            aabb_world = aabb_world.unsqueeze(0)  # (1, 2, 3)
        aabb_min_w = aabb_world[0, 0]  # (3,)
        aabb_max_w = aabb_world[0, 1]  # (3,)
        link_pos = self._link.get_pos(0)
        link_quat = self._link.get_quat(0)
        aabb_min_local = gu.inv_transform_by_trans_quat(aabb_min_w, link_pos, link_quat)
        aabb_max_local = gu.inv_transform_by_trans_quat(aabb_max_w, link_pos, link_quat)
        aabb_extent = (aabb_max_local - aabb_min_local).reshape(3)
        self._shared_metadata.aabb_min = concat_with_tensor(
            self._shared_metadata.aabb_min, aabb_min_local, expand=(1, 3), dim=0
        )
        self._shared_metadata.aabb_extent = concat_with_tensor(
            self._shared_metadata.aabb_extent, aabb_extent, expand=(1, 3), dim=0
        )

        self._shared_metadata.grid_size = concat_with_tensor(
            self._shared_metadata.grid_size, self._options.grid_size, expand=(1, 3), dim=0
        )

        grid_size_tensor = torch.tensor(self._options.grid_size, dtype=gs.tc_int, device=gs.device)
        voxel_size = aabb_extent / grid_size_tensor
        self._shared_metadata.voxel_size = concat_with_tensor(
            self._shared_metadata.voxel_size, voxel_size, expand=(1, 3), dim=0
        )
        self._shared_metadata.voxel_volume = concat_with_tensor(
            self._shared_metadata.voxel_volume, voxel_size.prod(), expand=(1,), dim=0
        )
        self._shared_metadata.rho_cp = concat_with_tensor(
            self._shared_metadata.rho_cp, props.density * props.specific_heat, expand=(1,), dim=0
        )
        self._shared_metadata.convection_coeff = concat_with_tensor(
            self._shared_metadata.convection_coeff, self._options.convection_coefficient, expand=(1,), dim=0
        )
        self._shared_metadata.sensor_time_const = concat_with_tensor(
            self._shared_metadata.sensor_time_const, self._options.sensor_time_constant, expand=(1,), dim=0
        )
        self._shared_metadata.contact_depth_weight = concat_with_tensor(
            self._shared_metadata.contact_depth_weight, self._options.contact_depth_weight, expand=(1,), dim=0
        )

        dx, dy, dz = voxel_size.tolist()
        nx, ny, nz = grid_size_tensor.tolist()
        K2_padded = _compute_K2_rfft3(nx * 2, ny * 2, nz * 2, dx, dy, dz)
        self._shared_metadata.K2_spectral.append(K2_padded)

        surface_mask = _compute_surface_mask(nx, ny, nz)
        self._shared_metadata.sensor_surface_mask.append(surface_mask)

        self._shared_metadata.heat_generation.append(
            torch.tensor(self._options.heat_generation, dtype=gs.tc_float, device=gs.device)
            if self._options.heat_generation is not None
            else None
        )

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start, current_cache_start, expand=(1,), dim=0
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
            link_idx = shared_metadata.links_idx[i_s].item()
            mat_idx = shared_metadata.link_to_material_idx[link_idx].item()
            base_T = shared_metadata.link_material_properties[PROP_IDX_MAP["base_temperature"]][mat_idx].item()
            start = shared_metadata.sensor_cache_start[i_s].item()
            shared_ground_truth_cache[envs_idx, start : start + shared_metadata.cache_sizes[i_s]] = base_T

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: TemperatureGridSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver
        dt = solver._sim.dt
        props = shared_metadata.link_material_properties
        link_conductivity = props[PROP_IDX_MAP["conductivity"]]
        link_base_temperature = props[PROP_IDX_MAP["base_temperature"]]
        link_emissivity = props[PROP_IDX_MAP["emissivity"]]

        # 1) Batched FFT semi-implicit diffusion + 2) Heat generation
        _apply_diffusion_and_heat_generation(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.grid_size,
            shared_metadata.rho_cp,
            shared_metadata.heat_generation,
            shared_metadata.voxel_size,
            shared_metadata.links_idx,
            shared_metadata.link_to_material_idx,
            link_conductivity,
            shared_metadata.K2_spectral,
            dt,
            shared_ground_truth_cache,
        )
        # 3) Contact heat transfer
        collider_state = solver.collider._collider_state
        if collider_state.contact_data.link_a.shape[0] > 0:
            n_c_max = collider_state.contact_data.link_a.shape[0]
            n_batches = collider_state.contact_data.link_a.shape[1]
            n_sensors = shared_metadata.links_idx.shape[0]
            contact_area = torch.zeros(n_c_max, n_batches, n_sensors, device=gs.device, dtype=gs.tc_float)

            _kernel_compute_contact_areas(
                solver.links_state,
                collider_state,
                shared_metadata.links_idx,
                contact_area,
                shared_metadata.contact_area_scratch,
                gs.EPS,
            )
            output = shared_ground_truth_cache.contiguous()
            _kernel_contact_heat(
                solver.links_state,
                collider_state,
                shared_metadata.links_idx,
                shared_metadata.aabb_min,
                shared_metadata.grid_size,
                shared_metadata.voxel_size,
                shared_metadata.voxel_volume,
                shared_metadata.sensor_cache_start,
                shared_metadata.rho_cp,
                shared_metadata.contact_depth_weight,
                shared_metadata.link_to_material_idx,
                link_conductivity,
                link_base_temperature,
                contact_area,
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
            shared_metadata.rho_cp,
            shared_metadata.voxel_volume,
            shared_metadata.convection_coeff,
            shared_metadata.links_idx,
            shared_metadata.link_to_material_idx,
            link_emissivity,
            shared_metadata.ambient_temperature,
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
        dt = shared_metadata.solver._sim.dt
        buffered_data.set(shared_ground_truth_cache)
        _apply_T_measured_filter(
            shared_metadata.sensor_cache_start,
            shared_metadata.cache_sizes,
            shared_metadata.sensor_time_const,
            dt,
            shared_ground_truth_cache,
            buffered_data.at(0),
        )
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
        aabb_min = tensor_to_array(self._shared_metadata.aabb_min[i_s]).reshape(3)
        aabb_extent = tensor_to_array(self._shared_metadata.aabb_extent[i_s]).reshape(3)
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
        colors = np.column_stack((norm, np.zeros_like(norm), 1.0 - norm, np.full_like(norm, 0.5)))
        for i, pose in enumerate(poses):
            mesh = mu.create_box(extents=grid_cell_size, color=tuple(float(c) for c in colors[i]))
            self._debug_objects.append(context.draw_debug_mesh(mesh, T=pose))
