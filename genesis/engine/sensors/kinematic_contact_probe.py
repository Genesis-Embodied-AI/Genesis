"""Kinematic contact probe for contact detection without physics side effects."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import support_field
from genesis.options.sensors import KinematicContactProbe as KinematicContactProbeOptions
from genesis.utils.geom import transform_by_quat
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
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


class KinematicContactProbeData(NamedTuple):
    """Data returned by the kinematic contact probe."""

    penetration: torch.Tensor  # (n_envs, n_probes) or (n_probes,) - depth in meters (0 if no contact)
    position: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - contact position in link frame
    normal: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - contact normal in link frame
    force: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - estimated force (not physical, see options)


@dataclass
class KinematicContactProbeMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all kinematic contact probes."""

    radii: torch.Tensor = make_tensor_field((0,))
    stiffness: torch.Tensor = make_tensor_field((0,))
    contypes: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    conaffinities: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)

    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_normals: torch.Tensor = make_tensor_field((0, 3))

    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    total_n_probes: int = 0


@register_sensor(KinematicContactProbeOptions, KinematicContactProbeMetadata, KinematicContactProbeData)
@ti.data_oriented
class KinematicContactProbe(
    RigidSensorMixin[KinematicContactProbeMetadata],
    NoisySensorMixin[KinematicContactProbeMetadata],
    Sensor[KinematicContactProbeMetadata],
):
    """Kinematic contact probe using support function queries for contact detection."""

    _update_on_read: bool = True

    def __init__(
        self,
        sensor_options: KinematicContactProbeOptions,
        sensor_idx: int,
        data_cls: Type[KinematicContactProbeData],
        sensor_manager: "SensorManager",
    ):
        # Store n_probes before super().__init__() since _get_return_format() is called there
        self._n_probes = sensor_options.n_probes
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self.debug_sphere_objects: list["Mesh | None"] = []
        self.debug_contact_objects: list["Mesh | None"] = []

    def build(self):
        super().build()

        probe_positions = self._options.get_probe_positions()
        probe_normals = self._options.get_probe_normals()
        n_probes = len(probe_positions)
        sensor_idx = self._idx

        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor,
            torch.tensor([n_probes], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            torch.tensor([current_cache_start], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

        current_probe_start = self._shared_metadata.total_n_probes
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start,
            torch.tensor([current_probe_start], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((n_probes,), sensor_idx, dtype=torch.int32, device=gs.device),
            expand=(n_probes,),
            dim=0,
        )

        positions_tensor = torch.tensor(probe_positions, dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.probe_positions = torch.cat(
            [self._shared_metadata.probe_positions, positions_tensor], dim=0
        )

        normals_tensor = torch.tensor(probe_normals, dtype=gs.tc_float, device=gs.device)
        norms = normals_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals_tensor = normals_tensor / norms
        self._shared_metadata.probe_normals = torch.cat([self._shared_metadata.probe_normals, normals_tensor], dim=0)

        self._shared_metadata.total_n_probes += n_probes

        self._shared_metadata.radii = concat_with_tensor(
            self._shared_metadata.radii, self._options.radius, expand=(1,), dim=0
        )
        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, self._options.stiffness, expand=(1,), dim=0
        )
        self._shared_metadata.contypes = concat_with_tensor(
            self._shared_metadata.contypes,
            torch.tensor([self._options.contype], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.conaffinities = concat_with_tensor(
            self._shared_metadata.conaffinities,
            torch.tensor([self._options.conaffinity], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        n = self._n_probes
        return (n,), (n, 3), (n, 3), (n, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: KinematicContactProbeMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        solver = shared_metadata.solver

        n_sensors = len(shared_metadata.radii)
        total_n_probes = shared_metadata.total_n_probes
        B = max(solver.n_envs, 1)
        n_geoms = solver.n_geoms

        if n_sensors == 0 or total_n_probes == 0:
            return

        links_pos = solver.get_links_pos(links_idx=shared_metadata.links_idx)
        links_quat = solver.get_links_quat(links_idx=shared_metadata.links_idx)

        if solver.n_envs == 0:
            links_pos = links_pos[None]
            links_quat = links_quat[None]

        links_pos = links_pos.reshape(B, n_sensors, 3)
        links_quat = links_quat.reshape(B, n_sensors, 4)

        shared_ground_truth_cache.zero_()

        _kernel_kinematic_contact_probe_support_query(
            probe_positions_local=shared_metadata.probe_positions.contiguous(),
            probe_normals_local=shared_metadata.probe_normals.contiguous(),
            probe_sensor_idx=shared_metadata.probe_sensor_idx.contiguous(),
            links_pos=links_pos.contiguous(),
            links_quat=links_quat.contiguous(),
            radii=shared_metadata.radii.contiguous(),
            stiffness=shared_metadata.stiffness.contiguous(),
            links_idx=shared_metadata.links_idx.contiguous(),
            contypes=shared_metadata.contypes.contiguous(),
            conaffinities=shared_metadata.conaffinities.contiguous(),
            n_probes_per_sensor=shared_metadata.n_probes_per_sensor.contiguous(),
            sensor_cache_start=shared_metadata.sensor_cache_start.contiguous(),
            sensor_probe_start=shared_metadata.sensor_probe_start.contiguous(),
            n_geoms=n_geoms,
            geoms_state=solver.geoms_state,
            geoms_info=solver.geoms_info,
            rigid_global_info=solver._rigid_global_info,
            constraint_state=solver.constraint_solver.constraint_state,
            equalities_info=solver.equalities_info,
            support_field_info=solver.collider._support_field._support_field_info,
            output=shared_ground_truth_cache.contiguous(),
        )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: KinematicContactProbeMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.set(shared_ground_truth_cache)
        torch.normal(0.0, shared_metadata.jitter_ts, out=shared_metadata.cur_jitter_ts)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.cur_jitter_ts,
            shared_metadata.interpolate,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _read_internal(self, envs_idx=None, is_ground_truth=False):
        if not self._manager._has_delay_configured(type(self)):
            self._manager.update_sensor_type_cache(type(self))
        return self._get_formatted_data(
            self._manager.get_cloned_from_cache(self, is_ground_truth=is_ground_truth), envs_idx
        )

    @gs.assert_built
    def read(self, envs_idx=None):
        """Read sensor data with noise applied."""
        return self._read_internal(envs_idx, is_ground_truth=False)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None):
        """Read ground truth sensor data without noise."""
        return self._read_internal(envs_idx, is_ground_truth=True)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        for obj in self.debug_sphere_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self.debug_sphere_objects = []
        for obj in self.debug_contact_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self.debug_contact_objects = []

        if self._link is None:
            return

        link_pos = self._link.get_pos(env_idx).reshape((3,))
        link_quat = self._link.get_quat(env_idx).reshape((4,))
        probe_positions = self._options.get_probe_positions()
        data = self.read(env_idx)

        for i, pos in enumerate(probe_positions):
            offset_pos = torch.tensor(pos, dtype=gs.tc_float, device=gs.device)
            probe_world = link_pos + transform_by_quat(offset_pos, link_quat)

            sphere_obj = context.draw_debug_sphere(
                pos=tensor_to_array(probe_world),
                radius=self._options.radius,
                color=self._options.debug_sphere_color,
            )
            self.debug_sphere_objects.append(sphere_obj)

            penetration = data.penetration[i].item() if data.penetration.dim() > 0 else data.penetration.item()

            if penetration > 0:
                contact_pos_local = data.position[i].reshape((3,))
                contact_pos_world = link_pos + transform_by_quat(contact_pos_local, link_quat)

                contact_obj = context.draw_debug_sphere(
                    pos=tensor_to_array(contact_pos_world),
                    radius=self._options.radius * 0.3,
                    color=self._options.debug_contact_color,
                )
                self.debug_contact_objects.append(contact_obj)
            else:
                self.debug_contact_objects.append(None)


@ti.func
def _func_point_in_expanded_aabb(
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_state: array_class.GeomsState,
    point: ti.types.vector(3, gs.ti_float),
    expansion: gs.ti_float,
):
    aabb_min = geoms_state.aabb_min[i_g, i_b] - expansion
    aabb_max = geoms_state.aabb_max[i_g, i_b] + expansion
    return (point >= aabb_min).all() and (point <= aabb_max).all()


@ti.func
def _func_check_collision_filter(
    i_g: ti.i32,
    i_b: ti.i32,
    sensor_link_idx: ti.i32,
    sensor_contype: ti.i32,
    sensor_conaffinity: ti.i32,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
):
    """Check collision filter: self-detection, contype/conaffinity bitmasks, weld constraints."""
    is_valid = True

    geom_link_idx = geoms_info.link_idx[i_g]
    if geom_link_idx == sensor_link_idx:
        is_valid = False

    if is_valid:
        geom_contype = geoms_info.contype[i_g]
        geom_conaffinity = geoms_info.conaffinity[i_g]
        cond1 = (geom_contype & sensor_conaffinity) != 0
        cond2 = (sensor_contype & geom_conaffinity) != 0
        if not (cond1 and cond2):
            is_valid = False

    if is_valid:
        for i_eq in range(rigid_global_info.n_equalities[None], constraint_state.ti_n_equalities[i_b]):
            if equalities_info.eq_type[i_eq, i_b] == gs.EQUALITY_TYPE.WELD:
                weld_link_a = equalities_info.eq_obj1id[i_eq, i_b]
                weld_link_b = equalities_info.eq_obj2id[i_eq, i_b]
                if (weld_link_a == sensor_link_idx and weld_link_b == geom_link_idx) or (
                    weld_link_a == geom_link_idx and weld_link_b == sensor_link_idx
                ):
                    is_valid = False
                    break

    return is_valid


@ti.func
def _func_support_point_for_probe(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    support_field_info: array_class.SupportFieldInfo,
    direction: gs.ti_vec3,
    i_g: ti.i32,
    i_b: ti.i32,
):
    """Get support point on a geom in the given direction."""
    geom_type = geoms_info.type[i_g]
    g_pos = geoms_state.pos[i_g, i_b]

    support_world = ti.Vector.zero(gs.ti_float, 3)

    if geom_type == gs.GEOM_TYPE.SPHERE:
        sphere_radius = geoms_info.data[i_g][0]
        dir_norm = direction.norm()
        if dir_norm > 1e-10:
            support_world = g_pos + sphere_radius * (direction / dir_norm)
        else:
            support_world = g_pos

    elif geom_type == gs.GEOM_TYPE.PLANE:
        support_world = g_pos

    elif geom_type == gs.GEOM_TYPE.BOX:
        support_world, _, _ = support_field._func_support_box(geoms_state, geoms_info, direction, i_g, i_b)

    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        support_world = support_field._func_support_capsule(geoms_state, geoms_info, direction, i_g, i_b, False)

    else:
        support_world, _, _ = support_field._func_support_world(
            geoms_state, geoms_info, support_field_info, direction, i_g, i_b
        )

    return support_world


@ti.kernel
def _kernel_kinematic_contact_probe_support_query(
    probe_positions_local: ti.types.ndarray(),
    probe_normals_local: ti.types.ndarray(),
    probe_sensor_idx: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    radii: ti.types.ndarray(),
    stiffness: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    contypes: ti.types.ndarray(),
    conaffinities: ti.types.ndarray(),
    n_probes_per_sensor: ti.types.ndarray(),
    sensor_cache_start: ti.types.ndarray(),
    sensor_probe_start: ti.types.ndarray(),
    n_geoms: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    support_field_info: array_class.SupportFieldInfo,
    output: ti.types.ndarray(),
):
    """Compute contact probe readings using support functions."""
    n_batches = links_pos.shape[0]
    total_n_probes = probe_positions_local.shape[0]

    for i_b, i_p in ti.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = ti.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = ti.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        radius = radii[i_s]
        stiff = stiffness[i_s]
        sensor_link_idx = links_idx[i_s]
        sensor_contype = contypes[i_s]
        sensor_conaffinity = conaffinities[i_s]

        link_pos = ti.Vector([links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2]])
        link_quat = ti.Vector(
            [links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]]
        )

        probe_pos = link_pos + gu.ti_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.ti_transform_by_quat(probe_normal_local, link_quat)

        max_penetration = gs.ti_float(0.0)
        best_contact_pos_world = ti.Vector.zero(gs.ti_float, 3)
        support_dir = -probe_normal

        for i_g in range(n_geoms):
            geom_type = geoms_info.type[i_g]

            if geom_type == gs.GEOM_TYPE.TERRAIN:
                continue

            if not _func_check_collision_filter(
                i_g,
                i_b,
                sensor_link_idx,
                sensor_contype,
                sensor_conaffinity,
                geoms_info,
                rigid_global_info,
                constraint_state,
                equalities_info,
            ):
                continue

            if not _func_point_in_expanded_aabb(i_g, i_b, geoms_state, probe_pos, radius):
                continue

            support_pos = ti.Vector.zero(gs.ti_float, 3)

            if geom_type == gs.GEOM_TYPE.PLANE:
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                geom_data = geoms_info.data[i_g]
                plane_normal_local = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
                plane_normal_world = gu.ti_transform_by_quat(plane_normal_local, g_quat)
                dist_to_plane = (probe_pos - g_pos).dot(plane_normal_world)
                support_pos = probe_pos - dist_to_plane * plane_normal_world
            else:
                support_pos = _func_support_point_for_probe(
                    geoms_state, geoms_info, support_field_info, support_dir, i_g, i_b
                )

            dist_to_probe = (probe_pos - support_pos).norm()
            if dist_to_probe <= radius:
                penetration = (probe_pos - support_pos).dot(probe_normal)
                if penetration > max_penetration:
                    max_penetration = penetration
                    best_contact_pos_world = support_pos

        contact_pos_local = ti.Vector.zero(gs.ti_float, 3)
        normal_local = ti.Vector.zero(gs.ti_float, 3)
        force_local = ti.Vector.zero(gs.ti_float, 3)

        if max_penetration > 0:
            contact_pos_local = gu.ti_inv_transform_by_trans_quat(best_contact_pos_world, link_pos, link_quat)
            normal_local = probe_normal_local
            force_local = stiff * max_penetration * probe_normal_local

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]

        output[i_b, cache_start + probe_idx_in_sensor] = max_penetration
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 0] = contact_pos_local[0]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 1] = contact_pos_local[1]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 2] = contact_pos_local[2]
        output[i_b, cache_start + n_probes + n_probes * 3 + probe_idx_in_sensor * 3 + 0] = normal_local[0]
        output[i_b, cache_start + n_probes + n_probes * 3 + probe_idx_in_sensor * 3 + 1] = normal_local[1]
        output[i_b, cache_start + n_probes + n_probes * 3 + probe_idx_in_sensor * 3 + 2] = normal_local[2]
        output[i_b, cache_start + n_probes + n_probes * 6 + probe_idx_in_sensor * 3 + 0] = force_local[0]
        output[i_b, cache_start + n_probes + n_probes * 6 + probe_idx_in_sensor * 3 + 1] = force_local[1]
        output[i_b, cache_start + n_probes + n_probes * 6 + probe_idx_in_sensor * 3 + 2] = force_local[2]
