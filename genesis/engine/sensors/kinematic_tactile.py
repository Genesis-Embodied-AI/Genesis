import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Type

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_all_verts
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ElastomerTactileSensor as ElastomerTactileSensorOptions
from genesis.options.sensors import KinematicContactProbe as KinematicContactProbeOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_qd import get_triangle_vertices, ray_triangle_intersection

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


@qd.func
def _func_probe_geom_penetration(
    probe_pos: gs.qd_vec3,
    probe_normal: gs.qd_vec3,
    probe_radius: gs.qd_float,
    max_range: gs.qd_float,
    i_g: gs.qd_int,
    i_b: gs.qd_int,
    geoms_info: array_class.GeomsInfo,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    eps: gs.qd_float,
):
    best = gs.qd_float(0.0)
    neg_normal = -probe_normal
    face_start = geoms_info.face_start[i_g]
    face_end = geoms_info.face_end[i_g]
    radius_sq = probe_radius * probe_radius

    for i_f in range(face_start, face_end):
        tri_verts = get_triangle_vertices(i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state)
        v0 = tri_verts[:, 0]
        v1 = tri_verts[:, 1]
        v2 = tri_verts[:, 2]

        if probe_radius > gs.EPS:
            # Sphere-triangle test (closest point)
            closest_point = _func_closest_point_on_triangle(probe_pos, v0, v1, v2)
            diff = closest_point - probe_pos
            dist_sq = diff.dot(diff)
            if dist_sq <= radius_sq:
                penetration = diff.dot(neg_normal)
                if penetration > best:
                    best = penetration

        # Raycast test (ray along -normal)
        result = ray_triangle_intersection(probe_pos, neg_normal, v0, v1, v2, eps)
        if result[3] > 0.5 and result[0] <= max_range:
            t = result[0]
            if best == 0.0 or t < best:
                best = t

    return best


@qd.func
def _func_closest_point_on_triangle(
    point: gs.qd_vec3,
    v0: gs.qd_vec3,
    v1: gs.qd_vec3,
    v2: gs.qd_vec3,
) -> gs.qd_vec3:
    """
    Find the point on the surface of a triangle closest to a given point.

    Reference: Christer Ericson, *Real-Time Collision Detection*, §5.1.5.
    """
    ab = v1 - v0
    ac = v2 - v0
    ap = point - v0

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    # Region A (vertex v0)
    closest = v0
    if not (d1 <= 0.0 and d2 <= 0.0):
        bp = point - v1
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        # Region B (vertex v1)
        if d3 >= 0.0 and d4 <= d3:
            closest = v1
        else:
            cp = point - v2
            d5 = ab.dot(cp)
            d6 = ac.dot(cp)

            # Region C (vertex v2)
            if d6 >= 0.0 and d5 <= d6:
                closest = v2
            else:
                vc = d1 * d4 - d3 * d2
                # Region AB (edge v0-v1)
                if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                    w = d1 / (d1 - d3)
                    closest = v0 + w * ab
                else:
                    vb = d5 * d2 - d1 * d6
                    # Region AC (edge v0-v2)
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        closest = v0 + w * ac
                    else:
                        va = d3 * d6 - d5 * d4
                        # Region BC (edge v1-v2)
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            closest = v1 + w * (v2 - v1)
                        else:
                            # Inside the triangle face
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac

    return closest


@qd.func
def _func_query_contact_depth(
    i_b: gs.qd_int,
    probe_pos: gs.qd_vec3,
    probe_normal: gs.qd_vec3,
    probe_radius: gs.qd_float,
    probe_max_raycast_range: gs.qd_float,
    sensor_link_idx: gs.qd_int,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    collider_state: array_class.ColliderState,
    eps: gs.qd_float,
):
    max_penetration = gs.qd_float(0.0)
    contact_link = gs.qd_int(-1)

    # Iterate over contacts directly from collider state
    n_contacts = collider_state.n_contacts[i_b]
    for i_c in range(n_contacts):
        c_link_a = collider_state.contact_data.link_a[i_c, i_b]
        c_link_b = collider_state.contact_data.link_b[i_c, i_b]
        c_geom_a = collider_state.contact_data.geom_a[i_c, i_b]
        c_geom_b = collider_state.contact_data.geom_b[i_c, i_b]

        # Check if either side of this contact involves one of our sensor links;
        for side in qd.static(range(2)):
            c_link = c_link_a if side == 0 else c_link_b
            i_g = c_geom_b if side == 0 else c_geom_a

            # Is this contact relevant to this sensor?
            if c_link == sensor_link_idx and func_point_in_geom_aabb(geoms_state, i_g, i_b, probe_pos, probe_radius):
                # Raycast + sphere penetration test per geom
                penetration = _func_probe_geom_penetration(
                    probe_pos,
                    probe_normal,
                    probe_radius,
                    probe_max_raycast_range,
                    i_g,
                    i_b,
                    geoms_info,
                    faces_info,
                    verts_info,
                    fixed_verts_state,
                    free_verts_state,
                    eps,
                )
                if penetration > max_penetration:
                    max_penetration = penetration
                    contact_link = c_link_b if side == 0 else c_link_a

    return max_penetration, contact_link


@qd.kernel
def _kernel_kinematic_contact_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    links_state: array_class.LinksState,
    radii: qd.types.ndarray(),
    stiffness: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    output: qd.types.ndarray(),
    eps: gs.qd_float,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output.shape[0]

    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = qd.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        radius = radii[i_p]
        stiff = stiffness[i_s]
        sensor_link_idx = links_idx[i_s]

        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.qd_transform_by_quat(probe_normal_local, link_quat)

        max_penetration, _ = _func_query_contact_depth(
            i_b,
            probe_pos,
            probe_normal,
            radius,
            probe_max_raycast_range,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            faces_info,
            verts_info,
            fixed_verts_state,
            free_verts_state,
            collider_state,
            eps,
        )

        force_local = qd.Vector.zero(gs.qd_float, 3)
        if max_penetration > 0:
            force_local = stiff * max_penetration * -probe_normal_local

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]

        output[i_b, cache_start + probe_idx_in_sensor] = max_penetration
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 0] = force_local[0]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 1] = force_local[1]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 2] = force_local[2]


@qd.kernel
def _kernel_elastomer_tactile(
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    links_state: array_class.LinksState,
    radii: qd.types.ndarray(),
    dilate_coefficients: qd.types.ndarray(),
    shear_coefficients: qd.types.ndarray(),
    twist_coefficients: qd.types.ndarray(),
    sensor_normals: qd.types.ndarray(),
    shear_max_delta: qd.types.ndarray(),
    twist_max_delta: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    contact_buf: qd.types.ndarray(),
    contact_link_buf: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    output: qd.types.ndarray(),
    dt: gs.qd_float,
    eps: gs.qd_float,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output.shape[0]

    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )

    # Phase 1: for each probe, query contact and store C_i (contact point), Δh_i (penetration), contact_link.
    # Eq (11): Δdd = Σ_i Δh_i · (M - C_i) · exp(-λd ||M - C_i||²) — we need all C_i, Δh_i for the sensor.
    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = qd.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.qd_transform_by_quat(probe_normal_local, link_quat)

        max_penetration, contact_link = _func_query_contact_depth(
            i_b,
            probe_pos,
            probe_normal,
            radii[i_p],
            probe_max_raycast_range,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            faces_info,
            verts_info,
            fixed_verts_state,
            free_verts_state,
            collider_state,
            eps,
        )

        if max_penetration > 0:
            contact_point = probe_pos - probe_normal * max_penetration
            contact_buf[i_b, i_p, 0] = contact_point[0]
            contact_buf[i_b, i_p, 1] = contact_point[1]
            contact_buf[i_b, i_p, 2] = contact_point[2]
            contact_buf[i_b, i_p, 3] = max_penetration
        else:
            contact_buf[i_b, i_p, 3] = gs.qd_float(0.0)
        contact_link_buf[i_b, i_p] = contact_link

    # Phase 2: for each marker M (probe), dilate = Σ over all contacts in same sensor; shear/twist from own contact.
    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = qd.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        dilate_coeff = dilate_coefficients[i_s]
        shear_coeff = shear_coefficients[i_s]
        twist_coeff = twist_coefficients[i_s]
        sensor_link_idx = links_idx[i_s]

        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        # FOTS Eq (11): Δdd = Σ_i Δh_i · (M - C_i) · exp(-λd ||M - C_i||²)
        dilate_disp = qd.Vector.zero(gs.qd_float, 3)
        probe_start = sensor_probe_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        for j in range(n_probes):
            j_p = probe_start + j
            delta_h = contact_buf[i_b, j_p, 3]
            if delta_h > gs.qd_float(0.0):
                C_j = qd.Vector(
                    [contact_buf[i_b, j_p, 0], contact_buf[i_b, j_p, 1], contact_buf[i_b, j_p, 2]], dt=gs.qd_float
                )
                M_minus_C = probe_pos - C_j
                dist_sq = M_minus_C[0] * M_minus_C[0] + M_minus_C[1] * M_minus_C[1] + M_minus_C[2] * M_minus_C[2]
                scale = delta_h * qd.exp(-dilate_coeff * dist_sq)
                dilate_disp = dilate_disp + M_minus_C * scale

        displacement_world = dilate_disp

        contact_link = contact_link_buf[i_b, i_p]
        max_penetration = contact_buf[i_b, i_p, 3]
        if max_penetration > gs.qd_float(0.0) and contact_link >= 0:
            # FOTS: shear and twist from contact body velocity * dt (relative to sensor).
            contact_pos = links_state.pos[contact_link, i_b]
            contact_vel = links_state.cd_vel[contact_link, i_b] + links_state.cd_ang[contact_link, i_b].cross(
                contact_pos - links_state.root_COM[contact_link, i_b]
            )
            sensor_vel = links_state.cd_vel[sensor_link_idx, i_b] + links_state.cd_ang[sensor_link_idx, i_b].cross(
                link_pos - links_state.root_COM[sensor_link_idx, i_b]
            )
            rel_vel = contact_vel - sensor_vel
            delta_s_world = rel_vel * dt
            delta_s_local = gu.qd_inv_transform_by_quat(delta_s_world, link_quat)
            delta_s_x = delta_s_local[0]
            delta_s_y = delta_s_local[1]
            sensor_normal_world = gu.qd_transform_by_quat(
                qd.Vector([sensor_normals[i_s, 0], sensor_normals[i_s, 1], sensor_normals[i_s, 2]], dt=gs.qd_float),
                link_quat,
            )
            rel_ang = links_state.cd_ang[contact_link, i_b] - links_state.cd_ang[sensor_link_idx, i_b]
            delta_theta = (
                rel_ang[0] * sensor_normal_world[0]
                + rel_ang[1] * sensor_normal_world[1]
                + rel_ang[2] * sensor_normal_world[2]
            ) * dt
            rel_pos = gu.qd_inv_transform_by_trans_quat(contact_pos, link_pos, link_quat)
            G_local = qd.Vector([rel_pos[0], rel_pos[1], 0.0], dt=gs.qd_float)
            G = link_pos + gu.qd_transform_by_quat(G_local, link_quat)
            M_minus_G = probe_pos - G
            mg_dist_sq = M_minus_G[0] * M_minus_G[0] + M_minus_G[1] * M_minus_G[1] + M_minus_G[2] * M_minus_G[2]
            shear_decay = qd.exp(-shear_coeff * mg_dist_sq)
            twist_decay = qd.exp(-twist_coeff * mg_dist_sq)
            delta_s_mag_sq = delta_s_x * delta_s_x + delta_s_y * delta_s_y
            delta_s_mag = qd.sqrt(delta_s_mag_sq + eps * eps)
            shear_cap = qd.min(delta_s_mag, shear_max_delta[i_s])
            shear_scale = shear_cap * shear_decay / (delta_s_mag + eps)
            shear_local = qd.Vector([shear_scale * delta_s_x, shear_scale * delta_s_y, 0.0], dt=gs.qd_float)
            displacement_world = displacement_world + gu.qd_transform_by_quat(shear_local, link_quat)
            theta_cap = qd.min(qd.abs(delta_theta), twist_max_delta[i_s]) * qd.select(delta_theta >= 0.0, 1.0, -1.0)
            c = qd.cos(theta_cap)
            s = qd.sin(theta_cap)
            M_minus_G_local = gu.qd_inv_transform_by_quat(M_minus_G, link_quat)
            mg_lx = M_minus_G_local[0]
            mg_ly = M_minus_G_local[1]
            twist_lx = (c - 1.0) * mg_lx - s * mg_ly
            twist_ly = s * mg_lx + (c - 1.0) * mg_ly
            twist_local = qd.Vector([twist_lx * twist_decay, twist_ly * twist_decay, 0.0], dt=gs.qd_float)
            displacement_world = displacement_world + gu.qd_transform_by_quat(twist_local, link_quat)

        displacement_local = gu.qd_inv_transform_by_quat(displacement_world, link_quat)

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]

        output[i_b, cache_start + probe_idx_in_sensor * 3 + 0] = displacement_local[0]
        output[i_b, cache_start + probe_idx_in_sensor * 3 + 1] = displacement_local[1]
        output[i_b, cache_start + probe_idx_in_sensor * 3 + 2] = displacement_local[2]


class KinematicContactProbeData(NamedTuple):
    """
    Data returned by the kinematic contact probe.

    Parameters
    ----------
    penetration: torch.Tensor, shape ([n_envs,] n_probes)
        Depth of penetration in meters (0 if no contact).
    force: torch.Tensor, shape ([n_envs,] n_probes, 3)
        Estimated contact force based on penetration and stiffness (non-physical) in the link frame.
    """

    penetration: torch.Tensor
    force: torch.Tensor


@dataclass
class KinematicContactProbeMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all kinematic contact probes."""

    radii: torch.Tensor = make_tensor_field((0,))
    stiffness: torch.Tensor = make_tensor_field((0,))

    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_normals: torch.Tensor = make_tensor_field((0, 3))
    probe_max_raycast_range: float = 0.0

    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    total_n_probes: int = 0


@register_sensor(KinematicContactProbeOptions, KinematicContactProbeMetadata, KinematicContactProbeData)
@qd.data_oriented
class KinematicContactProbe(
    RigidSensorMixin[KinematicContactProbeMetadata],
    NoisySensorMixin[KinematicContactProbeMetadata],
    Sensor[KinematicContactProbeMetadata],
):
    """Kinematic contact probe measuring penetration depth along the probe normal on collisions."""

    def __init__(
        self,
        sensor_options: KinematicContactProbeOptions,
        sensor_idx: int,
        data_cls: Type[KinematicContactProbeData],
        sensor_manager: "SensorManager",
    ):
        # Store n_probes before super().__init__() since _get_return_format() is called there
        self._n_probes = len(sensor_options.probe_local_pos)

        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self._debug_objects: list["Mesh | None"] = []
        self._probe_local_pos = torch.tensor(self._options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._probe_local_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        self._probe_local_normal /= self._probe_local_normal.norm(dim=1, keepdim=True).clamp(min=gs.EPS)

    def build(self):
        super().build()

        n_probes = len(self._probe_local_pos)
        sensor_idx = self._idx

        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor, n_probes, expand=(1,), dim=0
        )

        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start, current_cache_start, expand=(1,), dim=0
        )

        current_probe_start = self._shared_metadata.total_n_probes
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start, current_probe_start, expand=(1,), dim=0
        )

        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((n_probes,), sensor_idx, dtype=gs.tc_int, device=gs.device),
            expand=(n_probes,),
            dim=0,
        )

        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(n_probes, 3), dim=0
        )

        self._shared_metadata.probe_normals = concat_with_tensor(
            self._shared_metadata.probe_normals, self._probe_local_normal, expand=(n_probes, 3), dim=0
        )

        if self._shared_metadata.probe_max_raycast_range < gs.EPS:
            link_aabb = self._link.get_vAABB()
            max_range = torch.linalg.norm(link_aabb[1] - link_aabb[0], dim=-1).max()
            self._shared_metadata.probe_max_raycast_range = max_range.item()

        self._shared_metadata.total_n_probes += n_probes

        if isinstance(self._options.radius, float):
            radii_tensor = torch.full((n_probes,), self._options.radius, dtype=gs.tc_float, device=gs.device)
        else:
            radii_tensor = torch.tensor(self._options.radius, dtype=gs.tc_float, device=gs.device)

        self._shared_metadata.radii = concat_with_tensor(
            self._shared_metadata.radii, radii_tensor, expand=(n_probes,), dim=0
        )
        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, self._options.stiffness, expand=(1,), dim=0
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        n = self._n_probes
        return (n,), (n, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: KinematicContactProbeMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver
        collider_state = solver.collider._collider_state

        shared_ground_truth_cache.zero_()

        _kernel_kinematic_contact_probe(
            shared_metadata.probe_positions,
            shared_metadata.probe_normals,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_max_raycast_range,
            solver.links_state,
            shared_metadata.radii,
            shared_metadata.stiffness,
            shared_metadata.links_idx,
            shared_metadata.n_probes_per_sensor,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            collider_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.fixed_verts_state,
            solver.free_verts_state,
            solver._static_rigid_sim_config,
            solver.verts_info,
            solver.faces_info,
            shared_ground_truth_cache,
            gs.EPS,
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

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        for obj in self._debug_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_objects = []

        if self._link is None:
            return

        link_pos = self._link.get_pos(env_idx).reshape((3,))
        link_quat = self._link.get_quat(env_idx).reshape((4,))
        data = self.read_ground_truth(env_idx)

        for i, pos in enumerate(self._probe_local_pos):
            probe_world = link_pos + gu.transform_by_quat(pos, link_quat)

            probe_global_idx = self._shared_metadata.sensor_probe_start[self._idx].item() + i
            probe_radius = self._shared_metadata.radii[probe_global_idx].item()

            penetration = data.penetration[i].item() if data.penetration.dim() > 0 else data.penetration.item()

            sphere_obj = context.draw_debug_sphere(
                pos=tensor_to_array(probe_world),
                radius=probe_radius,
                color=self._options.debug_sphere_color if penetration <= gs.EPS else self._options.debug_contact_color,
            )
            self._debug_objects.append(sphere_obj)


@dataclass
class ElastomerTactileSensorMetadata(KinematicContactProbeMetadata):
    """Shared metadata for elastomer tactile (FOTS-style) sensors."""

    dilate_coefficient: torch.Tensor = make_tensor_field((0,))
    shear_coefficient: torch.Tensor = make_tensor_field((0,))
    twist_coefficient: torch.Tensor = make_tensor_field((0,))
    shear_max_delta: torch.Tensor = make_tensor_field((0,))
    twist_max_delta: torch.Tensor = make_tensor_field((0,))
    sensor_normal: torch.Tensor = make_tensor_field((0, 3))


@register_sensor(ElastomerTactileSensorOptions, ElastomerTactileSensorMetadata, tuple)
@qd.data_oriented
class ElastomerTactileSensor(KinematicContactProbe):
    def __init__(
        self,
        sensor_options: KinematicContactProbeOptions,
        sensor_idx: int,
        data_cls: Type[KinematicContactProbeData],
        sensor_manager: "SensorManager",
    ):
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

    def build(self):
        super().build()

        self._shared_metadata.dilate_coefficient = concat_with_tensor(
            self._shared_metadata.dilate_coefficient, self._options.dilate_coefficient, expand=(1,), dim=0
        )
        self._shared_metadata.shear_coefficient = concat_with_tensor(
            self._shared_metadata.shear_coefficient, self._options.shear_coefficient, expand=(1,), dim=0
        )
        self._shared_metadata.twist_coefficient = concat_with_tensor(
            self._shared_metadata.twist_coefficient, self._options.twist_coefficient, expand=(1,), dim=0
        )
        self._shared_metadata.shear_max_delta = concat_with_tensor(
            self._shared_metadata.shear_max_delta,
            torch.tensor([self._options.shear_max_delta], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        self._shared_metadata.twist_max_delta = concat_with_tensor(
            self._shared_metadata.twist_max_delta,
            torch.tensor([math.radians(self._options.twist_max_delta)], dtype=gs.tc_float, device=gs.device),
            expand=(1,),
            dim=0,
        )
        # sensor normal should be the average of the probe normals
        self._shared_metadata.sensor_normal = concat_with_tensor(
            self._shared_metadata.sensor_normal, self._probe_local_normal.mean(dim=0), expand=(1, 3), dim=0
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        n = self._n_probes
        return (n, 3)

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: ElastomerTactileSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver

        shared_ground_truth_cache.zero_()

        n_batches = shared_ground_truth_cache.shape[0]
        total_n_probes = shared_metadata.total_n_probes
        contact_buf = torch.empty(
            (n_batches, total_n_probes, 4), dtype=gs.tc_float, device=shared_ground_truth_cache.device
        )
        contact_link_buf = torch.empty(
            (n_batches, total_n_probes), dtype=gs.tc_int, device=shared_ground_truth_cache.device
        )

        _kernel_elastomer_tactile(
            shared_metadata.probe_positions,
            shared_metadata.probe_normals,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_max_raycast_range,
            solver.links_state,
            shared_metadata.radii,
            shared_metadata.dilate_coefficient,
            shared_metadata.shear_coefficient,
            shared_metadata.twist_coefficient,
            shared_metadata.sensor_normal,
            shared_metadata.shear_max_delta,
            shared_metadata.twist_max_delta,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            contact_buf,
            contact_link_buf,
            solver.collider._collider_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.fixed_verts_state,
            solver.free_verts_state,
            solver._static_rigid_sim_config,
            solver.verts_info,
            solver.faces_info,
            shared_ground_truth_cache,
            solver._sim.dt,
            gs.EPS,
        )

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ElastomerTactileSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        super()._update_shared_cache(shared_metadata, shared_ground_truth_cache, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        for obj in self._debug_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_objects = []

        if self._link is None:
            return

        link_pos = self._link.get_pos(env_idx).reshape((3,))
        link_quat = self._link.get_quat(env_idx).reshape((4,))
        data = self.read_ground_truth(env_idx)

        for i, pos in enumerate(self._probe_local_pos):
            probe_world = link_pos + gu.transform_by_quat(pos, link_quat)

            probe_global_idx = self._shared_metadata.sensor_probe_start[self._idx].item() + i
            probe_radius = self._shared_metadata.radii[probe_global_idx].item()

            magnitude = torch.linalg.norm(data[i])

            sphere_obj = context.draw_debug_sphere(
                pos=tensor_to_array(probe_world),
                radius=probe_radius,
                color=self._options.debug_sphere_color if magnitude <= gs.EPS else self._options.debug_contact_color,
            )
            self._debug_objects.append(sphere_obj)
