import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, Type, TypeVar

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.abd.forward_kinematics import func_update_all_verts
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ElastomerDisplacement as ElastomerDisplacementSensorOptions
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

if TYPE_CHECKING:
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def _func_probe_geom_penetration(
    i_b: gs.qd_int,
    i_g: gs.qd_int,
    probe_pos: gs.qd_vec3,
    probe_normal: gs.qd_vec3,
    probe_radius: gs.qd_float,
    max_range: gs.qd_float,
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

        if probe_radius > eps:
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
                    i_b,
                    i_g,
                    probe_pos,
                    probe_normal,
                    probe_radius,
                    probe_max_raycast_range,
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


@qd.func
def _func_shear_twist_displacement(
    i_b: gs.qd_int,
    probe_pos: gs.qd_vec3,
    link_pos: gs.qd_vec3,
    link_quat: gs.qd_vec4,
    contact_link: gs.qd_int,
    links_state: array_class.LinksState,
    sensor_link_idx: gs.qd_int,
    sensor_normal_local: gs.qd_vec3,
    shear_coeff: gs.qd_float,
    twist_coeff: gs.qd_float,
    shear_max_delta: gs.qd_float,
    twist_max_delta: gs.qd_float,
    dt: gs.qd_float,
    eps: gs.qd_float,
) -> gs.qd_vec3:
    displacement_world = qd.Vector.zero(gs.qd_float, 3)

    contact_pos = links_state.pos[contact_link, i_b]
    contact_vel = links_state.cd_vel[contact_link, i_b] + links_state.cd_ang[contact_link, i_b].cross(
        contact_pos - links_state.root_COM[contact_link, i_b]
    )
    sensor_vel = links_state.cd_vel[sensor_link_idx, i_b] + links_state.cd_ang[sensor_link_idx, i_b].cross(
        link_pos - links_state.root_COM[sensor_link_idx, i_b]
    )
    sensor_normal_world = gu.qd_transform_by_quat(sensor_normal_local, link_quat)
    rel_vel = contact_vel - sensor_vel

    rel_pos = gu.qd_inv_transform_by_trans_quat(contact_pos, link_pos, link_quat)
    G_local = qd.Vector([rel_pos[0], rel_pos[1], 0.0], dt=gs.qd_float)
    G = link_pos + gu.qd_transform_by_quat(G_local, link_quat)
    M_minus_G = probe_pos - G
    mg_dist_sq = M_minus_G[0] * M_minus_G[0] + M_minus_G[1] * M_minus_G[1] + M_minus_G[2] * M_minus_G[2]

    # shear_displacement = min(shear_max_delta, shear_velocity * dt) * exp(-λs ||M - G||²)
    if shear_max_delta > 0.0:
        delta_s_world = rel_vel * dt
        delta_s_local = gu.qd_inv_transform_by_quat(delta_s_world, link_quat)

        shear_decay = qd.exp(-shear_coeff * mg_dist_sq)
        delta_s_mag = qd.sqrt(delta_s_local[0] * delta_s_local[0] + delta_s_local[1] * delta_s_local[1] + eps * eps)
        shear_scale = qd.min(delta_s_mag, shear_max_delta) * shear_decay / (delta_s_mag + eps)
        shear_local = qd.Vector(
            [
                shear_scale * delta_s_local[0],
                shear_scale * delta_s_local[1],
                0.0,
            ],
            dt=gs.qd_float,
        )
        displacement_world += gu.qd_transform_by_quat(shear_local, link_quat)

    # twist_displacement = min(twist_max_delta, twist_angle) * (M - G) * exp(-λt ||M - G||²)
    if twist_max_delta > 0.0:
        rel_ang = links_state.cd_ang[contact_link, i_b] - links_state.cd_ang[sensor_link_idx, i_b]
        delta_theta = (
            rel_ang[0] * sensor_normal_world[0]
            + rel_ang[1] * sensor_normal_world[1]
            + rel_ang[2] * sensor_normal_world[2]
        ) * dt
        theta_cap = qd.min(qd.abs(delta_theta), twist_max_delta) * qd.select(delta_theta >= 0.0, 1.0, -1.0)
        cos_theta = qd.cos(theta_cap)
        sin_theta = qd.sin(theta_cap)
        twist_decay = qd.exp(-twist_coeff * mg_dist_sq)
        M_minus_G_local = gu.qd_inv_transform_by_quat(M_minus_G, link_quat)
        twist_local = qd.Vector(
            [
                twist_decay * (cos_theta - 1.0) * M_minus_G_local[0] - sin_theta * M_minus_G_local[1],
                twist_decay * (sin_theta * M_minus_G_local[0] + (cos_theta - 1.0) * M_minus_G_local[1]),
                0.0,
            ],
            dt=gs.qd_float,
        )
        displacement_world += gu.qd_transform_by_quat(twist_local, link_quat)

    return displacement_world


@qd.func
def _func_kinematic_contact_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radius: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    stiffness: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
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

        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.qd_transform_by_quat(probe_normal_local, link_quat)

        max_penetration, _ = _func_query_contact_depth(
            i_b,
            probe_pos,
            probe_normal,
            probe_radius[i_p],
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
            force_local = stiffness[i_s] * max_penetration * -probe_normal_local

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        output[i_b, cache_start + probe_idx_in_sensor] = max_penetration
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 0] = force_local[0]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 1] = force_local[1]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 2] = force_local[2]


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_kinematic_contact_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radius: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    stiffness: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    _func_kinematic_contact_probe(
        probe_positions_local,
        probe_normals_local,
        probe_sensor_idx,
        probe_radius,
        probe_max_raycast_range,
        stiffness,
        links_idx,
        sensor_cache_start,
        sensor_probe_start,
        n_probes_per_sensor,
        static_rigid_sim_config,
        links_state,
        collider_state,
        geoms_state,
        geoms_info,
        fixed_verts_state,
        free_verts_state,
        verts_info,
        faces_info,
        eps,
        output,
    )


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_elastomer_displacement(
    skip_sensor: qd.types.ndarray(),
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radius: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    dilate_coefficients: qd.types.ndarray(),
    dilate_max_delta: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    contact_buf: qd.types.ndarray(),
    contact_link_buf: qd.types.ndarray(),
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    _func_query_contact_probes(
        probe_positions_local,
        probe_normals_local,
        probe_sensor_idx,
        probe_radius,
        probe_max_raycast_range,
        links_idx,
        static_rigid_sim_config,
        links_state,
        collider_state,
        geoms_state,
        geoms_info,
        fixed_verts_state,
        free_verts_state,
        verts_info,
        faces_info,
        contact_buf,
        contact_link_buf,
        eps,
    )

    total_n_probes = probe_positions_local.shape[0]
    n_batches = output.shape[0]

    # Phase 2: for each marker M (probe), dilate = Σ over all contacts in same sensor; shear/twist from own contact.
    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]

        if not skip_sensor[i_s]:
            dilate_max_delta_s = dilate_max_delta[i_s]
            displacement = qd.Vector.zero(gs.qd_float, 3)

            probe_pos_local = qd.Vector(
                [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
            )
            sensor_link_idx = links_idx[i_s]
            link_pos = links_state.pos[sensor_link_idx, i_b]
            link_quat = links_state.quat[sensor_link_idx, i_b]
            probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

            probe_start = sensor_probe_start[i_s]
            n_probes = n_probes_per_sensor[i_s]

            if dilate_max_delta_s > 0.0:
                # dilate_displacement = Σ_i min(dilate_max_delta, Δh_i) * (M - C_i) * exp(-λd ||M - C_i||²)
                for j in range(n_probes):
                    j_p = probe_start + j
                    delta_h = qd.min(dilate_max_delta_s, contact_buf[i_b, j_p, 3])
                    if delta_h > gs.qd_float(0.0):
                        C_j = qd.Vector(
                            [contact_buf[i_b, j_p, 0], contact_buf[i_b, j_p, 1], -contact_buf[i_b, j_p, 2]],
                            dt=gs.qd_float,
                        )
                        M_minus_C = probe_pos - C_j
                        dist_sq = (
                            M_minus_C[0] * M_minus_C[0] + M_minus_C[1] * M_minus_C[1] + M_minus_C[2] * M_minus_C[2]
                        )
                        displacement += M_minus_C * delta_h * qd.exp(-dilate_coefficients[i_s] * dist_sq)

            displacement_local = gu.qd_inv_transform_by_quat(displacement, link_quat)

            probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
            cache_start = sensor_cache_start[i_s]

            output[i_b, cache_start + probe_idx_in_sensor * 3 + 0] = displacement_local[0]
            output[i_b, cache_start + probe_idx_in_sensor * 3 + 1] = displacement_local[1]
            output[i_b, cache_start + probe_idx_in_sensor * 3 + 2] = qd.min(
                dilate_max_delta_s, contact_buf[i_b, i_p, 3]
            )


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (1 if n==0)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


@torch.jit.script
def _precompute_dilate_kernel_fft(
    dilate_coeff: float, grid_spacing: tuple[float, float], fft_n: tuple[int, int]
) -> torch.Tensor:
    """
    Build 2D convolution kernels Kx, Ky for the dilate sum (see _elastomer_displacement_grid_fft_dilate).
    K_x(Δx, Δy) = Δx*exp(-λ*r²), K_y = Δy*exp(-λ*r²) with r² = Δx²+Δy².

    Parameters
    ----------
    dilate_coeff: float
        Dilate coefficient λ
    grid_spacing: tuple[float, float]
        Grid spacing in x and y direction
    fft_n: tuple[int, int]
        FFT size in x and y direction

    Returns
    -------
    kernel_fft: torch.Tensor, shape (4, fft_nx * fft_ny)
        FFT kernels [Kx, Ky] as complex tensors
    """
    i = torch.arange(fft_n[0], dtype=gs.tc_float, device=gs.device)
    j = torch.arange(fft_n[1], dtype=gs.tc_float, device=gs.device)
    xx, yy = torch.meshgrid((i - fft_n[0] // 2) * grid_spacing[0], (j - fft_n[1] // 2) * grid_spacing[1], indexing="ij")
    g = torch.exp(torch.tensor(-dilate_coeff, dtype=gs.tc_float) * (xx * xx + yy * yy))
    k = torch.stack((xx * g, yy * g), dim=0)
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    return torch.fft.fft2(k)


@qd.func
def _func_query_contact_probes(
    probe_positions_local: qd.types.ndarray(),
    probe_normals_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radius: qd.types.ndarray(),
    probe_max_raycast_range: gs.qd_float,
    links_idx: qd.types.ndarray(),
    static_rigid_sim_config: qd.template(),
    links_state: array_class.LinksState,
    collider_state: array_class.ColliderState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    contact_buf: qd.types.ndarray(),
    contact_link_buf: qd.types.ndarray(),
    eps: gs.qd_float,
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = contact_buf.shape[0]

    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )

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

        contact_depth, contact_link = _func_query_contact_depth(
            i_b,
            probe_pos,
            probe_normal,
            probe_radius[i_p],
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

        if contact_depth > 0:
            contact_point = probe_pos - probe_normal * contact_depth
            contact_buf[i_b, i_p, 0] = contact_point[0]
            contact_buf[i_b, i_p, 1] = contact_point[1]
            contact_buf[i_b, i_p, 2] = contact_point[2]
            contact_buf[i_b, i_p, 3] = contact_depth
        else:
            contact_buf[i_b, i_p, 3] = gs.qd_float(0.0)
        contact_link_buf[i_b, i_p] = contact_link


def _elastomer_displacement_grid_fft_dilate(
    is_grid: torch.Tensor,
    contact_buf: torch.Tensor,
    sensor_probe_start: torch.Tensor,
    sensor_cache_start: torch.Tensor,
    grid_n: torch.Tensor,
    fft_depth_buffer: torch.Tensor,
    fft_kernel_list: list[torch.Tensor],
    dilate_coefficient: torch.Tensor,
    dilate_max_delta: torch.Tensor,
    grid_dilate_out_buffer: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Grid dilate via 2D FFT. Uses in-plane distance only in the kernel; scale by exp(-λ*H²)
    so magnitude matches non-grid formulation which uses full 3D distance (r²_2D + Δz²).
    """
    n_batches = contact_buf.shape[0]
    n_sensors = grid_n.shape[0]

    for i_s in range(n_sensors):
        if not is_grid[i_s]:
            continue

        g_nx = grid_n[i_s, 0].item()
        g_ny = grid_n[i_s, 1].item()

        probe_start = sensor_probe_start[i_s].item()
        cache_start = sensor_cache_start[i_s].item()
        lam = dilate_coefficient[i_s].item()
        max_h = dilate_max_delta[i_s].item()

        Kx = fft_kernel_list[i_s][0]
        Ky = fft_kernel_list[i_s][1]
        fft_nx, fft_ny = Kx.shape[0], Kx.shape[1]

        # H = contact depth grid (ix, iy); zero-pad to fft size for linear convolution.
        # Probes from generate_grid_points_on_plane flat as iy*nx+ix; view (g_ny, g_nx) then transpose -> (g_nx, g_ny).
        contact_slice = contact_buf[:, probe_start : probe_start + g_nx * g_ny, 3]
        H_buf = fft_depth_buffer[:, i_s, :fft_nx, :fft_ny]
        H_buf.zero_()
        H_buf[:, :g_nx, :g_ny].copy_(contact_slice.view(n_batches, g_ny, g_nx).transpose(1, 2))
        H_buf[:, :g_nx, :g_ny].clamp_(max=max_h)

        # D_x = real(ifft(fft(H) * fft(K_x))), D_y = real(ifft(fft(H) * fft(K_y)))
        H_fft = torch.fft.fft2(H_buf)
        disp_x = torch.fft.ifft2(H_fft * Kx).real[:, :g_nx, :g_ny]
        disp_y = torch.fft.ifft2(H_fft * Ky).real[:, :g_nx, :g_ny]

        # Match non-grid magnitude: non-grid uses exp(-λ*(r²_2D + Δz²)); 2D kernel only has r²_2D.
        H_cell = H_buf[:, :g_nx, :g_ny]
        scale = torch.exp(-lam * H_cell * H_cell)
        disp_x = disp_x * scale
        disp_y = disp_y * scale

        # Cache order is probe flat index iy*nx+ix; (g_nx, g_ny) transpose(1,2).reshape gives (g_ny, g_nx) -> iy*nx+ix.
        # Write to pre-allocated contiguous block then copy back to avoid illegal memory access on GPU.
        grid_size = g_nx * g_ny * 3
        out_block = grid_dilate_out_buffer[:, :grid_size]
        out_block[:, 0:grid_size:3] = disp_x.transpose(1, 2).reshape(n_batches, -1)
        out_block[:, 1:grid_size:3] = disp_y.transpose(1, 2).reshape(n_batches, -1)
        out_block[:, 2:grid_size:3] = torch.clamp(contact_slice, max=max_h)
        output[:, cache_start : cache_start + grid_size].copy_(out_block)


@qd.kernel(fastcache=gs.use_fastcache)
def _kernel_elastomer_displacement_grid_shear_twist(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    sensor_normals: qd.types.ndarray(),
    shear_coefficients: qd.types.ndarray(),
    twist_coefficients: qd.types.ndarray(),
    shear_max_delta: qd.types.ndarray(),
    twist_max_delta: qd.types.ndarray(),
    contact_buf: qd.types.ndarray(),
    contact_link_buf: qd.types.ndarray(),
    links_state: array_class.LinksState,
    dt: gs.qd_float,
    eps: gs.qd_float,
    output: qd.types.ndarray(),
):
    """Phase 3: add shear/twist displacement to grid elastomer output."""
    total_n_probes = probe_positions_local.shape[0]
    n_batches = contact_buf.shape[0]

    for i_b, i_p in qd.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]
        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        contact_link = contact_link_buf[i_b, i_p]
        contact_depth = contact_buf[i_b, i_p, 3]
        sensor_normal_local = qd.Vector(
            [sensor_normals[i_s, 0], sensor_normals[i_s, 1], sensor_normals[i_s, 2]], dt=gs.qd_float
        )

        if (
            contact_depth > gs.qd_float(0.0)
            and contact_link >= 0
            and (shear_max_delta[i_s] > 0.0 or twist_max_delta[i_s] > 0.0)
        ):
            shear_twist = _func_shear_twist_displacement(
                i_b,
                probe_pos,
                link_pos,
                link_quat,
                contact_link,
                links_state,
                sensor_link_idx,
                sensor_normal_local,
                shear_coefficients[i_s],
                twist_coefficients[i_s],
                shear_max_delta[i_s],
                twist_max_delta[i_s],
                dt,
                eps,
            )
            shear_twist_local = gu.qd_inv_transform_by_quat(shear_twist, link_quat)
            probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
            cache_start = sensor_cache_start[i_s]
            output[i_b, cache_start + probe_idx_in_sensor * 3 + 0] += shear_twist_local[0]
            output[i_b, cache_start + probe_idx_in_sensor * 3 + 1] += shear_twist_local[1]
            output[i_b, cache_start + probe_idx_in_sensor * 3 + 2] += shear_twist_local[2]


@dataclass
class KinematicTactileSensorMetadataMixin:
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_normals: torch.Tensor = make_tensor_field((0, 3))
    probe_radius: torch.Tensor = make_tensor_field((0,))
    probe_max_raycast_range: float = 0.0
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_int)
    total_n_probes: int = 0


KinematicTactileSensorMetadataMixinT = TypeVar(
    "KinematicTactileSensorMetadataMixinT", bound=KinematicTactileSensorMetadataMixin
)


class KinematicTactileSensorMixin(Generic[KinematicTactileSensorMetadataMixinT]):
    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        # Store n_probes before super().__init__() since _get_return_format() is called there
        self._probe_local_pos = torch.tensor(sensor_options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._n_probes = int(np.prod(self._probe_local_pos.shape[:-1]))

        super().__init__(sensor_options, sensor_idx, sensor_manager)

        self._debug_objects: list["Mesh | None"] = []
        self._probe_local_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        if self._probe_local_normal.ndim == 1:
            self._probe_local_normal = self._probe_local_normal.expand(self._n_probes, 3).contiguous()

    def build(self):
        super().build()

        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(self._n_probes, 3)
        )

        self._shared_metadata.probe_normals = concat_with_tensor(
            self._shared_metadata.probe_normals, self._probe_local_normal, expand=(self._n_probes, 3)
        )

        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor, self._n_probes, expand=(1,)
        )
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0,
            expand=(1,),
        )
        self._shared_metadata.sensor_probe_start = concat_with_tensor(
            self._shared_metadata.sensor_probe_start,
            self._shared_metadata.total_n_probes,
            expand=(1,),
        )
        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((self._n_probes,), self._idx, dtype=gs.tc_int, device=gs.device),
            expand=(self._n_probes,),
        )

        if self._shared_metadata.probe_max_raycast_range < gs.EPS:
            aabb = self._link.get_vAABB()
            self._shared_metadata.probe_max_raycast_range = torch.linalg.norm(aabb[1] - aabb[0], dim=-1).max().item()

        self._shared_metadata.total_n_probes += self._n_probes

        if isinstance(self._options.probe_radius, float):
            probe_radius_tensor = torch.full(
                (self._n_probes,), self._options.probe_radius, dtype=gs.tc_float, device=gs.device
            )
        else:
            probe_radius_tensor = torch.tensor(self._options.probe_radius, dtype=gs.tc_float, device=gs.device)

        self._shared_metadata.probe_radius = concat_with_tensor(
            self._shared_metadata.probe_radius, probe_radius_tensor, expand=(self._n_probes,)
        )

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: KinematicTactileSensorMetadataMixinT,
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

    def _draw_debug_probes(self, context: "RasterizerContext", get_magnitude: Callable[[int], float]):
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        link_pos = self._link.get_pos(env_idx).squeeze()
        link_quat = self._link.get_quat(env_idx).squeeze()

        data = self.read_ground_truth(env_idx)
        magnitude = tensor_to_array(get_magnitude(data))

        probe_world = tensor_to_array(gu.transform_by_trans_quat(self._probe_local_pos, link_pos, link_quat))
        probe_global_idx = int(self._shared_metadata.sensor_probe_start[self._idx])
        probe_radius = float(self._shared_metadata.probe_radius[probe_global_idx])
        for is_contact in (False, True):
            (probes_idx,) = np.nonzero(magnitude >= gs.EPS if is_contact else magnitude < gs.EPS)
            if probes_idx.size > 0:
                spheres_obj = context.draw_debug_spheres(
                    poss=probe_world[probes_idx],
                    radius=probe_radius,
                    color=self._options.debug_contact_color if is_contact else self._options.debug_sphere_color,
                )
                self._debug_objects.append(spheres_obj)

    @property
    def probe_local_pos(self) -> torch.Tensor:
        return self._probe_local_pos

    @property
    def probe_local_normal(self) -> torch.Tensor:
        return self._probe_local_normal

    @property
    def n_probes(self) -> int:
        return self._n_probes


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
class KinematicContactProbeMetadata(
    KinematicTactileSensorMetadataMixin, RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata
):
    stiffness: torch.Tensor = make_tensor_field((0,))


class KinematicContactProbe(
    KinematicTactileSensorMixin[KinematicContactProbeMetadata],
    RigidSensorMixin[KinematicContactProbeMetadata],
    NoisySensorMixin[KinematicContactProbeMetadata],
    Sensor[KinematicContactProbeOptions, KinematicContactProbeMetadata, KinematicContactProbeData],
):
    """Kinematic contact probe measuring penetration depth along the probe normal on collisions."""

    def __init__(self, sensor_options: KinematicContactProbeOptions, sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (self._n_probes,), (self._n_probes, 3)

    def build(self):
        super().build()

        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, self._options.stiffness, expand=(1,)
        )

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: KinematicTactileSensorMetadataMixinT, shared_ground_truth_cache: torch.Tensor
    ):
        solver = shared_metadata.solver
        collider_state = solver.collider._collider_state

        shared_ground_truth_cache.zero_()

        output = shared_ground_truth_cache.contiguous()
        _kernel_kinematic_contact_probe(
            shared_metadata.probe_positions,
            shared_metadata.probe_normals,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radius,
            shared_metadata.probe_max_raycast_range,
            shared_metadata.stiffness,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            solver._static_rigid_sim_config,
            solver.links_state,
            collider_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.fixed_verts_state,
            solver.free_verts_state,
            solver.verts_info,
            solver.faces_info,
            gs.EPS,
            output,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        self._draw_debug_probes(context, lambda data: data.penetration)


@dataclass
class ElastomerDisplacementSensorMetadata(
    KinematicTactileSensorMetadataMixin, RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata
):
    contact_buf: torch.Tensor = make_tensor_field((0, 0, 4))
    contact_link_buf: torch.Tensor = make_tensor_field((0, 0))

    dilate_coefficient: torch.Tensor = make_tensor_field((0,))
    shear_coefficient: torch.Tensor = make_tensor_field((0,))
    twist_coefficient: torch.Tensor = make_tensor_field((0,))

    dilate_max_delta: torch.Tensor = make_tensor_field((0,))
    shear_max_delta: torch.Tensor = make_tensor_field((0,))
    twist_max_delta: torch.Tensor = make_tensor_field((0,))

    sensor_normal: torch.Tensor = make_tensor_field((0, 3))

    # grid fft for dilation displacement
    is_grid: torch.Tensor = make_tensor_field((0,), dtype=gs.tc_bool)
    grid_n: torch.Tensor = make_tensor_field((0, 2), dtype=gs.tc_int)  # (nx, ny) per sensor
    grid_spacing: torch.Tensor = make_tensor_field((0, 2))  # (dx, dy) per sensor
    fft_kernel_list: list[torch.Tensor] = field(default_factory=list)  # each entry shape (4, fft_size)
    fft_depth_buffer: torch.Tensor = make_tensor_field((0, 0, 0, 0))  # used as H buffer in torch.fft dilate step
    grid_dilate_out_buffer: torch.Tensor = make_tensor_field((0, 0))  # buffer (B, max_grid_size) for FFT dilate copy


class ElastomerDisplacementSensor(
    KinematicTactileSensorMixin[ElastomerDisplacementSensorMetadata],
    RigidSensorMixin[ElastomerDisplacementSensorMetadata],
    NoisySensorMixin[ElastomerDisplacementSensorMetadata],
    Sensor[ElastomerDisplacementSensorOptions, ElastomerDisplacementSensorMetadata],
):
    def __init__(
        self, sensor_options: ElastomerDisplacementSensorOptions, sensor_idx: int, sensor_manager: "SensorManager"
    ):
        super().__init__(sensor_options, sensor_idx, sensor_manager)

        self._is_grid = self._probe_local_pos.ndim > 2
        self._shape = self._probe_local_pos.shape[:-1]
        self._probe_local_pos = self._probe_local_pos.reshape(-1, 3)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._n_probes, 3)

    def build(self):
        super().build()

        B = self._manager._sim._B
        total_n_probes = self._shared_metadata.total_n_probes
        self._shared_metadata.contact_buf = torch.empty((B, total_n_probes, 4), dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.contact_link_buf = torch.empty((B, total_n_probes), dtype=gs.tc_int, device=gs.device)

        self._shared_metadata.dilate_coefficient = concat_with_tensor(
            self._shared_metadata.dilate_coefficient, self._options.dilate_coefficient
        )
        self._shared_metadata.shear_coefficient = concat_with_tensor(
            self._shared_metadata.shear_coefficient, self._options.shear_coefficient
        )
        self._shared_metadata.twist_coefficient = concat_with_tensor(
            self._shared_metadata.twist_coefficient, self._options.twist_coefficient
        )
        self._shared_metadata.dilate_max_delta = concat_with_tensor(
            self._shared_metadata.dilate_max_delta, self._options.dilate_max_delta
        )
        self._shared_metadata.shear_max_delta = concat_with_tensor(
            self._shared_metadata.shear_max_delta, self._options.shear_max_delta
        )
        self._shared_metadata.twist_max_delta = concat_with_tensor(
            self._shared_metadata.twist_max_delta, math.radians(self._options.twist_max_delta)
        )
        self._shared_metadata.sensor_normal = concat_with_tensor(
            self._shared_metadata.sensor_normal,
            self._probe_local_normal.mean(dim=0) / self._probe_local_normal.norm(),
            expand=(1, 3),
        )
        self._shared_metadata.is_grid = concat_with_tensor(self._shared_metadata.is_grid, self._is_grid)

        grid_n = torch.tensor((0, 0), dtype=gs.tc_int, device=gs.device)
        grid_spacing = torch.tensor((0, 0), dtype=gs.tc_float, device=gs.device)
        kernel_fft = torch.tensor(0, dtype=gs.tc_float, device=gs.device)
        if self._is_grid:
            nx, ny = int(self._shape[1]), int(self._shape[0])
            grid_n = torch.tensor((nx, ny), dtype=gs.tc_int, device=gs.device)
            pos_xy = self._probe_local_pos[:, :2]
            extent = pos_xy.amax(dim=0) - pos_xy.amin(dim=0)
            grid_spacing = extent.to(gs.tc_float) / torch.tensor(
                [max(nx - 1, 1), max(ny - 1, 1)], dtype=gs.tc_float, device=gs.device
            )
            # Use linear-convolution padding (>= 2*g - 1) to avoid circular wrap across sensor borders.
            fft_n = tuple(_next_pow2(2 * n - 1) for n in (nx, ny))
            kernel_fft = _precompute_dilate_kernel_fft(self._options.dilate_coefficient, grid_spacing.tolist(), fft_n)
            n_sensors = len(self._shared_metadata.dilate_coefficient)
            prev = self._shared_metadata.fft_depth_buffer.shape
            max_fft_n = (max(fft_n[0], prev[2]), max(fft_n[1], prev[3]))
            self._shared_metadata.fft_depth_buffer = torch.zeros(
                (self._manager._sim._B, n_sensors, max_fft_n[0], max_fft_n[1]),
                dtype=gs.tc_float,
                device=gs.device,
            )
            grid_size = nx * ny * 3
            out_buf = self._shared_metadata.grid_dilate_out_buffer
            if out_buf.numel() == 0 or out_buf.shape[1] < grid_size:
                new_size = max(out_buf.shape[1] if out_buf.numel() > 0 else 0, grid_size)
                self._shared_metadata.grid_dilate_out_buffer = torch.empty(
                    (B, new_size), dtype=gs.tc_float, device=gs.device
                )

        self._shared_metadata.grid_n = concat_with_tensor(self._shared_metadata.grid_n, grid_n, expand=(1, 2))
        self._shared_metadata.grid_spacing = concat_with_tensor(
            self._shared_metadata.grid_spacing, grid_spacing, expand=(1, 2)
        )
        self._shared_metadata.fft_kernel_list.append(kernel_fft)

    @classmethod
    def _update_shared_ground_truth_cache(
        cls,
        shared_metadata: ElastomerDisplacementSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
    ):
        solver = shared_metadata.solver

        output = shared_ground_truth_cache.contiguous()
        _kernel_elastomer_displacement(
            shared_metadata.is_grid,
            shared_metadata.probe_positions,
            shared_metadata.probe_normals,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radius,
            shared_metadata.probe_max_raycast_range,
            shared_metadata.dilate_coefficient,
            shared_metadata.dilate_max_delta,
            shared_metadata.links_idx,
            shared_metadata.sensor_probe_start,
            shared_metadata.sensor_cache_start,
            shared_metadata.n_probes_per_sensor,
            solver._static_rigid_sim_config,
            solver.links_state,
            solver.collider._collider_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.fixed_verts_state,
            solver.free_verts_state,
            solver.verts_info,
            solver.faces_info,
            shared_metadata.contact_buf,
            shared_metadata.contact_link_buf,
            gs.EPS,
            output,
        )

        _elastomer_displacement_grid_fft_dilate(
            shared_metadata.is_grid,
            shared_metadata.contact_buf,
            shared_metadata.sensor_probe_start,
            shared_metadata.sensor_cache_start,
            shared_metadata.grid_n,
            shared_metadata.fft_depth_buffer,
            shared_metadata.fft_kernel_list,
            shared_metadata.dilate_coefficient,
            shared_metadata.dilate_max_delta,
            shared_metadata.grid_dilate_out_buffer,
            output,
        )
        _kernel_elastomer_displacement_grid_shear_twist(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.sensor_normal,
            shared_metadata.shear_coefficient,
            shared_metadata.twist_coefficient,
            shared_metadata.shear_max_delta,
            shared_metadata.twist_max_delta,
            shared_metadata.contact_buf,
            shared_metadata.contact_link_buf,
            solver.links_state,
            solver._sim.dt,
            gs.EPS,
            output,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        self._draw_debug_probes(context, lambda data: torch.linalg.norm(data, dim=-1))
