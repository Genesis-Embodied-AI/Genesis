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
def _probe_geom_penetration(
    probe_pos: gs.qd_vec3,
    probe_normal: gs.qd_vec3,
    radius: gs.qd_float,
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
    radius_sq = radius * radius

    for i_f in range(face_start, face_end):
        tri_verts = get_triangle_vertices(i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state)
        v0 = tri_verts[:, 0]
        v1 = tri_verts[:, 1]
        v2 = tri_verts[:, 2]

        if radius > gs.EPS:
            # Sphere-triangle test (closest point)
            closest_point = _closest_point_on_triangle(probe_pos, v0, v1, v2)
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
def _closest_point_on_triangle(
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

        max_penetration = gs.qd_float(0.0)

        # Iterate over contacts directly from collider state
        n_contacts = collider_state.n_contacts[i_b]
        for i_c in range(n_contacts):
            c_link_a = collider_state.contact_data.link_a[i_c, i_b]
            c_link_b = collider_state.contact_data.link_b[i_c, i_b]
            c_geom_a = collider_state.contact_data.geom_a[i_c, i_b]
            c_geom_b = collider_state.contact_data.geom_b[i_c, i_b]

            # Check if either side of this contact involves one of our sensor links;
            for side in qd.static(range(2)):
                contact_link = c_link_a if side == 0 else c_link_b
                i_g = c_geom_b if side == 0 else c_geom_a

                # Is this contact relevant to this sensor?
                if contact_link == sensor_link_idx and func_point_in_geom_aabb(
                    geoms_state, i_g, i_b, probe_pos, radius
                ):
                    # Raycast + sphere penetration test per geom
                    penetration = _probe_geom_penetration(
                        probe_pos,
                        probe_normal,
                        radius,
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
