"""Kinematic contact probe for contact detection without physics side effects."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Sequence, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import func_check_weld_constraint, func_point_in_geom_aabb, mpr
from genesis.engine.solvers.rigid.rigid_solver import func_update_all_verts
from genesis.options.sensors import KinematicContactProbe as KinematicContactProbeOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array
from genesis.utils.raycast_ti import get_triangle_vertices, ray_triangle_intersection

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
    contypes: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    conaffinities: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)

    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    probe_normals: torch.Tensor = make_tensor_field((0, 3))

    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    sensor_probe_start: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)
    total_n_probes: int = 0

    default_raycast_depth: float = 1.0

    # Contact filtering - stores geoms in contact per env
    contact_geoms: torch.Tensor = make_tensor_field((0, 0), dtype=torch.int32)
    n_contact_geoms: torch.Tensor = make_tensor_field((0,), dtype=torch.int32)


@register_sensor(KinematicContactProbeOptions, KinematicContactProbeMetadata, KinematicContactProbeData)
@ti.data_oriented
class KinematicContactProbe(
    RigidSensorMixin[KinematicContactProbeMetadata],
    NoisySensorMixin[KinematicContactProbeMetadata],
    Sensor[KinematicContactProbeMetadata],
):
    """Kinematic contact probe measuring penetration depth along the probe normal on collisions."""

    _update_on_read: bool = True

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

        self._debug_sphere_objects: list["Mesh | None"] = []
        self._debug_contact_objects: list["Mesh | None"] = []
        self._probe_local_pos = torch.tensor(self._options.probe_local_pos, dtype=gs.tc_float, device=gs.device)
        self._probe_local_normal = torch.tensor(self._options.probe_local_normal, dtype=gs.tc_float, device=gs.device)
        norms = self._probe_local_normal.norm(dim=1, keepdim=True).clamp(min=gs.EPS)
        self._probe_local_normal /= norms

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
            torch.full((n_probes,), sensor_idx, dtype=torch.int32, device=gs.device),
            expand=(n_probes,),
            dim=0,
        )

        self._shared_metadata.probe_positions = concat_with_tensor(
            self._shared_metadata.probe_positions, self._probe_local_pos, expand=(n_probes, 3), dim=0
        )

        self._shared_metadata.probe_normals = concat_with_tensor(
            self._shared_metadata.probe_normals, self._probe_local_normal, expand=(n_probes, 3), dim=0
        )

        self._shared_metadata.total_n_probes += n_probes

        # Handle radius as either single float or per-probe sequence
        if isinstance(self._options.radius, Sequence):
            radii_tensor = torch.tensor(self._options.radius, dtype=gs.tc_float, device=gs.device)
        else:
            radii_tensor = torch.full((n_probes,), self._options.radius, dtype=gs.tc_float, device=gs.device)

        self._shared_metadata.radii = concat_with_tensor(
            self._shared_metadata.radii, radii_tensor, expand=(n_probes,), dim=0
        )
        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, self._options.stiffness, expand=(1,), dim=0
        )
        self._shared_metadata.contypes = concat_with_tensor(
            self._shared_metadata.contypes, self._options.contype, expand=(1,), dim=0
        )
        self._shared_metadata.conaffinities = concat_with_tensor(
            self._shared_metadata.conaffinities, self._options.conaffinity, expand=(1,), dim=0
        )

        if self._shared_metadata.contact_geoms.numel() == 0:
            solver = self._shared_metadata.solver

            self._shared_metadata.contact_geoms = torch.full(
                (solver._B, min(solver.n_geoms, solver.max_collision_pairs)),
                -1,
                dtype=torch.int32,
                device=gs.device,
            )
            self._shared_metadata.n_contact_geoms = torch.zeros(solver._B, dtype=torch.int32, device=gs.device)

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

        shared_ground_truth_cache.zero_()

        # Get contacts and extract unique geoms in contact with probe links
        contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)

        shared_metadata.contact_geoms.fill_(-1)
        shared_metadata.n_contact_geoms.zero_()

        link_a, link_b = contacts["link_a"], contacts["link_b"]
        geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
        n_batches = link_a.shape[0]
        max_slots = shared_metadata.contact_geoms.shape[1]

        candidates = torch.stack(
            (
                geom_b.masked_fill(~torch.isin(link_a, shared_metadata.links_idx), -1),
                geom_a.masked_fill(~torch.isin(link_b, shared_metadata.links_idx), -1),
            ),
            dim=2,
        ).reshape(n_batches, -1)

        valid = candidates >= 0
        if valid.any():
            env_idx, col_idx = valid.nonzero(as_tuple=True)
            geom_vals = candidates[env_idx, col_idx]

            if geom_vals.numel() > 0:
                max_geoms = solver.n_geoms
                unique_keys = torch.unique(env_idx * max_geoms + geom_vals)

                unique_env = unique_keys // max_geoms
                unique_geom = unique_keys % max_geoms

                sort_idx = torch.argsort(unique_env)
                unique_env = unique_env[sort_idx]
                unique_geom = unique_geom[sort_idx]

                env_vals, counts = torch.unique_consecutive(unique_env, return_counts=True)
                group_starts = torch.cumsum(counts, 0) - counts
                positions = torch.arange(unique_env.numel(), device=unique_env.device) - torch.repeat_interleave(
                    group_starts, counts
                )
                keep = positions < max_slots

                if keep.any():
                    shared_metadata.contact_geoms[unique_env[keep], positions[keep]] = unique_geom[keep].to(torch.int32)

                shared_metadata.n_contact_geoms[env_vals] = torch.clamp(counts, max=max_slots).to(torch.int32)

        _kernel_kinematic_contact_probe(
            probe_positions_local=shared_metadata.probe_positions,
            probe_normals_local=shared_metadata.probe_normals,
            probe_sensor_idx=shared_metadata.probe_sensor_idx,
            links_state=solver.links_state,
            radii=shared_metadata.radii,
            stiffness=shared_metadata.stiffness,
            links_idx=shared_metadata.links_idx,
            contypes=shared_metadata.contypes,
            conaffinities=shared_metadata.conaffinities,
            n_probes_per_sensor=shared_metadata.n_probes_per_sensor,
            sensor_cache_start=shared_metadata.sensor_cache_start,
            sensor_probe_start=shared_metadata.sensor_probe_start,
            contact_geoms=shared_metadata.contact_geoms,
            n_contact_geoms=shared_metadata.n_contact_geoms,
            geoms_state=solver.geoms_state,
            geoms_info=solver.geoms_info,
            static_rigid_sim_config=shared_metadata.solver._static_rigid_sim_config,
            rigid_global_info=solver._rigid_global_info,
            constraint_state=solver.constraint_solver.constraint_state,
            equalities_info=solver.equalities_info,
            fixed_verts_state=solver.fixed_verts_state,
            free_verts_state=solver.free_verts_state,
            verts_info=solver.verts_info,
            faces_info=solver.faces_info,
            default_raycast_depth=shared_metadata.default_raycast_depth,
            output=shared_ground_truth_cache,
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

        for obj in self._debug_sphere_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_sphere_objects = []
        for obj in self._debug_contact_objects:
            if obj is not None:
                context.clear_debug_object(obj)
        self._debug_contact_objects = []

        if self._link is None:
            return

        link_pos = self._link.get_pos(env_idx).reshape((3,))
        link_quat = self._link.get_quat(env_idx).reshape((4,))
        data = self.read_ground_truth(env_idx)

        for i, pos in enumerate(self._probe_local_pos):
            probe_world = link_pos + gu.transform_by_quat(pos, link_quat)

            # Get per-probe radius from shared metadata
            probe_global_idx = self._shared_metadata.sensor_probe_start[self._idx].item() + i
            probe_radius = self._shared_metadata.radii[probe_global_idx].item()

            sphere_obj = context.draw_debug_sphere(
                pos=tensor_to_array(probe_world),
                radius=probe_radius,
                color=self._options.debug_sphere_color,
            )
            self._debug_sphere_objects.append(sphere_obj)

            penetration = data.penetration[i].item() if data.penetration.dim() > 0 else data.penetration.item()

            if penetration > gs.EPS:
                contact_obj = context.draw_debug_sphere(
                    pos=probe_world,
                    radius=probe_radius * 0.3,
                    color=self._options.debug_contact_color,
                )
                self._debug_contact_objects.append(contact_obj)
            else:
                self._debug_contact_objects.append(None)


@ti.func
def _raycast_geom_faces(
    probe_pos: ti.types.vector(3, gs.ti_float),
    ray_dir: ti.types.vector(3, gs.ti_float),
    max_range: gs.ti_float,
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_info: array_class.GeomsInfo,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
):
    """Raycast against all faces of a single geom. Returns closest hit distance or -1."""
    closest_dist = gs.ti_float(-1.0)
    face_start = geoms_info.face_start[i_g]
    face_end = geoms_info.face_end[i_g]

    for i_f in range(face_start, face_end):
        tri_verts = get_triangle_vertices(i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state)
        result = ti.Vector.zero(gs.ti_float, 4)
        result = ray_triangle_intersection(probe_pos, ray_dir, tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2])
        hit = result[3]  # 1.0 if hit, 0.0 if no hit
        t = result[0]
        if hit > 0.5 and t <= max_range:
            if closest_dist < 0.0 or t < closest_dist:
                closest_dist = t
    return closest_dist


@ti.func
def _sphere_geom_faces_intersection(
    sphere_pos: ti.types.vector(3, gs.ti_float),
    radius: gs.ti_float,
    selection_vector: ti.types.vector(3, gs.ti_float),
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_info: array_class.GeomsInfo,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
):
    """Find the point of intersection between between the sphere and geom. Return the intersection point that is furthest along the selection_vector direction."""
    max_penetration = gs.ti_float(-1.0)
    face_start = geoms_info.face_start[i_g]
    face_end = geoms_info.face_end[i_g]

    for i_f in range(face_start, face_end):
        tri_verts = get_triangle_vertices(i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state)
        v0 = tri_verts[:, 0]
        v1 = tri_verts[:, 1]
        v2 = tri_verts[:, 2]

        # Find closest point on triangle to sphere center
        closest_point = _closest_point_on_triangle(sphere_pos, v0, v1, v2)

        # Check if sphere intersects this triangle
        diff = closest_point - sphere_pos
        dist_sq = diff.dot(diff)

        if dist_sq <= radius * radius:
            # The penetration depth we care about is the distance along the selection vector
            penetration = diff.dot(selection_vector)
            if penetration > max_penetration:
                max_penetration = penetration

    return max_penetration


@ti.func
def _closest_point_on_triangle(
    point: ti.types.vector(3, gs.ti_float),
    v0: ti.types.vector(3, gs.ti_float),
    v1: ti.types.vector(3, gs.ti_float),
    v2: ti.types.vector(3, gs.ti_float),
) -> ti.types.vector(3, gs.ti_float):
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
                            denom = gs.ti_float(1.0) / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            closest = v0 + v * ab + w * ac

    return closest


@ti.kernel
def _kernel_kinematic_contact_probe(
    probe_positions_local: ti.types.ndarray(),
    probe_normals_local: ti.types.ndarray(),
    probe_sensor_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    radii: ti.types.ndarray(),
    stiffness: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    contypes: ti.types.ndarray(),
    conaffinities: ti.types.ndarray(),
    n_probes_per_sensor: ti.types.ndarray(),
    sensor_cache_start: ti.types.ndarray(),
    sensor_probe_start: ti.types.ndarray(),
    # Contact filtering
    contact_geoms: ti.types.ndarray(),
    n_contact_geoms: ti.types.ndarray(),
    # Geometry data for support functions and raycasting
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    default_raycast_depth: ti.f32,
    output: ti.types.ndarray(),
):
    func_update_all_verts(
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output.shape[0]

    for i_b, i_p in ti.ndrange(n_batches, total_n_probes):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = ti.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = ti.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        radius = radii[i_p]
        stiff = stiffness[i_s]
        sensor_link_idx = links_idx[i_s]
        sensor_contype = contypes[i_s]
        sensor_conaffinity = conaffinities[i_s]

        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.ti_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.ti_transform_by_quat(probe_normal_local, link_quat)

        max_penetration = gs.ti_float(0.0)

        for i_cg in range(n_contact_geoms[i_b]):
            i_g = contact_geoms[i_b, i_cg]

            # Filter geoms based on AABB check, self-link, contype/conaffinity
            if not func_check_collision_filter(
                i_g,
                i_b,
                probe_pos,
                radius,
                sensor_link_idx,
                sensor_contype,
                sensor_conaffinity,
                geoms_state,
                geoms_info,
                rigid_global_info,
                constraint_state,
                equalities_info,
            ):
                continue

            # Step 1: Direct raycast against faces of each contact geom
            hit_dist = _raycast_geom_faces(
                probe_pos,
                -probe_normal,
                default_raycast_depth,
                i_g,
                i_b,
                geoms_info,
                faces_info,
                verts_info,
                fixed_verts_state,
                free_verts_state,
            )
            if hit_dist >= 0.0:
                if max_penetration == 0.0 or hit_dist < max_penetration:
                    max_penetration = hit_dist

            # Step 2: Collision with probe sphere for probes
            penetration = _sphere_geom_faces_intersection(
                probe_pos,
                radius,
                -probe_normal,
                i_g,
                i_b,
                geoms_info,
                faces_info,
                verts_info,
                fixed_verts_state,
                free_verts_state,
            )
            if penetration >= 0.0:
                if penetration > max_penetration:
                    max_penetration = penetration

        force_local = ti.Vector.zero(gs.ti_float, 3)
        if max_penetration > 0:
            force_local = stiff * max_penetration * -probe_normal_local

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]

        output[i_b, cache_start + probe_idx_in_sensor] = max_penetration
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 0] = force_local[0]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 1] = force_local[1]
        output[i_b, cache_start + n_probes + probe_idx_in_sensor * 3 + 2] = force_local[2]


@ti.func
def func_check_collision_filter(
    i_g: ti.i32,
    i_b: ti.i32,
    probe_pos: ti.types.vector(3, gs.ti_float),
    probe_radius: gs.ti_float,
    other_link_idx: ti.i32,
    other_contype: ti.i32,
    other_conaffinity: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
):
    valid = True
    # Early AABB check to skip unnecessary filtering for geoms that are too far from the probe
    if not func_point_in_geom_aabb(i_g, i_b, geoms_state, probe_pos, probe_radius):
        valid = False

    # Self-link filtering
    geom_link_idx = geoms_info.link_idx[i_g]
    if valid:
        if geom_link_idx == other_link_idx:
            valid = False

    # Contype and conaffinity bitmask filtering
    if valid:
        geom_contype = geoms_info.contype[i_g]
        geom_conaffinity = geoms_info.conaffinity[i_g]
        cond1 = (geom_contype & other_conaffinity) != 0
        cond2 = (other_contype & geom_conaffinity) != 0
        if not (cond1 and cond2):
            valid = False

    # Weld constraint filtering
    if valid:
        if not func_check_weld_constraint(
            geom_link_idx,
            other_link_idx,
            i_b,
            rigid_global_info,
            constraint_state,
            equalities_info,
        ):
            valid = False

    return valid
