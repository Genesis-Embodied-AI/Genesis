"""
Kinematic contact probe for contact detection without physics side effects.

This module implements a kinematic contact probe that uses SDF-based queries to detect
contact information (penetration, area, position, normal, force) for a virtual
sensing sphere attached to a rigid link.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
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
    from genesis.engine.solvers import RigidSolver
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


class KinematicContactProbeData(NamedTuple):
    """Data returned by the kinematic contact probe."""

    penetration: torch.Tensor  # (n_envs, n_probes) or (n_probes,) - depth in meters (0 if no contact)
    position: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - contact position in link frame
    normal: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - contact normal in link frame
    force: torch.Tensor  # (n_envs, n_probes, 3) or (n_probes, 3) - F = stiffness * penetration * probe_normal


@dataclass
class KinematicContactProbeMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all kinematic contact probes."""

    # Per-sensor parameters (n_sensors,)
    radii: torch.Tensor = make_tensor_field((0,))
    stiffness: torch.Tensor = make_tensor_field((0,))
    contypes: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.int32)
    conaffinities: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.int32)

    # Per-probe metadata for global probe indexing (total_n_probes,)
    # Which sensor each probe belongs to
    probe_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.int32)
    # Probe positions in link-local frame (total_n_probes, 3)
    probe_positions: torch.Tensor = make_tensor_field((0, 3))
    # Probe normals in link-local frame (total_n_probes, 3)
    probe_normals: torch.Tensor = make_tensor_field((0, 3))

    # Per-sensor indexing info
    # Number of probes per sensor (n_sensors,)
    n_probes_per_sensor: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.int32)
    # Starting index in the output cache for each sensor (n_sensors,)
    sensor_cache_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: torch.int32)
    # Total number of probes across all sensors
    total_n_probes: int = 0


@register_sensor(KinematicContactProbeOptions, KinematicContactProbeMetadata, KinematicContactProbeData)
@ti.data_oriented
class KinematicContactProbe(
    RigidSensorMixin[KinematicContactProbeMetadata],
    NoisySensorMixin[KinematicContactProbeMetadata],
    Sensor[KinematicContactProbeMetadata],
):
    """
    Kinematic contact probe that detects contact information using SDF queries.

    The probe performs SDF-based contact queries without affecting the physics simulation.
    It returns penetration depth, contact area, position, normal, and estimated force.

    This probe uses lazy evaluation: collision queries are only executed when read() is called,
    not on every simulation step.
    """

    # Enable lazy evaluation - skip updates in SensorManager.step()
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
        # Note: solver assignment is handled by parent RigidSensorMixin.build()

        # Get probe positions and normals from options
        probe_positions = self._options.get_probe_positions()
        probe_normals = self._options.get_probe_normals()
        n_probes = len(probe_positions)

        # Store number of probes for this sensor
        self._shared_metadata.n_probes_per_sensor = concat_with_tensor(
            self._shared_metadata.n_probes_per_sensor,
            torch.tensor([n_probes], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

        # Store starting index in the cache for this sensor
        # Cache layout per sensor: n_probes * 10 values
        current_cache_start = sum(self._shared_metadata.cache_sizes[:-1]) if self._shared_metadata.cache_sizes else 0
        self._shared_metadata.sensor_cache_start = concat_with_tensor(
            self._shared_metadata.sensor_cache_start,
            torch.tensor([current_cache_start], dtype=torch.int32, device=gs.device),
            expand=(1,),
            dim=0,
        )

        # Batch store per-probe metadata (more efficient than loop)
        sensor_idx = self._idx

        # Batch append sensor indices for all probes
        self._shared_metadata.probe_sensor_idx = concat_with_tensor(
            self._shared_metadata.probe_sensor_idx,
            torch.full((n_probes,), sensor_idx, dtype=torch.int32, device=gs.device),
            expand=(n_probes,),
            dim=0,
        )

        # Batch append positions
        positions_tensor = torch.tensor(probe_positions, dtype=gs.tc_float, device=gs.device)
        self._shared_metadata.probe_positions = torch.cat(
            [self._shared_metadata.probe_positions, positions_tensor], dim=0
        )

        # Batch append normals with safe normalization (avoid division by zero)
        normals_tensor = torch.tensor(probe_normals, dtype=gs.tc_float, device=gs.device)
        norms = normals_tensor.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normals_tensor = normals_tensor / norms
        self._shared_metadata.probe_normals = torch.cat([self._shared_metadata.probe_normals, normals_tensor], dim=0)

        self._shared_metadata.total_n_probes += n_probes

        # Store sensor-specific parameters
        self._shared_metadata.radii = concat_with_tensor(
            self._shared_metadata.radii, self._options.radius, expand=(1,), dim=0
        )
        self._shared_metadata.stiffness = concat_with_tensor(
            self._shared_metadata.stiffness, self._options.stiffness, expand=(1,), dim=0
        )
        # Store collision filtering bitmasks
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

        # Note: Support field is required for non-primitive geoms (MESH, CONVEX)
        # The support field is initialized automatically by the collider

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        # For n_probes:
        # penetration (n_probes,), position (n_probes, 3), normal (n_probes, 3), force (n_probes, 3)
        n = self._n_probes
        return (n,), (n, 3), (n, 3), (n, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: KinematicContactProbeMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Compute kinematic contact probe readings using support function queries.

        Uses a Taichi kernel to query support points for all geoms and all probes.
        """
        assert shared_metadata.solver is not None
        solver = shared_metadata.solver

        n_sensors = len(shared_metadata.radii)
        total_n_probes = shared_metadata.total_n_probes
        B = max(solver.n_envs, 1)
        n_geoms = solver.n_geoms

        if n_sensors == 0 or total_n_probes == 0:
            return

        # Get link transforms for each sensor
        links_pos = solver.get_links_pos(links_idx=shared_metadata.links_idx)
        links_quat = solver.get_links_quat(links_idx=shared_metadata.links_idx)

        if solver.n_envs == 0:
            links_pos = links_pos[None]
            links_quat = links_quat[None]

        # Reshape to (B, n_sensors, 3) and (B, n_sensors, 4)
        links_pos = links_pos.reshape(B, n_sensors, 3)
        links_quat = links_quat.reshape(B, n_sensors, 4)

        # Initialize output cache
        shared_ground_truth_cache.zero_()

        # Call Taichi kernel for support-based queries
        # The kernel iterates over all probes globally
        _kernel_kinematic_contact_probe_support_query(
            # Per-probe data
            probe_positions_local=shared_metadata.probe_positions.contiguous(),  # (total_n_probes, 3)
            probe_normals_local=shared_metadata.probe_normals.contiguous(),  # (total_n_probes, 3)
            probe_sensor_idx=shared_metadata.probe_sensor_idx.contiguous(),  # (total_n_probes,)
            # Per-sensor data
            links_pos=links_pos.contiguous(),  # (B, n_sensors, 3)
            links_quat=links_quat.contiguous(),  # (B, n_sensors, 4)
            radii=shared_metadata.radii.contiguous(),  # (n_sensors,)
            stiffness=shared_metadata.stiffness.contiguous(),  # (n_sensors,)
            links_idx=shared_metadata.links_idx.contiguous(),  # (n_sensors,)
            contypes=shared_metadata.contypes.contiguous(),  # (n_sensors,)
            conaffinities=shared_metadata.conaffinities.contiguous(),  # (n_sensors,)
            n_probes_per_sensor=shared_metadata.n_probes_per_sensor.contiguous(),  # (n_sensors,)
            sensor_cache_start=shared_metadata.sensor_cache_start.contiguous(),  # (n_sensors,)
            # Scene data
            n_geoms=n_geoms,
            geoms_state=solver.geoms_state,
            geoms_info=solver.geoms_info,
            rigid_global_info=solver._rigid_global_info,
            constraint_state=solver.constraint_solver.constraint_state,
            equalities_info=solver.equalities_info,
            support_field_info=solver.collider._support_field._support_field_info,
            # Output
            output=shared_ground_truth_cache.contiguous(),  # (B, total_cache_size)
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
        """Internal method to read sensor data, shared by read() and read_ground_truth()."""
        # Trigger lazy cache update for this sensor type (if not already updated in step())
        if not self._manager._has_delay_configured(type(self)):
            self._manager.update_sensor_type_cache(type(self))
        return self._get_formatted_data(
            self._manager.get_cloned_from_cache(self, is_ground_truth=is_ground_truth), envs_idx
        )

    @gs.assert_built
    def read(self, envs_idx=None):
        """
        Read the sensor data (with noise applied if applicable).

        This triggers a lazy evaluation - the collision query is executed on demand
        rather than on every simulation step (unless delay is configured, in which case
        updates happen every step to maintain the delay buffer).
        """
        return self._read_internal(envs_idx, is_ground_truth=False)

    @gs.assert_built
    def read_ground_truth(self, envs_idx=None):
        """
        Read the ground truth sensor data (without noise).

        This triggers a lazy evaluation - the collision query is executed on demand
        rather than on every simulation step (unless delay is configured, in which case
        updates happen every step to maintain the delay buffer).
        """
        return self._read_internal(envs_idx, is_ground_truth=True)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug visualization of the kinematic contact probes.

        Shows the sensing spheres and contact points (if in contact) for all probes.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        # Clear previous debug objects
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

        # Get link transform
        link_pos = self._link.get_pos(env_idx).reshape((3,))
        link_quat = self._link.get_quat(env_idx).reshape((4,))

        # Get probe positions and read sensor data
        probe_positions = self._options.get_probe_positions()
        data = self.read(env_idx)

        # Draw each probe
        for i, pos in enumerate(probe_positions):
            # Compute probe world position
            offset_pos = torch.tensor(pos, dtype=gs.tc_float, device=gs.device)
            probe_world = link_pos + transform_by_quat(offset_pos, link_quat)

            # Draw sensing sphere
            sphere_obj = context.draw_debug_sphere(
                pos=tensor_to_array(probe_world),
                radius=self._options.radius,
                color=self._options.debug_sphere_color,
            )
            self.debug_sphere_objects.append(sphere_obj)

            # Check for contact on this probe
            penetration = data.penetration[i].item() if data.penetration.dim() > 0 else data.penetration.item()

            if penetration > 0:
                # Transform contact position from link frame to world frame
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
    """Check if a point is within the expanded AABB of a geom (inclusive)."""
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
    """
    Check if a geom should be considered for collision with the sensor.

    Uses proper MuJoCo-style bitmask collision filtering:
    - (geom.contype & sensor.conaffinity) != 0 AND
    - (sensor.contype & geom.conaffinity) != 0

    Also skips:
    - Geoms belonging to the sensor's own link (self-detection filter)
    - Geoms whose link is in a weld equality constraint with the sensor's link

    Returns True if the geom should be queried, False if it should be skipped.
    """
    is_valid = True

    # Skip geoms belonging to the sensor's own link (avoid self-detection)
    geom_link_idx = geoms_info.link_idx[i_g]
    if geom_link_idx == sensor_link_idx:
        is_valid = False

    # Apply MuJoCo-style bitmask collision filtering
    if is_valid:
        geom_contype = geoms_info.contype[i_g]
        geom_conaffinity = geoms_info.conaffinity[i_g]

        # Both conditions must be true for collision to be considered:
        # 1. geom's contype overlaps with sensor's conaffinity
        # 2. sensor's contype overlaps with geom's conaffinity
        cond1 = (geom_contype & sensor_conaffinity) != 0
        cond2 = (sensor_contype & geom_conaffinity) != 0
        if not (cond1 and cond2):
            is_valid = False

    # Filter out geoms whose link is in a dynamic weld equality constraint with the sensor's link
    # This mirrors the logic in broadphase.py:73-79
    if is_valid:
        for i_eq in range(rigid_global_info.n_equalities[None], constraint_state.ti_n_equalities[i_b]):
            if equalities_info.eq_type[i_eq, i_b] == gs.EQUALITY_TYPE.WELD:
                weld_link_a = equalities_info.eq_obj1id[i_eq, i_b]
                weld_link_b = equalities_info.eq_obj2id[i_eq, i_b]
                # Check if sensor link and geom link are the pair in this weld constraint
                if (weld_link_a == sensor_link_idx and weld_link_b == geom_link_idx) or (
                    weld_link_a == geom_link_idx and weld_link_b == sensor_link_idx
                ):
                    is_valid = False
                    break  # No need to check remaining constraints

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
    """
    Get the support point on a geom in the given direction.

    Returns support_point_world.
    Uses the same approach as support_driver in GJK for all geom types.
    """
    geom_type = geoms_info.type[i_g]
    g_pos = geoms_state.pos[i_g, i_b]
    g_quat = geoms_state.quat[i_g, i_b]

    support_world = ti.Vector.zero(gs.ti_float, 3)

    if geom_type == gs.GEOM_TYPE.SPHERE:
        # Sphere support: center + radius * normalized(direction)
        sphere_radius = geoms_info.data[i_g][0]
        dir_norm = direction.norm()
        if dir_norm > 1e-10:
            support_world = g_pos + sphere_radius * (direction / dir_norm)
        else:
            support_world = g_pos

    elif geom_type == gs.GEOM_TYPE.PLANE:
        # For probe detection, we need the closest point on the plane to the probe
        # The plane extends infinitely, so the standard support function doesn't apply
        # Instead, we project the probe position onto the plane
        # This requires probe_pos, which we'll get from the kernel
        # For now, return g_pos as placeholder - the kernel will handle plane specially
        support_world = g_pos

    elif geom_type == gs.GEOM_TYPE.BOX:
        # Box support: for each axis, pick the vertex in the direction of d
        support_world, _, _ = support_field._func_support_box(geoms_state, geoms_info, direction, i_g, i_b)

    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        # Capsule support
        support_world = support_field._func_support_capsule(geoms_state, geoms_info, direction, i_g, i_b, False)

    else:
        # MESH, CONVEX, CYLINDER, ELLIPSOID: use support field
        support_world, _, _ = support_field._func_support_world(
            geoms_state, geoms_info, support_field_info, direction, i_g, i_b
        )

    return support_world


@ti.kernel
def _kernel_kinematic_contact_probe_support_query(
    # Per-probe data
    probe_positions_local: ti.types.ndarray(),  # (total_n_probes, 3)
    probe_normals_local: ti.types.ndarray(),  # (total_n_probes, 3)
    probe_sensor_idx: ti.types.ndarray(),  # (total_n_probes,)
    # Per-sensor data
    links_pos: ti.types.ndarray(),  # (B, n_sensors, 3)
    links_quat: ti.types.ndarray(),  # (B, n_sensors, 4)
    radii: ti.types.ndarray(),  # (n_sensors,)
    stiffness: ti.types.ndarray(),  # (n_sensors,)
    links_idx: ti.types.ndarray(),  # (n_sensors,)
    contypes: ti.types.ndarray(),  # (n_sensors,)
    conaffinities: ti.types.ndarray(),  # (n_sensors,)
    n_probes_per_sensor: ti.types.ndarray(),  # (n_sensors,)
    sensor_cache_start: ti.types.ndarray(),  # (n_sensors,)
    # Scene data
    n_geoms: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    constraint_state: array_class.ConstraintState,
    equalities_info: array_class.EqualitiesInfo,
    support_field_info: array_class.SupportFieldInfo,
    # Output
    output: ti.types.ndarray(),  # (B, total_cache_size)
):
    """
    Taichi kernel to compute kinematic contact probe readings using support functions.

    Iterates over all probes globally. For each probe, finds the deepest penetration point
    on each geom using the support function in direction -probe_normal. Then checks if that
    point is inside the probe sphere and computes penetration depth.

    Penetration is computed as: dot(support_pos - probe_pos, probe_normal)
    This measures how far the geom's deepest point penetrates into the probe's tangent plane.

    Output layout per sensor with n probes:
    - penetration: n values
    - position: n * 3 values
    - normal: n * 3 values
    - force: n * 3 values
    Total: n * 10 values per sensor
    """
    n_batches = links_pos.shape[0]
    n_sensors = links_pos.shape[1]
    total_n_probes = probe_positions_local.shape[0]

    for i_b, i_p in ti.ndrange(n_batches, total_n_probes):
        # Get which sensor this probe belongs to
        i_s = probe_sensor_idx[i_p]

        # Load probe local position and normal
        probe_pos_local = ti.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )
        probe_normal_local = ti.Vector(
            [probe_normals_local[i_p, 0], probe_normals_local[i_p, 1], probe_normals_local[i_p, 2]]
        )

        # Load sensor parameters
        radius = radii[i_s]
        stiff = stiffness[i_s]
        sensor_link_idx = links_idx[i_s]
        sensor_contype = contypes[i_s]
        sensor_conaffinity = conaffinities[i_s]

        # Load link transform
        link_pos = ti.Vector([links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2]])
        link_quat = ti.Vector(
            [links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]]
        )

        # Transform probe position and normal to world frame
        probe_pos = link_pos + gu.ti_transform_by_quat(probe_pos_local, link_quat)
        probe_normal = gu.ti_transform_by_quat(probe_normal_local, link_quat)

        # Track best (deepest) penetration across all geoms
        max_abs_penetration = gs.ti_float(0.0)
        best_contact_pos_world = ti.Vector.zero(gs.ti_float, 3)

        # Direction for support query: -probe_normal (find point furthest in opposite direction)
        support_dir = -probe_normal

        # Query support function for each geom
        for i_g in range(n_geoms):
            geom_type = geoms_info.type[i_g]

            # Skip terrain (not supported)
            if geom_type == gs.GEOM_TYPE.TERRAIN:
                continue

            # Check collision filtering with proper bitmasks and weld/equality constraints
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

            # AABB broadphase check - expand AABB by sensor radius
            if not _func_point_in_expanded_aabb(i_g, i_b, geoms_state, probe_pos, radius):
                continue

            # Get support point on geom in direction -probe_normal
            support_pos = ti.Vector.zero(gs.ti_float, 3)

            if geom_type == gs.GEOM_TYPE.PLANE:
                # For plane, project probe position onto plane surface
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                geom_data = geoms_info.data[i_g]
                plane_normal_local = gs.ti_vec3([geom_data[0], geom_data[1], geom_data[2]])
                plane_normal_world = gu.ti_transform_by_quat(plane_normal_local, g_quat)

                # Project probe onto plane: closest_point = probe - (probe - g_pos)·n * n
                dist_to_plane = (probe_pos - g_pos).dot(plane_normal_world)
                support_pos = probe_pos - dist_to_plane * plane_normal_world
            else:
                support_pos = _func_support_point_for_probe(
                    geoms_state, geoms_info, support_field_info, support_dir, i_g, i_b
                )

            # Check if support point is inside or on the probe sphere
            dist_to_probe = (support_pos - probe_pos).norm()
            if dist_to_probe <= radius:
                # Compute penetration: dot(support_pos - probe_pos, probe_normal)
                penetration = (support_pos - probe_pos).dot(probe_normal)

                # Track by absolute penetration
                abs_pen = ti.abs(penetration)
                if abs_pen > max_abs_penetration:
                    max_abs_penetration = abs_pen
                    best_contact_pos_world = support_pos

        # Transform results to link-local frame
        contact_pos_local = ti.Vector.zero(gs.ti_float, 3)
        normal_local = ti.Vector.zero(gs.ti_float, 3)
        force_local = ti.Vector.zero(gs.ti_float, 3)

        if max_abs_penetration > 0:
            contact_pos_local = gu.ti_inv_transform_by_trans_quat(best_contact_pos_world, link_pos, link_quat)
            normal_local = probe_normal_local  # Already in local frame
            force_local = stiff * max_abs_penetration * probe_normal_local

        # Compute output position within this sensor's cache region
        # Find which probe index within this sensor (0-based within sensor)
        probe_idx_in_sensor = gs.ti_int(0)
        probe_start = gs.ti_int(0)
        for s in range(i_s):
            probe_start += n_probes_per_sensor[s]
        probe_idx_in_sensor = i_p - probe_start

        # Cache layout for this sensor:
        # [pen_0, pen_1, ..., pen_n-1, pos_0_x, pos_0_y, pos_0_z, ..., norm_0_x, ..., force_0_x, ...]
        # i.e., n penetrations, then n*3 positions, then n*3 normals, then n*3 forces
        n_probes = n_probes_per_sensor[i_s]
        cache_start = sensor_cache_start[i_s]
        p_idx = probe_idx_in_sensor

        # Write penetration
        output[i_b, cache_start + p_idx] = max_abs_penetration
        # Write position (offset by n_probes)
        output[i_b, cache_start + n_probes + p_idx * 3 + 0] = contact_pos_local[0]
        output[i_b, cache_start + n_probes + p_idx * 3 + 1] = contact_pos_local[1]
        output[i_b, cache_start + n_probes + p_idx * 3 + 2] = contact_pos_local[2]
        # Write normal (offset by n_probes + n_probes*3)
        output[i_b, cache_start + n_probes + n_probes * 3 + p_idx * 3 + 0] = normal_local[0]
        output[i_b, cache_start + n_probes + n_probes * 3 + p_idx * 3 + 1] = normal_local[1]
        output[i_b, cache_start + n_probes + n_probes * 3 + p_idx * 3 + 2] = normal_local[2]
        # Write force (offset by n_probes + n_probes*3 + n_probes*3)
        output[i_b, cache_start + n_probes + n_probes * 6 + p_idx * 3 + 0] = force_local[0]
        output[i_b, cache_start + n_probes + n_probes * 6 + p_idx * 3 + 1] = force_local[1]
        output[i_b, cache_start + n_probes + n_probes * 6 + p_idx * 3 + 2] = force_local[2]
