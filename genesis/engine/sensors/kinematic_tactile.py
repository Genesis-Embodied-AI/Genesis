from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf
from genesis.engine.solvers.rigid.collider.utils import func_point_in_geom_aabb
from genesis.options.sensors import ContactDepthProbe as ContactDepthProbeOptions
from genesis.options.sensors import ContactProbe as ContactProbeOptions
from genesis.options.sensors import KinematicTaxel as KinematicTaxelOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, SimpleSensor, SimpleSensorMetadata
from .probe import (
    ProbeSensorMetadataMixin,
    ProbeSensorMixin,
    ProbesWithNormalSensorMetadataMixin,
    ProbesWithNormalSensorMixin,
    ProbesWithNormalSensorSharedMetadataT,
    func_noised_probe_radius,
)

if TYPE_CHECKING:
    from genesis.options.sensors import SensorOptions
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.func
def _func_query_contact_depth_penetration(
    i_b: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_link_idx: int,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    sdf_info: array_class.SDFInfo,
):
    """
    Max probe penetration from SDF for contacts involving the sensor link, dual-radius.

    Returns ``(max_pen_gt, max_pen_m)`` from a single SDF pass: both penetrations come from the same ``sd`` per contact
    via ``pen = radius - sd``. Callers that do not need the noised-radius branch pass ``probe_radius_m ==
    probe_radius_gt`` and ignore the second return.
    """
    max_pen_gt = gs.qd_float(0.0)
    max_pen_m = gs.qd_float(0.0)

    n_contacts = collider_state.n_contacts[i_b]
    for i_c in range(n_contacts):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        c_link_a = collider_state.contact_data.link_a[i_col, i_b]
        c_link_b = collider_state.contact_data.link_b[i_col, i_b]
        c_geom_a = collider_state.contact_data.geom_a[i_col, i_b]
        c_geom_b = collider_state.contact_data.geom_b[i_col, i_b]

        for side in qd.static(range(2)):
            c_link = c_link_a if side == 0 else c_link_b
            i_g = c_geom_b if side == 0 else c_geom_a

            if c_link == sensor_link_idx:
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                sd = sdf.sdf_func_world_local(geoms_info, sdf_info, probe_pos, i_g, g_pos, g_quat)
                pen_gt = probe_radius_gt - sd
                if pen_gt > max_pen_gt:
                    max_pen_gt = pen_gt
                pen_m = probe_radius_m - sd
                if pen_m > max_pen_m:
                    max_pen_m = pen_m

    return max_pen_gt, max_pen_m


@qd.func
def _func_query_contact_depth(
    i_b: int,
    probe_pos: qd.types.vector(3),
    probe_radius_gt: float,
    probe_radius_m: float,
    sensor_link_idx: int,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    collider_state: array_class.ColliderState,
    sdf_info: array_class.SDFInfo,
    eps: float,
):
    """
    Dual-radius probe query: single SDF + normal pass yielding both GT and noised-radius results.

    Returns ``(max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m)``. AABB
    pre-filter expands by ``max(probe_radius_gt, probe_radius_m)`` so neither branch is silently skipped. Callers
    without a noised radius pass ``probe_radius_m == probe_radius_gt``.
    """
    max_pen_gt = gs.qd_float(0.0)
    contact_link_gt = gs.qd_int(-1)
    contact_normal_gt = qd.Vector.zero(gs.qd_float, 3)
    max_pen_m = gs.qd_float(0.0)
    contact_link_m = gs.qd_int(-1)
    contact_normal_m = qd.Vector.zero(gs.qd_float, 3)

    aabb_expansion = qd.max(probe_radius_gt, probe_radius_m)
    # Iterate over contacts directly from collider state; each contact may have the sensor link on either side.
    n_contacts = collider_state.n_contacts[i_b]
    for i_c in range(n_contacts):
        i_col = collider_state.contact_sort_idx[i_c, i_b]
        c_link_a = collider_state.contact_data.link_a[i_col, i_b]
        c_link_b = collider_state.contact_data.link_b[i_col, i_b]
        c_geom_a = collider_state.contact_data.geom_a[i_col, i_b]
        c_geom_b = collider_state.contact_data.geom_b[i_col, i_b]

        # Check if either side of this contact involves the sensor link.
        for side in qd.static(range(2)):
            c_link = c_link_a if side == 0 else c_link_b
            i_g = c_geom_b if side == 0 else c_geom_a

            if c_link == sensor_link_idx and func_point_in_geom_aabb(geoms_state, i_g, i_b, probe_pos, aabb_expansion):
                g_pos = geoms_state.pos[i_g, i_b]
                g_quat = geoms_state.quat[i_g, i_b]
                sd = sdf.sdf_func_world_local(geoms_info, sdf_info, probe_pos, i_g, g_pos, g_quat)
                pen_gt = probe_radius_gt - sd
                pen_m = probe_radius_m - sd
                # Compute the SDF normal at most once across both branches.
                need_normal = (pen_gt > max_pen_gt and pen_gt > eps) or (pen_m > max_pen_m and pen_m > eps)
                if need_normal:
                    normal = sdf.sdf_func_normal_world_local(
                        geoms_info, rigid_global_info, collider_static_config, sdf_info, probe_pos, i_g, g_pos, g_quat
                    )
                    if pen_gt > max_pen_gt and pen_gt > eps:
                        max_pen_gt = pen_gt
                        contact_link_gt = c_link_b if side == 0 else c_link_a
                        contact_normal_gt = normal
                    if pen_m > max_pen_m and pen_m > eps:
                        max_pen_m = pen_m
                        contact_link_m = c_link_b if side == 0 else c_link_a
                        contact_normal_m = normal

    return max_pen_gt, contact_link_gt, contact_normal_gt, max_pen_m, contact_link_m, contact_normal_m


@qd.kernel
def _kernel_kinematic_taxel(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    normal_stiffness: qd.types.ndarray(),
    normal_damping: qd.types.ndarray(),
    normal_exponent: qd.types.ndarray(),
    shear_scalar: qd.types.ndarray(),
    twist_scalar: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    n_probes_per_sensor: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    collider_static_config: qd.template(),
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    sdf_info: array_class.SDFInfo,
    eps: float,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        use_noised_radius = probe_radius_noise > eps
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise) if use_noised_radius else probe_radius
        )

        (
            max_penetration_gt,
            contact_link_gt,
            contact_normal_gt,
            max_penetration_m,
            contact_link_m,
            contact_normal_m,
        ) = _func_query_contact_depth(
            i_b,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            rigid_global_info,
            collider_static_config,
            collider_state,
            sdf_info,
            eps,
        )

        force_local_gt = qd.Vector.zero(gs.qd_float, 3)
        torque_local_gt = qd.Vector.zero(gs.qd_float, 3)
        if max_penetration_gt > 0:
            contact_normal_local = gu.qd_inv_transform_by_quat(contact_normal_gt, link_quat)
            s = qd.pow(max_penetration_gt, normal_exponent[i_s])
            force_local_gt = contact_normal_local * (normal_stiffness[i_s] * s)

            if contact_link_gt >= 0:
                contact_vel = links_state.cd_vel[contact_link_gt, i_b] + links_state.cd_ang[contact_link_gt, i_b].cross(
                    probe_pos - links_state.root_COM[contact_link_gt, i_b]
                )
                sensor_vel = links_state.cd_vel[sensor_link_idx, i_b] + links_state.cd_ang[sensor_link_idx, i_b].cross(
                    probe_pos - links_state.root_COM[sensor_link_idx, i_b]
                )
                rel_vel_world = contact_vel - sensor_vel
                rel_vel_local = gu.qd_inv_transform_by_quat(rel_vel_world, link_quat)

                vn_dot = rel_vel_local.dot(contact_normal_local)
                v_t_local = rel_vel_local - contact_normal_local * vn_dot
                force_local_gt += (
                    contact_normal_local * (normal_damping[i_s] * s * vn_dot) - shear_scalar[i_s] * v_t_local
                )

                rel_ang_world = links_state.cd_ang[contact_link_gt, i_b] - links_state.cd_ang[sensor_link_idx, i_b]
                omega_n = rel_ang_world.dot(contact_normal_gt)
                torque_local_gt = probe_pos_local.cross(force_local_gt) - contact_normal_local * (
                    twist_scalar[i_s] * omega_n
                )
            else:
                torque_local_gt = probe_pos_local.cross(force_local_gt)

        force_local_m = qd.Vector.zero(gs.qd_float, 3)
        torque_local_m = qd.Vector.zero(gs.qd_float, 3)
        if not use_noised_radius:
            for j in qd.static(range(3)):
                force_local_m[j] = force_local_gt[j]
                torque_local_m[j] = torque_local_gt[j]
        elif max_penetration_m > 0:
            contact_normal_local = gu.qd_inv_transform_by_quat(contact_normal_m, link_quat)
            s = qd.pow(max_penetration_m, normal_exponent[i_s])
            force_local_m = contact_normal_local * (normal_stiffness[i_s] * s)

            if contact_link_m >= 0:
                contact_vel = links_state.cd_vel[contact_link_m, i_b] + links_state.cd_ang[contact_link_m, i_b].cross(
                    probe_pos - links_state.root_COM[contact_link_m, i_b]
                )
                sensor_vel = links_state.cd_vel[sensor_link_idx, i_b] + links_state.cd_ang[sensor_link_idx, i_b].cross(
                    probe_pos - links_state.root_COM[sensor_link_idx, i_b]
                )
                rel_vel_world = contact_vel - sensor_vel
                rel_vel_local = gu.qd_inv_transform_by_quat(rel_vel_world, link_quat)

                vn_dot = rel_vel_local.dot(contact_normal_local)
                v_t_local = rel_vel_local - contact_normal_local * vn_dot
                force_local_m += (
                    contact_normal_local * (normal_damping[i_s] * s * vn_dot) - shear_scalar[i_s] * v_t_local
                )

                rel_ang_world = links_state.cd_ang[contact_link_m, i_b] - links_state.cd_ang[sensor_link_idx, i_b]
                omega_n = rel_ang_world.dot(contact_normal_m)
                torque_local_m = probe_pos_local.cross(force_local_m) - contact_normal_local * (
                    twist_scalar[i_s] * omega_n
                )
            else:
                torque_local_m = probe_pos_local.cross(force_local_m)

        probe_idx_in_sensor = i_p - sensor_probe_start[i_s]
        cache_start = sensor_cache_start[i_s]
        n_probes = n_probes_per_sensor[i_s]
        force_start = cache_start + probe_idx_in_sensor * 3
        torque_start = cache_start + n_probes * 3 + probe_idx_in_sensor * 3
        for j in qd.static(range(3)):
            output_gt[force_start + j, i_b] = force_local_gt[j]
            output_gt[torque_start + j, i_b] = torque_local_gt[j]
            output_measured[force_start + j, i_b] = force_local_m[j]
            output_measured[torque_start + j, i_b] = torque_local_m[j]


@qd.kernel
def _kernel_contact_depth_probe(
    probe_positions_local: qd.types.ndarray(),
    probe_sensor_idx: qd.types.ndarray(),
    probe_radii: qd.types.ndarray(),
    probe_radii_noise: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    sensor_cache_start: qd.types.ndarray(),
    sensor_probe_start: qd.types.ndarray(),
    collider_state: array_class.ColliderState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    sdf_info: array_class.SDFInfo,
    output_gt: qd.types.ndarray(),
    output_measured: qd.types.ndarray(),
):
    total_n_probes = probe_positions_local.shape[0]
    n_batches = output_gt.shape[-1]

    for i_p, i_b in qd.ndrange(total_n_probes, n_batches):
        i_s = probe_sensor_idx[i_p]

        probe_pos_local = qd.Vector(
            [probe_positions_local[i_p, 0], probe_positions_local[i_p, 1], probe_positions_local[i_p, 2]]
        )

        sensor_link_idx = links_idx[i_s]
        link_pos = links_state.pos[sensor_link_idx, i_b]
        link_quat = links_state.quat[sensor_link_idx, i_b]

        probe_pos = link_pos + gu.qd_transform_by_quat(probe_pos_local, link_quat)

        probe_radius = probe_radii[i_p]
        probe_radius_noise = probe_radii_noise[i_p]
        probe_radius_m = (
            func_noised_probe_radius(probe_radius, probe_radius_noise) if probe_radius_noise > gs.EPS else probe_radius
        )

        max_penetration_gt, max_penetration_m = _func_query_contact_depth_penetration(
            i_b,
            probe_pos,
            probe_radius,
            probe_radius_m,
            sensor_link_idx,
            geoms_info,
            geoms_state,
            collider_state,
            sdf_info,
        )
        cache_idx = sensor_cache_start[i_s] + i_p - sensor_probe_start[i_s]
        output_gt[cache_idx, i_b] = max_penetration_gt
        output_measured[cache_idx, i_b] = max_penetration_m


class KinematicTactileSensorMixin(ProbeSensorMixin[ProbesWithNormalSensorSharedMetadataT]):
    def __init__(self, sensor_options: "SensorOptions", sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)
        self._debug_objects: list = []

    def build(self):
        super().build()
        self._shared_metadata.solver.collider.activate_sdf()

    def _draw_debug_probes(self, context: "RasterizerContext", get_is_contact: Callable[[object], object]):
        for obj in self._debug_objects:
            context.clear_debug_object(obj)
        self._debug_objects.clear()

        envs_idx, n_debug_envs, _, probe_world = self._compute_probes_world_pos(context)
        data = self.read_ground_truth(envs_idx)
        is_contact = np.asarray(tensor_to_array(get_is_contact(data)), dtype=bool).reshape(-1)
        probe_global_idx = int(self._shared_metadata.sensor_probe_start[self._idx])
        probe_radius = float(self._shared_metadata.probe_radii[probe_global_idx])
        for is_contact_state in (False, True):
            (probes_idx,) = np.nonzero(is_contact == is_contact_state)
            if probes_idx.size > 0:
                spheres_obj = context.draw_debug_spheres(
                    poss=probe_world[probes_idx],
                    radius=probe_radius,
                    color=self._options.debug_contact_color if is_contact_state else self._options.debug_probe_color,
                )
                self._debug_objects.append(spheres_obj)


@dataclass
class ContactDepthProbeMetadata(ProbeSensorMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata):
    pass


class ContactDepthProbeSensor(
    KinematicTactileSensorMixin[ContactDepthProbeMetadata],
    RigidSensorMixin[ContactDepthProbeMetadata],
    SimpleSensor[ContactDepthProbeOptions, ContactDepthProbeMetadata, tuple],
):
    """Returns contact depth in meters per probe."""

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._n_probes,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_metadata: ContactDepthProbeMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver

        current_ground_truth_data_T.zero_()
        measured = measured_data_timeline.at(0, copy=False)
        measured.zero_()
        if shared_metadata.measured_scratch_T.shape != current_ground_truth_data_T.shape:
            shared_metadata.measured_scratch_T = torch.empty_like(current_ground_truth_data_T)
        measured_cols_b = shared_metadata.measured_scratch_T

        _kernel_contact_depth_probe(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            solver.collider._collider_state,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver.collider._sdf._sdf_info,
            current_ground_truth_data_T,
            measured_cols_b,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        self._draw_debug_probes(context, lambda depth: depth >= gs.EPS)


@dataclass
class ContactProbeMetadata(ContactDepthProbeMetadata):
    contact_threshold: torch.Tensor = make_tensor_field((0,))
    # Per-probe threshold scattered into intermediate-cache layout, computed lazily on first `_post_process`.
    threshold_row: torch.Tensor = make_tensor_field((0,))


class ContactProbeSensor(ContactDepthProbeSensor, SimpleSensor[ContactProbeOptions, ContactProbeMetadata, tuple]):
    """Returns boolean contact per probe (depth > threshold). Shares the depth-probe kernel."""

    def build(self):
        super().build()
        self._shared_metadata.contact_threshold = concat_with_tensor(
            self._shared_metadata.contact_threshold, self._options.contact_threshold, expand=(1,)
        )

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _post_process(
        cls,
        shared_metadata: ContactProbeMetadata,
        tensor: torch.Tensor,
        timeline: "TensorRingBuffer",
        *,
        is_measured: bool,
    ) -> torch.Tensor:
        if (
            shared_metadata.threshold_row.shape != (tensor.shape[1],)
            or shared_metadata.threshold_row.dtype != tensor.dtype
        ):
            i_p = torch.arange(shared_metadata.total_n_probes, device=gs.device, dtype=gs.tc_int)
            i_s = shared_metadata.probe_sensor_idx
            cache_idx = shared_metadata.sensor_cache_start[i_s] + i_p - shared_metadata.sensor_probe_start[i_s]
            row = torch.zeros((tensor.shape[1],), dtype=tensor.dtype, device=gs.device)
            row.scatter_(
                0, cache_idx.to(dtype=torch.int64), shared_metadata.contact_threshold[i_s].to(dtype=tensor.dtype)
            )
            shared_metadata.threshold_row = row
        return tensor > shared_metadata.threshold_row.unsqueeze(0)

    def _draw_debug(self, context: "RasterizerContext"):
        self._draw_debug_probes(context, lambda data: data)


class KinematicTaxelData(NamedTuple):
    """
    Parameters
    ----------
    force: torch.Tensor, shape ([n_envs,] n_probes, 3)
        Estimated contact force in the link frame from the kinematic spring-damper model.
    torque: torch.Tensor, shape ([n_envs,] n_probes, 3)
    """

    force: torch.Tensor
    torque: torch.Tensor


@dataclass
class KinematicTaxelMetadata(ProbesWithNormalSensorMetadataMixin, RigidSensorMetadataMixin, SimpleSensorMetadata):
    normal_stiffness: torch.Tensor = make_tensor_field((0,))
    normal_damping: torch.Tensor = make_tensor_field((0,))
    normal_exponent: torch.Tensor = make_tensor_field((0,))
    shear_scalar: torch.Tensor = make_tensor_field((0,))
    twist_scalar: torch.Tensor = make_tensor_field((0,))


class KinematicTaxelSensor(
    KinematicTactileSensorMixin[KinematicTaxelMetadata],
    ProbesWithNormalSensorMixin[KinematicTaxelMetadata],
    RigidSensorMixin[KinematicTaxelMetadata],
    SimpleSensor[KinematicTaxelOptions, KinematicTaxelMetadata, KinematicTaxelData],
):
    """Kinematic taxels: spring-damper force and torque per probe from contact geometry and relative motion."""

    def __init__(self, sensor_options: KinematicTaxelOptions, sensor_idx: int, sensor_manager: "SensorManager"):
        super().__init__(sensor_options, sensor_idx, sensor_manager)

    def build(self):
        super().build()

        self._shared_metadata.normal_stiffness = concat_with_tensor(
            self._shared_metadata.normal_stiffness, float(self._options.normal_stiffness), expand=(1,)
        )
        self._shared_metadata.normal_damping = concat_with_tensor(
            self._shared_metadata.normal_damping, float(self._options.normal_damping), expand=(1,)
        )
        self._shared_metadata.normal_exponent = concat_with_tensor(
            self._shared_metadata.normal_exponent, float(self._options.normal_exponent), expand=(1,)
        )
        self._shared_metadata.shear_scalar = concat_with_tensor(
            self._shared_metadata.shear_scalar, float(self._options.shear_scalar), expand=(1,)
        )
        self._shared_metadata.twist_scalar = concat_with_tensor(
            self._shared_metadata.twist_scalar, float(self._options.twist_scalar), expand=(1,)
        )

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (self._n_probes, 3), (self._n_probes, 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_current_timestep_data(
        cls,
        shared_metadata: KinematicTaxelMetadata,
        current_ground_truth_data_T: torch.Tensor,
        ground_truth_data_timeline: "TensorRingBuffer | None",
        measured_data_timeline: "TensorRingBuffer",
    ):
        solver = shared_metadata.solver

        current_ground_truth_data_T.zero_()
        measured = measured_data_timeline.at(0, copy=False)
        measured.zero_()
        if shared_metadata.measured_scratch_T.shape != current_ground_truth_data_T.shape:
            shared_metadata.measured_scratch_T = torch.empty_like(current_ground_truth_data_T)
        measured_cols_b = shared_metadata.measured_scratch_T

        _kernel_kinematic_taxel(
            shared_metadata.probe_positions,
            shared_metadata.probe_sensor_idx,
            shared_metadata.probe_radii,
            shared_metadata.probe_radii_noise,
            shared_metadata.normal_stiffness,
            shared_metadata.normal_damping,
            shared_metadata.normal_exponent,
            shared_metadata.shear_scalar,
            shared_metadata.twist_scalar,
            shared_metadata.links_idx,
            shared_metadata.sensor_cache_start,
            shared_metadata.sensor_probe_start,
            shared_metadata.n_probes_per_sensor,
            solver.collider._collider_state,
            solver.collider._collider_static_config,
            solver.links_state,
            solver.geoms_state,
            solver.geoms_info,
            solver._rigid_global_info,
            solver.collider._sdf._sdf_info,
            gs.EPS,
            current_ground_truth_data_T,
            measured_cols_b,
        )
        if ground_truth_data_timeline is not None:
            ground_truth_data_timeline.at(0, copy=False).copy_(current_ground_truth_data_T.T)
        measured.copy_(measured_cols_b.T)

    def _draw_debug(self, context: "RasterizerContext"):
        self._draw_debug_probes(context, lambda data: torch.linalg.norm(data.force, dim=-1) >= gs.EPS)
