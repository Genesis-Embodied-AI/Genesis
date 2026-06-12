from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.options.sensors import Contact as ContactSensorOptions
from genesis.options.sensors import ContactForce as ContactForceSensorOptions
from genesis.utils.geom import inv_transform_by_quat, qd_inv_transform_by_quat, transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, qd_to_torch, tensor_to_array
from genesis.utils.ring_buffer import TensorRingBuffer

from .base_sensor import RigidSensorMetadataMixin, RigidSensorMixin, Sensor, SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@qd.kernel
def _kernel_get_contacts_forces(
    contact_forces: qd.types.ndarray(),
    link_a: qd.types.ndarray(),
    link_b: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    sensors_link_idx: qd.types.ndarray(),
    output: qd.types.ndarray(),
):
    for i_c, i_s, i_b in qd.ndrange(link_a.shape[-1], sensors_link_idx.shape[-1], output.shape[-1]):
        contact_data_link_a = link_a[i_b, i_c]
        contact_data_link_b = link_b[i_b, i_c]
        if contact_data_link_a == sensors_link_idx[i_s] or contact_data_link_b == sensors_link_idx[i_s]:
            j_s = i_s * 3  # per-sensor output dimension is 3

            quat_a = qd.Vector.zero(gs.qd_float, 4)
            quat_b = qd.Vector.zero(gs.qd_float, 4)
            for j in qd.static(range(4)):
                quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                quat_b[j] = links_quat[i_b, contact_data_link_b, j]

            force_vec = qd.Vector.zero(gs.qd_float, 3)
            for j in qd.static(range(3)):
                force_vec[j] = contact_forces[i_b, i_c, j]

            force_a = qd_inv_transform_by_quat(-force_vec, quat_a)
            force_b = qd_inv_transform_by_quat(force_vec, quat_b)

            if contact_data_link_a == sensors_link_idx[i_s]:
                for j in qd.static(range(3)):
                    output[j_s + j, i_b] += force_a[j]
            if contact_data_link_b == sensors_link_idx[i_s]:
                for j in qd.static(range(3)):
                    output[j_s + j, i_b] += force_b[j]


@dataclass
class ContactSensorMetadata(SimpleSensorMetadata):
    """
    Metadata for all rigid contact sensors.
    """

    solver: "RigidSolver | None" = None
    expanded_links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # (num_contact_sensors, max_num_filter_links); unused slots are -1.
    filter_links_idx: torch.Tensor = make_tensor_field((0, 0), dtype_factory=lambda: gs.tc_int)
    # Indices into expanded_links_idx of sensors that have at least one filter link. Lets the GT update skip the 4D
    # contact-vs-filter comparison for the (typically larger) subset of sensors with no filter.
    filtered_sensor_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Per-sensor bool threshold (broadcast over B); _post_process returns `tensor > thresholds`.
    thresholds: torch.Tensor = make_tensor_field((0,))


class ContactSensor(SimpleSensor[ContactSensorOptions, None, ContactSensorMetadata]):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.
    """

    def __init__(
        self,
        options: ContactSensorOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self._link: "RigidLink | None" = None
        self.debug_object: "Mesh | None" = None

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        entity = self._shared_metadata.solver.entities[self._options.entity_idx]
        link_idx = self._options.link_idx_local + entity.link_start
        self._link = entity.links[self._options.link_idx_local]

        self._shared_metadata.expanded_links_idx = concat_with_tensor(
            self._shared_metadata.expanded_links_idx, link_idx, expand=(1,), dim=0
        )

        num_sensors, cur_num_filter_links = self._shared_metadata.filter_links_idx.shape
        max_num_filter_links = max(cur_num_filter_links, len(self._options.filter_link_idx))
        filter_links_idx = torch.full((num_sensors + 1, max_num_filter_links), -1, dtype=gs.tc_int, device=gs.device)
        filter_links_idx[:num_sensors, :cur_num_filter_links] = self._shared_metadata.filter_links_idx
        filter_links_idx[num_sensors, : len(self._options.filter_link_idx)] = torch.tensor(
            self._options.filter_link_idx, dtype=gs.tc_int, device=gs.device
        )
        self._shared_metadata.filter_links_idx = filter_links_idx

        if len(self._options.filter_link_idx) > 0:
            self._shared_metadata.filtered_sensor_idx = concat_with_tensor(
                self._shared_metadata.filtered_sensor_idx, num_sensors, expand=(1,), dim=0
            )

        self._shared_metadata.thresholds = concat_with_tensor(
            self._shared_metadata.thresholds, float(self._options.threshold), expand=(1,)
        )

    def _get_return_format(self) -> tuple[int, ...]:
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        # Float kernel output; bool projection happens in `_post_process`. Shape matches `_get_return_format`.
        return gs.tc_float

    @classmethod
    def _update_raw_data(cls, shared_context: None, shared_metadata: ContactSensorMetadata, raw_data_T: torch.Tensor):
        assert shared_metadata.solver is not None
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b = all_contacts["link_a"], all_contacts["link_b"]
        if link_a.shape[-1] == 0:
            raw_data_T.zero_()
            return
        if shared_metadata.solver.n_envs == 0:
            link_a, link_b = link_a[None], link_b[None]

        is_contact_a = link_a[..., None, :] == shared_metadata.expanded_links_idx[..., None]
        is_contact_b = link_b[..., None, :] == shared_metadata.expanded_links_idx[..., None]
        # Float-valued contact count per sensor (intermediate cache is float; bool projection in `_post_process`).
        result = (is_contact_a | is_contact_b).sum(dim=-1).to(dtype=gs.tc_float)
        # Apply the (more expensive) filter-aware update only on sensors that declared a filter; other sensors keep the
        # cheap aggregate result above.
        if shared_metadata.filtered_sensor_idx.numel() > 0:
            filt = shared_metadata.filtered_sensor_idx
            sub_filter = shared_metadata.filter_links_idx[filt][None, :, None, :]
            filtered_a = (link_b[:, None, :, None] == sub_filter).any(dim=-1)
            filtered_b = (link_a[:, None, :, None] == sub_filter).any(dim=-1)
            sub_is_a = is_contact_a[:, filt, :]
            sub_is_b = is_contact_b[:, filt, :]
            result[:, filt] = ((sub_is_a & ~filtered_a) | (sub_is_b & ~filtered_b)).sum(dim=-1).to(dtype=gs.tc_float)
        raw_data_T[:] = result.T

    @classmethod
    def _post_process(
        cls, shared_metadata: ContactSensorMetadata, tensor: torch.Tensor, timeline, *, is_measured: bool
    ) -> torch.Tensor:
        return tensor > shared_metadata.thresholds

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug sphere when the sensor detects contact.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx, relative=False).reshape((3,))
        is_contact = self.read(env_idx)

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        if is_contact:
            self.debug_object = context.draw_debug_sphere(
                pos=pos, radius=self._options.debug_sphere_radius, color=self._options.debug_color
            )


# ==========================================================================================================


@dataclass
class ContactForceSensorMetadata(RigidSensorMetadataMixin, SimpleSensorMetadata):
    """
    Shared metadata for all contact force sensors.
    """

    min_force: torch.Tensor = make_tensor_field((0, 3))
    max_force: torch.Tensor = make_tensor_field((0, 3))


class ContactForceSensor(
    RigidSensorMixin[ContactForceSensorMetadata],
    SimpleSensor[ContactForceSensorOptions, None, ContactForceSensorMetadata],
):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.
    """

    def __init__(
        self,
        options: ContactForceSensorOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)

        self.debug_object: "Mesh" | None = None

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.min_force = concat_with_tensor(
            self._shared_metadata.min_force, self._options.min_force, expand=(1, 3)
        )
        self._shared_metadata.max_force = concat_with_tensor(
            self._shared_metadata.max_force, self._options.max_force, expand=(1, 3)
        )

    def _get_return_format(self) -> tuple[int, ...]:
        return (3,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _get_intermediate_dtype(cls) -> torch.dtype:
        # Required override because `_post_process` is overridden, even though shape and dtype coincide with return. The
        # intermediate buffer must be a distinct buffer (the timeline ring is in intermediate space).
        return cls._get_cache_dtype()

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: ContactForceSensorMetadata, raw_data_T: torch.Tensor
    ):
        assert shared_metadata.solver is not None

        # Note that forcing GPU sync to operate on `slice(0, max(n_contacts))` is usually faster overall.
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = all_contacts["force"], all_contacts["link_a"], all_contacts["link_b"]
        if shared_metadata.solver.n_envs == 0:
            force, link_a, link_b = force[None], link_a[None], link_b[None]

        # Short-circuit if no contacts
        if link_a.shape[-1] == 0:
            raw_data_T.zero_()
            return

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            links_quat = links_quat[None]

        if gs.use_zerocopy:
            # Forces are aggregated BEFORE moving them in local frame for efficiency.
            force_mask_a = link_a[:, None] == shared_metadata.links_idx[None, :, None]
            force_mask_b = link_b[:, None] == shared_metadata.links_idx[None, :, None]
            force_mask = force_mask_b.to(dtype=gs.tc_float) - force_mask_a.to(dtype=gs.tc_float)
            sensors_force = (force_mask[..., None] * force[:, None]).sum(dim=2)
            sensors_quat = links_quat[:, shared_metadata.links_idx]
            n_envs = max(shared_metadata.solver.n_envs, 1)
            result = inv_transform_by_quat(sensors_force, sensors_quat)  # (B, n_sensors, 3)
            raw_data_T[:] = result.permute(1, 2, 0).reshape(-1, n_envs)
        else:
            raw_data_T.zero_()
            _kernel_get_contacts_forces(
                force.contiguous(),
                link_a.contiguous(),
                link_b.contiguous(),
                links_quat.contiguous(),
                shared_metadata.links_idx,
                raw_data_T,
            )

    @classmethod
    def _post_process(
        cls, shared_metadata: ContactForceSensorMetadata, tensor: torch.Tensor, timeline, *, is_measured: bool
    ) -> torch.Tensor:
        # Saturate at max_force and zero out values below the min_force dead band. Applied after quantization (which
        # happens upstream in `_apply_hardware_imperfections`); for max_force values that are not multiples of
        # resolution this produces a non-quantized saturation value, accepted as minor drift in that edge case.
        per_sensor = tensor.reshape((tensor.shape[0], -1, 3))
        out = per_sensor.clamp(min=-shared_metadata.max_force, max=shared_metadata.max_force)
        out = out.masked_fill(out.abs() < shared_metadata.min_force, 0.0)
        return out.reshape(tensor.shape)

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw debug arrow representing the contact force.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx, relative=False).reshape((3,))
        quat = self._link.get_quat(env_idx, relative=False).reshape((4,))

        force = self.read(env_idx).reshape((3,))
        vec = tensor_to_array(transform_by_quat(force * self._options.debug_scale, quat))

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        self.debug_object = context.draw_debug_arrow(pos=pos, vec=vec, color=self._options.debug_color)
