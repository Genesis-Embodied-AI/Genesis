from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.options.sensors import (
    Contact as ContactSensorOptions,
    ContactForce as ContactForceSensorOptions,
)
from genesis.utils.geom import inv_transform_by_quat, ti_inv_transform_by_quat, transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array, ti_to_torch

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


@ti.kernel
def _kernel_get_contacts_forces(
    contact_forces: ti.types.ndarray(),
    link_a: ti.types.ndarray(),
    link_b: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    sensors_link_idx: ti.types.ndarray(),
    output: ti.types.ndarray(),
):
    for i_b, i_c, i_s in ti.ndrange(output.shape[0], link_a.shape[-1], sensors_link_idx.shape[-1]):
        contact_data_link_a = link_a[i_b, i_c]
        contact_data_link_b = link_b[i_b, i_c]
        if contact_data_link_a == sensors_link_idx[i_s] or contact_data_link_b == sensors_link_idx[i_s]:
            j_s = i_s * 3  # per-sensor output dimension is 3

            quat_a = ti.Vector.zero(gs.ti_float, 4)
            quat_b = ti.Vector.zero(gs.ti_float, 4)
            for j in ti.static(range(4)):
                quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                quat_b[j] = links_quat[i_b, contact_data_link_b, j]

            force_vec = ti.Vector.zero(gs.ti_float, 3)
            for j in ti.static(range(3)):
                force_vec[j] = contact_forces[i_b, i_c, j]

            force_a = ti_inv_transform_by_quat(-force_vec, quat_a)
            force_b = ti_inv_transform_by_quat(force_vec, quat_b)

            if contact_data_link_a == sensors_link_idx[i_s]:
                for j in ti.static(range(3)):
                    output[i_b, j_s + j] += force_a[j]
            if contact_data_link_b == sensors_link_idx[i_s]:
                for j in ti.static(range(3)):
                    output[i_b, j_s + j] += force_b[j]


@dataclass
class ContactSensorMetadata(SharedSensorMetadata):
    """
    Metadata for all rigid contact sensors.
    """

    solver: "RigidSolver | None" = None
    expanded_links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Flag to skip filtering logic when no sensors use filtering options
    needs_filtering: bool = False
    # Per-sensor filter options: link range of the sensor's own entity
    sensor_entity_link_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    sensor_entity_link_end: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Per-sensor filter options: link range of with_entity (-1 means no filter)
    with_entity_link_start: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    with_entity_link_end: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Per-sensor flag: exclude self contact
    exclude_self_contact: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_bool)


@register_sensor(ContactSensorOptions, ContactSensorMetadata, tuple)
@ti.data_oriented
class ContactSensor(Sensor):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.
    """

    def __init__(
        self,
        sensor_options: ContactSensorOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

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

        # Store sensor's own entity link range for exclude_self_contact
        self._shared_metadata.sensor_entity_link_start = concat_with_tensor(
            self._shared_metadata.sensor_entity_link_start, entity.link_start, expand=(1,), dim=0
        )
        self._shared_metadata.sensor_entity_link_end = concat_with_tensor(
            self._shared_metadata.sensor_entity_link_end, entity.link_end, expand=(1,), dim=0
        )

        # Store with_entity link range (-1 means no filter)
        if self._options.with_entity_idx is not None:
            with_entity = self._shared_metadata.solver.entities[self._options.with_entity_idx]
            with_link_start = with_entity.link_start
            with_link_end = with_entity.link_end
        else:
            with_link_start = -1
            with_link_end = -1
        self._shared_metadata.with_entity_link_start = concat_with_tensor(
            self._shared_metadata.with_entity_link_start, with_link_start, expand=(1,), dim=0
        )
        self._shared_metadata.with_entity_link_end = concat_with_tensor(
            self._shared_metadata.with_entity_link_end, with_link_end, expand=(1,), dim=0
        )

        # Store exclude_self_contact flag
        self._shared_metadata.exclude_self_contact = concat_with_tensor(
            self._shared_metadata.exclude_self_contact, self._options.exclude_self_contact, expand=(1,), dim=0
        )

        # Update needs_filtering flag if this sensor uses any filtering
        if self._options.with_entity_idx is not None or self._options.exclude_self_contact:
            self._shared_metadata.needs_filtering = True

    def _get_return_format(self) -> tuple[int, ...]:
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: ContactSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        assert shared_metadata.solver is not None
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b = all_contacts["link_a"], all_contacts["link_b"]

        n_contacts = link_a.shape[-1]

        # Handle empty contacts case
        if n_contacts == 0:
            shared_ground_truth_cache[:] = False
            return

        # Add batch dimension when n_envs=0 (non-batched mode)
        if shared_metadata.solver.n_envs == 0:
            link_a, link_b = link_a[None], link_b[None]

        # Base check: sensor link is involved in contact
        # Shape: link_a is (n_envs, n_contacts), expanded_links_idx is (n_sensors,)
        # Result shape: (n_envs, n_sensors, n_contacts) -> any over contacts -> (n_envs, n_sensors)
        sensor_in_a = (
            link_a[..., None, :] == shared_metadata.expanded_links_idx[..., None]
        )  # (n_envs, n_sensors, n_contacts)
        sensor_in_b = link_b[..., None, :] == shared_metadata.expanded_links_idx[..., None]

        # Fast path: no filtering options used by any sensor
        if not shared_metadata.needs_filtering:
            shared_ground_truth_cache[:] = sensor_in_a.any(dim=-1) | sensor_in_b.any(dim=-1)
            return

        # Slow path: filtering logic for with_entity_idx and/or exclude_self_contact
        has_with_filter = shared_metadata.with_entity_link_start >= 0  # (n_sensors,)
        has_exclude_self = shared_metadata.exclude_self_contact  # (n_sensors,)

        # Check if link_a/link_b are in with_entity range
        link_a_in_with = (link_a[..., None, :] >= shared_metadata.with_entity_link_start[..., None]) & (
            link_a[..., None, :] < shared_metadata.with_entity_link_end[..., None]
        )
        link_b_in_with = (link_b[..., None, :] >= shared_metadata.with_entity_link_start[..., None]) & (
            link_b[..., None, :] < shared_metadata.with_entity_link_end[..., None]
        )

        # Check if link_a/link_b are in sensor's own entity range (for exclude_self)
        link_a_in_self = (link_a[..., None, :] >= shared_metadata.sensor_entity_link_start[..., None]) & (
            link_a[..., None, :] < shared_metadata.sensor_entity_link_end[..., None]
        )
        link_b_in_self = (link_b[..., None, :] >= shared_metadata.sensor_entity_link_start[..., None]) & (
            link_b[..., None, :] < shared_metadata.sensor_entity_link_end[..., None]
        )

        # Valid contact for each sensor:
        # - sensor link in A, other (B) passes filter
        # - OR sensor link in B, other (A) passes filter

        # with_entity filter for "other": -1 means no filter (accept all)
        # Expand has_with_filter to match (n_envs, n_sensors, n_contacts)
        has_with_3d = has_with_filter[None, :, None].expand_as(link_b_in_with)
        other_passes_with_a = ~has_with_3d | link_b_in_with  # B is other when sensor in A
        other_passes_with_b = ~has_with_3d | link_a_in_with  # A is other when sensor in B

        # exclude_self filter for "other"
        exclude_self_3d = has_exclude_self[None, :, None].expand_as(link_b_in_self)
        other_passes_self_a = ~exclude_self_3d | ~link_b_in_self  # B not in self entity
        other_passes_self_b = ~exclude_self_3d | ~link_a_in_self  # A not in self entity

        # Combine: sensor in contact AND other passes all filters
        valid_a = sensor_in_a & other_passes_with_a & other_passes_self_a
        valid_b = sensor_in_b & other_passes_with_b & other_passes_self_b

        # Any valid contact per sensor per env
        shared_ground_truth_cache[:] = (valid_a | valid_b).any(dim=-1)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ContactSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.set(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    @gs.assert_built
    def read(self, envs_idx=None, force: bool = False):
        """
        Read the sensor data.

        Parameters
        ----------
        envs_idx : array_like, optional
            The indices of the environments to read. Defaults to all environments.
        force : bool, optional
            If True, run collision detection before reading. Defaults to False.
        """
        if force:
            solver = self._shared_metadata.solver
            # Update geoms state from links state
            solver._func_update_geoms(solver._scene._envs_idx)
            # Run collision detection
            solver.collider.clear()
            solver.collider.detection()

            # Update the ground truth cache for this sensor type
            dtype = self._get_cache_dtype()
            cache_slice = self._manager._cache_slices_by_type[type(self)]
            ground_truth_slice = self._manager._ground_truth_cache[dtype][:, cache_slice]
            self._update_shared_ground_truth_cache(self._shared_metadata, ground_truth_slice)

            # Copy ground truth to regular cache (skip delay/noise for force mode)
            self._manager._cache[dtype][:, cache_slice] = ground_truth_slice

            # Invalidate cloned cache so next read fetches fresh data
            self._manager._is_last_cache_cloned[(False, dtype)] = False
            self._manager._is_last_cache_cloned[(True, dtype)] = False

        return super().read(envs_idx)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug sphere when the sensor detects contact.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx).reshape((3,))
        is_contact = self.read(env_idx)

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_objects = None

        if is_contact:
            self.debug_object = context.draw_debug_sphere(
                pos=pos, radius=self._options.debug_sphere_radius, color=self._options.debug_color
            )


# ==========================================================================================================


@dataclass
class ContactForceSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata for all contact force sensors.
    """

    min_force: torch.Tensor = make_tensor_field((0, 3))
    max_force: torch.Tensor = make_tensor_field((0, 3))


@register_sensor(ContactForceSensorOptions, ContactForceSensorMetadata, tuple)
@ti.data_oriented
class ContactForceSensor(
    RigidSensorMixin[ContactForceSensorMetadata],
    NoisySensorMixin[ContactForceSensorMetadata],
    Sensor[ContactForceSensorMetadata],
):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.
    """

    def __init__(
        self,
        sensor_options: ContactForceSensorOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self.debug_object: "Mesh" | None = None

    def build(self):
        if not (isinstance(self._options.resolution, tuple) and len(self._options.resolution) == 3):
            self._options.resolution = tuple([self._options.resolution] * 3)

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
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: ContactForceSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        assert shared_metadata.solver is not None

        # Note that forcing GPU sync to operate on `slice(0, max(n_contacts))` is usually faster overall
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = all_contacts["force"], all_contacts["link_a"], all_contacts["link_b"]
        if shared_metadata.solver.n_envs == 0:
            force, link_a, link_b = force[None], link_a[None], link_b[None]

        # Short-circuit if no contacts
        if link_a.shape[-1] == 0:
            shared_ground_truth_cache.zero_()
            return

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            links_quat = links_quat[None]

        if gs.use_zerocopy:
            # Forces are aggregated BEFORE moving them in local frame for efficiency
            force_mask_a = link_a[:, None] == shared_metadata.links_idx[None, :, None]
            force_mask_b = link_b[:, None] == shared_metadata.links_idx[None, :, None]
            force_mask = force_mask_b.to(dtype=gs.tc_float) - force_mask_a.to(dtype=gs.tc_float)
            sensors_force = (force_mask[..., None] * force[:, None]).sum(dim=2)
            sensors_quat = links_quat[:, shared_metadata.links_idx]
            output_forces = shared_ground_truth_cache.reshape((max(shared_metadata.solver.n_envs, 1), -1, 3))
            output_forces[:] = inv_transform_by_quat(sensors_force, sensors_quat)
            return

        output_forces = shared_ground_truth_cache.contiguous()
        output_forces.zero_()
        _kernel_get_contacts_forces(
            force.contiguous(),
            link_a.contiguous(),
            link_b.contiguous(),
            links_quat.contiguous(),
            shared_metadata.links_idx,
            output_forces,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache.copy_(output_forces)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ContactForceSensorMetadata,
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
        shared_cache_per_sensor = shared_cache.reshape((shared_cache.shape[0], -1, 3))  # B, n_sensors * 3
        # clip for max force
        shared_cache_per_sensor.clamp_(min=-shared_metadata.max_force, max=shared_metadata.max_force)
        # set to 0 for undetectable force
        shared_cache_per_sensor.masked_fill_(torch.abs(shared_cache_per_sensor) < shared_metadata.min_force, 0.0)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug arrow representing the contact force.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(env_idx).reshape((3,))
        quat = self._link.get_quat(env_idx).reshape((4,))

        force = self.read(env_idx).reshape((3,))
        vec = tensor_to_array(transform_by_quat(force * self._options.debug_scale, quat))

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        self.debug_object = context.draw_debug_arrow(pos=pos, vec=vec, color=self._options.debug_color)
