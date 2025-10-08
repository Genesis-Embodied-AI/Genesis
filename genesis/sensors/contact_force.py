from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import ti_inv_transform_by_quat, trans_to_T, transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import (
    MaybeTuple3FType,
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    NoisySensorOptionsMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    RigidSensorOptionsMixin,
    Sensor,
    SensorOptions,
    SharedSensorMetadata,
    _to_tuple,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
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


class ContactSensorOptions(RigidSensorOptionsMixin, SensorOptions):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    debug_sphere_radius : float, optional
        The radius of the debug sphere. Defaults to 0.05.
    debug_color : float, optional
        The rgba color of the debug sphere. Defaults to (1.0, 0.0, 1.0, 0.5).
    """

    debug_sphere_radius: float = 0.05
    debug_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.5)


@dataclass
class ContactSensorMetadata(SharedSensorMetadata):
    """
    Metadata for all rigid contact sensors.
    """

    solver: RigidSolver | None = None
    expanded_links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)


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

        self._link: "RigidLink" | None = None
        self.debug_object: "Mesh" | None = None

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
        if all_contacts["link_a"].numel() == 0:
            shared_ground_truth_cache.fill_(False)
        else:
            contact_links = torch.cat([all_contacts["link_a"], all_contacts["link_b"]], dim=-1)
            is_contact = (contact_links.unsqueeze(-2) == shared_metadata.expanded_links_idx.unsqueeze(-1)).any(-1)
            shared_ground_truth_cache.copy_(is_contact)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ContactSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug sphere when the sensor detects contact.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None

        pos = self._link.get_pos(envs_idx=env_idx).squeeze(0)
        is_contact = self.read(envs_idx=env_idx).item()

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_objects = None

        if is_contact:
            self.debug_object = context.draw_debug_sphere(
                pos=pos, radius=self._options.debug_sphere_radius, color=self._options.debug_color
            )


# ==========================================================================================================


class ContactForceSensorOptions(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    min_force : float | tuple[float, float, float], optional
        The minimum detectable absolute force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | tuple[float, float, float], optional
        The maximum output absolute force per each axis. Values above this will be clipped. Default is infinity.
    debug_color : float, optional
        The rgba color of the debug arrow. Defaults to (1.0, 0.0, 1.0, 0.5).
    debug_scale : float, optional
        The scale factor for the debug force arrow. Defaults to 0.01.
    """

    min_force: MaybeTuple3FType = 0.0
    max_force: MaybeTuple3FType = np.inf

    debug_color: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.5)
    debug_scale: float = 0.01

    def model_post_init(self, _):
        if not (
            isinstance(self.min_force, float) or (isinstance(self.min_force, Sequence) and len(self.min_force) == 3)
        ):
            gs.raise_exception(f"min_force must be a float or tuple of 3 floats, got: {self.min_force}")
        if not (
            isinstance(self.max_force, float) or (isinstance(self.max_force, Sequence) and len(self.max_force) == 3)
        ):
            gs.raise_exception(f"max_force must be a float or tuple of 3 floats, got: {self.max_force}")
        if np.any(np.array(self.min_force) < 0):
            gs.raise_exception(f"min_force must be non-negative, got: {self.min_force}")
        if np.any(np.array(self.max_force) <= np.array(self.min_force)):
            gs.raise_exception(f"min_force should be less than max_force, got: {self.min_force} and {self.max_force}")
        if self.resolution is not None and not (
            isinstance(self.resolution, float) or (isinstance(self.resolution, Sequence) and len(self.resolution) == 3)
        ):
            gs.raise_exception(f"resolution must be a float or tuple of 3 floats, got: {self.resolution}")


@dataclass
class ContactForceSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata for all contact force sensors.
    """

    min_force: torch.Tensor = make_tensor_field((0, 3))
    max_force: torch.Tensor = make_tensor_field((0, 3))
    output_forces: torch.Tensor = make_tensor_field((0, 0))  # FIXME: remove once we have contiguous cache slices


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

        if self._shared_metadata.output_forces.numel() == 0:
            self._shared_metadata.output_forces.reshape(self._manager._sim._B, 0)
        self._shared_metadata.output_forces = concat_with_tensor(
            self._shared_metadata.output_forces,
            torch.empty((self._manager._sim._B, 3), dtype=gs.tc_float, device=gs.device),
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
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        force, link_a, link_b = all_contacts["force"], all_contacts["link_a"], all_contacts["link_b"]

        if not shared_ground_truth_cache.is_contiguous():
            shared_metadata.output_forces.fill_(0.0)
        else:
            shared_ground_truth_cache.fill_(0.0)

        if link_a.shape[-1] == 0:
            return  # no contacts

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            force = force.unsqueeze(0)
            link_a = link_a.unsqueeze(0)
            link_b = link_b.unsqueeze(0)
            links_quat = links_quat.unsqueeze(0)

        _kernel_get_contacts_forces(
            force.contiguous(),
            link_a.contiguous(),
            link_b.contiguous(),
            links_quat.contiguous(),
            shared_metadata.links_idx,
            shared_ground_truth_cache if shared_ground_truth_cache.is_contiguous() else shared_metadata.output_forces,
        )
        if not shared_ground_truth_cache.is_contiguous():
            shared_ground_truth_cache[:] = shared_metadata.output_forces

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: ContactForceSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0.0, shared_metadata.jitter_ts, out=shared_metadata.cur_jitter_ts)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.cur_jitter_ts,
            shared_metadata.interpolate,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        shared_cache_per_sensor = shared_cache.reshape(shared_cache.shape[0], -1, 3)  # B, n_sensors * 3
        # clip for max force
        shared_cache_per_sensor.clamp_(min=-shared_metadata.max_force, max=shared_metadata.max_force)
        # set to 0 for undetectable force
        shared_cache_per_sensor[torch.abs(shared_cache_per_sensor) < shared_metadata.min_force] = 0.0
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug arrow representing the contact force.

        Only draws for first rendered environment.
        """
        env_idx = context.rendered_envs_idx[0]

        pos = self._link.get_pos(envs_idx=env_idx).squeeze(0)
        quat = self._link.get_quat(envs_idx=env_idx).squeeze(0)

        force = self.read(envs_idx=env_idx if self._manager._sim.n_envs > 0 else None)
        vec = tensor_to_array(transform_by_quat(force.squeeze(0) * self._options.debug_scale, quat))

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None

        self.debug_object = context.draw_debug_arrow(pos=pos, vec=vec, color=self._options.debug_color)
