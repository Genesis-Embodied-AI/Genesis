from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.engine.solvers import RigidSolver
from genesis.utils.geom import (
    ti_inv_transform_by_quat,
)

from .base_sensor import NoisySensorOptionsBase, RigidSensorOptionsBase, Sensor, SharedSensorMetadata
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


@ti.kernel
def _kernel_get_contacts_forces(
    contact_forces: ti.types.ndarray(),
    contact_normals: ti.types.ndarray(),
    link_a: ti.types.ndarray(),
    link_b: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    min_max_force: ti.types.ndarray(),
    sensors_link_idx: ti.types.ndarray(),
    return_normtan: ti.types.ndarray(),
    output: ti.types.ndarray(),
):
    for i_b, i_c, i_s in ti.ndrange(output.shape[0], link_a.shape[-1], sensors_link_idx.shape[-1]):
        contact_data_link_a = link_a[i_b, i_c]
        contact_data_link_b = link_b[i_b, i_c]
        if contact_data_link_a == sensors_link_idx[i_s] or contact_data_link_b == sensors_link_idx[i_s]:
            j_s = i_s * 4

            quat_a = ti.Vector.zero(ti.f32, 4)
            quat_b = ti.Vector.zero(ti.f32, 4)
            for j in ti.static(range(4)):
                quat_a[j] = links_quat[i_b, contact_data_link_a, j]
                quat_b[j] = links_quat[i_b, contact_data_link_b, j]

            force_vec = ti.Vector.zero(ti.f32, 3)
            for j in ti.static(range(3)):
                force_vec[j] = contact_forces[i_b, i_c, j]

            force_a = ti_inv_transform_by_quat(-force_vec, quat_a)
            force_b = ti_inv_transform_by_quat(force_vec, quat_b)

            if return_normtan[i_s]:  # |Fn|, |Ft|, tx, ty
                normal_vec = ti.Vector.zero(ti.f32, 3)
                for j in ti.static(range(3)):
                    normal_vec[j] = contact_normals[i_b, i_c, j]

                if contact_data_link_a == sensors_link_idx[i_s]:
                    normal_a_local = ti_inv_transform_by_quat(normal_vec, quat_a)
                    fn_a = ti.abs(force_a.dot(normal_a_local))
                    force_tangential_a = force_a - force_a.dot(normal_a_local) * normal_a_local

                    output[i_b, j_s] += fn_a
                    output[i_b, j_s + 2] += force_tangential_a[0]
                    output[i_b, j_s + 3] += force_tangential_a[1]

                if contact_data_link_b == sensors_link_idx[i_s]:
                    normal_b_local = ti_inv_transform_by_quat(-normal_vec, quat_b)
                    fn_b = ti.abs(force_b.dot(normal_b_local))
                    force_tangential_b = force_b - force_b.dot(normal_b_local) * normal_b_local

                    output[i_b, j_s] += fn_b
                    output[i_b, j_s + 2] += force_tangential_b[0]
                    output[i_b, j_s + 3] += force_tangential_b[1]

            else:  # Fx, Fy, Fz, |F|
                if contact_data_link_a == sensors_link_idx[i_s]:
                    for j in ti.static(range(3)):
                        output[i_b, j_s + j] += force_a[j]
                if contact_data_link_b == sensors_link_idx[i_s]:
                    for j in ti.static(range(3)):
                        output[i_b, j_s + j] += force_b[j]

    for i_b, i_s in ti.ndrange(output.shape[0], sensors_link_idx.shape[-1]):
        j_s = i_s * 4
        min_normal_force = min_max_force[j_s]
        max_normal_force = min_max_force[j_s + 1]
        min_shear_force = min_max_force[j_s + 2]
        max_shear_force = min_max_force[j_s + 3]

        if return_normtan[i_s]:  # |Fn|, |Ft|, tx, ty
            if output[i_b, j_s] < min_normal_force:
                output[i_b, j_s] = 0.0
            if output[i_b, j_s] > max_normal_force:
                output[i_b, j_s] = max_normal_force

            ft_mag = ti.sqrt(output[i_b, j_s + 2] ** 2 + output[i_b, j_s + 3] ** 2)
            if ft_mag > min_shear_force:
                output[i_b, j_s + 1] = ti.max(ft_mag, min_shear_force)
                output[i_b, j_s + 2] /= ft_mag
                output[i_b, j_s + 3] /= ft_mag
            else:
                output[i_b, j_s + 1] = 0.0
                output[i_b, j_s + 2] = 0.0
                output[i_b, j_s + 3] = 0.0

        else:  # Fx, Fy, Fz, |F|
            for j in ti.static(range(2)):
                if output[i_b, j_s + j] < min_shear_force:
                    output[i_b, j_s + j] = 0.0
                if output[i_b, j_s + j] > max_shear_force:
                    output[i_b, j_s + j] = max_shear_force

            if output[i_b, j_s + 2] < min_normal_force:
                output[i_b, j_s + 2] = 0.0
            if output[i_b, j_s + 2] > max_normal_force:
                output[i_b, j_s + 2] = max_normal_force

            output[i_b, j_s + 3] = ti.sqrt(
                output[i_b, j_s] ** 2 + output[i_b, j_s + 1] ** 2 + output[i_b, j_s + 2] ** 2
            )


class ContactSensorOptions(RigidSensorOptionsBase):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    delay : float
        The delay in seconds before the sensor data is read.
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    """


@dataclass
class ContactSensorMetadata(SharedSensorMetadata):
    """
    Metadata for all rigid contact sensors.
    """

    solver: RigidSolver | None = None
    expanded_links_idx: torch.Tensor = torch.tensor([])


@register_sensor(ContactSensorOptions, ContactSensorMetadata)
@ti.data_oriented
class ContactSensor(Sensor):
    """
    Sensor that returns bool based on whether associated RigidLink is in contact.
    """

    def build(self):
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.expanded_links_idx = torch.cat(
            [
                self._shared_metadata.expanded_links_idx.to(gs.device),
                torch.tensor(
                    [self._options.entity_idx + self._options.link_idx_local], dtype=gs.tc_int, device=gs.device
                )
                .unsqueeze(0)
                .expand(self._manager._sim._B, -1, -1),
            ],
            dim=-1,
        )

    def get_return_format(self) -> tuple[int, ...]:
        return (1,)

    def get_cache_length(self) -> int:
        return 1

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_bool

    @classmethod
    def update_shared_ground_truth_cache(
        cls, shared_metadata: ContactSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        contact_links = torch.cat([all_contacts["link_a"], all_contacts["link_b"]], dim=-1)
        is_contact = (contact_links.unsqueeze(-2) == shared_metadata.expanded_links_idx).any(dim=-1)
        shared_ground_truth_cache.copy_(is_contact)

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: ContactSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)


# ==========================================================================================================


class ForceSensorOptions(NoisySensorOptionsBase, RigidSensorOptionsBase):
    """
    Sensor that returns contact force in the associated RigidLink's local frame.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    min_force : float, tuple[float, float, float], optional
        The minimum detectable force for the sensor. Values below this will be treated as 0.
    max_force : float, tuple[float, float, float], optional
        The maximum detectable force for the sensor. Values above this will be clipped.
    return_normtan : bool, optional
        By default, `read()` will return a dict {"force": (Fx, Fy, Fz), "magnitude": |F|}. By convention, Fz represents
        the normal direction, but the actual z direction is defined by link frame rotation. If return_normtan is True,
        the returned dict will contain {"normal": |Fn|, "tangential": |Ft|, "tangential_direction": (tx, ty)}.
    noise_std : float | tuple[float, float, float] | tuple[float, float]
        The standard deviation of the noise.
    bias : float | tuple[float, float, float] | tuple[float, float]
        The bias of the sensor.
    bias_drift_std : float | tuple[float, float, float] | tuple[float, float]
        The standard deviation of the bias drift.
    delay : float
        The delay in seconds before the sensor data is read.
    jitter : float
        The time jitter standard deviation in seconds before the sensor data is read.
    delay : float
        The read delay time in seconds. Data read will be outdated by this amount.
    interpolate_for_delay : bool
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used.
    update_ground_truth_only : bool
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    """

    min_normal_force: float = 0.0
    max_normal_force: float = np.inf
    min_shear_force: float = 0.0
    max_shear_force: float = np.inf
    return_normtan: bool = False

    def validate(self, scene):
        super().validate(scene)
        assert self.min_normal_force >= 0, "Minimum normal force must be non-negative."
        assert self.min_normal_force < self.max_normal_force, "Min normal force should be less than max normal force."
        assert self.min_shear_force >= 0, "Minimum shear force must be non-negative."
        assert self.min_shear_force < self.max_shear_force, "Min shear force should be less than max shear force."


@dataclass
class ForceSensorMetadata(ContactSensorMetadata):
    """
    Metadata for all rigid contact force sensors.
    """

    links_idx: torch.Tensor = torch.tensor([])
    min_max_force: torch.Tensor = torch.tensor([])
    return_normtan: list[bool] = field(default_factory=list)


@register_sensor(ForceSensorOptions, ForceSensorMetadata)
@ti.data_oriented
class ForceSensor(Sensor):
    """
    Sensor that returns the contact force in the associated RigidLink's local frame.
    """

    def build(self):
        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        self._shared_metadata.links_idx = torch.cat(
            [
                self._shared_metadata.links_idx.to(gs.device),
                torch.tensor(
                    [self._options.entity_idx + self._options.link_idx_local], dtype=gs.tc_int, device=gs.device
                ),
            ],
            dim=-1,
        )
        self._shared_metadata.min_max_force = torch.cat(
            [
                self._shared_metadata.min_max_force.to(gs.device),
                torch.tensor(
                    [
                        self._options.min_normal_force,
                        self._options.max_normal_force,
                        self._options.min_shear_force,
                        self._options.max_shear_force,
                    ],
                    dtype=gs.tc_float,
                    device=gs.device,
                ),
            ],
            dim=-1,
        )
        self._shared_metadata.return_normtan.append(self._options.return_normtan)

    def get_return_format(self) -> dict[str, tuple[int, ...]]:
        if self._options.return_normtan:
            return {"normal": (1,), "tangential": (1,), "tangential_direction": (2,)}
        else:
            return {"force": (3,), "magnitude": (1,)}

    def get_cache_length(self) -> int:
        return 1

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def update_shared_ground_truth_cache(
        cls, shared_metadata: ForceSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        shared_ground_truth_cache.fill_(0.0)
        if all_contacts["link_a"].shape[-1] == 0:
            return  # no contacts

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            all_contacts["force"] = all_contacts["force"].unsqueeze(0)
            all_contacts["normal"] = all_contacts["normal"].unsqueeze(0)
            all_contacts["link_a"] = all_contacts["link_a"].unsqueeze(0)
            all_contacts["link_b"] = all_contacts["link_b"].unsqueeze(0)
            links_quat = links_quat.unsqueeze(0)

        _kernel_get_contacts_forces(
            all_contacts["force"].contiguous(),
            all_contacts["normal"].contiguous(),
            all_contacts["link_a"].contiguous(),
            all_contacts["link_b"].contiguous(),
            links_quat.contiguous(),
            shared_metadata.min_max_force,
            shared_metadata.links_idx,
            np.array(shared_metadata.return_normtan),
            shared_ground_truth_cache,
        )

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: ForceSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)
