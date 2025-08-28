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

from .base_sensor import (
    NoisySensorBase,
    NoisySensorMetadataBase,
    NoisySensorOptionsBase,
    RigidSensorOptionsBase,
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer


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

            if contact_data_link_a == sensors_link_idx[i_s]:
                for j in ti.static(range(3)):
                    output[i_b, j_s + j] += force_a[j]
            if contact_data_link_b == sensors_link_idx[i_s]:
                for j in ti.static(range(3)):
                    output[i_b, j_s + j] += force_b[j]


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
    expanded_links_idx: torch.Tensor = field(
        default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device)
    )


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
                self._shared_metadata.expanded_links_idx,
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


class ContactForceSensorOptions(NoisySensorOptionsBase):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    min_force : float | tuple[float, float, float], optional
        The minimum detectable force per each axis. Values below this will be treated as 0. Default is 0.
    max_force : float | tuple[float, float, float], optional
        The maximum detectable force per each axis. Values above this will be clipped. Default is infinity.
    noise_std : float | tuple[float, float, float], optional
        The standard deviation of the noise.
    bias : float | tuple[float, float, float], optional
        The bias of the sensor.
    bias_drift_std : float | tuple[float, float, float], optional
        The standard deviation of the bias drift.
    delay : float, optional
        The delay in seconds before the sensor data is read.
    jitter : float, optional
        The time jitter standard deviation in seconds before the sensor data is read.
    interpolate_for_delay : bool, optional
        If True, the sensor data is interpolated between data points for delay + jitter.
        Otherwise, the sensor data at the closest time step will be used.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    """

    min_force: float | tuple[float, float, float] = 0.0
    max_force: float | tuple[float, float, float] = np.inf

    def validate(self, scene):
        super().validate(scene)
        assert (
            isinstance(self.min_force, float) or len(self.min_force) == 3
        ), "Min force must be a float or tuple of 3 floats."
        assert (
            isinstance(self.max_force, float) or len(self.max_force) == 3
        ), "Max force must be a float or tuple of 3 floats."
        assert np.all(np.array(self.min_force) >= 0), "Min/max force must be non-negative."
        assert np.all(np.array(self.max_force) > np.array(self.min_force)), "Min force should be less than max force."


@dataclass
class ContactForceSensorMetadata(NoisySensorMetadataBase):
    """
    Shared metadata for all contact force sensors.
    """

    min_max_force: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=gs.tc_float, device=gs.device))


@register_sensor(ContactForceSensorOptions, ContactForceSensorMetadata)
@ti.data_oriented
class ContactForceSensor(NoisySensorBase):
    """
    Sensor that returns the total contact force being applied to the associated RigidLink in its local frame.
    """

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        # shape of min_max_force is (n_sensors, 2, 3)
        min_max_force = torch.zeros((2, 3), dtype=gs.tc_float, device=gs.device)
        min_max_force[0, :] = (
            self._options.min_force
            if isinstance(self._options.min_force, float)
            else torch.tensor(self._options.min_force, dtype=gs.tc_float, device=gs.device)
        )
        min_max_force[1, :] = (
            self._options.max_force
            if isinstance(self._options.max_force, float)
            else torch.tensor(self._options.max_force, dtype=gs.tc_float, device=gs.device)
        )
        self._shared_metadata.min_max_force = (
            min_max_force.unsqueeze(0)
            if self._shared_metadata.min_max_force.numel() == 0
            else torch.stack((self._shared_metadata.min_max_force, min_max_force))
        )

    def get_return_format(self) -> dict[str, tuple[int, ...]]:
        return (3,)

    def get_cache_length(self) -> int:
        return 1

    @classmethod
    def get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def update_shared_ground_truth_cache(
        cls, shared_metadata: ContactForceSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        all_contacts = shared_metadata.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        shared_ground_truth_cache.fill_(0.0)
        if all_contacts["link_a"].shape[-1] == 0:
            return  # no contacts

        links_quat = shared_metadata.solver.get_links_quat()
        if shared_metadata.solver.n_envs == 0:
            all_contacts["force"] = all_contacts["force"].unsqueeze(0)
            all_contacts["link_a"] = all_contacts["link_a"].unsqueeze(0)
            all_contacts["link_b"] = all_contacts["link_b"].unsqueeze(0)
            links_quat = links_quat.unsqueeze(0)

        _kernel_get_contacts_forces(
            all_contacts["force"].contiguous(),
            all_contacts["link_a"].contiguous(),
            all_contacts["link_b"].contiguous(),
            links_quat.contiguous(),
            np.array(shared_metadata.links_idx, dtype=gs.np_int),
            shared_ground_truth_cache,
        )

    @classmethod
    def update_shared_cache(
        cls,
        shared_metadata: ContactForceSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        torch.normal(0, shared_metadata.jitter_std_in_steps, out=shared_metadata.jitter_in_steps)
        cls._apply_delay_to_shared_cache(
            shared_metadata,
            shared_cache,
            buffered_data,
            shared_metadata.jitter_in_steps,
            shared_metadata.interpolate_for_delay,
        )
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        reshaped_cache = shared_cache.reshape(shared_cache.shape[0], -1, 3)  # B, n_sensors * 3
        reshaped_cache.clamp_(max=shared_metadata.min_max_force[:, 1, :])  # clip for max force
        reshaped_cache[reshaped_cache < shared_metadata.min_max_force[:, 0, :]] = 0.0  # set to 0 for undetectable force
        cls._quantize_to_resolution(shared_metadata, shared_cache)
