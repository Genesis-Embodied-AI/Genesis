import numpy as np
from pydantic import conlist

import genesis as gs
from genesis.constants import FArrayType, MaybeVec3FArrayType, Vec3FArrayType, Vec3FType, Vec4FType

from .options import NoisySensorOptionsMixin, RigidSensorOptionsMixin, SensorOptions


def _validate_probe_arrays(values: Vec3FArrayType) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        gs.raise_exception(f"Probe locals array must have shape (N, 3), got: {array.shape}")
    if array.shape[0] == 0:
        gs.raise_exception("Probe locals array must have at least one entry")
    return array


def _validate_probe_pos_and_norm(
    probe_local_pos: Vec3FArrayType, probe_local_normal: MaybeVec3FArrayType, probe_radius: float | FArrayType
):
    probe_local_pos = _validate_probe_arrays(probe_local_pos)
    np_probe_normal = np.array(probe_local_normal)

    norms = np.linalg.norm(np_probe_normal, axis=-1)
    if np.any(norms < gs.EPS):
        gs.raise_exception(f"probe_local_normal must be non-zero vectors, got: {probe_local_normal}")

    if np_probe_normal.ndim > 1:
        np_probe_normal = _validate_probe_arrays(np_probe_normal)

        if len(probe_local_pos) != len(np_probe_normal):
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {len(probe_local_pos)} positions and {len(np_probe_normal)} normals."
            )
    if not isinstance(probe_radius, float) and len(probe_radius) != len(probe_local_pos):
        gs.raise_exception(
            "If radius is array-like, it must have the same length as probe_local_pos. "
            f"Got {len(probe_radius)} radii and {len(probe_local_pos)} probe positions."
        )


class KinematicTactileSensorMixin:
    """
    Parameters
    ----------
    probe_local_normal : array-like[array-like[float, float, float]]
        Probe sensing directions in link-local frame. Penetration is measured along this axis.
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. Objects within this distance are detected. Default: 0.005 (5mm)
    debug_sphere_color: array-like[float, float, float, float]
        The color of the debug sphere. Defaults to (1.0, 0.5, 0.0, 0.4).
    debug_contact_color: array-like[float, float, float, float]
        The color of the debug contact. Defaults to (1.0, 0.2, 0.0, 0.8).
    """

    probe_local_normal: MaybeVec3FArrayType = (0.0, 0.0, 1.0)
    probe_radius: float | FArrayType = 0.005

    debug_sphere_color: Vec4FType = (1.0, 0.5, 0.0, 0.4)
    debug_contact_color: Vec4FType = (1.0, 0.2, 0.0, 0.8)

    def model_post_init(self, _):
        super().model_post_init(_)

        if np.any(np.array(self.probe_radius) < 0):
            gs.raise_exception(f"radius must be non-negative, got: {self.probe_radius}")


class KinematicContactProbe(
    RigidSensorOptionsMixin, NoisySensorOptionsMixin, KinematicTactileSensorMixin, SensorOptions
):
    """
    A tactile sensor which queries contact depth relative to given probe normals and within the radius of the probe
    positions along a rigid entity link.

    The returned force is an spring-like (kinematic) estimate based on contact depth, computed as
    F = stiffness * penetration * probe_normal, as opposed to the actual impulse force on the link from the contact
    obtained from the physics solver.

    Note
    ----
    If this sensor is attached to a fixed entity, it will not detect contacts with other fixed entities.

    Parameters
    ----------
    probe_local_pos : array-like[array-like[float, float, float]]
        Probe positions in link-local frame. One (x, y, z) per probe.
    stiffness : float
        User-defined coefficient for force estimation. Default: 1000.0.
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    stiffness: float = 1000.0

    def model_post_init(self, _):
        super().model_post_init(_)
        _validate_probe_pos_and_norm(self.probe_local_pos, self.probe_local_normal, self.probe_radius)
        if self.stiffness < 0:
            gs.raise_exception(f"stiffness must be non-negative, got: {self.stiffness}")


class ElastomerDisplacement(
    RigidSensorOptionsMixin,
    NoisySensorOptionsMixin,
    KinematicTactileSensorMixin,
    SensorOptions,
):
    """
    A tactile sensor which estimates the displacement of the elastomer based on the contact force and depth of
    penetration along the probe normal on collisions.

    Note
    ----
    When probe_local_pos is a 2D array, the displacement is computed by FFT-based convolution over the grid.
    This provides a significant speedup when the number of probes is large.

    The equations for the displacement are as follows (from FOTS paper https://arxiv.org/pdf/2404.19217):
    dilate_displacement = Σ_i min(dilate_max_delta, penetration_depth) * (M - C_i) * exp(-λd ||M - C_i||²)
    shear_displacement = min(shear_max_delta, shear_velocity * dt) * exp(-λs ||M - G||²)
    twist_displacement = min(twist_max_delta, twist_angle) * (M - G) * exp(-λt ||M - G||²)

    Parameters
    ----------
    probe_local_pos: array-like[array-like[float, float, float]], shape (N, 3) or (M, N, 3)
        Probe positions in link-local frame.
    dilate_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by dilate motion.
    shear_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by shear motion.
    twist_coefficient: float
        The coefficient of the exponential function that can affect the displacement caused by twist motion.
    dilate_max_delta: float
        The maximum dilate depth in meters.
    shear_max_delta: float
        The maximum shear magnitude in meters.
    twist_max_delta: float
        The maximum twist angle in degrees.
    """

    probe_local_pos: Vec3FArrayType | conlist(Vec3FArrayType, min_length=1) = [(0.0, 0.0, 0.0)]
    dilate_coefficient: float = 1e-2
    shear_coefficient: float = 1e-2
    twist_coefficient: float = 1e-2
    dilate_max_delta: float = 0.1
    shear_max_delta: float = 0.1
    twist_max_delta: float = 50.0

    def model_post_init(self, _):
        super().model_post_init(_)

        probe_local_pos = self.probe_local_pos
        if np.array(self.probe_local_pos).ndim > 3:
            gs.raise_exception(
                f"probe_local_pos should have shape (N, 3) or (M, N, 3), got: {np.array(self.probe_local_pos).shape}"
            )
        elif np.array(self.probe_local_pos).ndim == 3:
            probe_local_pos = np.array(probe_local_pos).reshape(-1, 3)

        _validate_probe_pos_and_norm(probe_local_pos, self.probe_local_normal, self.probe_radius)

        if np.any(
            np.array(
                (
                    self.dilate_coefficient,
                    self.shear_coefficient,
                    self.twist_coefficient,
                    self.dilate_max_delta,
                    self.shear_max_delta,
                    self.twist_max_delta,
                )
            )
            < 0
        ):
            gs.raise_exception("Elastomer displacement coefficients and max_deltas should be non-negative.")
