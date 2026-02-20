import numpy as np
from pydantic import conlist

import genesis as gs
from genesis.constants import FArrayType, Vec3FArrayType, Vec3FType, Vec4FType

from .options import NoisySensorOptionsMixin, RigidSensorOptionsMixin, SensorOptions


class KinematicTactileSensorMixin:
    """
    Parameters
    ----------
    probe_radius : float | array-like[float]
        Probe sensing radius in meters. Objects within this distance are detected. Default: 0.005 (5mm)
    debug_sphere_color: array-like[float, float, float, float]
        The color of the debug sphere. Defaults to (1.0, 0.5, 0.0, 0.4).
    debug_contact_color: array-like[float, float, float, float]
        The color of the debug contact. Defaults to (1.0, 0.2, 0.0, 0.8).
    """

    probe_radius: float | FArrayType = 0.005

    debug_sphere_color: Vec4FType = (1.0, 0.5, 0.0, 0.4)
    debug_contact_color: Vec4FType = (1.0, 0.2, 0.0, 0.8)

    def _validate_probe_arrays(self, values: Vec3FArrayType) -> np.ndarray:
        array = np.array(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            gs.raise_exception(f"Probe locals array must have shape (N, 3), got: {array.shape}")
        if array.shape[0] == 0:
            gs.raise_exception("Probe locals array must have at least one entry")
        return array

    def model_post_init(self, _):
        super().model_post_init(_)

        if np.any(np.array(self.probe_radius) < 0):
            gs.raise_exception(f"radius must be non-negative, got: {self.probe_radius}")


class ElastomerDisplacementSensorMixin:
    """
    Parameters
    ----------
    dilate_coefficient: float
        The coefficient for the Gaussian decay in distance of the displacement caused by dilate motion.
        Higher values result in more localized displacement.
    shear_coefficient: float
        The coefficient for the effect of displacement caused by shear motion.
    twist_coefficient: float
        The coefficient for the effect of displacement caused by twist motion.
    shear_max_delta: float
        Maximum shear magnitude in meters.
    twist_max_delta: float
        Maximum twist angle in degrees.
    """

    dilate_coefficient: float = 1.25e-3
    shear_coefficient: float = 2.10e-4
    twist_coefficient: float = 3.80e-4
    shear_max_delta: float = 0.1
    twist_max_delta: float = 50.0


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
    probe_local_normal : array-like[array-like[float, float, float]]
        Probe sensing directions in link-local frame. Penetration is measured along this axis.
    stiffness : float
        User-defined coefficient for force estimation. Default: 1000.0.
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    probe_local_normal: Vec3FArrayType = [(0.0, 0.0, 1.0)]
    stiffness: float = 1000.0

    def model_post_init(self, _):
        super().model_post_init(_)

        if self.stiffness < 0:
            gs.raise_exception(f"stiffness must be non-negative, got: {self.stiffness}")

        probe_local_pos = self._validate_probe_arrays(self.probe_local_pos)
        probe_local_normal = self._validate_probe_arrays(self.probe_local_normal)
        norms = np.linalg.norm(probe_local_normal, axis=1)
        if np.any(norms < gs.EPS):
            gs.raise_exception(f"probe_local_normal must be non-zero vectors, got: {probe_local_normal}")

        if len(probe_local_pos) != len(probe_local_normal):
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {len(probe_local_pos)} positions and {len(probe_local_normal)} normals."
            )
        if not isinstance(self.probe_radius, float) and len(self.probe_radius) != len(probe_local_pos):
            gs.raise_exception(
                "If radius is array-like, it must have the same length as probe_local_pos. "
                f"Got {len(self.probe_radius)} radii and {len(probe_local_pos)} probe positions."
            )


class ElastomerDisplacementSensor(
    RigidSensorOptionsMixin,
    NoisySensorOptionsMixin,
    KinematicTactileSensorMixin,
    ElastomerDisplacementSensorMixin,
    SensorOptions,
):
    """
    A tactile sensor which estimates the displacement of the elastomer based on the contact force and depth of
    penetration along the probe normal on collisions.

    Note
    ----
    For large number of probes, consider using ElastomerDisplacementGridSensor instead for better performance.
    """

    probe_local_pos: Vec3FArrayType = [(0.0, 0.0, 0.0)]
    probe_local_normal: Vec3FArrayType = [(0.0, 0.0, 1.0)]

    def model_post_init(self, _):
        super().model_post_init(_)

        probe_local_pos = self._validate_probe_arrays(self.probe_local_pos)
        probe_local_normal = self._validate_probe_arrays(self.probe_local_normal)
        norms = np.linalg.norm(probe_local_normal, axis=1)
        if np.any(norms < gs.EPS):
            gs.raise_exception(f"probe_local_normal must be non-zero vectors, got: {probe_local_normal}")

        if len(probe_local_pos) != len(probe_local_normal):
            gs.raise_exception(
                "probe_local_pos and probe_local_normal must have the same length. "
                f"Got {len(probe_local_pos)} positions and {len(probe_local_normal)} normals."
            )
        if not isinstance(self.probe_radius, float) and len(self.probe_radius) != len(probe_local_pos):
            gs.raise_exception(
                "If radius is array-like, it must have the same length as probe_local_pos. "
                f"Got {len(self.probe_radius)} radii and {len(probe_local_pos)} probe positions."
            )


class ElastomerDisplacementGridSensor(
    RigidSensorOptionsMixin,
    NoisySensorOptionsMixin,
    KinematicTactileSensorMixin,
    ElastomerDisplacementSensorMixin,
    SensorOptions,
):
    """
    A tactile sensor which estimates the displacement of the elastomer based on the contact force and depth of
    penetration along the probe normal on collisions.

    Probe positions are generated on a regular 2D grid from bounds and grid size.
    Displacement is computed by FFT-based convolution over the grid.

    Parameters
    ----------
    probe_local_pos_grid_bounds: array-like[[float, float, float], [float, float, float]]
        The min and max bounds of the probe positions in link-local frame.
    probe_local_normal: array-like[float, float, float]
        The normal of all probes in link-local frame.
    probe_grid_size: array-like[int, int]
        Number of probes along x and y.
    """

    probe_local_pos_grid_bounds: conlist(Vec3FType, min_length=2, max_length=2)
    probe_grid_size: tuple[int, int]
    probe_local_normal: Vec3FType

    def model_post_init(self, _):
        super().model_post_init(_)

        if np.linalg.norm(self.probe_local_normal) < gs.EPS:
            gs.raise_exception(f"probe_local_normal must be non-zero, got: {self.probe_local_normal}")

        nx, ny = self.probe_grid_size[0], self.probe_grid_size[1]
        if nx < 1 or ny < 1:
            gs.raise_exception(f"probe_grid_size must be at least (1, 1), got: {self.probe_grid_size}")
