"""
Camera sensor options for Rasterizer, Raytracer, and Batch Renderer backends.
"""

from typing import Any, Optional

import genesis as gs
from .options import SensorOptions, Tuple3FType, RigidSensorOptionsMixin


class BaseCameraOptions(RigidSensorOptionsMixin, SensorOptions):
    """
    Base class for camera sensor options containing common properties.

    Parameters
    ----------
    res : tuple[int, int]
        Resolution as (width, height). Default is (512, 512).
    pos : tuple[float, float, float]
        Camera position offset. If attached to a link, this is relative to the link frame.
        If not attached, this is relative to the world origin. Default is (3.5, 0.0, 1.5).
    lookat : tuple[float, float, float]
        Point the camera looks at in world frame.
    up : tuple[float, float, float]
        Up vector for camera orientation. Default is (0, 0, 1).
    fov : float
        Vertical field of view in degrees. Default is 60.0.
    lights : list[dict], optional
        List of lights to add for this camera backend. Each light is a dict with
        backend-specific parameters. Default is empty list.
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached. -1 or None for static sensors.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    """

    res: tuple[int, int] = (512, 512)
    pos: Tuple3FType = (3.5, 0.0, 1.5)
    lookat: Tuple3FType = (0.0, 0.0, 0.0)
    up: Tuple3FType = (0.0, 0.0, 1.0)
    fov: float = 60.0
    lights: list[dict] = []

    def model_post_init(self, _):
        if not isinstance(self.res, (tuple, list)) or len(self.res) != 2:
            gs.raise_exception(f"res must be a tuple of (width, height), got: {self.res}")
        if self.res[0] <= 0 or self.res[1] <= 0:
            gs.raise_exception(f"res must have positive dimensions, got: {self.res}")
        if self.fov <= 0 or self.fov >= 180:
            gs.raise_exception(f"fov must be between 0 and 180 degrees, got: {self.fov}")
        if not isinstance(self.lights, list):
            gs.raise_exception(f"lights must be a list, got: {type(self.lights)}")
        for i, light in enumerate(self.lights):
            if not isinstance(light, dict):
                gs.raise_exception(f"lights[{i}] must be a dict, got: {type(light)}")


class RasterizerCameraOptions(BaseCameraOptions):
    """
    Options for Rasterizer camera sensor (OpenGL-based rendering).

    Parameters
    ----------
    near : float
        Near clipping plane distance. Default is 0.01.
    far : float
        Far clipping plane distance. Default is 100.0.
    """

    near: float = 0.01
    far: float = 100.0
    # Camera images are updated lazily on read(), so skip per-step measured-cache updates
    update_ground_truth_only: bool = True

    def model_post_init(self, _):
        super().model_post_init(_)
        if self.near <= 0:
            gs.raise_exception(f"near must be positive, got: {self.near}")
        if self.far <= self.near:
            gs.raise_exception(f"far must be greater than near, got near={self.near}, far={self.far}")


class RaytracerCameraOptions(BaseCameraOptions):
    """
    Options for Raytracer camera sensor (LuisaRender path tracing).

    Parameters
    ----------
    model : str
        Camera model: "pinhole" or "thinlens". Default is "pinhole".
    spp : int
        Samples per pixel for path tracing. Default is 256.
    denoise : bool
        Whether to apply denoising. Default is False.
    aperture : float
        Aperture size for thinlens camera (depth of field). Default is 2.8.
    focal_len : float
        Focal length in meters for thinlens camera. Default is 0.05.
    focus_dist : float
        Focus distance in meters for thinlens camera. Default is 3.0.
    env_surface : gs.surfaces.Surface | None
        Environment surface for skybox. Default is None.
    env_radius : float
        Environment sphere radius. Default is 15.0.
    env_pos : tuple[float, float, float]
        Environment sphere position. Default is (0, 0, 0).
    env_quat : tuple[float, float, float, float]
        Environment sphere quaternion (w, x, y, z). Default is (1, 0, 0, 0).
    """

    model: str = "pinhole"
    spp: int = 256
    denoise: bool = False
    aperture: float = 2.8
    focal_len: float = 0.05
    focus_dist: float = 3.0
    env_surface: Any = None  # gs.surfaces.Surface
    env_radius: float = 15.0
    env_pos: Tuple3FType = (0.0, 0.0, 0.0)
    env_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    update_ground_truth_only: bool = True

    def model_post_init(self, _):
        super().model_post_init(_)
        if self.model not in ("pinhole", "thinlens"):
            gs.raise_exception(f"model must be 'pinhole' or 'thinlens', got: {self.model}")
        if self.spp <= 0:
            gs.raise_exception(f"spp must be positive, got: {self.spp}")


class BatchRendererCameraOptions(BaseCameraOptions):
    """
    Options for Batch Renderer camera sensor (Madrona GPU batch rendering).

    Note: All batch renderer cameras must have the same resolution.

    Parameters
    ----------
    use_rasterizer : bool
        Whether to use rasterizer mode. Default is True.
    """

    near: float = 0.01
    far: float = 100.0
    use_rasterizer: bool = True
    update_ground_truth_only: bool = True

    def model_post_init(self, _):
        super().model_post_init(_)
        if self.near <= 0:
            gs.raise_exception(f"near must be positive, got: {self.near}")
        if self.far <= self.near:
            gs.raise_exception(f"far must be greater than near, got near={self.near}, far={self.far}")
