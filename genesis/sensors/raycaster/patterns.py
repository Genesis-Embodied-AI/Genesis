import math
from dataclasses import dataclass
from typing import Sequence

import torch

import genesis as gs
from genesis.utils.geom import spherical_to_cartesian

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern


def _generate_uniform_angles(
    n: int | None = None,
    fov: float | tuple[float, float] | None = None,
    res: float | None = None,
    angles: Sequence[float] | None = None,
    use_degrees: bool = True,
) -> torch.Tensor:

    if angles is None:
        assert fov is not None, "FOV should be provided if angles not given."

        if res is not None:
            if isinstance(fov, tuple):
                f_min, f_max = fov
            else:
                f_max = fov / 2.0
                f_min = -f_max
            n = math.ceil((f_max - f_min) / res) + 1

        assert n is not None

        if isinstance(fov, tuple):
            f_min, f_max = fov
            fov_size = f_max - f_min
        else:
            f_max = fov / 2.0
            f_min = -f_max
            fov_size = fov

        full_rotation = 360.0 if use_degrees else math.pi
        assert fov_size <= full_rotation + gs.EPS, "FOV should not be larger than a full rotation."

        # avoid duplicate angle at 0/360 degrees
        if fov_size >= full_rotation - gs.EPS:
            f_max -= fov_size / (n - 1) * 0.5

        angles = torch.linspace(f_min, f_max, n, dtype=gs.tc_float, device=gs.device)

    if use_degrees:
        angles = torch.deg2rad(angles)

    return angles


def _compute_focal_lengths(
    width: int, height: int, fov_horizontal: float | None, fov_vertical: float | None
) -> tuple[float, float]:
    if fov_horizontal is not None and fov_vertical is None:
        fh_rad = math.radians(fov_horizontal)
        fv_rad = 2.0 * math.atan((height / width) * math.tan(fh_rad / 2.0))
    elif fov_vertical is not None and fov_horizontal is None:
        fv_rad = math.radians(fov_vertical)
        fh_rad = 2.0 * math.atan((width / height) * math.tan(fv_rad / 2.0))
    else:
        fh_rad = math.radians(fov_horizontal)
        fv_rad = math.radians(fov_vertical)

    fx = width / (2.0 * math.tan(fh_rad / 2.0))
    fy = height / (2.0 * math.tan(fv_rad / 2.0))

    return fx, fy


# ============================== Generic Patterns ==============================
@dataclass
class GridPattern(RaycastPattern):
    """
    Configuration for grid-based ray casting.

    Defines a 2D grid of rays in the sensor coordinate system.

    Parameters
    ----------
    resolution : float
        Grid spacing in meters.
    size : tuple[float, float]
        Grid dimensions (length, width) in meters.
    direction : tuple[float, float, float]
        Ray direction vector.
    ordering : str
        Point ordering, either "xy" or "yx".
    """

    resolution: float = 0.1
    size: tuple[float, float] = (2.0, 2.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    ordering: str = "xy"

    def validate(self):
        if self.ordering not in ["xy", "yx"]:
            raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{self.ordering}'.")
        if self.resolution <= 0:
            raise ValueError(f"Resolution must be greater than 0. Received: '{self.resolution}'.")

    def get_return_shape(self) -> tuple[int, ...]:
        num_x = math.ceil(self.size[0] / self.resolution) + 1
        num_y = math.ceil(self.size[1] / self.resolution) + 1
        return (num_x, num_y)


@register_pattern(GridPattern, "grid")
class GridPatternGenerator(RaycastPatternGenerator):
    """Generator for 2D grid ray patterns."""

    def __init__(self, options: GridPattern):
        super().__init__(options)
        self.x_coords = torch.arange(
            -options.size[0] / 2, options.size[0] / 2 + gs.EPS, options.resolution, dtype=gs.tc_float, device=gs.device
        )
        self.y_coords = torch.arange(
            -options.size[1] / 2, options.size[1] / 2 + gs.EPS, options.resolution, dtype=gs.tc_float, device=gs.device
        )
        self.direction = torch.tensor(options.direction, dtype=gs.tc_float, device=gs.device)

    def get_ray_directions(self) -> torch.Tensor:
        return self.direction.expand((*self._return_shape, 3))

    def get_ray_starts(self) -> torch.Tensor:
        if self._options.ordering == "xy":
            grid_x, grid_y = torch.meshgrid(self.x_coords, self.y_coords, indexing="xy")
        else:
            grid_x, grid_y = torch.meshgrid(self.x_coords, self.y_coords, indexing="ij")

        starts = torch.empty((*self._return_shape, 3), dtype=gs.tc_float, device=gs.device)
        starts[..., 0] = grid_x
        starts[..., 1] = grid_y
        starts[..., 2] = 0.0

        return starts


@dataclass
class SphericalPattern(RaycastPattern):
    """
    Configuration for spherical ray pattern.

    Either specify:
    - (n_vertical, n_horizontal, fov_vertical, fov_horizontal) for uniform spacing by count.
    - (vertical_res, horizontal_res, fov_vertical, fov_horizontal) for uniform spacing by resolution.
    - (vertical_angles, horizontal_angles) for custom angles.


    Parameters
    ----------
    fov_vertical: float | tuple[float, float]
        Vertical field of view.
    fov_horizontal: float | tuple[float, float]
        Horizontal field of view.
    n_vertical : int
        Number of vertical/elevation scan lines.
    n_horizontal : int
        Number of horizontal/azimuth points per scan line.
    res_vertical : float, optional
        Vertical angular resolution in degrees. Overrides n_vertical if provided.
    res_horizontal : float, optional
        Horizontal angular resolution in degrees. Overrides n_horizontal if provided.
    angles_vertical : Sequence[float], optional
        Array of elevation angles. Overrides n_vertical/res_vertical/fov_vertical if provided.
    angles_horizontal: Sequence[float], optional
        Array of azimuth angles. Overrides n_horizontal/res_horizontal/fov_horizontal if provided.
    use_degrees : bool, optional
        Whether the provided angles are in degrees or radians. Defaults to True.
    """

    fov_vertical: float | tuple[float, float] = 30.0
    fov_horizontal: float | tuple[float, float] = 360.0
    n_vertical: int = 64
    n_horizontal: int = 128
    res_vertical: float | None = None
    res_horizontal: float | None = None
    vertical_angles: Sequence[float] | None = None
    horizontal_angles: Sequence[float] | None = None
    use_degrees: bool = True

    def validate(self):
        full_rotation = 360.0 if self.use_degrees else math.pi * 2.0
        for fov in (self.fov_vertical, self.fov_horizontal):
            if (isinstance(fov, float) and (fov < 0 or fov > full_rotation + gs.EPS)) or (
                isinstance(fov, tuple) and (fov[1] - fov[0] > full_rotation + gs.EPS)
            ):
                gs.raise_exception(f"[{type(self).__class__}] FOV should not be <0 or >{full_rotation}. Got: {fov}.")

    def get_return_shape(self) -> tuple[int, ...]:
        if self.vertical_angles is not None and self.horizontal_angles is not None:
            return (len(self.vertical_angles), len(self.horizontal_angles))
        else:
            return (self.n_vertical, self.n_horizontal)


@register_pattern(SphericalPattern, "spherical")
class SphericalPatternGenerator(RaycastPatternGenerator):
    """Generator for uniform or custom spherical ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        v_angles = _generate_uniform_angles(
            n=self._options.n_vertical,
            fov=self._options.fov_vertical,
            res=self._options.res_vertical,
            angles=self._options.vertical_angles,
            use_degrees=self._options.use_degrees,
        )
        h_angles = _generate_uniform_angles(
            n=self._options.n_horizontal,
            fov=self._options.fov_horizontal,
            res=self._options.res_horizontal,
            angles=self._options.horizontal_angles,
            use_degrees=self._options.use_degrees,
        )

        h_grid, v_grid = torch.meshgrid(h_angles, v_angles, indexing="ij")
        return spherical_to_cartesian(h_grid, v_grid)


# ============================== Camera Patterns ==============================


@dataclass
class DepthCameraPattern(RaycastPattern):
    """Configuration for pinhole depth camera ray casting.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fx : float | None
        Focal length in x direction (pixels). Computed from FOV if None.
    fy : float | None
        Focal length in y direction (pixels). Computed from FOV if None.
    cx : float | None
        Principal point x coordinate (pixels). Defaults to image center if None.
    cy : float | None
        Principal point y coordinate (pixels). Defaults to image center if None.
    fov_horizontal : float
        Horizontal field of view in degrees. Used to compute fx if fx is None.
    fov_vertical : float | None
        Vertical field of view in degrees. Used to compute fy if fy is None.
    """

    width: int = 128
    height: int = 96
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    fov_horizontal: float = 90.0
    fov_vertical: float | None = None

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.height, self.width)


@register_pattern(DepthCameraPattern, "depth_camera")
class DepthCameraPatternGenerator(RaycastPatternGenerator):
    """Generator for pinhole depth camera ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        W, H = int(self._options.width), int(self._options.height)

        if W <= 0 or H <= 0:
            raise ValueError("Image dimensions must be positive")

        fx, fy, cx, cy = self._options.fx, self._options.fy, self._options.cx, self._options.cy
        if fx is None or fy is None:
            fx, fy = _compute_focal_lengths(W, H, self._options.fov_horizontal, self._options.fov_vertical)
        if cx is None:
            cx = W * 0.5
        if cy is None:
            cy = H * 0.5

        u = torch.arange(0, W, dtype=gs.tc_float, device=gs.device) + 0.5
        v = torch.arange(0, H, dtype=gs.tc_float, device=gs.device) + 0.5
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        # standard camera frame coordinates
        x_c = (uu - cx) / fx
        y_c = (vv - cy) / fy
        z_c = torch.ones_like(x_c, dtype=gs.tc_float, device=gs.device)

        # transform to robotics camera frame
        dirs = torch.stack([z_c, -x_c, -y_c], dim=-1)
        dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)

        return dirs
