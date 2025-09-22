import math
from dataclasses import dataclass
from typing import Sequence

import torch

import genesis as gs
from genesis.utils.geom import spherical_to_cartesian

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern


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
            -options.size[0] / 2, options.size[0] / 2 + 1e-9, options.resolution, dtype=gs.tc_float, device=gs.device
        )
        self.y_coords = torch.arange(
            -options.size[1] / 2, options.size[1] / 2 + 1e-9, options.resolution, dtype=gs.tc_float, device=gs.device
        )
        self.direction = torch.tensor(options.direction, dtype=gs.tc_float, device=gs.device)

    def get_ray_directions(self) -> torch.Tensor:
        return self.direction.expand((*self._return_shape, 3))

    def get_ray_starts(self) -> torch.Tensor:
        if self.config.ordering == "xy":
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
    Configuration for spherical uniform ray pattern.

    Parameters
    ----------
    n_scan_lines : int
        Number of vertical scan lines.
    n_points_per_line : int
        Number of horizontal points per scan line.
    fov_vertical : float
        Vertical field of view in degrees.
    fov_horizontal : float
        Horizontal field of view in degrees.
    """

    n_scan_lines: int = 32
    n_points_per_line: int = 64
    fov_vertical: float = 30.0
    fov_horizontal: float = 360.0

    def get_return_shape(self) -> tuple[int, ...]:
        return (self.n_scan_lines, self.n_points_per_line)


@register_pattern(SphericalPattern, "spherical")
class SphericalPatternGenerator(RaycastPatternGenerator):
    """Generator for uniform spherical ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        """Generate uniform spherical ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (n_scan_lines, n_points_per_line, 3).
        """
        vertical_angles = torch.linspace(
            -self.config.fov_vertical / 2,
            self.config.fov_vertical / 2,
            self.config.n_scan_lines,
            dtype=gs.tc_float,
            device=gs.device,
        )
        horizontal_angles = torch.linspace(
            -self.config.fov_horizontal / 2,
            self.config.fov_horizontal / 2,
            self.config.n_points_per_line,
            dtype=gs.tc_float,
            device=gs.device,
        )

        v_rad = torch.deg2rad(vertical_angles)
        h_rad = torch.deg2rad(horizontal_angles)
        h_angles, v_angles = torch.meshgrid(h_rad, v_rad, indexing="ij")

        x, y, z = spherical_to_cartesian(h_angles, v_angles)
        ray_vectors = torch.stack([x, y, z], dim=-1)

        return ray_vectors


@dataclass
class AngleGridPattern(RaycastPattern):
    """
    Pattern for a spherical grid of angles for ray casting.

    Parameters
    ----------
    vertical_angles : Sequence[float]
        Array of elevation angles in degrees.
    horizontal_angles : Sequence[float]
        Array of azimuth angles in degrees.
    """

    vertical_angles: Sequence[float]
    horizontal_angles: Sequence[float]

    def get_return_shape(self) -> tuple[int, ...]:
        return (len(self.vertical_angles), len(self.horizontal_angles))


@register_pattern(AngleGridPattern, "angle_grid")
class AngleGridPatternGenerator(RaycastPatternGenerator):
    """Generator for multi-channel LiDAR ray patterns."""

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spherical ray pattern for multi-channel LiDAR.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (vertical_angles, horizontal_angles, 3).
        """
        v_angles_rad = torch.deg2rad(torch.tensor(self.config.vertical_angles, device=gs.device, dtype=gs.tc_float))
        h_angles_rad = torch.deg2rad(torch.tensor(self.config.horizontal_angles, device=gs.device, dtype=gs.tc_float))
        v_angles, h_angles = torch.meshgrid(v_angles_rad, h_angles_rad, indexing="ij")
        x, y, z = spherical_to_cartesian(h_angles, v_angles)
        return torch.stack([x, y, z], dim=-1)
