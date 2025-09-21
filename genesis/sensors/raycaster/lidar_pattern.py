import math
from collections.abc import Sequence
from dataclasses import dataclass

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
class LidarAnglesPattern(RaycastPattern):
    """
    Pattern for multi-channel LiDAR ray casting.

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


@register_pattern(LidarAnglesPattern, "lidar_angles")
class LidarAnglesPatternGenerator(RaycastPatternGenerator):
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


@dataclass
class LidarPattern(RaycastPattern):
    """
    Configuration for multi-channel LiDAR ray casting.

    Parameters
    ----------
    num_channels : int
        Number of vertical scanning channels.
    vertical_fov_range : tuple[float, float]
        Vertical field of view limits in degrees (min, max).
    horizontal_fov_range : tuple[float, float]
        Horizontal field of view limits in degrees (min, max).
    horizontal_res : float
        Horizontal angular resolution in degrees.
    """

    vertical_fov_range: tuple[float, float] = (-15.0, 15.0)
    num_channels: int = 32
    horizontal_fov_range: tuple[float, float] = (-180.0, 180.0)
    horizontal_res: float = 1.0

    def get_return_shape(self) -> tuple[int, ...]:
        h_range = self.horizontal_fov_range[1] - self.horizontal_fov_range[0]
        num_horizontal_angles = math.ceil(h_range / self.horizontal_res)
        return (self.num_channels, num_horizontal_angles)


@register_pattern(LidarPattern, "lidar")
class LidarPatternGenerator(RaycastPatternGenerator):
    """Generator for multi-channel LiDAR ray patterns."""

    def __init__(self, config: LidarPattern):
        super().__init__(config)
        self.h_range = config.horizontal_fov_range[1] - config.horizontal_fov_range[0]
        self.num_horizontal_angles = math.ceil(self.h_range / config.horizontal_res)
        if abs(abs(self.h_range) - 360.0) < 1e-6:
            self.num_horizontal_angles -= 1  # remove duplicate angle
        self.vertical_fov_range = torch.tensor(config.vertical_fov_range, device=gs.device, dtype=gs.tc_float)
        self.horizontal_fov_range = torch.tensor(config.horizontal_fov_range, device=gs.device, dtype=gs.tc_float)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spherical ray pattern for multi-channel LiDAR.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (channels, angles_per_channel, 3).
        """
        vertical_angles = torch.linspace(
            torch.deg2rad(self.vertical_fov_range[0]),
            torch.deg2rad(self.vertical_fov_range[1]),
            self.config.num_channels,
        )
        horizontal_angles = torch.linspace(
            torch.deg2rad(self.horizontal_fov_range[0]),
            torch.deg2rad(self.horizontal_fov_range[1]),
            self.num_horizontal_angles,
        )

        vertical_meshgrid, horizontal_meshgrid = torch.meshgrid(vertical_angles, horizontal_angles, indexing="ij")

        x, y, z = spherical_to_cartesian(horizontal_meshgrid, vertical_meshgrid)
        return torch.stack([x, y, z], dim=-1)


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
class SpinningLidarPattern(RaycastPattern):
    """
    Configuration for traditional spinning LiDAR sensors.

    Parameters
    ----------
    f_rot : float
        Rotation frequency in Hz. Defaults to 10 Hz.
    sample_rate : float
        Sample rate in samples per rotation. Defaults to 1.0e5.
    n_channels : int, optional
        Number of vertical channels. Defaults to 32.
        Used along with `vertical_fov` to generate vertical angles if not provided.
    vertical_fov : tuple[float, float], optional
        Vertical field of view limits in degrees (min, max). Defaults to (-20.0, 20.0).
        Used along with `n_channels` to generate vertical angles if not provided.
    vertical_angles: Sequence[float], optional
        Vertical angles in degrees. Replaces `n_channels` and `vertical_fov` if provided.
    """

    f_rot: float = 10.0
    sample_rate: float = 1.0e5
    n_channels: int = 32
    vertical_fov: tuple[float, float] = (-20.0, 20.0)
    vertical_angles: Sequence[float] | None = None

    def get_return_shape(self) -> tuple[int, ...]:
        n_channels = len(self.vertical_angles) if self.vertical_angles is not None else self.n_channels
        n_time_steps = int(self.sample_rate / (self.f_rot * n_channels))
        return (n_time_steps * n_channels,)


@register_pattern(SpinningLidarPattern, "spinning")
class SpinningLidarPatternGenerator(RaycastPatternGenerator):
    """Generator for traditional spinning LiDAR patterns."""

    def __init__(self, options: SpinningLidarPattern):
        super().__init__(options)

    def get_ray_directions(self) -> torch.Tensor:
        """Generate spinning LiDAR ray pattern.

        Returns
        -------
        torch.Tensor
            Ray directions with shape (1, n_rays, 3).
        """

        if self.config.vertical_angles is not None:
            phi = torch.deg2rad(torch.tensor(self.config.vertical_angles, device=gs.device, dtype=gs.tc_float))
            n_channels = len(self.config.vertical_angles)
        else:
            n_channels = self.config.n_channels
            phi_min, phi_max = torch.deg2rad(
                torch.tensor(self.config.vertical_fov, dtype=gs.tc_float, device=gs.device)
            )
            phi = torch.linspace(phi_min, phi_max, n_channels, dtype=gs.tc_float, device=gs.device)

        t = torch.arange(
            0.0, 1.0 / self.config.f_rot, n_channels / self.config.sample_rate, dtype=gs.tc_float, device=gs.device
        )[:, None]
        theta = (2.0 * torch.pi * self.config.f_rot * t) % (2.0 * torch.pi)

        theta_grid = theta + torch.zeros((1, n_channels), dtype=gs.tc_float, device=gs.device)
        phi_grid = torch.zeros_like(theta, dtype=gs.tc_float) + phi

        theta_flat = theta_grid.reshape(-1)
        phi_flat = phi_grid.reshape(-1)

        x, y, z = spherical_to_cartesian(theta_flat, phi_flat)
        dirs = torch.stack([x, y, z], dim=1)

        norms = torch.linalg.norm(dirs, dim=1, keepdim=True)
        dirs = dirs / torch.maximum(norms, torch.tensor(1e-8, device=gs.device))

        return dirs.reshape(1, -1, 3)
