import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import genesis as gs
from genesis.utils.geom import spherical_to_cartesian

from .base_pattern import RaycastPattern, RaycastPatternGenerator, register_pattern


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
            self.num_horizontal_angles -= 1  # remove duplicate angle at 0/360 degrees
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
