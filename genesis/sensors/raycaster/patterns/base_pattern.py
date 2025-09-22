from dataclasses import dataclass

import torch

import genesis as gs


@dataclass
class RaycastPattern:
    """
    Base class for raycast pattern options.
    """

    def validate(self):
        """Validate the pattern options."""
        pass

    def get_return_shape(self) -> tuple[int, ...]:
        """Get the shape of the ray vectors, e.g. (n_scan_lines, n_points_per_line) or (n_rays,)"""
        raise NotImplementedError(f"{type(self).__name__} must implement `get_return_shape()`.")


class RaycastPatternGenerator:
    """Base class for raycast pattern generators."""

    def __init__(self, config: RaycastPattern):
        self.config: RaycastPattern = config
        self._return_shape: tuple[int, ...] = self.config.get_return_shape()

    def get_ray_directions(self) -> torch.Tensor:
        """
        Get the local direction vectors of the rays.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `get_ray_directions()`.")

    def get_ray_starts(self) -> torch.Tensor:
        """
        Get the local start positions of the rays.

        As a default, return zeros which means all rays start at the local origin.
        """
        return torch.zeros((*self._return_shape, 3), dtype=gs.tc_float, device=gs.device)


class DynamicPatternGenerator(RaycastPatternGenerator):
    """Base class for dynamic raycast pattern generators."""

    def __init__(self, options: RaycastPattern):
        super().__init__(options)
        self._cached_ray_directions: torch.Tensor = torch.empty(self._return_shape, dtype=gs.tc_float, device=gs.device)

    def update_ray_directions(self, cur_t: float):
        """
        Compute the local direction vectors of the rays and save them to `self._cached_ray_dirs`.

        Parameters
        ----------
        cur_t : float
            Current simulation time in seconds.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `update_ray_directions()`.")

    def get_ray_directions(self) -> torch.Tensor:
        """
        Get the local direction vectors of the rays.
        """
        if self._cached_ray_directions.isnan().any():
            self.update_ray_directions(self._cached_ray_directions)
        return self._cached_ray_directions


RAYCAST_PATTERN_NAME_TO_OPTIONS_MAP: dict[str, type[RaycastPattern]] = {}
RAYCAST_PATTERN_OPTIONS_TO_GENERATOR_MAP: dict[type[RaycastPattern], type[RaycastPatternGenerator]] = {}


def register_pattern(pattern_type: type[RaycastPattern], name: str):
    def _impl(generator_type: type[RaycastPatternGenerator]):
        RAYCAST_PATTERN_NAME_TO_OPTIONS_MAP[name] = pattern_type
        RAYCAST_PATTERN_OPTIONS_TO_GENERATOR_MAP[pattern_type] = generator_type
        return generator_type

    return _impl


def create_pattern_generator(pattern: RaycastPattern) -> RaycastPatternGenerator:
    generator_cls = RAYCAST_PATTERN_OPTIONS_TO_GENERATOR_MAP[type(pattern)]
    return generator_cls(pattern)
