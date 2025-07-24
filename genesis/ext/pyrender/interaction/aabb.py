from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .ray import Ray, RayHit, EPSILON
from .vec3 import Pose, Vec3

class AABB:
    v: NDArray[np.float32]

    def __init__(self, v: NDArray[np.float32]):
        assert v.shape == (2, 3,), f"Aabb must be initialized with a (2,3)-element array, got {v.shape}"
        assert v.dtype == np.float32, f"Aabb must be initialized with a float32 array, got {v.dtype}"
        self.v = v

    @property
    def min(self) -> Vec3:
        return Vec3(self.v[0])

    @property
    def max(self) -> Vec3:
        return Vec3(self.v[1])

    @property
    def extents(self) -> Vec3:
        return self.max - self.min

    def expand(self, padding: float) -> None:
        self.v[0] -= padding
        self.v[1] += padding

    def raycast(self, ray: Ray) -> RayHit:
        """
        Standard AABB slab implementation. Early-exits and returns no-hit for rays withing the XY, XZ, or YZ planes.
        Ignores hits for rays originating inside the AABB.
        """
        if (np.abs(ray.direction.v) < EPSILON).any():
            # unhandled ray case: early-exit
            return RayHit.no_hit()

        tmin = (self.v[0] - ray.origin.v) / ray.direction.v
        tmax = (self.v[1] - ray.origin.v) / ray.direction.v
        mmin = np.minimum(tmin, tmax)
        mmax = np.maximum(tmin, tmax)
        min_idx = np.argmax(mmin)
        max_idx = np.argmin(mmax)
        tnear = mmin[min_idx]
        tfar = mmax[max_idx]

        # Drop hits coming from inside
        if tfar < tnear or tnear < 0:  # tfar < 0
            return RayHit.no_hit()

        # Calculate enter point and normal
        enter = tnear  # if 0 <= tnear else tfar
        normal = Vec3.zero()
        normal.v[min_idx] = -np.sign(ray.direction.v[min_idx])

        hit_pos = ray.origin + ray.direction * enter
        return RayHit(enter, hit_pos, normal)

    def __repr__(self) -> str:
        return f"AABB: Min({self.min.x}, {self.min.y}, {self.min.z}) Max({self.max.x}, {self.max.y}, {self.max.z})"

    @classmethod
    def from_min_max(cls, min: Vec3, max: Vec3) -> 'AABB':
        bounds = np.stack((min.v, max.v), axis=0)
        return cls(bounds)

    @classmethod
    def from_center_and_half_extents(cls, center: Vec3, half_extents: Vec3) -> 'AABB':
        min = center - half_extents
        max = center + half_extents
        bounds = np.stack((min.v, max.v), axis=0)
        return cls(bounds)


@dataclass
class OBB():
    pose: Pose
    half_extents: Vec3

    def raycast(self, ray: Ray) -> RayHit:
        origin2 = self.pose.inverse_transform_point(ray.origin)
        direction2 = self.pose.inverse_transform_direction(ray.direction)
        ray2 = Ray(origin2, direction2)
        aabb = AABB.from_center_and_half_extents(Vec3.zero(), self.half_extents)
        ray_hit = aabb.raycast(ray2)
        if ray_hit.is_hit:
            ray_hit.position = self.pose.transform_point(ray_hit.position)
            ray_hit.normal = self.pose.transform_direction(ray_hit.normal)
        return ray_hit

    def __repr__(self) -> str:
        return f"OBB(pose={self.pose}, half_extents={self.half_extents})"
