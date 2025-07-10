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

    def raycast_oobb(self, pose: Pose, ray: Ray) -> RayHit:
        inv_pose = pose.get_inverse()
        origin2 = inv_pose.transform_point(ray.origin)
        direction2 = inv_pose.transform_direction(ray.direction)
        ray2 = Ray(origin2, direction2)
        ray_hit = self.raycast(ray2)
        if ray_hit.is_hit:
            ray_hit.position = pose.transform_point(ray_hit.position)
            ray_hit.normal = pose.transform_direction(ray_hit.normal)
        return ray_hit

    def __repr__(self) -> str:
        return f"Min({self.min.x}, {self.min.y}, {self.min.z}) Max({self.max.x}, {self.max.y}, {self.max.z})"

    @classmethod
    def from_min_max(cls, min: Vec3, max: Vec3) -> 'AABB':
        bounds = np.stack((min.v, max.v))
        return cls(bounds)

    @classmethod
    def from_center_and_size(cls, center: Vec3, size: Vec3) -> 'AABB':
        min = center - 0.5 * size
        max = center + 0.5 * size
        bounds = np.stack((min.v, max.v))
        return cls(bounds)
