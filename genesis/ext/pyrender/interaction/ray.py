from dataclasses import dataclass

from .vec3 import Vec3


EPSILON = 1e-6


class Ray:
    origin: Vec3
    direction: Vec3

    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalized()

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"


@dataclass
class RayHit:
    is_hit: bool
    distance: float
    normal: Vec3
    position: Vec3
    object_idx: int


class Plane:
    normal: Vec3
    distance: float  # distance from plane to origin along normal

    def __init__(self, normal: Vec3, point: Vec3):
        self.normal = normal
        self.distance = -normal.dot(point)

    def raycast(self, ray: Ray) -> RayHit:
        dot = ray.direction.dot(self.normal)
        dist = ray.origin.dot(self.normal) + self.distance

        if -EPSILON < dot or dist < EPSILON:
            return RayHit(is_hit=False, distance=0, normal=Vec3.zero(), position=Vec3.zero(), object_idx=-1)

        dist_along_ray = dist / -dot

        return RayHit(
            is_hit=True,
            distance=dist_along_ray,
            normal=self.normal,
            position=ray.origin + ray.direction * dist_along_ray,
            object_idx=0
        )

