from dataclasses import dataclass
import sys

from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom

from .vec3 import Vec3


EPSILON = 1e-6
EPSILON2 = EPSILON * EPSILON

_MAX_RAY_DISTANCE = sys.float_info.max


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
    distance: float
    position: Vec3
    normal: Vec3
    geom: RigidGeom | None = None

    @property
    def is_hit(self) -> bool:
        assert 0.0 <= self.distance
        return self.distance < _MAX_RAY_DISTANCE

    @classmethod
    def no_hit(cls) -> 'RayHit':
        return RayHit(_MAX_RAY_DISTANCE, Vec3.zero(), Vec3.zero(), None)


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
            return RayHit.no_hit()
        else:
            dist_along_ray = dist / -dot
            return RayHit(dist_along_ray, ray.origin + ray.direction * dist_along_ray, self.normal, None)
