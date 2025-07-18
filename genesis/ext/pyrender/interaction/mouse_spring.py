
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity
from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom

from .ray import Plane, Ray, RayHit
from .vec3 import Pose, Quat, Vec3, Color

from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

MOUSE_SPRING_POSITION_CORRECTION_FACTOR = 1.0
MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR = 1.0

def _ensure_torch_imported() -> None:
    global torch
    import torch

class MouseSpring:
    def __init__(self):
        self.held_geom: RigidGeom | None = None
        self.held_point_in_local: Vec3 | None = None
        self.prev_control_point: Vec3 | None = None

    def attach(self, picked_entity: RigidEntity, control_point: Vec3):
        # for now, we just pick the first geometry
        self.held_geom = picked_entity.geoms[0]
        pose: Pose = Pose.from_geom(self.held_geom)
        self.held_point_in_local = pose.inverse_transform_point(control_point)
        self.prev_control_point = control_point

    def detach(self):
        self.held_geom = None

    def apply_force(self, control_point: Vec3, delta_time: float):
        _ensure_torch_imported()
        
        # works ok:
        # delta: Vec3 = control_point - self.prev_control_point
        # pos = Vec3.from_tensor(self.held_geom.entity.get_pos())
        # pos = pos + delta
        # self.held_geom.entity.set_pos(pos.as_tensor())
        self.prev_control_point = control_point

        # do simple force on COM only:
        link: RigidLink = self.held_geom.link
        link_pos: Vec3 = Vec3.from_tensor(link.get_pos())
        lin_vel: Vec3 = Vec3.from_tensor(link.get_vel())
        ang_vel: Vec3 = Vec3.from_tensor(link.get_ang())

        pos_err_v: Vec3 = control_point - link_pos
        vel_err_v: Vec3 = Vec3.zero() - lin_vel
        inv_mass: float = float(1.0 / link.get_mass() if link.get_mass() > 0.0 else 0.0)

        inv_dt: float = 1.0 / delta_time
        # these are temporary values, till we fix an issue with apply_links_external_force.
        # after fixing it, use tau = damp = 1.0:
        tau: float = MOUSE_SPRING_POSITION_CORRECTION_FACTOR
        damp: float = MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR

        total_impulse: Vec3 = Vec3.zero()

        for i in range(3):
            dir: Vec3 = Vec3.zero()
            dir.v[i] = 1.0
            pos_err: float = dir.dot(pos_err_v)
            vel_err: float = dir.dot(vel_err_v)
            error: float = tau * pos_err * inv_dt + damp * vel_err
            virtual_mass: float = 1.0 / (inv_mass + 1e-24)
            impulse: float = error * virtual_mass

            lin_vel += impulse * dir * inv_mass
            total_impulse.v[i] = impulse

        # Apply the new force
        total_force = total_impulse * inv_dt
        force_tensor: torch.Tensor = total_force.as_tensor().unsqueeze(0)
        link.solver.apply_links_external_force(force_tensor, (link.idx,), ref='link_com', local=False)

    @property
    def is_attached(self) -> bool:
        return self.held_geom is not None
