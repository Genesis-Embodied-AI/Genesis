from threading import Lock as threading_Lock
from typing import TYPE_CHECKING

import torch
from typing_extensions import override  # Made it into standard lib from Python 3.12

import genesis as gs
from genesis.options.viewer_plugins import MouseSpringPlugin as MouseSpringPluginOptions

from ..utils import AABB, OBB, Color, Plane, Pose, Quat, Ray, RayHit, Vec3, ViewerRaycaster
from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, ViewerPlugin, register_viewer_plugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidGeom, RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


MOUSE_SPRING_POSITION_CORRECTION_FACTOR = 1.0
MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR = 1.0

class MouseSpring:
    def __init__(self) -> None:
        self.held_link: "RigidLink | None" = None
        self.held_point_in_local: Vec3 | None = None
        self.prev_control_point: Vec3 | None = None

    def attach(self, picked_link: "RigidLink", control_point: Vec3) -> None:
        # for now, we just pick the first geometry
        self.held_link = picked_link
        pose: Pose = Pose.from_link(self.held_link)
        self.held_point_in_local = pose.inverse_transform_point(control_point)
        self.prev_control_point = control_point

    def detach(self) -> None:
        self.held_link = None

    def apply_force(self, control_point: Vec3, delta_time: float) -> None:
        # note when threaded: apply_force is called before attach!
        # note2: that was before we added a lock to ViewerInteraction; this migth be fixed now
        if not self.held_link:
            return

        self.prev_control_point = control_point

        # do simple force on COM only:
        link: "RigidLink" = self.held_link
        lin_vel: Vec3 = Vec3.from_tensor(link.get_vel())
        ang_vel: Vec3 = Vec3.from_tensor(link.get_ang())
        link_pose: Pose = Pose.from_link(link)
        held_point_in_world: Vec3 = link_pose.transform_point(self.held_point_in_local)

        # note: we should assert earlier that link inertial_pos/quat are not None
        # todo: verify inertial_pos/quat are stored in local frame
        link_T_principal: Pose = Pose(Vec3.from_arraylike(link.inertial_pos), Quat.from_arraylike(link.inertial_quat))
        world_T_principal: Pose = link_pose * link_T_principal

        arm_in_principal: Vec3 = link_T_principal.inverse_transform_point(self.held_point_in_local)   # for non-spherical inertia
        arm_in_world: Vec3 = world_T_principal.rot * arm_in_principal  # for spherical inertia

        pos_err_v: Vec3 = control_point - held_point_in_world
        inv_mass: float = float(1.0 / link.get_mass() if link.get_mass() > 0.0 else 0.0)
        inv_spherical_inertia: float = float(1.0 / link.inertial_i[0, 0] if link.inertial_i[0, 0] > 0.0 else 0.0)

        inv_dt: float = 1.0 / delta_time
        tau: float = MOUSE_SPRING_POSITION_CORRECTION_FACTOR
        damp: float = MOUSE_SPRING_VELOCITY_CORRECTION_FACTOR

        total_impulse: Vec3 = Vec3.zero()
        total_torque_impulse: Vec3 = Vec3.zero()

        for i in range(3*4):
            body_point_vel: Vec3 = lin_vel + ang_vel.cross(arm_in_world)
            vel_err_v: Vec3 = Vec3.zero() - body_point_vel

            dir: Vec3 = Vec3.zero()
            dir.v[i % 3] = 1.0
            pos_err: float = dir.dot(pos_err_v)
            vel_err: float = dir.dot(vel_err_v)
            error: float = tau * pos_err * inv_dt + damp * vel_err

            arm_x_dir: Vec3 = arm_in_world.cross(dir)
            virtual_mass: float = 1.0 / (inv_mass + arm_x_dir.sqr_magnitude() * inv_spherical_inertia + 1e-24)
            impulse: float = error * virtual_mass

            lin_vel += impulse * inv_mass * dir
            ang_vel += impulse * inv_spherical_inertia * arm_x_dir
            total_impulse.v[i % 3] += impulse
            total_torque_impulse += impulse * arm_x_dir

        # Apply the new force
        total_force = total_impulse * inv_dt
        total_torque = total_torque_impulse * inv_dt
        force_tensor: torch.Tensor = total_force.as_tensor()[None]
        torque_tensor: torch.Tensor = total_torque.as_tensor()[None]
        link.solver.apply_links_external_force(force_tensor, (link.idx,), ref='link_com', local=False)
        link.solver.apply_links_external_torque(torque_tensor, (link.idx,), ref='link_com', local=False)

    @property
    def is_attached(self) -> bool:
        return self.held_link is not None


@register_viewer_plugin(MouseSpringPluginOptions)
class MouseSpringPlugin(ViewerPlugin):
    """
    Basic interactive viewer plugin that enables using mouse to apply spring force on rigid entities.
    """

    def __init__(
        self,
        viewer,
        options: MouseSpringPluginOptions,
        camera: "Node",
        scene: "Scene",
        viewport_size: tuple[int, int],
    ) -> None:
        super().__init__(viewer, options, camera, scene, viewport_size)
        self.prev_mouse_pos: tuple[int, int] = (viewport_size[0] // 2, viewport_size[1] // 2)

        self.picked_link: RigidLink | None = None
        self.picked_point_in_local: Vec3 | None = None
        self.mouse_drag_plane: Plane | None = None
        self.prev_mouse_3d_pos: Vec3 | None = None

        self.mouse_spring: MouseSpring = MouseSpring()
        self.lock = threading_Lock()

        self.raycaster: ViewerRaycaster = ViewerRaycaster(self.scene)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_motion(x, y, dx, dy)
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        self.prev_mouse_pos = (x, y)
        if self.picked_link:
            # actual processing done in update_on_sim_step()

            return EVENT_HANDLED

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_press(x, y, button, modifiers)
        if button == 1: # left mouse button

            ray_hit = self.raycaster.cast_ray(self._screen_position_to_ray(x, y).origin.v, self._screen_position_to_ray(x, y).direction.v)
            with self.lock:
                if ray_hit.geom:
                    self.picked_link = ray_hit.geom.link
                    assert self.picked_link is not None

                    temp_fwd = self._get_camera_forward()
                    temp_back = -temp_fwd

                    self.mouse_drag_plane = Plane(temp_back, ray_hit.position)
                    self.prev_mouse_3d_pos = ray_hit.position

                    pose: Pose = Pose.from_link(self.picked_link)
                    self.picked_point_in_local = pose.inverse_transform_point(ray_hit.position)

                    self.mouse_spring.attach(self.picked_link, ray_hit.position)

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_release(x, y, button, modifiers)
        if button == 1: # left mouse button
            with self.lock:
                self.picked_link = None
                self.picked_point_in_local = None
                self.mouse_drag_plane = None
                self.prev_mouse_3d_pos = None

                self.mouse_spring.detach()

    @override
    def update_on_sim_step(self) -> None:
        self.raycaster.update_bvh()

        with self.lock:
            if self.picked_link:
                mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_pos)
                ray_hit: RayHit = self.mouse_drag_plane.raycast(mouse_ray)
                assert ray_hit.is_hit
                if ray_hit.is_hit:
                    new_mouse_3d_pos: Vec3 = ray_hit.position
                    delta_3d_pos: Vec3 = new_mouse_3d_pos - self.prev_mouse_3d_pos
                    self.prev_mouse_3d_pos = new_mouse_3d_pos

                    use_force: bool = True
                    if use_force:
                        # apply force
                        self.mouse_spring.apply_force(new_mouse_3d_pos, self.scene.sim.dt)
                    else:
                        # apply displacement
                        pos = Vec3.from_tensor(self.picked_link.entity.get_pos())
                        pos += delta_3d_pos
                        self.picked_link.entity.set_pos(pos.as_tensor())

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_pos)
            closest_hit = self.raycaster.cast_ray(mouse_ray.origin.v, mouse_ray.direction.v)

            with self.lock:
                if self.picked_link:
                    assert self.mouse_drag_plane is not None
                    assert self.picked_point_in_local is not None

                    # draw held point
                    pose: Pose = Pose.from_link(self.picked_link)
                    held_point: Vec3 = pose.transform_point(self.picked_point_in_local)
                    self.scene.draw_debug_sphere(held_point.v, 0.02, Color.red().tuple())

                    plane_hit: RayHit = self.mouse_drag_plane.raycast(mouse_ray)
                    if plane_hit.is_hit:
                        self.scene.draw_debug_sphere(plane_hit.position.v, 0.02, Color.red().tuple())
                        self.scene.draw_debug_line(held_point.v, plane_hit.position.v, color=Color.red().tuple())
                else:
                    if closest_hit.is_hit:
                        self.scene.draw_debug_sphere(closest_hit.position.v, 0.01, (0, 1, 0, 1))
                        self._draw_arrow(closest_hit.position, 0.25 * closest_hit.normal, (0, 1, 0, 1))
                    if closest_hit.geom:
                        self._draw_entity_unrotated_obb(closest_hit.geom)



    def _get_box_obb(self, box_entity: "RigidEntity") -> OBB:
        box: gs.morphs.Box = box_entity.morph
        pose = Pose.from_link(box_entity.links[0])
        half_extents = 0.5 * Vec3.from_xyz(*box.size)
        return OBB(pose, half_extents)

    def _get_geom_placeholder_obb(self, geom: 'RigidGeom') -> OBB:
        pose = Pose.from_geom(geom)
        half_extents = Vec3.full(0.5 * 0.125)
        return OBB(pose, half_extents)

    def _draw_arrow(
        self, pos: Vec3, dir: Vec3, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.scene.draw_debug_arrow(pos.v, dir.v, color=color)

    def _draw_entity_unrotated_obb(self, geom: 'RigidGeom') -> None:
        obb: OBB | None = None
        if isinstance(geom.entity.morph, gs.morphs.Box):
            obb = self._get_box_obb(geom.entity)
        else:
            obb = self._get_geom_placeholder_obb(geom)

        if obb:
            aabb: AABB = AABB.from_center_and_half_extents(obb.pose.pos, obb.half_extents)
            aabb.expand(padding=0.01)
            self.scene.draw_debug_box(aabb.v, color=Color.red().with_alpha(0.5).tuple(), wireframe=False)
