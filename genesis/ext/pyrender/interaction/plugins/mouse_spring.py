from threading import Lock as threading_Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.options.viewer_plugins import MouseSpringPlugin as MouseSpringPluginOptions
from genesis.utils.misc import tensor_to_array

from ..raycaster import Ray, RayHit, ViewerRaycaster
from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, ViewerPlugin, register_viewer_plugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


def plane_raycast(normal: np.ndarray, distance: float, ray: Ray) -> RayHit:
    """Cast a ray against a plane defined by its normal and distance."""
    assert normal.shape == ray.direction.shape == ray.origin.shape == (3,)
    dot = np.dot(ray.direction, normal)
    dist = np.dot(ray.origin, normal) + distance

    if -gs.EPS < dot or dist < gs.EPS:
        return None
    else:
        dist_along_ray = dist / -dot
        hit_pos = ray.origin + ray.direction * dist_along_ray
        return RayHit(distance=dist_along_ray, position=hit_pos, normal=normal, geom=None)


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
    ) -> None:
        super().__init__(viewer, options, camera, scene)
        self.prev_mouse_pos: tuple[int, int] = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

        self.held_link: "RigidLink | None" = None
        self.held_point_in_local: tuple[float, float, float] | None = None
        self.mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self.prev_mouse_3d_pos: tuple[float, float, float] | None = None

        self.lock = threading_Lock()

        self.raycaster: ViewerRaycaster = ViewerRaycaster(self.scene)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        self.prev_mouse_pos = (x, y)
        if self.held_link:
            return EVENT_HANDLED

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == 1:  # left mouse button
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self.raycaster.cast_ray(ray.origin, ray.direction)
            with self.lock:
                if ray_hit.geom and ray_hit.geom.link is not None and not ray_hit.geom.link.is_fixed:
                    self.held_link = ray_hit.geom.link

                    cam_backward = self.camera.matrix[:3, 2]
                    self.mouse_drag_plane = (cam_backward, np.dot(cam_backward, ray_hit.position))
                    self.prev_mouse_3d_pos = ray_hit.position

                    link_pos = tensor_to_array(self.held_link.get_pos())
                    link_quat = tensor_to_array(self.held_link.get_quat())
                    hit_pos_arr = np.array(ray_hit.position, dtype=np.float32)
                    self.held_point_in_local = tuple(gu.inv_transform_by_trans_quat(hit_pos_arr, link_pos, link_quat))

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == 1:  # left mouse button
            with self.lock:
                self.held_link = None
                self.held_point_in_local = None
                self.mouse_drag_plane = None
                self.prev_mouse_3d_pos = None

    @override
    def update_on_sim_step(self) -> None:
        self.raycaster.update_bvh()

        with self.lock:
            if self.held_link:
                mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_pos)
                assert self.mouse_drag_plane is not None
                ray_hit: RayHit = plane_raycast(*self.mouse_drag_plane, mouse_ray)
                assert ray_hit is not None

                new_mouse_3d_pos = ray_hit.position
                new_pos_arr = np.array(new_mouse_3d_pos, dtype=np.float32)
                prev_pos_arr = np.array(self.prev_mouse_3d_pos, dtype=np.float32)
                delta_3d_pos = new_pos_arr - prev_pos_arr
                self.prev_mouse_3d_pos = new_mouse_3d_pos

                use_force: bool = True
                if use_force:
                    self._apply_spring_force(new_mouse_3d_pos, self.scene.sim.dt)
                else:
                    # apply displacement
                    pos = tensor_to_array(self.held_link.entity.get_pos())
                    pos = pos + delta_3d_pos
                    self.held_link.entity.set_pos(torch.from_numpy(pos))

    @override
    def on_draw(self) -> None:
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_pos)
            closest_hit = self.raycaster.cast_ray(mouse_ray.origin, mouse_ray.direction)

            with self.lock:
                if self.held_link:
                    assert self.mouse_drag_plane is not None
                    assert self.held_point_in_local is not None

                    # draw held point
                    link_pos = tensor_to_array(self.held_link.get_pos())
                    link_quat = tensor_to_array(self.held_link.get_quat())
                    held_point_local_arr = np.array(self.held_point_in_local, dtype=np.float32)
                    held_point = gu.transform_by_trans_quat(held_point_local_arr, link_pos, link_quat)
                    self.scene.draw_debug_sphere(held_point, 0.02, self.options.spring_color)

                    plane_hit = plane_raycast(*self.mouse_drag_plane, mouse_ray)
                    if plane_hit is not None:
                        self.scene.draw_debug_sphere(
                            np.array(plane_hit.position, dtype=np.float32),
                            0.02,
                            self.options.spring_color,
                        )
                        self.scene.draw_debug_line(
                            np.array(held_point, dtype=np.float32),
                            np.array(plane_hit.position, dtype=np.float32),
                            color=self.options.spring_color,
                        )
                else:
                    if closest_hit is not None:
                        self.scene.draw_debug_arrow(
                            np.array(closest_hit.position, dtype=np.float32),
                            np.array(closest_hit.normal, dtype=np.float32) * 0.25,
                            color=self.options.normal_color,
                        )

    def _apply_spring_force(self, control_point: tuple[float, float, float], delta_time: float) -> None:
        if not self.held_link:
            return

        link: "RigidLink" = self.held_link
        lin_vel = tensor_to_array(link.get_vel())
        ang_vel = tensor_to_array(link.get_ang())

        link_pos = tensor_to_array(link.get_pos())
        link_quat = tensor_to_array(link.get_quat())
        held_point_local_arr = np.array(self.held_point_in_local, dtype=np.float32)
        held_point_in_world = gu.transform_by_trans_quat(held_point_local_arr, link_pos, link_quat)

        inertial_pos = tensor_to_array(link.inertial_pos)
        inertial_quat = tensor_to_array(link.inertial_quat)

        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        arm_in_principal = gu.inv_transform_by_trans_quat(held_point_local_arr, inertial_pos, inertial_quat)
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        control_point_arr = np.array(control_point, dtype=np.float32)
        pos_err_v = control_point_arr - held_point_in_world
        inv_mass: float = float(1.0 / link.get_mass() if link.get_mass() > 0.0 else 0.0)
        inv_spherical_inertia: float = float(1.0 / link.inertial_i[0, 0] if link.inertial_i[0, 0] > 0.0 else 0.0)

        inv_dt: float = 1.0 / delta_time
        tau: float = self.options.spring_const
        damp: float = self.options.spring_damping

        total_impulse = np.zeros(3, dtype=np.float32)
        total_torque_impulse = np.zeros(3, dtype=np.float32)

        for i in range(3 * 4):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            dir = np.zeros(3, dtype=np.float32)
            dir[i % 3] = 1.0

            pos_err = np.dot(dir, pos_err_v)
            vel_err = np.dot(dir, vel_err_v)
            error = tau * pos_err * inv_dt + damp * vel_err

            arm_x_dir = np.cross(arm_in_world, dir)
            virtual_mass = 1.0 / (inv_mass + np.dot(arm_x_dir, arm_x_dir) * inv_spherical_inertia + 1e-24)
            impulse = error * virtual_mass

            lin_vel = lin_vel + dir * impulse * inv_mass
            ang_vel = ang_vel + arm_x_dir * impulse * inv_spherical_inertia

            total_impulse[i % 3] += impulse
            total_torque_impulse += arm_x_dir * impulse

        # Apply the new force
        total_force = total_impulse * inv_dt
        total_torque = total_torque_impulse * inv_dt
        force_tensor: torch.Tensor = torch.tensor(total_force, dtype=torch.float32)[None]
        torque_tensor: torch.Tensor = torch.tensor(total_torque, dtype=torch.float32)[None]
        link.solver.apply_links_external_force(force_tensor, (link.idx,), ref="link_com", local=False)
        link.solver.apply_links_external_torque(torque_tensor, (link.idx,), ref="link_com", local=False)
