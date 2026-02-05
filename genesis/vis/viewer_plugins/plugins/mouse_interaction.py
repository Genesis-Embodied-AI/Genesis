from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import create_plane
from genesis.utils.misc import tensor_to_array
from genesis.utils.raycast import Ray, RayHit, plane_raycast
from genesis.vis.keybindings import MouseButton

from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class MouseInteractionPlugin(RaycasterViewerPlugin):
    """
    Basic interactive viewer plugin that enables using mouse to apply spring force on rigid entities.
    """

    def __init__(
        self,
        use_force: bool = True,
        spring_const: float = 1.0,
        spring_damping: float = 1.0,
        color: tuple[float, float, float, float] = (0.1, 0.6, 0.8, 0.6),
    ) -> None:
        super().__init__()
        self.spring_const = float(spring_const)
        self.spring_damping = float(spring_damping)
        self.color = tuple(color)
        self.plane_color = (color[0], color[1], color[2], color[3] * 0.4)
        self.use_force = bool(use_force)

        self.held_link: "RigidLink | None" = None
        self.held_point_in_local: np.ndarray | None = None
        self.mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self.prev_mouse_screen_pos: tuple[int, int] = (0, 0)
        self.prev_mouse_scene_pos: np.ndarray | None = None
        self.surface_normal: np.ndarray | None = None
        self.plane_rotation_angle: float = 0.0

        self.lock = Lock()

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self.prev_mouse_screen_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self.prev_mouse_screen_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        self.prev_mouse_screen_pos = (x, y)
        if self.held_link:
            return EVENT_HANDLED

    @override
    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> EVENT_HANDLE_STATE:
        if self.held_link and self.surface_normal is not None:
            with self.lock:
                # Rotate the drag plane around the surface normal
                self.plane_rotation_angle += scroll_y * 0.1  # 0.1 radians per scroll unit
                self._update_drag_plane()
            return EVENT_HANDLED

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == 1:  # left mouse button
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(ray[0], ray[1])
            with self.lock:
                if ray_hit.geom and ray_hit.geom.link is not None and not ray_hit.geom.link.is_fixed:
                    link = ray_hit.geom.link

                    # Validate mass is not too small to prevent numerical instability
                    if link.get_mass() < 1e-3:
                        gs.logger.warning(
                            f"Link '{link.name}' has very small mass ({link.get_mass():.2e}). "
                            "Skipping interaction to avoid numerical instability."
                        )
                        return

                    self.held_link = link

                    # Store the surface normal for rotation
                    self.surface_normal = ray_hit.normal
                    self.plane_rotation_angle = 0.0
                    self.prev_mouse_scene_pos = ray_hit.position

                    # Create drag plane perpendicular to surface normal
                    self._update_drag_plane()

                    link_pos = tensor_to_array(self.held_link.get_pos())
                    link_quat = tensor_to_array(self.held_link.get_quat())
                    hit_pos_arr = ray_hit.position
                    self.held_point_in_local = gu.inv_transform_by_trans_quat(hit_pos_arr, link_pos, link_quat)

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT:
            with self.lock:
                self.held_link = None
                self.held_point_in_local = None
                self.mouse_drag_plane = None
                self.prev_mouse_scene_pos = None
                self.surface_normal = None
                self.plane_rotation_angle = 0.0

    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        with self.lock:
            if self.held_link:
                mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_screen_pos)
                assert self.mouse_drag_plane is not None
                ray_hit: RayHit = plane_raycast(*self.mouse_drag_plane, mouse_ray)

                # If ray doesn't hit the plane, skip this update
                if ray_hit is None:
                    return

                new_mouse_3d_pos = ray_hit.position
                prev_pos = self.prev_mouse_scene_pos
                delta_3d_pos = new_mouse_3d_pos - prev_pos
                self.prev_mouse_scene_pos = new_mouse_3d_pos

                if self.use_force:
                    self._apply_spring_force(new_mouse_3d_pos, self.scene.sim.dt)
                else:
                    # apply displacement
                    pos = tensor_to_array(self.held_link.entity.get_pos())
                    pos = pos + delta_3d_pos
                    self.held_link.entity.set_pos(pos)

    @override
    def on_draw(self) -> None:
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self._screen_position_to_ray(*self.prev_mouse_screen_pos)
            closest_hit: RayHit = self._raycaster.cast(mouse_ray[0], mouse_ray[1])

            with self.lock:
                if self.held_link:
                    assert self.mouse_drag_plane is not None
                    assert self.held_point_in_local is not None

                    # draw held point
                    link_pos = tensor_to_array(self.held_link.get_pos())
                    link_quat = tensor_to_array(self.held_link.get_quat())
                    held_point = gu.transform_by_trans_quat(self.held_point_in_local, link_pos, link_quat)

                    plane_hit: RayHit | None = plane_raycast(*self.mouse_drag_plane, mouse_ray)
                    if plane_hit is not None:
                        self.scene.draw_debug_sphere(
                            plane_hit.position,
                            0.01,
                            self.color,
                        )
                        self.scene.draw_debug_line(
                            held_point,
                            plane_hit.position,
                            color=self.color,
                        )
                        # draw the mouse drag plane as a flat box around the mouse position
                        plane_normal, plane_dist = self.mouse_drag_plane
                        self._draw_plane(
                            plane_normal,
                            plane_hit.position,
                            size=1.0,
                            color=self.plane_color,
                        )

                else:
                    if closest_hit is not None:
                        self.scene.draw_debug_arrow(
                            closest_hit.position,
                            closest_hit.normal * 0.25,
                            color=self.color,
                        )

    def _draw_plane(
        self,
        normal: np.ndarray,
        center: np.ndarray | tuple[float, float, float],
        size: float = 1.0,
        color: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 0.2),
    ) -> None:
        vmesh, _ = create_plane(plane_size=(size, size), color_or_texture=color, double_sided=True)
        normal_arr = np.ascontiguousarray(normal, dtype=gs.np_float)
        T = gu.trans_R_to_T(center, gu.z_up_to_R(normal_arr))
        self.scene.draw_debug_mesh(vmesh, T=T)

    def _update_drag_plane(self) -> None:
        """Update the drag plane based on surface normal and rotation angle."""
        if self.surface_normal is None or self.prev_mouse_scene_pos is None:
            return

        # Get camera direction
        cam_forward = np.ascontiguousarray(-self.camera.matrix[:3, 2], dtype=gs.np_float)
        surface_normal_contig = np.ascontiguousarray(self.surface_normal, dtype=gs.np_float)

        # Create orthonormal basis with surface_normal as z-axis
        R = gu.z_up_to_R(surface_normal_contig, up=cam_forward)

        plane_normal = R[:, 0] * np.dot(R[:, 0], cam_forward) + R[:, 1] * np.dot(R[:, 1], cam_forward)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + gs.EPS)

        if abs(self.plane_rotation_angle) > gs.EPS:
            rotation_matrix = gu.axis_angle_to_R(surface_normal_contig, self.plane_rotation_angle)
            plane_normal = gu.transform_by_R(plane_normal, rotation_matrix)

        # Set the drag plane (perpendicular to surface normal)
        self.mouse_drag_plane = (plane_normal, -np.dot(plane_normal, self.prev_mouse_scene_pos))

    def _apply_spring_force(self, control_point: np.ndarray, delta_time: float) -> None:
        if not self.held_link:
            return

        link: "RigidLink" = self.held_link
        lin_vel = tensor_to_array(link.get_vel())
        ang_vel = tensor_to_array(link.get_ang())

        link_pos = tensor_to_array(link.get_pos())
        link_quat = tensor_to_array(link.get_quat())
        held_point_in_world = gu.transform_by_trans_quat(self.held_point_in_local, link_pos, link_quat)

        inertial_pos = tensor_to_array(link.inertial_pos)
        inertial_quat = tensor_to_array(link.inertial_quat)

        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        arm_in_principal = gu.inv_transform_by_trans_quat(self.held_point_in_local, inertial_pos, inertial_quat)
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        pos_err_v = control_point - held_point_in_world
        inv_mass: float = float(1.0 / link.get_mass())

        R_world = gu.quat_to_R(world_principal_quat)
        inertia_world = R_world @ link.inertial_i @ R_world.T
        inv_inertia_world = np.linalg.inv(inertia_world)

        inv_dt: float = 1.0 / delta_time
        tau: float = self.spring_const
        damp: float = self.spring_damping

        total_impulse = np.zeros(3, dtype=gs.np_float)
        total_torque_impulse = np.zeros(3, dtype=gs.np_float)

        for i in range(3 * 4):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            direction = np.zeros(3, dtype=gs.np_float)
            direction[i % 3] = 1.0

            pos_err = np.dot(direction, pos_err_v)
            vel_err = np.dot(direction, vel_err_v)
            error = tau * pos_err * inv_dt + damp * vel_err

            arm_x_dir = np.cross(arm_in_world, direction)
            rotational_mass_contribution = np.dot(arm_x_dir, inv_inertia_world @ arm_x_dir)
            virtual_mass = 1.0 / (inv_mass + rotational_mass_contribution + gs.EPS)
            impulse = error * virtual_mass

            lin_vel = lin_vel + direction * impulse * inv_mass
            ang_vel = ang_vel + inv_inertia_world @ (arm_x_dir * impulse)

            total_impulse[i % 3] += impulse
            total_torque_impulse += arm_x_dir * impulse

        # Apply the new force
        total_force = total_impulse * inv_dt
        total_torque = total_torque_impulse * inv_dt
        link.solver.apply_links_external_force(total_force, (link.idx,), ref="link_com", local=False)
        link.solver.apply_links_external_torque(total_torque, (link.idx,), ref="link_com", local=False)
