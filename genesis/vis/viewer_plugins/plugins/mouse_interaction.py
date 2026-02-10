from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Type

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


def with_lock(fun: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fun)
    def fun_safe(self: "MouseInteractionPlugin", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fun(self, *args, **kwargs)

    return fun_safe


class MouseInteractionPlugin(RaycasterViewerPlugin):
    """
    Basic interactive viewer plugin that enables using mouse to apply spring force on rigid entities.
    """

    def __init__(
        self,
        use_force: bool = True,
        spring_const: float = 1000.0,
        color: tuple[float, float, float, float] = (0.2, 0.8, 0.8, 0.6),
    ) -> None:
        super().__init__()
        self.use_force = bool(use_force)
        self.spring_const = float(spring_const)
        self.color = tuple(color)
        self.plane_color = (color[0], color[1], color[2], color[3] * 0.5)

        self._lock: Lock = Lock()
        self._held_link: "RigidLink | None" = None
        self._held_point_local: np.ndarray | None = None  # Held point in link-local frame
        self._mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self._prev_mouse_screen_pos: tuple[int, int] = (0, 0)
        self._prev_mouse_scene_pos: np.ndarray | None = None
        self._surface_normal: np.ndarray | None = None
        self._plane_rotation_angle: float = 0.0

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self._prev_mouse_screen_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_screen_pos = (x, y)

    @with_lock
    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_screen_pos = (x, y)
        if self._held_link:
            return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> EVENT_HANDLE_STATE:
        if self._held_link and self._surface_normal is not None:
            # Rotate the drag plane around the surface normal
            self._plane_rotation_angle += scroll_y * 0.1  # 0.1 radians per scroll unit
            self._update_drag_plane()
            return EVENT_HANDLED

    @with_lock
    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT:  # left mouse button
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(ray[0], ray[1])

            if ray_hit.geom and ray_hit.geom.link is not None and not ray_hit.geom.link.is_fixed:
                link = ray_hit.geom.link

                # Validate mass is not too small to prevent numerical instability
                if link.get_mass() < 1e-3:
                    gs.logger.warning(
                        f"Link '{link.name}' has very small mass ({link.get_mass():.2e}). "
                        "Skipping interaction to avoid numerical instability."
                    )
                    return

                self._held_link = link

                # Store the surface normal for rotation
                self._surface_normal = ray_hit.normal
                self._plane_rotation_angle = 0.0
                self._prev_mouse_scene_pos = ray_hit.position

                # Create drag plane perpendicular to surface normal
                self._update_drag_plane()

                # Store held point in link-local frame
                link_pos = tensor_to_array(link.get_pos())
                link_quat = tensor_to_array(link.get_quat())
                self._held_point_local = gu.inv_transform_by_trans_quat(ray_hit.position, link_pos, link_quat)

    @with_lock
    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT:
            self._held_link = None
            self._held_point_local = None
            self._mouse_drag_plane = None
            self._prev_mouse_scene_pos = None
            self._surface_normal = None
            self._plane_rotation_angle = 0.0

    @with_lock
    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        if self._held_link:
            mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)
            assert self._mouse_drag_plane is not None
            ray_hit: RayHit = plane_raycast(*self._mouse_drag_plane, mouse_ray)

            # If ray doesn't hit the plane, skip this update
            if ray_hit is None:
                return

            new_mouse_3d_pos = ray_hit.position
            prev_pos = self._prev_mouse_scene_pos
            delta_3d_pos = new_mouse_3d_pos - prev_pos
            self._prev_mouse_scene_pos = new_mouse_3d_pos

            if self.use_force:
                self._apply_spring_force(new_mouse_3d_pos, self.scene.sim.dt)
            else:
                # apply displacement
                pos = tensor_to_array(self._held_link.entity.get_pos())
                pos = pos + delta_3d_pos
                self._held_link.entity.set_pos(pos)

    @with_lock
    @override
    def on_draw(self) -> None:
        if self.scene._visualizer is not None and self.scene._visualizer.is_built:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)
            closest_hit: RayHit = self._raycaster.cast(mouse_ray[0], mouse_ray[1])

            if self._held_link:
                assert self._mouse_drag_plane is not None
                assert self._held_point_local is not None

                # Draw held point
                link_pos = tensor_to_array(self._held_link.get_pos())
                link_quat = tensor_to_array(self._held_link.get_quat())
                held_point_world = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)

                plane_hit: RayHit | None = plane_raycast(*self._mouse_drag_plane, mouse_ray)
                if plane_hit is not None:
                    self.scene.draw_debug_sphere(
                        plane_hit.position,
                        radius=0.01,
                        color=self.color,
                    )
                    self.scene.draw_debug_line(
                        held_point_world,
                        plane_hit.position,
                        radius=0.005,
                        color=self.color,
                    )
                    # draw the mouse drag plane as a flat box around the mouse position
                    plane_normal, _plane_dist = self._mouse_drag_plane
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
        if self._surface_normal is None or self._prev_mouse_scene_pos is None:
            return

        # Get camera direction
        cam_forward = np.ascontiguousarray(-self.camera.matrix[:3, 2], dtype=gs.np_float)
        surface_normal_contig = np.ascontiguousarray(self._surface_normal, dtype=gs.np_float)

        # Create orthonormal basis with surface_normal as z-axis
        R = gu.z_up_to_R(surface_normal_contig, up=cam_forward)

        plane_normal = R[:, 0] * np.dot(R[:, 0], cam_forward) + R[:, 1] * np.dot(R[:, 1], cam_forward)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + gs.EPS)

        if abs(self._plane_rotation_angle) > gs.EPS:
            rotation_matrix = gu.axis_angle_to_R(surface_normal_contig, self._plane_rotation_angle)
            plane_normal = gu.transform_by_R(plane_normal, rotation_matrix)

        # Set the drag plane (perpendicular to surface normal)
        self._mouse_drag_plane = (plane_normal, -np.dot(plane_normal, self._prev_mouse_scene_pos))

    def _apply_spring_force(self, control_point: np.ndarray, dt: float) -> None:
        if not self._held_link:
            return

        # Get current link state
        link_pos = tensor_to_array(self._held_link.get_pos())
        link_quat = tensor_to_array(self._held_link.get_quat())
        lin_vel = tensor_to_array(self._held_link.get_vel())
        ang_vel = tensor_to_array(self._held_link.get_ang())

        # Compute current world position of held point
        held_point_world = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)

        # Compute inertial frame properties
        inertial_pos = tensor_to_array(self._held_link.inertial_pos)
        inertial_quat = tensor_to_array(self._held_link.inertial_quat)
        world_principal_quat = gu.transform_quat_by_quat(inertial_quat, link_quat)

        # Compute arm from COM to held point in world frame
        arm_in_principal = gu.inv_transform_by_trans_quat(self._held_point_local, inertial_pos, inertial_quat)
        arm_in_world = gu.transform_by_quat(arm_in_principal, world_principal_quat)

        # Compute inverse inertia in world frame
        R_world = gu.quat_to_R(world_principal_quat)
        inertia_world = R_world @ self._held_link.inertial_i @ R_world.T
        inv_inertia_world = np.linalg.inv(inertia_world)

        pos_err_v = control_point - held_point_world
        inv_mass = float(1.0 / self._held_link.get_mass())

        total_impulse = np.zeros(3, dtype=gs.np_float)
        total_torque_impulse = np.zeros(3, dtype=gs.np_float)

        # Approximate spring-damper in each axis
        for i in range(3):
            body_point_vel = lin_vel + np.cross(ang_vel, arm_in_world)
            vel_err_v = -body_point_vel

            direction = np.zeros(3, dtype=gs.np_float)
            direction[i % 3] = 1.0

            pos_err = np.dot(direction, pos_err_v)
            vel_err = np.dot(direction, vel_err_v)

            # Compute virtual mass (effective inertia for this constraint direction)
            arm_x_dir = np.cross(arm_in_world, direction)
            rot_mass = np.dot(arm_x_dir, inv_inertia_world @ arm_x_dir)
            virtual_mass = 1.0 / (inv_mass + rot_mass + gs.EPS)

            # Critical damping
            damping_coeff = 2.0 * np.sqrt(self.spring_const * virtual_mass)
            # Impulse: J = F*dt = k*x*dt + c*v*dt
            impulse = (self.spring_const * pos_err + damping_coeff * vel_err) * dt

            lin_vel += direction * impulse * inv_mass
            ang_vel += inv_inertia_world @ (arm_x_dir * impulse)

            total_impulse[i % 3] += impulse
            total_torque_impulse += arm_x_dir * impulse

        # Apply the new force
        self._held_link.solver.apply_links_external_force(
            total_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )
        self._held_link.solver.apply_links_external_torque(
            total_torque_impulse / dt, (self._held_link.idx,), ref="link_com", local=False
        )
