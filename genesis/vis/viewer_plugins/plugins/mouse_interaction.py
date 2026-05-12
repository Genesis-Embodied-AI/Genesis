from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Type

import numpy as np
from typing_extensions import override

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import create_cylinder, create_plane
from genesis.utils.misc import qd_to_numpy, tensor_to_array, with_lock
from genesis.utils.raycast import Ray, RayHit, plane_raycast
from genesis.vis.keybindings import MouseButton

from ..viewer_plugin import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


MIN_PICKABLE_MASS = 1e-3  # kg - links below this threshold are skipped to avoid numerical instability


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

        self._lock: Lock = Lock()
        self._held_link: "RigidLink | None" = None
        self._interact_env_idx: int | None = None
        self._env_offset: np.ndarray = np.zeros(3, dtype=gs.np_float)
        self._held_point_local: np.ndarray | None = None
        self._mouse_drag_plane: tuple[np.ndarray, float] | None = None
        self._prev_mouse_screen_pos: tuple[int, int] = (0, 0)
        self._prev_mouse_scene_pos: np.ndarray | None = None
        self._surface_normal: np.ndarray | None = None
        self._plane_rotation_angle: float = 0.0

        # Persistent debug nodes (lifecycle managed in on_draw)
        self._debug_interact_nodes: list["Node"] = []
        self._debug_normal_node: "Node | None" = None
        self._unit_cylinder_mesh = None
        self._plane_mesh = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        self._prev_mouse_screen_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

        self._unit_cylinder_mesh = create_cylinder(radius=0.005, height=1.0, color=self.color)
        plane_color = (*self.color[:3], self.color[3] * 0.5)
        plane_vmesh, _ = create_plane(plane_size=(0.3, 0.3), color_or_texture=plane_color, double_sided=True)
        self._plane_mesh = plane_vmesh

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
            if ray_hit is None:
                return

            if ray_hit.geom and ray_hit.geom.link is not None and not ray_hit.geom.link.is_fixed:
                link = ray_hit.geom.link
                hit_env_idx = self._get_last_raycast_env_idx()
                mass = float(link.get_mass())

                # Validate mass is not too small to prevent numerical instability
                if mass < MIN_PICKABLE_MASS:
                    gs.logger.warning(
                        f"Link '{link.name}' has very small mass ({mass:.2e}). "
                        "Skipping interaction to avoid numerical instability."
                    )
                    return

                self._held_link = link
                self._interact_env_idx = hit_env_idx
                self._env_offset = (
                    self.scene.envs_offset[hit_env_idx].astype(gs.np_float, copy=False)
                    if hit_env_idx is not None
                    else np.zeros(3, dtype=gs.np_float)
                )

                # Store the surface normal for rotation
                self._surface_normal = ray_hit.normal
                self._plane_rotation_angle = 0.0
                self._prev_mouse_scene_pos = ray_hit.position

                # Create drag plane perpendicular to surface normal
                self._update_drag_plane()

                # Store held point in link-local frame. ray_hit.position is in world coords (with envs_offset baked in
                # by the raycaster), while link.get_pos returns env-local coords, so subtract the offset to align.
                link_pos = tensor_to_array(link.get_pos(envs_idx=hit_env_idx).squeeze(0))
                link_quat = tensor_to_array(link.get_quat(envs_idx=hit_env_idx).squeeze(0))
                hit_env_local = ray_hit.position - self._env_offset
                self._held_point_local = gu.inv_transform_by_trans_quat(hit_env_local, link_pos, link_quat)

    @with_lock
    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if button == MouseButton.LEFT:
            self._held_link = None
            self._interact_env_idx = None
            self._env_offset = np.zeros(3, dtype=gs.np_float)
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
            ray_hit_plane: RayHit | None = plane_raycast(*self._mouse_drag_plane, mouse_ray)

            # If ray doesn't hit the plane, skip this update
            if ray_hit_plane is None:
                return

            self._prev_mouse_scene_pos = ray_hit_plane.position
            envs_idx = self._interact_env_idx

            if self.use_force:
                self._apply_spring_force(ray_hit_plane.position, self.scene.sim.dt)
            else:
                assert self._held_point_local is not None
                link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx).squeeze(0))
                offset_world = gu.transform_by_quat(self._held_point_local, link_quat)
                # Mouse target is world; entity position is env-local, so strip the env offset before setting.
                new_link_pos = ray_hit_plane.position - self._env_offset - offset_world
                self._held_link.entity.set_pos(new_link_pos, envs_idx=envs_idx)

    @with_lock
    @override
    def on_draw(self) -> None:
        if self.scene._visualizer is None or not self.scene._visualizer.is_built:
            return

        mouse_ray: Ray = self._screen_position_to_ray(*self._prev_mouse_screen_pos)

        if self._held_link:
            # Clean up hover arrow when transitioning to hold state
            if self._debug_normal_node is not None:
                self.scene.clear_debug_object(self._debug_normal_node)
                self._debug_normal_node = None

            assert self._mouse_drag_plane is not None
            assert self._held_point_local is not None

            envs_idx = self._interact_env_idx
            link_pos = tensor_to_array(self._held_link.get_pos(envs_idx=envs_idx).squeeze(0))
            link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx).squeeze(0))
            held_point_env_local = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)
            held_point_world = held_point_env_local + self._env_offset

            plane_hit: RayHit | None = plane_raycast(*self._mouse_drag_plane, mouse_ray)
            if plane_hit is not None:
                control_point = plane_hit.position

                # Sphere at clamped control point (translation only, no rotation)
                sphere_T = gu.trans_to_T(control_point)
                # Cylinder from held point to clamped control point
                line_T = self._compute_line_T(held_point_world, control_point)
                # Drag plane visualization centered on clamped control point
                plane_T = gu.trans_R_to_T(control_point, gu.z_up_to_R(self._mouse_drag_plane[0]))

                if not self._debug_interact_nodes:
                    self._debug_interact_nodes.append(
                        self.scene.draw_debug_sphere(control_point, radius=0.01, color=self.color)
                    )
                    self._debug_interact_nodes.append(self.scene.draw_debug_mesh(self._unit_cylinder_mesh, T=line_T))
                    self._debug_interact_nodes.append(self.scene.draw_debug_mesh(self._plane_mesh, T=plane_T))
                else:
                    self.scene.update_debug_objects(self._debug_interact_nodes, (sphere_T, line_T, plane_T))
            else:
                # No plane hit: hide held visualization nodes
                for node in self._debug_interact_nodes:
                    self.scene.clear_debug_object(node)
                self._debug_interact_nodes.clear()

        else:
            # Clean up held visualization nodes
            for node in self._debug_interact_nodes:
                self.scene.clear_debug_object(node)
            self._debug_interact_nodes.clear()

            # Hover arrow: only show for pickable (non-fixed, sufficient mass) entities
            closest_hit: RayHit = self._raycaster.cast(mouse_ray[0], mouse_ray[1])
            link = closest_hit.geom.link if closest_hit is not None and closest_hit.geom is not None else None
            is_pickable = link is not None and not link.is_fixed and float(link.get_mass()) >= MIN_PICKABLE_MASS
            if is_pickable:
                arrow_T = gu.trans_R_to_T(closest_hit.position, gu.z_up_to_R(closest_hit.normal))
                if self._debug_normal_node is None:
                    self._debug_normal_node = self.scene.draw_debug_arrow(
                        closest_hit.position, closest_hit.normal * 0.25, color=self.color
                    )
                else:
                    self.scene.update_debug_objects((self._debug_normal_node,), (arrow_T,))
            elif self._debug_normal_node is not None:
                self.scene.clear_debug_object(self._debug_normal_node)
                self._debug_normal_node = None

    def _compute_line_T(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Compute transform for unit cylinder (height=1, centered at z=0) from start to end."""
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < gs.EPS:
            return gu.trans_to_T(start)
        R_basis = gu.z_up_to_R(direction / length)
        T = np.eye(4, dtype=gs.np_float)
        T[:3, 0] = R_basis[:, 0]
        T[:3, 1] = R_basis[:, 1]
        T[:3, 2] = direction  # scaled z: maps local +/-0.5 to world start/end
        T[:3, 3] = (start + end) * 0.5
        return T

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

    def _get_last_raycast_env_idx(self) -> int | None:
        if self.scene.n_envs == 0:
            return None
        env_idx = int(qd_to_numpy(self._raycaster.result.env_idx))
        return env_idx if env_idx >= 0 else None

    def _apply_spring_force(self, control_point: np.ndarray, dt: float) -> None:
        if not self._held_link:
            return
        envs_idx = self._interact_env_idx

        # Current link state in env-local frame
        link_pos = tensor_to_array(self._held_link.get_pos(envs_idx=envs_idx).squeeze(0))
        link_quat = tensor_to_array(self._held_link.get_quat(envs_idx=envs_idx).squeeze(0))
        lin_vel = tensor_to_array(self._held_link.get_vel(envs_idx=envs_idx).squeeze(0))
        ang_vel = tensor_to_array(self._held_link.get_ang(envs_idx=envs_idx).squeeze(0))

        # Held point in env-local frame; control point comes from a world-space plane raycast, so strip the offset.
        held_point_env_local = gu.transform_by_trans_quat(self._held_point_local, link_pos, link_quat)
        control_point_env_local = control_point - self._env_offset

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

        pos_err_v = control_point_env_local - held_point_env_local
        inv_mass = 1.0 / float(self._held_link.get_mass())

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
            total_impulse / dt,
            (self._held_link.idx,),
            envs_idx=envs_idx,
            ref="link_com",
            local=False,
        )
        self._held_link.solver.apply_links_external_torque(
            total_torque_impulse / dt,
            (self._held_link.idx,),
            envs_idx=envs_idx,
            ref="link_com",
            local=False,
        )
