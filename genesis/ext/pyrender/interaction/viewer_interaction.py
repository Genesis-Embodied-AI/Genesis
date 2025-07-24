from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

import genesis as gs
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

from .aabb import AABB
from .mouse_spring import MouseSpring
from .ray import Plane, Ray, RayHit
from .vec3 import Pose, Quat, Vec3, Color
from .viewer_interaction_base import ViewerInteractionBase, EVENT_HANDLE_STATE, EVENT_HANDLED

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class ViewerInteraction(ViewerInteractionBase):
    """Functionalities to be implemented:
    - mouse picking
    - mouse dragging
    """

    def __init__(self, 
        camera: 'Node', 
        scene: 'Scene', 
        viewport_size: tuple[int, int], 
        camera_yfov: float,
        log_events: bool = False,
        camera_fov: float = 60.0,
    ):
        super().__init__(log_events)
        self.camera: 'Node' = camera
        self.scene: 'Scene' = scene
        self.viewport_size: tuple[int, int] = viewport_size
        self.camera_yfov: float = camera_yfov

        self.tan_half_fov: float = np.tan(0.5 * self.camera_yfov)
        self.prev_mouse_pos: tuple[int, int] = (viewport_size[0] // 2, viewport_size[1] // 2)

        self.picked_entity: RigidEntity | None = None
        self.picked_point_in_local: Vec3 | None = None
        self.mouse_drag_plane: Plane | None = None
        self.prev_mouse_3d_pos: Vec3 | None = None

        self.mouse_spring: MouseSpring = MouseSpring()

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_motion(x, y, dx, dy)
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        self.prev_mouse_pos = (x, y)
        if self.picked_entity:
            # actual processing done in update_on_sim_step()

            return EVENT_HANDLED

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_press(x, y, button, modifiers)
        if button == 1: # left mouse button
            (ray_hit, self.picked_entity) = self.raycast_against_entities(self.screen_position_to_ray(x, y))
            if self.picked_entity and ray_hit:
                temp_fwd = self.get_camera_forward()
                temp_back = -temp_fwd

                self.mouse_drag_plane = Plane(temp_back, ray_hit.position)
                self.prev_mouse_3d_pos = ray_hit.position

                pose: Pose = self.get_pose_of_first_geom(self.picked_entity)
                self.picked_point_in_local = pose.inverse_transform_point(ray_hit.position)

                self.mouse_spring.attach(self.picked_entity, ray_hit.position)

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_release(x, y, button, modifiers)
        if button == 1: # left mouse button
            self.picked_entity = None
            self.picked_point_in_local = None

            self.mouse_spring.detach()

    @override
    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        super().on_resize(width, height)
        self.viewport_size = (width, height)
        self.tan_half_fov = np.tan(0.5 * self.camera_yfov)

    @override
    def update_on_sim_step(self) -> None:
        if self.picked_entity:
            mouse_ray: Ray = self.screen_position_to_ray(*self.prev_mouse_pos)
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
                    #apply displacement
                    pos = Vec3.from_tensor(self.picked_entity.get_pos())
                    pos = pos + delta_3d_pos
                    self.picked_entity.set_pos(pos.as_tensor())

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.viewer_lock is not None:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self.screen_position_to_ray(*self.prev_mouse_pos)
            closest_hit = None
            hit_entity: RigidEntity | None = None

            ray_hit = self._raycast_against_ground_plane(mouse_ray)
            if ray_hit.is_hit:
                closest_hit = ray_hit

            for entity in self.get_entities():
                ray_hit = self.raycast_against_entity_obb(entity, mouse_ray)
                if ray_hit.is_hit:
                    if closest_hit is None or ray_hit.distance < closest_hit.distance:
                        closest_hit = ray_hit
                        hit_entity = entity

            if closest_hit:
                self.scene.draw_debug_sphere(closest_hit.position.v, 0.01, (0, 1, 0, 1))
                self._draw_arrow(closest_hit.position, 0.25 * closest_hit.normal, (0, 1, 0, 1))
            if hit_entity:
                self._draw_entity_unrotated_obb(hit_entity)

            if self.picked_entity:
                assert self.mouse_drag_plane is not None
                assert self.picked_point_in_local is not None

                # draw held point
                pose: Pose = self.get_pose_of_first_geom(self.picked_entity)
                held_point: Vec3 = pose.transform_point(self.picked_point_in_local)
                self.scene.draw_debug_sphere(held_point.v, 0.02, Color.red().tuple())

                plane_hit: RayHit = self.mouse_drag_plane.raycast(mouse_ray)
                if plane_hit.is_hit:
                    self.scene.draw_debug_sphere(plane_hit.position.v, 0.02, Color.red().tuple())
                    self.scene.draw_debug_line(held_point.v, plane_hit.position.v, color=Color.red().tuple())

    def screen_position_to_ray(self, x: float, y: float) -> Ray:
        # convert screen position to ray
        if True:
            x = x - 0.5 * self.viewport_size[0]
            y = y - 0.5 * self.viewport_size[1]
            x = 2.0 * x / self.viewport_size[1] * self.tan_half_fov
            y = 2.0 * y / self.viewport_size[1] * self.tan_half_fov
        else:
            # alternative way
            projection_matrix = self.camera.camera.get_projection_matrix(*self.viewport_size)
            x = x - 0.5 * self.viewport_size[0]
            y = y - 0.5 * self.viewport_size[1]
            x = 2.0 * x / self.viewport_size[0] / projection_matrix[0, 0]
            y = 2.0 * y / self.viewport_size[1] / projection_matrix[1, 1]

        # Note: ignoring pixel aspect ratio

        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        right = Vec3.from_array(mtx[:3, 0])
        up = Vec3.from_array(mtx[:3, 1])

        direction = forward + right * x + up * y
        return Ray(position, direction)

    def get_camera_forward(self) -> Vec3:
        mtx = self.camera.matrix
        return Vec3.from_array(-mtx[:3, 2])

    def get_camera_ray(self) -> Ray:
        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        return Ray(position, forward)

    def _raycast_against_ground_plane(self, ray: Ray) -> RayHit:
        ground_plane = Plane(Vec3.from_xyz(0, 0, 1), Vec3.zero())
        return ground_plane.raycast(ray)

    def raycast_against_entity_obb(self, entity: RigidEntity, ray: Ray) -> RayHit:
        if isinstance(entity.morph, gs.morphs.Box):
            box: gs.morphs.Box = entity.morph
            size = Vec3.from_xyz(*box.size)
            pose = self.get_pose_of_first_geom(entity)
            aabb = AABB.from_center_and_size(Vec3.zero(), size)
            ray_hit = aabb.raycast_obb(pose, ray)
            return ray_hit
        else:
            return RayHit.no_hit()

    def raycast_against_entities(self, ray: Ray) -> tuple[RayHit | None, RigidEntity | None]:
        closest_hit = None
        hit_entity: RigidEntity | None = None
        for entity in self.get_entities():
            ray_hit = self.raycast_against_entity_obb(entity, ray)
            if ray_hit.is_hit and (closest_hit is None or ray_hit.distance < closest_hit.distance):
                closest_hit = ray_hit
                hit_entity = entity
        return (closest_hit, hit_entity)

    def get_entities(self) -> list[RigidEntity]:
        return self.scene.sim.rigid_solver.entities

    def get_pose_of_first_geom(self, entity: RigidEntity) -> 'Pose':
        return Pose.from_geom(entity.geoms[0])

    def _draw_arrow(
        self, pos: Vec3, dir: Vec3, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.scene.draw_debug_arrow(pos.v, dir.v, color=color)  # Only draws arrowhead -- bug?
        self.scene.draw_debug_line(pos.v, pos.v + dir.v, color=color)

    def _draw_entity_unrotated_obb(self, entity: RigidEntity) -> None:
        if self.picked_entity:
            return
        
        if isinstance(entity.morph, gs.morphs.Plane):
            plane: gs.morphs.Plane = entity.morph
            pass
        if isinstance(entity.morph, gs.morphs.Box):
            box: gs.morphs.Box = entity.morph
            size = Vec3.from_xyz(*box.size)
            geom: RigidGeom = entity.geoms[0]
            assert geom._solver.n_envs == 0, "ViewerInteraction only supports single-env for now"
            # geom.get_pos() and .get_quat() are squeezed if n_envs == 0
            pos = Vec3.from_tensor(geom.get_pos())
            quat = Quat.from_tensor(geom.get_quat())
            aabb = AABB.from_center_and_size(pos, size)
            aabb.expand(0.01)
            self.scene.draw_debug_box(aabb.v, color=Color.red().with_alpha(0.5).tuple(), wireframe=False)

