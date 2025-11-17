from threading import Lock as threading_Lock
from typing import TYPE_CHECKING

from genesis.options.viewer_plugins import MouseSpringViewerPlugin as MouseSpringViewerPluginOptions
from typing_extensions import override

import genesis as gs

from ..aabb import AABB, OBB
from ..mouse_spring import MouseSpring
from ..ray import Plane, Ray, RayHit
from ..raycaster import ViewerRaycaster
from ..vec3 import Color, Pose, Vec3
from ..viewer_plugin_base import EVENT_HANDLE_STATE, EVENT_HANDLED, ViewerPluginBase, register_viewer_plugin

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidGeom, RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


@register_viewer_plugin(MouseSpringViewerPluginOptions)
class MouseSpringViewerPlugin(ViewerPluginBase):
    """
    Basic interactive viewer plugin that enables using mouse to apply spring force on rigid entities.
    """

    def __init__(
        self,
        options: MouseSpringViewerPluginOptions,
        camera: "Node",
        scene: "Scene",
        viewport_size: tuple[int, int],
    ) -> None:
        super().__init__(options, camera, scene, viewport_size)
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
