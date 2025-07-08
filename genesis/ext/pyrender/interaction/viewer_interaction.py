from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from pyglet.event import EVENT_HANDLE_STATE

import genesis as gs
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

from .aabb import AABB
from .ray import Plane, Ray, RayHit
from .vec3 import Pose, Quat, Vec3, Color
from .viewer_interaction_base import ViewerInteractionBase, EVENT_HANDLE_STATE

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


class ViewerInteraction(ViewerInteractionBase):
    """Functionalities to be implemented:
    - mouse picking
    - mouse dragging
    """

    camera: 'Node'
    scene: 'Scene'
    viewport_size: tuple[int, int]
    camera_yfov: float

    tan_half_fov: float
    prev_mouse_pos: tuple[int, int]

    def __init__(self, 
        camera: 'Node', 
        scene: 'Scene', 
        viewport_size: tuple[int, int], 
        camera_yfov: float,
        log_events: bool = False,
        camera_fov: float = 60.0,
    ):
        super().__init__(log_events)
        self.camera = camera
        self.scene = scene
        self.viewport_size = viewport_size
        self.camera_yfov = camera_yfov

        self.tan_half_fov = np.tan(0.5 * self.camera_yfov)
        self.prev_mouse_pos = tuple(np.array(viewport_size) / 2)

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_motion(x, y, dx, dy)
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_press(x, y, button, modifiers)

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.viewer_lock is not None:
            self.scene.clear_debug_objects()
            mouse_ray = self.screen_position_to_ray(*self.prev_mouse_pos)
            closest_hit = None
            hit_entity: RigidEntity | None = None

            ray_hit = self._raycast_against_ground_plane(mouse_ray)
            if ray_hit.is_hit:
                closest_hit = ray_hit

            for entity in self.get_entities():
                ray_hit = self.raycast_against_entity_oobb(entity, mouse_ray)
                if ray_hit.is_hit:
                    if closest_hit is None or ray_hit.distance < closest_hit.distance:
                        closest_hit = ray_hit
                        hit_entity = entity

            if closest_hit:
                self.scene.draw_debug_sphere(closest_hit.position.v, 0.01, (0, 1, 0, 1))
                self._draw_arrow(closest_hit.position, 0.25 * closest_hit.normal, (0, 1, 0, 1))
            if hit_entity:
                self._draw_entity_unrotated_oobb(hit_entity)

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
        position = Vec3.from_float64(mtx[:3, 3])
        forward = Vec3.from_float64(-mtx[:3, 2])
        right = Vec3.from_float64(mtx[:3, 0])
        up = Vec3.from_float64(mtx[:3, 1])

        direction = forward + right * x + up * y
        return Ray(position, direction)

    def get_camera_ray(self) -> Ray:
        mtx = self.camera.matrix
        position = Vec3.from_float64(mtx[:3, 3])
        forward = Vec3.from_float64(-mtx[:3, 2])
        return Ray(position, forward)

    def _raycast_against_ground_plane(self, ray: Ray) -> RayHit:
        ground_plane = Plane(Vec3.from_xyz(0, 0, 1), Vec3.zero())
        return ground_plane.raycast(ray)

    def raycast_against_entity_oobb(self, entity: RigidEntity, ray: Ray) -> RayHit:
        if isinstance(entity.morph, gs.morphs.Box):
            box: gs.morphs.Box = entity.morph
            size = Vec3.from_xyz(*box.size)
            pose = self.get_pose_of_first_geom(entity)
            aabb = AABB.from_center_and_size(Vec3.zero(), size)
            ray_hit = aabb.raycast_oobb(pose, ray)
            return ray_hit
        else:
            return RayHit.no_hit()

    def get_entities(self) -> list[RigidEntity]:
        return self.scene.sim.rigid_solver.entities

    def get_pose_of_first_geom(self, entity: RigidEntity) -> 'Pose':
        geom: RigidGeom = entity.geoms[0]
        assert geom._solver.n_envs == 0, "ViewerInteraction only supports single-env for now"
        gpos = geom.get_pos()  # squeezed if n_envs == 0
        gquat = geom.get_quat()  # squeezed if n_envs == 0
        return Pose(Vec3(gpos.cpu().numpy()), Quat(gquat.cpu().numpy()))

    def _draw_arrow(
        self, pos: Vec3, dir: Vec3, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.scene.draw_debug_arrow(pos.v, dir.v, color=color)  # Only draws arrowhead -- bug?
        self.scene.draw_debug_line(pos.v, pos.v + dir.v, color=color)

    def _draw_entity_unrotated_oobb(self, entity: RigidEntity) -> None:
        if isinstance(entity.morph, gs.morphs.Plane):
            plane: gs.morphs.Plane = entity.morph
            pass
        if isinstance(entity.morph, gs.morphs.Box):
            box: gs.morphs.Box = entity.morph
            size = Vec3.from_xyz(*box.size)
            geom: RigidGeom = entity.geoms[0]
            assert geom._solver.n_envs == 0, "ViewerInteraction only supports single-env for now"
            gpos = geom.get_pos()  # squeezed if n_envs == 0
            gquat = geom.get_quat()  # squeezed if n_envs == 0
            pos = Vec3.from_any_array(gpos.cpu().numpy())
            quat = Quat.from_any_array(gquat.cpu().numpy())
            aabb = AABB.from_center_and_size(pos, size)
            aabb.expand(0.01)
            self.scene.draw_debug_box(aabb.v, color=Color.red().with_alpha(0.5).tuple(), wireframe=False)

