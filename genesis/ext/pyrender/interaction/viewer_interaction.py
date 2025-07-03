from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from .ray import Plane, Ray, RayHit
from .vec3 import Vec3
from .viewer_interaction_base import ViewerInteractionBase, EVENT_HANDLE_STATE

if TYPE_CHECKING:
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
        mouse_ray = self.screen_position_to_ray(x, y)
        # print(f"mouse_ray: {mouse_ray}")

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is not None and self.scene._visualizer.viewer_lock is not None:
            self.scene.clear_debug_objects()
        
            ray_hit = self._raycast_against_ground(self.screen_position_to_ray(*self.prev_mouse_pos))
            if ray_hit.is_hit:
                self.scene.draw_debug_sphere(ray_hit.position.v, 0.01, (0, 1, 0, 1))
                self._draw_arrow(ray_hit.position, ray_hit.normal, (0, 1, 0, 1))

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

    def _raycast_against_ground(self, ray: Ray) -> RayHit:
        ground_plane = Plane(Vec3.from_xyz(0, 0, 1), Vec3.zero())
        return ground_plane.raycast(ray)

    def _draw_arrow(
        self, pos: Vec3, dir: Vec3, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.scene.draw_debug_arrow(pos.v, dir.v, color=color)  # Only draws arrowhead -- bug?
        self.scene.draw_debug_line(pos.v, pos.v + dir.v, color=color)
