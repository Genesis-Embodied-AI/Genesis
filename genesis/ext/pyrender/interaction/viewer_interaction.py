from typing import override
from pyglet.event import EVENT_HANDLE_STATE

from genesis.engine.scene import Scene
from genesis.ext.pyrender.node import Node
from genesis.ext.pyrender.interaction.ray import Ray
from genesis.ext.pyrender.interaction.vec3 import Vec3
from genesis.ext.pyrender.viewer_interaction_base import ViewerInteractionBase


class ViewerInteraction(ViewerInteractionBase):
    """Functionalities to be implemented:
    - mouse picking
    - mouse dragging
    """

    camera: Node
    scene: Scene

    def __init__(self, camera: Node, scene: Scene, log_events: bool = False):
        super().__init__(log_events)
        self.camera = camera
        self.scene = scene

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        mouse_ray = self.screen_position_to_ray(x, y)
        print(f"mouse_ray: {mouse_ray}")

    def screen_position_to_ray(self, x: int, y: int) -> Ray:
        mtx = self.camera.matrix
        position = Vec3(mtx[:3, 3])
        forward = Vec3(-mtx[:3, 2])
        return Ray(position, forward)
