from typing import TYPE_CHECKING, Literal, Type

import numpy as np

from .ray import Ray
from .vec3 import Vec3

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.options.viewer_plugins import ViewerPlugin as ViewerPluginOptions


EVENT_HANDLE_STATE = Literal[True] | None
EVENT_HANDLED: Literal[True] = True

# Global map from options class to viewer plugin class
VIEWER_PLUGIN_MAP: dict[Type["ViewerPluginOptions"], Type["ViewerPluginBase"]] = {}


def register_viewer_plugin(options_cls: Type["ViewerPluginOptions"]):
    """
    Decorator to register a viewer plugin class with its corresponding options class.
    
    Parameters
    ----------
    options_cls : Type[ViewerPluginOptions]
        The options class that configures this viewer plugin.
        
    Returns
    -------
    Callable
        The decorator function that registers the plugin class.
        
    Example
    -------
    @register_viewer_plugin(ViewerInteractionOptions)
    class ViewerInteraction(ViewerInteractionBase):
        ...
    """
    def _impl(plugin_cls: Type["ViewerPluginBase"]):
        VIEWER_PLUGIN_MAP[options_cls] = plugin_cls
        return plugin_cls
    return _impl

# Note: Viewer window is based on pyglet.window.Window, mouse events are defined in pyglet.window.BaseWindow

class ViewerPluginBase():
    """
    Base class for handling pyglet.window.Window events.
    """

    def __init__(
        self,
        options: "ViewerPluginOptions",
        camera: "Node",
        scene: "Scene",
        viewport_size: tuple[int, int],
    ):
        self.options: "ViewerPluginOptions" = options
        self.camera: 'Node' = camera
        self.scene: 'Scene' = scene
        self.viewport_size: tuple[int, int] = viewport_size

        self.camera_yfov: float = camera.camera.yfov
        self.tan_half_fov: float = np.tan(0.5 * self.camera_yfov)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        self.viewport_size = (width, height)
        self.tan_half_fov = np.tan(0.5 * self.camera_yfov)

    def update_on_sim_step(self) -> None:
        pass

    def on_draw(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def _screen_position_to_ray(self, x: float, y: float) -> Ray:
        # convert screen position to ray
        x = x - 0.5 * self.viewport_size[0]
        y = y - 0.5 * self.viewport_size[1]
        x = 2.0 * x / self.viewport_size[1] * self.tan_half_fov
        y = 2.0 * y / self.viewport_size[1] * self.tan_half_fov

        # Note: ignoring pixel aspect ratio

        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        right = Vec3.from_array(mtx[:3, 0])
        up = Vec3.from_array(mtx[:3, 1])

        direction = forward + right * x + up * y
        return Ray(position, direction)

    def _get_camera_forward(self) -> Vec3:
        mtx = self.camera.matrix
        return Vec3.from_array(-mtx[:3, 2])

    def _get_camera_ray(self) -> Ray:
        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        return Ray(position, forward)