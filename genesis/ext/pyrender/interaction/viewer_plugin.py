from typing import TYPE_CHECKING, Literal, Type

import numpy as np

from .raycaster import Ray

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.options.viewer_plugins import ViewerPlugin as ViewerPluginOptions


EVENT_HANDLE_STATE = Literal[True] | None
EVENT_HANDLED: Literal[True] = True

# Global map from options class to viewer plugin class
VIEWER_PLUGIN_MAP: dict[Type["ViewerPluginOptions"], Type["ViewerPlugin"]] = {}


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

    def _impl(plugin_cls: Type["ViewerPlugin"]):
        VIEWER_PLUGIN_MAP[options_cls] = plugin_cls
        return plugin_cls

    return _impl


class ViewerPlugin:
    """
    Base class for handling pyglet.window.Window events.
    """

    def __init__(
        self,
        viewer,
        options: "ViewerPluginOptions",
        camera: "Node",
        scene: "Scene",
    ):
        self.viewer = viewer
        self.options: "ViewerPluginOptions" = options
        self.camera: "Node" = camera
        self.scene: "Scene" = scene

        self._camera_yfov: float = camera.camera.yfov
        self._tan_half_fov: float = np.tan(0.5 * self._camera_yfov)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_press(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_key_release(self, symbol: int, modifiers: int) -> EVENT_HANDLE_STATE:
        pass

    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        pass

    def update_on_sim_step(self) -> None:
        pass

    def on_draw(self) -> None:
        pass

    def on_close(self) -> None:
        pass

    def _screen_position_to_ray(self, x: float, y: float) -> Ray:
        # convert screen position to ray
        x = x - 0.5 * self.viewer._viewport_size[0]
        y = y - 0.5 * self.viewer._viewport_size[1]
        x = 2.0 * x / self.viewer._viewport_size[1] * self._tan_half_fov
        y = 2.0 * y / self.viewer._viewport_size[1] * self._tan_half_fov

        # Note: ignoring pixel aspect ratio
        mtx = self.camera.matrix
        position = mtx[:3, 3]
        forward = -mtx[:3, 2]
        right = mtx[:3, 0]
        up = mtx[:3, 1]

        direction = forward + right * x + up * y
        return Ray(origin=position, direction=direction / np.linalg.norm(direction))
