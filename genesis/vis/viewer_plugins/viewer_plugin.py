from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from genesis.utils.raycast import Ray

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.ext.pyrender.viewer import Viewer
    from genesis.utils.raycast_ti import Raycaster


EVENT_HANDLE_STATE = Literal[True] | None
EVENT_HANDLED: Literal[True] = True


class ViewerPlugin:
    """
    Base class for handling pyglet.window.Window events.
    """

    def __init__(self):
        self.viewer = None
        self.camera: "Node | None" = None
        self.scene: "Scene | None" = None
        self._camera_yfov: float = 0.0
        self._tan_half_fov: float = 0.0

    def build(self, viewer: "Viewer", camera: "Node", scene: "Scene"):
        """Build and initialize the plugin with pyrender viewer context."""

        self.viewer = viewer
        self.camera = camera
        self.scene = scene
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


class RaycasterViewerPlugin(ViewerPlugin):
    """
    Base viewer plugins using mouse raycast
    """

    def __init__(self) -> None:
        super().__init__()
        self._camera_tan_half_fov: float = 0.0
        self._raycaster: "Raycaster | None" = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        # NOTE: delayed import to avoid array_class import before gs is fully initialized
        from genesis.utils.raycast_ti import Raycaster

        self._raycaster = Raycaster(self.scene)
        self._camera_tan_half_fov = np.tan(0.5 * self.camera.camera.yfov)

    @override
    def update_on_sim_step(self) -> None:
        super().update_on_sim_step()

        self._raycaster.update()

    def _screen_position_to_ray(self, x: float, y: float) -> Ray:
        """
        Converts 2D screen position to a ray.

        Parameters
        ----------
        x : float
            The x coordinate on the screen.
        y : float
            The y coordinate on the screen.

        Returns
        -------
        origin : np.ndarray, shape (3,)
            The origin of the ray in world coordinates.
        direction : np.ndarray, shape (3,)
            The direction of the ray in world coordinates.
        """

        viewport_size = self.viewer._viewport_size
        x = x - 0.5 * viewport_size[0]
        y = y - 0.5 * viewport_size[1]
        x = 2.0 * x / viewport_size[1] * self._camera_tan_half_fov
        y = 2.0 * y / viewport_size[1] * self._camera_tan_half_fov

        # NOTE: ignoring pixel aspect ratio
        mtx = self.camera.matrix
        position = mtx[:3, 3]
        forward = -mtx[:3, 2]
        right = mtx[:3, 0]
        up = mtx[:3, 1]

        direction = forward + right * x + up * y
        direction /= np.linalg.norm(direction)

        return Ray(position, direction)
