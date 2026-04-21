from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import override

from genesis.ext.pyrender.camera import OrthographicCamera
from genesis.utils.raycast import Ray

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.ext.pyrender.viewer import Viewer
    from genesis.utils.raycast_qd import Raycaster


EVENT_HANDLE_STATE = Literal[True] | None
EVENT_HANDLED: Literal[True] = True


class ViewerPlugin:
    """
    Base class for handling pyglet.window.Window events.
    """

    def __init__(self):
        self.viewer: "Viewer | None" = None
        self.camera: "Node | None" = None
        self.scene: "Scene | None" = None

    def build(self, viewer: "Viewer", camera: "Node", scene: "Scene"):
        """Build and initialize the plugin with pyrender viewer context."""

        self.viewer = viewer
        self.camera = camera
        self.scene = scene

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
        self._raycaster: "Raycaster | None" = None

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)

        # NOTE: delayed import to avoid array_class import before gs is fully initialized
        from genesis.utils.raycast_qd import Raycaster

        self._raycaster = Raycaster(self.scene)

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
        w_raw = float(viewport_size[0])
        h_raw = float(viewport_size[1])
        h = max(h_raw, 1e-8)
        x_c = float(x) - 0.5 * w_raw
        y_c = float(y) - 0.5 * h_raw
        sx = 2.0 * x_c / h
        sy = 2.0 * y_c / h

        # NOTE: ignoring pixel aspect ratio; projection may change after build (e.g. O key)
        mtx = self.camera.matrix
        position = mtx[:3, 3]
        forward = -mtx[:3, 2]
        right = mtx[:3, 0]
        up = mtx[:3, 1]

        cam = self.camera.camera
        if isinstance(cam, OrthographicCamera):
            ymag = float(cam.ymag)
            origin = position + right * (sx * ymag) + up * (sy * ymag)
            direction = forward / np.linalg.norm(forward)
            return Ray(origin, direction)

        tan_half = float(np.tan(0.5 * float(cam.yfov)))
        direction = forward + right * (sx * tan_half) + up * (sy * tan_half)
        direction /= np.linalg.norm(direction)
        return Ray(position, direction)
