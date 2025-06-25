"""Wrapper for offscreen rendering.

Author: Matthew Matl
"""

import os

from OpenGL.GL import *

import genesis as gs

from .constants import RenderFlags
from .renderer import Renderer
from .shader_program import ShaderProgram, ShaderProgramCache


MODULE_DIR = os.path.dirname(__file__)


class OffscreenRenderer(object):
    """A wrapper for offscreen rendering.

    Parameters
    ----------
    viewport_width : int
        The width of the main viewport, in pixels.
    viewport_height : int
        The height of the main viewport, in pixels.
    point_size : float
        The size of screen-space points in pixels.
    """

    def __init__(self, point_size=1.0, pyopengl_platform="pyglet", seg_node_map=None):
        self.point_size = point_size
        self._platform = None
        self._is_software = False
        self._create(pyopengl_platform)
        self._seg_node_map = seg_node_map

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels."""
        return 32
        # return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = int(value)

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels."""
        return 32
        # return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = int(value)

    @property
    def point_size(self):
        """float : The pixel size of points in point clouds."""
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    def render(
        self,
        scene,
        renderer,
        flags=RenderFlags.NONE,
        camera_node=None,
        shadow=False,
        seg=False,
        ret_depth=False,
        plane_reflection=False,
        env_separate_rigid=False,
        normal=False,
    ):
        """Render a scene with the given set of flags.

        Parameters
        ----------
        scene : :class:`Scene`
            A scene to render.
        flags : int
            A bitwise or of one or more flags from :class:`.RenderFlags`.

        Returns
        -------
        color_im : (h, w, 3) uint8 or (h, w, 4) uint8
            The color buffer in RGB format, or in RGBA format if
            :attr:`.RenderFlags.RGBA` is set.
            Not returned if flags includes :attr:`.RenderFlags.DEPTH_ONLY`.
        depth_im : (h, w) float32
            The depth buffer in linear units.
        """

        if camera_node is not None:
            saved_camera_node = scene.main_camera_node
            scene.main_camera_node = camera_node

        self._platform.make_current()
        # If platform does not support dynamically-resizing framebuffers,
        # destroy it and restart it
        if (
            self._platform.viewport_height != self.viewport_height
            or self._platform.viewport_width != self.viewport_width
        ):
            if not self._platform.supports_framebuffers():
                self.delete()
                self._create()

        self._platform.make_current()

        # Forcibly disable shadow for software rendering as it may hang indefinitely
        if shadow and not self._is_software:
            flags |= RenderFlags.SHADOWS_ALL

        if plane_reflection and not self._is_software:
            flags |= RenderFlags.REFLECTIVE_FLOOR

        if env_separate_rigid:
            flags |= RenderFlags.ENV_SEPARATE

        if seg:
            seg_node_map = self._seg_node_map
            flags |= RenderFlags.SEG
        else:
            seg_node_map = None

        if ret_depth:
            flags |= RenderFlags.RET_DEPTH

        if self._platform.supports_framebuffers():
            flags |= RenderFlags.OFFSCREEN
            retval = renderer.render(scene, flags, seg_node_map)
        else:
            renderer.render(scene, flags, seg_node_map)
            depth = renderer.read_depth_buf()
            if flags & RenderFlags.DEPTH_ONLY:
                retval = depth
            else:
                color = renderer.read_color_buf()
                retval = color, depth

        if normal:
            class CustomShaderCache:
                def __init__(self):
                    self.program = None

                def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
                    if self.program is None:
                        self.program = ShaderProgram(
                            os.path.join(MODULE_DIR, "shaders/mesh_normal.vert"),
                            os.path.join(MODULE_DIR, "shaders/mesh_normal.frag"),
                            defines=defines,
                        )
                    return self.program

            old_cache = renderer._program_cache
            renderer._program_cache = CustomShaderCache()

            flags = RenderFlags.FLAT | RenderFlags.OFFSCREEN
            if env_separate_rigid:
                flags |= RenderFlags.ENV_SEPARATE
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            normal_arr, _ = renderer.render(scene, flags, is_first_pass=False)
            retval = retval + (normal_arr,)

            renderer._program_cache = old_cache

        # Make the platform not current
        self._platform.make_uncurrent()

        if camera_node is not None:
            scene.main_camera_node = saved_camera_node

        return retval

    def delete(self):
        """Free all OpenGL resources."""
        self._platform.make_current()
        self._platform.delete_context()
        del self._platform
        self._platform = None
        import gc

        gc.collect()

    def _create(self, platform):
        if platform == "pyglet":
            from .platforms.pyglet_platform import PygletPlatform

            self._platform = PygletPlatform(self.viewport_width, self.viewport_height)
        elif platform == "egl":
            from .platforms import egl

            if "EGL_DEVICE_ID" in os.environ:
                device_id = int(os.environ["EGL_DEVICE_ID"])
            else:
                device_id = None
            self._platform = egl.EGLPlatform(self.viewport_width, self.viewport_height, device_id)
        elif platform == "osmesa":
            from .platforms.osmesa import OSMesaPlatform

            self._platform = OSMesaPlatform(self.viewport_width, self.viewport_height)
        else:
            raise ValueError("Unsupported PyOpenGL platform: {}".format(os.environ["PYOPENGL_PLATFORM"]))
        self._platform.init_context()
        self._platform.make_current()

        try:
            from OpenGL.GL import glGetString, GL_RENDERER
            renderer = glGetString(GL_RENDERER).decode()
            self._is_software = "llvmpipe" in renderer
        except:
            pass
        if self._is_software:
            gs.logger.info(
                "Software rendering context detected. Shadows and plane reflection not supported. Beware rendering "
                "will be extremely slow."
            )

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


__all__ = ["OffscreenRenderer"]
