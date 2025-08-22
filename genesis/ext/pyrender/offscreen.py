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
        self._has_valid_context = False
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

    def make_current(self):
        """This function sets the current context and must be called before all rendering and GPU upload operations."""
        if self._has_valid_context:
            gs.raise_exception(
                "The method was called while having an other context current. Please call 'make_uncurrent' first."
            )

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

                # Only needs to happen if the context was deleted and created
                self._platform.make_current()

        self._has_valid_context = True

    def make_uncurrent(self):
        """This function unsets the current context and must be called after all rendering and GPU upload operations
        are done.
        """
        if not self._has_valid_context:
            gs.raise_exception("The method was called before making a context current.")
        self._platform.make_uncurrent()
        self._has_valid_context = False

    def render(
        self,
        scene,
        renderer,
        rgb=True,
        seg=False,
        normal=False,
        depth=False,
        flags=RenderFlags.NONE,
        camera_node=None,
        shadow=False,
        plane_reflection=False,
        env_separate_rigid=False,
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
        if seg and rgb:
            gs.raise_exception("RGB and segmentation map cannot be rendered in the same forward pass.")

        if not self._has_valid_context:
            gs.raise_exception(
                "Ensure that the right context is set before rendering. Please call the method 'make_current'."
            )

        if camera_node is not None:
            saved_camera_node = scene.main_camera_node
            scene.main_camera_node = camera_node

        # Forcibly disable shadow for software rendering as it may hang indefinitely
        if shadow and not self._is_software:
            flags |= RenderFlags.SHADOWS_ALL

        if depth and not (rgb or seg):
            flags |= RenderFlags.DEPTH_ONLY

        if plane_reflection and not self._is_software:
            flags |= RenderFlags.REFLECTIVE_FLOOR

        if env_separate_rigid:
            flags |= RenderFlags.ENV_SEPARATE

        if seg:
            seg_node_map = self._seg_node_map
            flags |= RenderFlags.SEG
        else:
            seg_node_map = None

        if depth:
            flags |= RenderFlags.RET_DEPTH

        if rgb or depth or seg:
            if self._platform.supports_framebuffers():
                flags |= RenderFlags.OFFSCREEN
                retval = renderer.render(scene, flags, seg_node_map)
            else:
                if flags & RenderFlags.ENV_SEPARATE:
                    gs.raise_exception("'env_separate_rigid=True' not supported on this platform.")
                renderer.render(scene, flags, seg_node_map)
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
                glReadBuffer(GL_FRONT)
                if depth:
                    z_near = scene.main_camera_node.camera.znear
                    z_far = scene.main_camera_node.camera.zfar
                    if z_far is None:
                        z_far = -1.0
                    depth_arr = renderer.jit.read_depth_buf(self.viewport_height, self.viewport_width, z_near, z_far)
                    depth_arr = renderer._resize_image(depth_arr, antialias=not seg)
                if flags & RenderFlags.DEPTH_ONLY:
                    retval = (depth_arr,)
                else:
                    color_arr = renderer.jit.read_color_buf(self.viewport_height, self.viewport_width, rgba=False)
                    color_arr = renderer._resize_image(color_arr, antialias=not seg)
                    retval = (color_arr, depth_arr) if depth else (color_arr,)
        else:
            retval = ()

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

            if self._platform.supports_framebuffers():
                normal_arr, *_ = renderer.render(scene, flags, is_first_pass=False, force_skip_shadows=True)
            else:
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
                glReadBuffer(GL_FRONT)
                renderer.render(scene, flags, is_first_pass=False, force_skip_shadows=True)
                normal_arr = renderer.jit.read_color_buf(self.viewport_height, self.viewport_width, rgba=False)
                normal_arr = renderer._resize_image(normal_arr, antialias=not seg)

            retval = (*retval, normal_arr)

            renderer._program_cache = old_cache

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
            raise ValueError("Unsupported PyOpenGL platform: {}".format(platform))
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
