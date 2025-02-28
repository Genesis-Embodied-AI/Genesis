"""A pyglet-based interactive 3D scene viewer."""

import copy
import os
import shutil
import sys
import time
from threading import Event, RLock, Semaphore, Thread

import imageio
import numpy as np
import OpenGL

import genesis as gs

import sys

if sys.platform.startswith("darwin"):
    # Mac OS
    from tkinter import Tk
    from tkinter import filedialog
else:
    try:
        from Tkinter import Tk
        from Tkinter import tkFileDialog as filedialog
    except Exception:
        try:
            from tkinter import Tk
            from tkinter import filedialog as filedialog
        except Exception:
            pass


try:
    root = Tk()
    root.withdraw()
except:
    pass

import pyglet
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from pyglet import clock

from .camera import IntrinsicsCamera, OrthographicCamera, PerspectiveCamera
from .constants import (
    DEFAULT_SCENE_SCALE,
    DEFAULT_Z_FAR,
    DEFAULT_Z_NEAR,
    MIN_OPEN_GL_MAJOR,
    MIN_OPEN_GL_MINOR,
    TARGET_OPEN_GL_MAJOR,
    TARGET_OPEN_GL_MINOR,
    TEXT_PADDING,
    RenderFlags,
    TextAlign,
)
from .light import DirectionalLight
from .node import Node
from .renderer import Renderer
from .shader_program import ShaderProgram, ShaderProgramCache
from .trackball import Trackball

pyglet.options["shadow_window"] = False


class Viewer(pyglet.window.Window):
    """An interactive viewer for 3D scenes.

    The viewer's camera is separate from the scene's, but will take on
    the parameters of the scene's main view camera and start in the same pose.
    If the scene does not have a camera, a suitable default will be provided.

    Parameters
    ----------
    scene : :class:`Scene`
        The scene to visualize.
    viewport_size : (2,) int
        The width and height of the initial viewing window.
    render_flags : dict
        A set of flags for rendering the scene. Described in the note below.
    viewer_flags : dict
        A set of flags for controlling the viewer's behavior.
        Described in the note below.
    registered_keys : dict
        A map from ASCII key characters to tuples containing:

        - A function to be called whenever the key is pressed,
          whose first argument will be the viewer itself.
        - (Optionally) A list of additional positional arguments
          to be passed to the function.
        - (Optionally) A dict of keyword arguments to be passed
          to the function.

    kwargs : dict
        Any keyword arguments left over will be interpreted as belonging to
        either the :attr:`.Viewer.render_flags` or :attr:`.Viewer.viewer_flags`
        dictionaries. Those flag sets will be updated appropriately.

    Note
    ----
    The basic commands for moving about the scene are given as follows:

    - **Rotating about the scene**: Hold the left mouse button and
      drag the cursor.
    - **Panning**:

      - Hold ALT, then hold the left mouse button and drag the cursor, or
      - Hold the middle mouse button and drag the cursor.

    - **Zooming**:

      - Scroll the mouse wheel, or
      - Hold the right mouse button and drag the cursor.

    Other keyboard commands are as follows:

    - ``a``: Toggles rotational animation mode.
    - ``c``: Toggles backface culling.
    - ``f``: Toggles fullscreen mode.
    - ``h``: Toggles shadow rendering.
    - ``i``: Toggles axis display mode.
    - ``l``: Toggles lighting mode
      (scene lighting, Raymond lighting, or direct lighting).
    - ``m``: Toggles face normal visualization.
    - ``n``: Toggles vertex normal visualization.
    - ``o``: Toggles orthographic mode.
    - ``q``: Quits the viewer.
    - ``r``: Starts recording a GIF, and pressing again stops recording
      and opens a file dialog.
    - ``s``: Opens a file dialog to save the current view as an image.
    - ``w``: Toggles wireframe mode
      (scene default, flip wireframes, all wireframe, or all solid).
    - ``z``: Resets the camera to the initial view.

    Note
    ----
    The valid keys for ``render_flags`` are as follows:

    - ``flip_wireframe``: `bool`, If `True`, all objects will have their
      wireframe modes flipped from what their material indicates.
      Defaults to `False`.
    - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
      in wireframe mode. Defaults to `False`.
    - ``all_solid``: `bool`, If `True`, all objects will be rendered in
      solid mode. Defaults to `False`.
    - ``shadows``: `bool`, If `True`, shadows will be rendered.
      Defaults to `False`.
    - ``vertex_normals``: `bool`, If `True`, vertex normals will be
      rendered as blue lines. Defaults to `False`.
    - ``face_normals``: `bool`, If `True`, face normals will be rendered as
      blue lines. Defaults to `False`.
    - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
      Defaults to `True`.
    - ``point_size`` : float, The point size in pixels. Defaults to 1px.

    Note
    ----
    The valid keys for ``viewer_flags`` are as follows:

    - ``rotate``: `bool`, If `True`, the scene's camera will rotate
      about an axis. Defaults to `False`.
    - ``rotate_rate``: `float`, The rate of rotation in radians per second.
      Defaults to `PI / 3.0`.
    - ``rotate_axis``: `(3,) float`, The axis in world coordinates to rotate
      about. Defaults to ``[0,0,1]``.
    - ``view_center``: `(3,) float`, The position to rotate the scene about.
      Defaults to the scene's centroid.
    - ``use_raymond_lighting``: `bool`, If `True`, an additional set of three
      directional lights that move with the camera will be added to the scene.
      Defaults to `False`.
    - ``use_direct_lighting``: `bool`, If `True`, an additional directional
      light that moves with the camera and points out of it will be added to
      the scene. Defaults to `False`.
    - ``lighting_intensity``: `float`, The overall intensity of the
      viewer's additional lights (when they're in use). Defaults to 3.0.
    - ``use_perspective_cam``: `bool`, If `True`, a perspective camera will
      be used. Otherwise, an orthographic camera is used. Defaults to `True`.
    - ``save_directory``: `str`, A directory to open the file dialogs in.
      Defaults to `None`.
    - ``window_title``: `str`, A title for the viewer's application window.
      Defaults to `"Scene Viewer"`.
    - ``refresh_rate``: `float`, A refresh rate for rendering, in Hertz.
      Defaults to `30.0`.
    - ``fullscreen``: `bool`, Whether to make viewer fullscreen.
      Defaults to `False`.
    - ``caption``: `list of dict`, Text caption(s) to display on the viewer.
      Defaults to `None`.

    Note
    ----
    Animation can be accomplished by running the viewer with ``run_in_thread``
    enabled. Then, just run a loop in your main thread, updating the scene as
    needed. Before updating the scene, be sure to acquire the
    :attr:`.Viewer.render_lock`, and release it  when your update is done.
    """

    def __init__(
        self,
        context,
        viewport_size=None,
        render_flags=None,
        viewer_flags=None,
        registered_keys=None,
        run_in_thread=False,
        auto_start=True,
        shadow=False,
        plane_reflection=False,
        env_separate_rigid=False,
        **kwargs,
    ):
        #######################################################################
        # Save attributes and flags
        #######################################################################
        if viewport_size is None:
            viewport_size = (640, 480)
        self.gs_context = context
        self._scene = context._scene
        self._viewport_size = viewport_size
        self._render_lock = RLock()
        self._offscreen_result_semaphore = Semaphore(0)
        self._offscreen_event = Event()
        self._initialized_event = Event()
        self._is_active = False
        self._run_in_thread = run_in_thread
        self._seg_node_map = context.seg_node_map

        self._video_saver = None

        self._default_render_flags = {
            "flip_wireframe": False,
            "all_wireframe": False,
            "all_solid": False,
            "shadows": shadow,
            "plane_reflection": plane_reflection,
            "env_separate_rigid": env_separate_rigid,
            "vertex_normals": False,
            "face_normals": False,
            "cull_faces": True,
            "offscreen": False,
            "point_size": 1.0,
            "seg": False,
            "depth": False,
        }
        self._default_viewer_flags = {
            "mouse_pressed": False,
            "rotate": False,
            "rotate_rate": np.pi / 3.0,
            "rotate_axis": np.array([0.0, 0.0, 1.0]),
            "view_center": None,
            "record": False,
            "use_raymond_lighting": False,
            "use_direct_lighting": False,
            "lighting_intensity": 3.0,
            "use_perspective_cam": True,
            "save_directory": None,
            "window_title": "Scene Viewer",
            "refresh_rate": 30.0,
            "fullscreen": False,
            "caption": None,
        }
        self._render_flags = self._default_render_flags.copy()
        self._viewer_flags = self._default_viewer_flags.copy()
        self._viewer_flags["rotate_axis"] = self._default_viewer_flags["rotate_axis"].copy()

        if render_flags is not None:
            self._render_flags.update(render_flags)
        if viewer_flags is not None:
            self._viewer_flags.update(viewer_flags)

        for key in kwargs:
            if key in self.render_flags:
                self._render_flags[key] = kwargs[key]
            elif key in self.viewer_flags:
                self._viewer_flags[key] = kwargs[key]

        # # TODO MAC OS BUG FOR SHADOWS
        # if sys.platform == 'darwin':
        #     self._render_flags['shadows'] = False

        self._registered_keys = {}
        if registered_keys is not None:
            self._registered_keys = {ord(k.lower()): registered_keys[k] for k in registered_keys}

        #######################################################################
        # Save internal settings
        #######################################################################

        # Set up caption stuff
        self._message_text = None
        self._ticks_till_fade = 2.0 / 3.0 * self.viewer_flags["refresh_rate"]
        self._message_opac = 1.0 + self._ticks_till_fade

        self._display_instr = False
        self._instr_texts = [
            ["> [i]: show keyboard instructions"],
            [
                "< [i]: hide keyboard instructions",
                "     [r]: record video",
                "     [s]: save image",
                "     [z]: reset camera",
                "     [a]: camera rotation",
                "     [h]: shadow",
                "     [f]: face normal",
                "     [v]: vertex normal",
                "     [w]: world frame",
                "     [l]: link frame",
                "     [d]: wireframe",
                "     [c]: camera & frustrum",
                "   [F11]: full-screen mode",
            ],
        ]

        # Set up raymond lights and direct lights
        self._raymond_lights = self._create_raymond_lights()
        self._direct_light = self._create_direct_light()

        #######################################################################
        # Set up camera node
        #######################################################################
        self._camera_node = None
        self._prior_main_camera_node = None
        self._default_camera_pose = None
        self._default_persp_cam = None
        self._default_orth_cam = None
        self._trackball = None

        # Extract main camera from scene and set up our mirrored copy
        znear = None
        zfar = None
        if self.scene.main_camera_node is not None:
            n = self.scene.main_camera_node
            camera = copy.copy(n.camera)
            if isinstance(camera, (PerspectiveCamera, IntrinsicsCamera)):
                self._default_persp_cam = camera
                znear = camera.znear
                zfar = camera.zfar
            elif isinstance(camera, OrthographicCamera):
                self._default_orth_cam = camera
                znear = camera.znear
                zfar = camera.zfar
            self._default_camera_pose = self.scene.get_pose(self.scene.main_camera_node)
            self._prior_main_camera_node = n

        # Set defaults as needed
        if zfar is None:
            zfar = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
        if znear is None or znear == 0:
            if self.scene.scale == 0:
                znear = DEFAULT_Z_NEAR
            else:
                znear = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)

        if self._default_persp_cam is None:
            self._default_persp_cam = PerspectiveCamera(yfov=np.pi / 3.0, znear=znear, zfar=zfar)
        if self._default_orth_cam is None:
            xmag = ymag = self.scene.scale
            if self.scene.scale == 0:
                xmag = ymag = 1.0
            self._default_orth_cam = OrthographicCamera(xmag=xmag, ymag=ymag, znear=znear, zfar=zfar)
        if self._default_camera_pose is None:
            self._default_camera_pose = self._compute_initial_camera_pose()

        # Pick camera
        if self.viewer_flags["use_perspective_cam"]:
            camera = self._default_persp_cam
        else:
            camera = self._default_orth_cam

        self._camera_node = Node(matrix=self._default_camera_pose, camera=camera)
        self.scene.add_node(self._camera_node)
        self.scene.main_camera_node = self._camera_node
        self._reset_view()

        #######################################################################
        # Initialize OpenGL context and renderer
        #######################################################################
        self._renderer = Renderer(
            self._viewport_size[0], self._viewport_size[1], context.jit, self.render_flags["point_size"]
        )
        self._is_active = True

        self.pending_offscreen_camera = None
        self.offscreen_result = None

        self.pending_buffer_updates = {}

        # Starting the viewer would raise an exception if the OpenGL context is invalid for some reason. This exception
        # must be caught in order to implement some fallback mechanism. One may want to start the viewer from the main
        # thread while the running loop would be running on a background thread. However, this approach is not possible
        # because all access to the OpenGL context must be done from the thread that created it in the first place. As
        # a result, the logic for catching an invalid OpenGL context must be implemented at the thread-level.
        self.auto_start = auto_start
        if self.run_in_thread:
            self._initialized_event.clear()
            self._thread = Thread(target=self.start, daemon=True)
            self._thread.start()
            self._initialized_event.wait()
            if not self._is_active:
                # TODO: For simplicity, the actual exception is not reported for now
                raise OpenGL.error.Error("Invalid OpenGL context.")
        else:
            self._thread = None
            if self.auto_start:
                self.start()

    @property
    def scene(self):
        """:class:`.Scene` : The scene being visualized."""
        return self._scene

    @property
    def viewport_size(self):
        """(2,) int : The width and height of the viewing window."""
        return self._viewport_size

    @property
    def render_lock(self):
        """:class:`threading.RLock` : If acquired, prevents the viewer from
        rendering until released.

        Run :meth:`.Viewer.render_lock.acquire` before making updates to
        the scene in a different thread, and run
        :meth:`.Viewer.render_lock.release` once you're done to let the viewer
        continue.
        """
        return self._render_lock

    @property
    def is_active(self):
        """bool : `True` if the viewer is active, or `False` if it has
        been closed.
        """
        return self._is_active

    @property
    def run_in_thread(self):
        """bool : Whether the viewer was run in a separate thread."""
        return self._run_in_thread

    @property
    def render_flags(self):
        """dict : Flags for controlling the renderer's behavior.

        - ``flip_wireframe``: `bool`, If `True`, all objects will have their
          wireframe modes flipped from what their material indicates.
          Defaults to `False`.
        - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
          in wireframe mode. Defaults to `False`.
        - ``all_solid``: `bool`, If `True`, all objects will be rendered in
          solid mode. Defaults to `False`.
        - ``shadows``: `bool`, If `True`, shadows will be rendered.
          Defaults to `False`.
        - ``vertex_normals``: `bool`, If `True`, vertex normals will be
          rendered as blue lines. Defaults to `False`.
        - ``face_normals``: `bool`, If `True`, face normals will be rendered as
          blue lines. Defaults to `False`.
        - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
          Defaults to `True`.
        - ``point_size`` : float, The point size in pixels. Defaults to 1px.

        """
        return self._render_flags

    @render_flags.setter
    def render_flags(self, value):
        self._render_flags = value

    @property
    def viewer_flags(self):
        """dict : Flags for controlling the viewer's behavior.

        The valid keys for ``viewer_flags`` are as follows:

        - ``rotate``: `bool`, If `True`, the scene's camera will rotate
          about an axis. Defaults to `False`.
        - ``rotate_rate``: `float`, The rate of rotation in radians per second.
          Defaults to `PI / 3.0`.
        - ``rotate_axis``: `(3,) float`, The axis in world coordinates to
          rotate about. Defaults to ``[0,0,1]``.
        - ``view_center``: `(3,) float`, The position to rotate the scene
          about. Defaults to the scene's centroid.
        - ``use_raymond_lighting``: `bool`, If `True`, an additional set of
          three directional lights that move with the camera will be added to
          the scene. Defaults to `False`.
        - ``use_direct_lighting``: `bool`, If `True`, an additional directional
          light that moves with the camera and points out of it will be
          added to the scene. Defaults to `False`.
        - ``lighting_intensity``: `float`, The overall intensity of the
          viewer's additional lights (when they're in use). Defaults to 3.0.
        - ``use_perspective_cam``: `bool`, If `True`, a perspective camera will
          be used. Otherwise, an orthographic camera is used. Defaults to
          `True`.
        - ``save_directory``: `str`, A directory to open the file dialogs in.
          Defaults to `None`.
        - ``window_title``: `str`, A title for the viewer's application window.
          Defaults to `"Scene Viewer"`.
        - ``refresh_rate``: `float`, A refresh rate for rendering, in Hertz.
          Defaults to `30.0`.
        - ``fullscreen``: `bool`, Whether to make viewer fullscreen.
          Defaults to `False`.
        - ``caption``: `list of dict`, Text caption(s) to display on
          the viewer. Defaults to `None`.

        """
        return self._viewer_flags

    @viewer_flags.setter
    def viewer_flags(self, value):
        self._viewer_flags = value

    @property
    def registered_keys(self):
        """dict : Map from ASCII key character to a handler function.

        This is a map from ASCII key characters to tuples containing:

        - A function to be called whenever the key is pressed,
          whose first argument will be the viewer itself.
        - (Optionally) A list of additional positional arguments
          to be passed to the function.
        - (Optionally) A dict of keyword arguments to be passed
          to the function.

        """
        return self._registered_keys

    @registered_keys.setter
    def registered_keys(self, value):
        self._registered_keys = value

    def close(self):
        """Close the viewer.

        This function will wait for the actual close, so you immediately
        manipulate the scene afterwards.
        """
        self.on_close()
        if self.run_in_thread:
            while self._is_active:
                time.sleep(1.0 / self.viewer_flags["refresh_rate"])

    def save_video(self, filename=None):
        """Save the stored frames to a video file.

        To use this asynchronously, run the viewer with the ``record``
        flag and the ``run_in_thread`` flags set.
        Kill the viewer after your desired time with
        :meth:`.Viewer.close_external`, and then call :meth:`.Viewer.save_video`.

        Parameters
        ----------
        filename : str
            The file to save the video to. If not specified,
            a file dialog will be opened to ask the user where
            to save the video file.
        """
        if filename is None:
            filename = self._get_save_filename(["mp4"])

        self.video_recorder.close()
        shutil.move(self.video_recorder.filename, filename)

    def on_close(self):
        """Exit the event loop when the window is closed."""
        # Always consider the viewer initialized at this point to avoid being stuck if starting fails
        if not self._initialized_event.is_set():
            self._initialized_event.set()

        # Early return if already closed
        if not self._is_active:
            return

        # Do not consider the viewer as active right away
        self._is_active = False

        # Remove our camera and restore the prior one
        try:
            if self._camera_node is not None:
                self.scene.remove_node(self._camera_node)
        except Exception:
            pass
        if self._prior_main_camera_node is not None:
            self.scene.main_camera_node = self._prior_main_camera_node

        # Delete any lighting nodes that we've attached
        if self.viewer_flags["use_raymond_lighting"]:
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)
        if self.viewer_flags["use_direct_lighting"]:
            if self.scene.has_node(self._direct_light):
                self.scene.remove_node(self._direct_light)

        # Delete renderer
        if self._renderer is not None:
            self._renderer.delete()
        self._renderer = None

        # Force clean-up of OpenGL context data
        try:
            OpenGL.contextdata.cleanupContext()
            super().close()
        except Exception:
            pass
        finally:
            super().on_close()
            pyglet.app.exit()

        self._offscreen_result_semaphore.release()

    def render_offscreen(self, camera_node, render_target, depth=False, seg=False, normal=False):
        if seg:
            self.render_flags["seg"] = True
        if depth:
            self.render_flags["depth"] = True
        self.pending_offscreen_camera = (camera_node, render_target, normal)
        if self.run_in_thread:
            # send_offscreen_request
            self._offscreen_event.set()
            # wait_for_offscreen
            self._offscreen_result_semaphore.acquire()
        else:
            # Force offscreen rendering synchronously
            self.draw_offscreen()
        if seg:
            self.render_flags["seg"] = False
        if depth:
            self.render_flags["depth"] = False
        return self.offscreen_result

    def update_buffers(self):
        self._renderer.jit.update_buffer(self.pending_buffer_updates)
        self.pending_buffer_updates = {}

    def wait_until_initialized(self):
        self._initialized_event.wait()

    def draw_offscreen(self):
        if self.pending_offscreen_camera is None:
            return

        if self.run_in_thread:
            self.render_lock.acquire()

        # Make OpenGL context current
        self.switch_to()
        self.update_buffers()

        self.offscreen_results = []
        self.render_flags["offscreen"] = True
        camera, target, normal = self.pending_offscreen_camera
        self.clear()
        retval = self._render(camera, target, normal)
        self.offscreen_result = retval if retval else [None, None]
        self.pending_offscreen_camera = None
        self.render_flags["offscreen"] = False
        self._offscreen_result_semaphore.release()

        if self.run_in_thread:
            self.render_lock.release()

    def on_draw(self):
        """Redraw the scene into the viewing window."""
        if self._renderer is None:
            return

        if self.run_in_thread or not self.auto_start:
            self.render_lock.acquire()

        # Make OpenGL context current
        self.switch_to()
        self.update_buffers()

        # Render the scene
        self.clear()
        self._render()

        if not self._initialized_event.is_set():
            self._initialized_event.set()

        if self._display_instr:
            self._renderer.render_texts(
                self._instr_texts[1],
                TEXT_PADDING,
                self.viewport_size[1] - TEXT_PADDING,
                font_pt=26,
                color=np.array([1.0, 1.0, 1.0, 0.85]),
            )
        else:
            self._renderer.render_texts(
                self._instr_texts[0],
                TEXT_PADDING,
                self.viewport_size[1] - TEXT_PADDING,
                font_pt=26,
                color=np.array([1.0, 1.0, 1.0, 0.85]),
            )

        if self._message_text is not None:
            self._renderer.render_text(
                self._message_text,
                self.viewport_size[0] - TEXT_PADDING,
                TEXT_PADDING,
                font_pt=20,
                color=np.array([0.1, 0.7, 0.2, np.clip(self._message_opac, 0.0, 1.0)]),
                align=TextAlign.BOTTOM_RIGHT,
            )

        if self.viewer_flags["caption"] is not None:
            for caption in self.viewer_flags["caption"]:
                xpos, ypos = self._location_to_x_y(caption["location"])
                self._renderer.render_text(
                    caption["text"],
                    xpos,
                    ypos,
                    font_name=caption["font_name"],
                    font_pt=caption["font_pt"],
                    color=caption["color"],
                    scale=caption["scale"],
                    align=caption["location"],
                )

        if self.run_in_thread or not self.auto_start:
            self.render_lock.release()

    def on_resize(self, width, height):
        """Resize the camera and trackball when the window is resized."""
        if self._renderer is None:
            return

        self._renderer._delete_shadow_framebuffer()
        self._renderer._delete_floor_framebuffer()

        self._viewport_size = (width, height)
        self._trackball.resize(self._viewport_size)
        self._renderer.viewport_width = self._viewport_size[0]
        self._renderer.viewport_height = self._viewport_size[1]
        self.on_draw()

    def on_mouse_press(self, x, y, buttons, modifiers):
        """Record an initial mouse press."""
        self._trackball.set_state(Trackball.STATE_ROTATE)
        if buttons == pyglet.window.mouse.LEFT:
            ctrl = modifiers & pyglet.window.key.MOD_CTRL
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            alt = modifiers & pyglet.window.key.MOD_ALT
            if ctrl:
                self._trackball.set_state(Trackball.STATE_ZOOM)
            elif alt or shift:
                self._trackball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.MIDDLE:
            self._trackball.set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.RIGHT:
            self._trackball.set_state(Trackball.STATE_ZOOM)

        self._trackball.down(np.array([x, y]))

        # Stop animating while using the mouse
        self.viewer_flags["mouse_pressed"] = True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Record a mouse drag."""
        self._trackball.drag(np.array([x, y]))

    def on_mouse_release(self, x, y, button, modifiers):
        """Record a mouse release."""
        self.viewer_flags["mouse_pressed"] = False

    def on_mouse_scroll(self, x, y, dx, dy):
        """Record a mouse scroll."""
        if self.viewer_flags["use_perspective_cam"]:
            self._trackball.scroll(dy)
        else:
            spfc = 0.95
            spbc = 1.0 / 0.95
            sf = 1.0
            if dy > 0:
                sf = spfc * dy
            elif dy < 0:
                sf = -spbc * dy

            c = self._camera_node.camera
            xmag = max(c.xmag * sf, 1e-8)
            ymag = max(c.ymag * sf, 1e-8 * c.ymag / c.xmag)
            c.xmag = xmag
            c.ymag = ymag

    def on_key_press(self, symbol, modifiers):
        """Record a key press."""
        # First, check for registered key callbacks
        if symbol in self.registered_keys:
            tup = self.registered_keys[symbol]
            callback = None
            args = []
            kwargs = {}
            if not isinstance(tup, (list, tuple, np.ndarray)):
                callback = tup
            else:
                callback = tup[0]
                if len(tup) == 2:
                    args = tup[1]
                if len(tup) == 3:
                    kwargs = tup[2]
            callback(self, *args, **kwargs)
            return

        # Otherwise, use default key functions

        # A causes the frame to rotate
        self._message_text = None
        if symbol == pyglet.window.key.A:
            self.viewer_flags["rotate"] = not self.viewer_flags["rotate"]
            if self.viewer_flags["rotate"]:
                self._message_text = "Rotation On"
            else:
                self._message_text = "Rotation Off"

        # F11 toggles face normals
        elif symbol == pyglet.window.key.F11:
            self.viewer_flags["fullscreen"] = not self.viewer_flags["fullscreen"]
            self.set_fullscreen(self.viewer_flags["fullscreen"])
            self.activate()
            if self.viewer_flags["fullscreen"]:
                self._message_text = "Fullscreen On"
            else:
                self._message_text = "Fullscreen Off"

        # H toggles shadows
        elif symbol == pyglet.window.key.H:
            self.render_flags["shadows"] = not self.render_flags["shadows"]
            if self.render_flags["shadows"]:
                self._message_text = "Shadows On"
            else:
                self._message_text = "Shadows Off"

        # W toggles world frame
        elif symbol == pyglet.window.key.W:
            if not self.gs_context.world_frame_shown:
                self.gs_context.on_world_frame()
                self._message_text = "World Frame On"
            else:
                self.gs_context.off_world_frame()
                self._message_text = "World Frame Off"

        # L toggles link frame
        elif symbol == pyglet.window.key.L:
            if not self.gs_context.link_frame_shown:
                self.gs_context.on_link_frame()
                self._message_text = "Link Frame On"
            else:
                self.gs_context.off_link_frame()
                self._message_text = "Link Frame Off"

        # C toggles camera frustum
        elif symbol == pyglet.window.key.C:
            if not self.gs_context.camera_frustum_shown:
                self.gs_context.on_camera_frustum()
                self._message_text = "Camera Frustrum On"
            else:
                self.gs_context.off_camera_frustum()
                self._message_text = "Camera Frustrum Off"

        # F toggles face normals
        elif symbol == pyglet.window.key.F:
            self.render_flags["face_normals"] = not self.render_flags["face_normals"]
            if self.render_flags["face_normals"]:
                self._message_text = "Face Normals On"
            else:
                self._message_text = "Face Normals Off"

        # V toggles vertex normals
        elif symbol == pyglet.window.key.V:
            self.render_flags["vertex_normals"] = not self.render_flags["vertex_normals"]
            if self.render_flags["vertex_normals"]:
                self._message_text = "Vert Normals On"
            else:
                self._message_text = "Vert Normals Off"

        # R starts recording frames
        elif symbol == pyglet.window.key.R:
            if self.viewer_flags["record"]:
                self.save_video()
                self.set_caption(self.viewer_flags["window_title"])
            else:
                self.video_recorder = FFMPEG_VideoWriter(
                    filename=os.path.join(gs.utils.misc.get_cache_dir(), "tmp_video.mp4"),
                    fps=self.viewer_flags["refresh_rate"],
                    size=self.viewport_size,
                )
                self.set_caption("{} (RECORDING)".format(self.viewer_flags["window_title"]))
            self.viewer_flags["record"] = not self.viewer_flags["record"]

        # S saves the current frame as an image
        elif symbol == pyglet.window.key.S:
            self._save_image()

        # T toggles through geom types
        # elif symbol == pyglet.window.key.T:
        #     if self.gs_context.rigid_shown == 'visual':
        #         self.gs_context.on_rigid('collision')
        #         self._message_text = "Geom Type: 'collision'"
        #     elif self.gs_context.rigid_shown == 'collision':
        #         self.gs_context.on_rigid('sdf')
        #         self._message_text = "Geom Type: 'sdf'"
        #     else:
        #         self.gs_context.on_rigid('visual')
        #         self._message_text = "Geom Type: 'visual'"

        # D toggles through wireframe modes
        elif symbol == pyglet.window.key.D:
            if self.render_flags["flip_wireframe"]:
                self.render_flags["flip_wireframe"] = False
                self.render_flags["all_wireframe"] = True
                self.render_flags["all_solid"] = False
                self._message_text = "All Wireframe"
            elif self.render_flags["all_wireframe"]:
                self.render_flags["flip_wireframe"] = False
                self.render_flags["all_wireframe"] = False
                self.render_flags["all_solid"] = True
                self._message_text = "All Solid"
            elif self.render_flags["all_solid"]:
                self.render_flags["flip_wireframe"] = False
                self.render_flags["all_wireframe"] = False
                self.render_flags["all_solid"] = False
                self._message_text = "Default Wireframe"
            else:
                self.render_flags["flip_wireframe"] = True
                self.render_flags["all_wireframe"] = False
                self.render_flags["all_solid"] = False
                self._message_text = "Flip Wireframe"

        # Z resets the camera viewpoint
        elif symbol == pyglet.window.key.Z:
            self._reset_view()

        # i toggles instruction display
        elif symbol == pyglet.window.key.I:
            self._display_instr = not self._display_instr

        elif symbol == pyglet.window.key.P:
            self._renderer.reload_program()

        if self._message_text is not None:
            self._message_opac = 1.0 + self._ticks_till_fade

    @staticmethod
    def _time_event(dt, self):
        """The timer callback."""
        # Don't run old dead events after we've already closed
        if not self._is_active:
            return

        if self.viewer_flags["record"]:
            self._record()
        if self.viewer_flags["rotate"] and not self.viewer_flags["mouse_pressed"]:
            self._rotate()

        # Manage message opacity
        if self._message_text is not None:
            if self._message_opac > 1.0:
                self._message_opac -= 1.0
            else:
                self._message_opac *= 0.90
            if self._message_opac < 0.05:
                self._message_opac = 1.0 + self._ticks_till_fade
                self._message_text = None

        # video saving warning
        if self._video_saver is not None:
            if self._video_saver.is_alive():
                self._message_text = "Saving video... Please don't exit."
                self._message_opac = 1.0
            else:
                self._message_text = f"Video saved to {self._video_file_name}"
                self._message_opac = self.viewer_flags["refresh_rate"] * 2
                self._video_saver = None

        self.on_draw()

    def _reset_view(self):
        """Reset the view to a good initial state.

        The view is initially along the positive x-axis at a
        sufficient distance from the scene.
        """
        # scale = self.scene.scale
        # if scale == 0.0:
        #     scale = DEFAULT_SCENE_SCALE
        scale = DEFAULT_SCENE_SCALE
        centroid = self.scene.centroid

        if self.viewer_flags["view_center"] is not None:
            centroid = self.viewer_flags["view_center"]

        self._camera_node.matrix = self._default_camera_pose
        self._trackball = Trackball(self._default_camera_pose, self.viewport_size, scale, centroid)

    def _get_save_filename(self, file_exts):
        file_types = {
            "mp4": ("video files", "*.mp4"),
            "png": ("png files", "*.png"),
            "jpg": ("jpeg files", "*.jpg"),
            "gif": ("gif files", "*.gif"),
            "all": ("all files", "*"),
        }
        filetypes = [file_types[x] for x in file_exts]
        try:
            save_dir = self.viewer_flags["save_directory"]
            if save_dir is None:
                save_dir = os.getcwd()
            filename = filedialog.asksaveasfilename(
                initialdir=save_dir, title="Select file save location", filetypes=filetypes
            )
        except Exception:
            return None

        if filename == ():
            return None
        return filename

    def _save_image(self):
        filename = self._get_save_filename(["png", "jpg", "gif", "all"])
        if filename is not None:
            self.viewer_flags["save_directory"] = os.path.dirname(filename)
            imageio.imwrite(filename, self._renderer.read_color_buf())

    def _record(self):
        """Save another frame for the GIF."""
        data = self._renderer.read_color_buf()
        if not np.all(data == 0.0):
            self.video_recorder.write_frame(data)

    def _rotate(self):
        """Animate the scene by rotating the camera."""
        az = self.viewer_flags["rotate_rate"] / self.viewer_flags["refresh_rate"]
        self._trackball.rotate(az, self.viewer_flags["rotate_axis"])

    def _render(self, camera_node=None, renderer=None, normal=False):
        """Render the scene into the framebuffer and flip."""
        scene = self.scene
        self._camera_node.matrix = self._trackball.pose.copy()

        if renderer is None:
            renderer = self._renderer

        if camera_node is not None:
            saved_camera_node = self.scene.main_camera_node
            self.scene.main_camera_node = camera_node

        # Set lighting
        vli = self.viewer_flags["lighting_intensity"]
        if self.viewer_flags["use_raymond_lighting"]:
            for n in self._raymond_lights:
                n.light.intensity = vli / 3.0
                if not self.scene.has_node(n):
                    scene.add_node(n, parent_node=self._camera_node)
        else:
            self._direct_light.light.intensity = vli
            for n in self._raymond_lights:
                if self.scene.has_node(n):
                    self.scene.remove_node(n)

        if self.viewer_flags["use_direct_lighting"]:
            if not self.scene.has_node(self._direct_light):
                scene.add_node(self._direct_light, parent_node=self._camera_node)
        elif self.scene.has_node(self._direct_light):
            self.scene.remove_node(self._direct_light)

        if normal:

            class CustomShaderCache:
                def __init__(self):
                    self.program = None

                def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
                    if self.program is None:
                        absolute_path = os.path.abspath(__file__)
                        print(absolute_path)
                        self.program = ShaderProgram(
                            os.path.join(absolute_path.replace("viewer.py", ""), "shaders/mesh_normal.vert"),
                            os.path.join(absolute_path.replace("viewer.py", ""), "shaders/mesh_normal.frag"),
                            defines=defines,
                        )
                    return self.program

            renderer._program_cache = CustomShaderCache()

            flags = RenderFlags.FLAT | RenderFlags.OFFSCREEN
            if self.render_flags["env_separate_rigid"]:
                flags |= RenderFlags.ENV_SEPARATE

            retval = renderer.render(scene, flags)
            renderer._program_cache = ShaderProgramCache()

        else:
            flags = RenderFlags.NONE
            if self.render_flags["flip_wireframe"]:
                flags |= RenderFlags.FLIP_WIREFRAME
            elif self.render_flags["all_wireframe"]:
                flags |= RenderFlags.ALL_WIREFRAME
            elif self.render_flags["all_solid"]:
                flags |= RenderFlags.ALL_SOLID

            if self.render_flags["shadows"]:
                flags |= RenderFlags.SHADOWS_ALL
            if self.render_flags["plane_reflection"]:
                flags |= RenderFlags.REFLECTIVE_FLOOR
            if self.render_flags["env_separate_rigid"]:
                flags |= RenderFlags.ENV_SEPARATE
            if self.render_flags["vertex_normals"]:
                flags |= RenderFlags.VERTEX_NORMALS
            if self.render_flags["face_normals"]:
                flags |= RenderFlags.FACE_NORMALS
            if not self.render_flags["cull_faces"]:
                flags |= RenderFlags.SKIP_CULL_FACES

            if self.render_flags["offscreen"]:
                flags |= RenderFlags.OFFSCREEN

            seg_node_map = None
            if self.render_flags["seg"]:
                flags |= RenderFlags.SEG
                seg_node_map = self._seg_node_map

            if self.render_flags["depth"]:
                flags |= RenderFlags.RET_DEPTH

            retval = renderer.render(self.scene, flags, seg_node_map=seg_node_map)

        if camera_node is not None:
            self.scene.main_camera_node = saved_camera_node

        return retval

    def start(self, auto_refresh=True):
        # Try multiple configs starting with target OpenGL version
        # and multisampling and removing these options if exception
        # Note: multisampling not available on all hardware
        from pyglet.gl import Config

        confs = [
            Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True,
                major_version=TARGET_OPEN_GL_MAJOR,
                minor_version=TARGET_OPEN_GL_MINOR,
            ),
            Config(
                depth_size=24,
                double_buffer=True,
                major_version=TARGET_OPEN_GL_MAJOR,
                minor_version=TARGET_OPEN_GL_MINOR,
            ),
            Config(
                sample_buffers=1,
                samples=4,
                depth_size=24,
                double_buffer=True,
                major_version=MIN_OPEN_GL_MAJOR,
                minor_version=MIN_OPEN_GL_MINOR,
            ),
            Config(depth_size=24, double_buffer=True, major_version=MIN_OPEN_GL_MAJOR, minor_version=MIN_OPEN_GL_MINOR),
        ]
        for conf in confs:
            # Keep the window invisible for now. It will be displayed only if everything is working fine.
            # This approach avoids "flickering" when creating and closing an invalid context. Besides, it avoids
            # "frozen" graphical window during compilation that would be interpreted as as bug by the end-user.
            try:
                super(Viewer, self).__init__(
                    config=conf,
                    visible=False,
                    resizable=True,
                    width=self._viewport_size[0],
                    height=self._viewport_size[1],
                )
                break
            except (pyglet.window.NoSuchConfigException, pyglet.gl.ContextException):
                pass

        if not self.context:
            raise ValueError("Unable to initialize an OpenGL 3+ context")
        clock.schedule_interval(Viewer._time_event, 1.0 / self.viewer_flags["refresh_rate"], self)
        self.switch_to()
        self.set_caption(self.viewer_flags["window_title"])

        # Model the complete scene once, to make sure that everything is fine.
        try:
            self.refresh()
        except OpenGL.error.Error:
            # Invalid OpenGL context. Closing before raising.
            self.close()
            return

        # At this point, we are all set to display the graphical window, finally!
        self.set_visible(True)
        self.activate()

        if auto_refresh:
            while self._is_active:
                try:
                    self.refresh()
                except AttributeError:
                    # The graphical window has been closed
                    self.on_close()
        else:
            self.refresh()

    def _run(self):
        while self._is_active:
            try:
                self.refresh()
            except AttributeError:
                # The graphical window has been closed
                self.on_close()

    def refresh(self):
        time_next_frame = time.time() + 1.0 / self.viewer_flags["refresh_rate"]
        while self._offscreen_event.wait(time_next_frame - time.time()):
            self.draw_offscreen()
            self._offscreen_event.clear()

        pyglet.clock.tick()

        if gs.platform != "Windows":
            pyglet.app.platform_event_loop.step(0.0)
        else:
            # even changing `platform_event_loop.step(0.0)` to 0.001 causes the viewer to hang on Windows
            # this is a workaround on Windows. not sure if it's correct
            time.sleep(0.001)

        self.switch_to()
        self.dispatch_pending_events()
        if self._is_active:
            self.dispatch_events()
        if self._is_active:
            self.flip()

    def _compute_initial_camera_pose(self):
        centroid = self.scene.centroid
        if self.viewer_flags["view_center"] is not None:
            centroid = self.viewer_flags["view_center"]
        scale = self.scene.scale
        if scale == 0.0:
            scale = DEFAULT_SCENE_SCALE

        s2 = 1.0 / np.sqrt(2.0)
        cp = np.eye(4)
        cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
        hfov = np.pi / 6.0
        dist = scale / (2.0 * np.tan(hfov))
        cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid

        return cp

    def _create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(Node(light=DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix))

        return nodes

    def _create_direct_light(self):
        light = DirectionalLight(color=np.ones(3), intensity=1.0)
        n = Node(light=light, matrix=np.eye(4))
        return n

    def _location_to_x_y(self, location):
        if location == TextAlign.CENTER:
            return (self.viewport_size[0] / 2.0, self.viewport_size[1] / 2.0)
        elif location == TextAlign.CENTER_LEFT:
            return (TEXT_PADDING, self.viewport_size[1] / 2.0)
        elif location == TextAlign.CENTER_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, self.viewport_size[1] / 2.0)
        elif location == TextAlign.BOTTOM_LEFT:
            return (TEXT_PADDING, TEXT_PADDING)
        elif location == TextAlign.BOTTOM_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, TEXT_PADDING)
        elif location == TextAlign.BOTTOM_CENTER:
            return (self.viewport_size[0] / 2.0, TEXT_PADDING)
        elif location == TextAlign.TOP_LEFT:
            return (TEXT_PADDING, self.viewport_size[1] - TEXT_PADDING)
        elif location == TextAlign.TOP_RIGHT:
            return (self.viewport_size[0] - TEXT_PADDING, self.viewport_size[1] - TEXT_PADDING)
        elif location == TextAlign.TOP_CENTER:
            return (self.viewport_size[0] / 2.0, self.viewport_size[1] - TEXT_PADDING)


__all__ = ["Viewer"]
