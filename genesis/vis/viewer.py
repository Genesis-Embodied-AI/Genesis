import os
import threading
import importlib
from typing import TYPE_CHECKING

import numpy as np
import OpenGL.error
import OpenGL.platform

import genesis as gs
import genesis.utils.geom as gu

from genesis.ext import pyrender
from genesis.repr_base import RBC
from genesis.utils.tools import Rate
from genesis.utils.misc import redirect_libc_stderr

if TYPE_CHECKING:
    from genesis.options.vis import ViewerOptions


class ViewerLock:
    def __init__(self, pyrender_viewer):
        self._pyrender_viewer = pyrender_viewer

    def __enter__(self):
        self._pyrender_viewer.render_lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self._pyrender_viewer.render_lock.release()


class Viewer(RBC):
    def __init__(self, options: "ViewerOptions", context):
        self._res = options.res
        self._run_in_thread = options.run_in_thread
        self._refresh_rate = options.refresh_rate
        self._max_FPS = options.max_FPS
        self._camera_init_pos = np.asarray(options.camera_pos, dtype=gs.np_float)
        self._camera_init_lookat = np.asarray(options.camera_lookat, dtype=gs.np_float)
        self._camera_up = np.asarray(options.camera_up, dtype=gs.np_float)
        self._camera_fov = options.camera_fov
        self._enable_interaction = options.enable_interaction

        if options.enable_interaction and gs.backend != gs.cpu:
            gs.logger.warning("Interaction code is slow on GPU. Switch to CPU backend or disable interaction.")

        self._pyrender_viewer = None
        self.context = context

        self._followed_entity = None
        self._follow_fixed_axis = None
        self._follow_smoothing = None
        self._follow_fix_orientation = None
        self._follow_lookat = None

        if self._max_FPS is not None:
            self.rate = Rate(self._max_FPS)

    def build(self, scene):
        self.scene = scene

        # set viewer camera
        self.setup_camera()

        # Try all candidate onscreen OpenGL "platforms" if none is specifically requested
        opengl_platform_orig = os.environ.get("PYOPENGL_PLATFORM")
        if opengl_platform_orig is None:
            if gs.platform == "Windows":
                all_opengl_platforms = ("wgl",)  # same as "native"
            elif gs.platform == "Linux":
                # "native" is platform-specific ("egl" or "glx")
                all_opengl_platforms = ("native", "egl", "glx", "osmesa")
            else:
                all_opengl_platforms = ("native",)
        else:
            if gs.platform == "Windows" and opengl_platform_orig == "osmesa":
                gs.raise_exception("PYOPENGL_PLATFORM='osmesa' is not supported on Windows OS.")
            all_opengl_platforms = (opengl_platform_orig,)

        for i, platform in enumerate(all_opengl_platforms):
            # Force re-import OpenGL platform
            os.environ["PYOPENGL_PLATFORM"] = platform
            importlib.reload(OpenGL.platform)

            try:
                gs.logger.debug(f"Trying to create OpenGL Context for PYOPENGL_PLATFORM='{platform}'...")
                with open(os.devnull, "w") as stderr, redirect_libc_stderr(stderr):
                    self._pyrender_viewer = pyrender.Viewer(
                        context=self.context,
                        viewport_size=self._res,
                        run_in_thread=self._run_in_thread,
                        auto_start=False,
                        view_center=self._camera_init_lookat,
                        shadow=self.context.shadow,
                        plane_reflection=self.context.plane_reflection,
                        env_separate_rigid=self.context.env_separate_rigid,
                        enable_interaction=self._enable_interaction,
                        viewer_flags={
                            "window_title": f"Genesis {gs.__version__}",
                            "refresh_rate": self._refresh_rate,
                        },
                    )
                    if not self._run_in_thread:
                        self._pyrender_viewer.start(auto_refresh=False)
                    self._pyrender_viewer.wait_until_initialized()
                break
            except OpenGL.error.Error:
                # Invalid OpenGL context. Trying another platform if any...
                gs.logger.debug(f"Invalid OpenGL context.")

                # Clear broken OpenGL context if it went this far
                if self._pyrender_viewer is not None:
                    self._pyrender_viewer.close()
                    self._pyrender_viewer = None

                if i == len(all_opengl_platforms) - 1:
                    raise

            # Select PyOpenGL backend compatible with `pyrender.OffscreenRenderer`
            if platform not in ("osmesa", "pyglet", "egl"):
                os.environ["PYOPENGL_PLATFORM"] = "pyglet"

        self.lock = ViewerLock(self._pyrender_viewer)

        gs.logger.info(f"Viewer created. Resolution: ~<{self._res[0]}×{self._res[1]}>~, max_FPS: ~<{self._max_FPS}>~.")

    def run(self):
        if self._pyrender_viewer is None:
            gs.raise_exception("Viewer must be built successfully before calling this method.")
        self._pyrender_viewer.run()

    def stop(self):
        if self.is_alive():
            self._pyrender_viewer.close()

    def is_alive(self):
        return self._pyrender_viewer is not None and self._pyrender_viewer.is_active

    def setup_camera(self):
        yfov = self._camera_fov / 180.0 * np.pi
        pose = gu.pos_lookat_up_to_T(self._camera_init_pos, self._camera_init_lookat, self._camera_up)
        self._camera_node = self.context.add_node(pyrender.PerspectiveCamera(yfov=yfov), pose=pose)

    def update(self, auto_refresh=None):
        if self._followed_entity is not None:
            self.update_following()

        self._pyrender_viewer.update_on_sim_step()

        with self.lock:
            # Update context
            self.context.update()

            # Refresh viewer by default if and if this is possible
            if auto_refresh is None:
                viewer_thread = self._pyrender_viewer._thread or threading.main_thread()
                auto_refresh = viewer_thread == threading.current_thread()

            if auto_refresh and not self._pyrender_viewer.run_in_thread:
                self._pyrender_viewer.refresh()

        # lock FPS
        if self._max_FPS is not None:
            self.rate.sleep()

    def render_offscreen(self, camera_node, render_target, rgb=True, depth=False, seg=False, normal=False):
        return self._pyrender_viewer.render_offscreen(camera_node, render_target, rgb, depth, seg, normal)

    def set_camera_pose(self, pose=None, pos=None, lookat=None):
        """
        Set viewer camera pose.

        Parameters
        ----------
        pose : [4,4] float, optional
            Camera-to-world pose. If provided, `pos` and `lookat` will be ignored.
        pos : (3,) float, optional
            Camera position.
        lookat : (3,) float, optional
            Camera lookat point.
        """
        if pose is None:
            if pos is None:
                pos = self._camera_init_pos
            if lookat is None:
                lookat = self._camera_init_lookat
            up = self._camera_up

            pose = gu.pos_lookat_up_to_T(pos, lookat, up)
        else:
            if np.array(pose).shape != (4, 4):
                gs.raise_exception("pose should be a 4x4 matrix.")

        self._pyrender_viewer._trackball.set_camera_pose(pose)

    def follow_entity(self, entity, fixed_axis=(None, None, None), smoothing=None, fix_orientation=False):
        """
        Set the viewer to follow a specified entity.
        Parameters
        ----------
        entity : genesis.Entity
            The entity to follow.
        fixed_axis : (float, float, float), optional
            The fixed axis for the viewer's movement. For each axis, if None, the viewer will move freely. If a float, the viewer will be fixed on at that value.
            For example, [None, None, None] will allow the viewer to move freely while following, [None, None, 0.5] will fix the viewer's z-axis at 0.5.
        smoothing : float, optional
            The smoothing factor in ]0,1[ for the viewer's movement. If None, no smoothing will be applied.
        fix_orientation : bool, optional
            If True, the viewer will maintain its orientation relative to the world. If False, the viewer will look at the base link of the entity.
        """
        self._followed_entity = entity
        self._follow_fixed_axis = fixed_axis
        self._follow_smoothing = smoothing
        self._follow_fix_orientation = fix_orientation
        self._follow_lookat = self._camera_init_lookat

    def update_following(self):
        """
        Update the viewer position to follow the specified entity.
        """
        entity_pos = self._followed_entity.get_pos().cpu().numpy()
        if entity_pos.ndim > 1:  # check for multiple envs
            entity_pos = entity_pos[0]
        camera_transform = np.asarray(self._pyrender_viewer._trackball.pose, copy=True)
        camera_pos = np.array(self._pyrender_viewer._trackball.pose[:3, 3])

        if self._follow_smoothing is not None:
            # Smooth viewer movement with a low-pass filter
            camera_pos = self._follow_smoothing * camera_pos + (1 - self._follow_smoothing) * (
                entity_pos + self._camera_init_pos
            )
            self._follow_lookat = (
                self._follow_smoothing * self._follow_lookat + (1 - self._follow_smoothing) * entity_pos
            )
        else:
            camera_pos = entity_pos + self._camera_init_pos
            self._follow_lookat = entity_pos

        for i, fixed_axis in enumerate(self._follow_fixed_axis):
            # Fix the camera's position along the specified axis
            if fixed_axis is not None:
                camera_pos[i] = fixed_axis

        if self._follow_fix_orientation:
            # Keep the camera orientation fixed by overriding the lookat point
            camera_transform[:3, 3] = camera_pos
            self.set_camera_pose(pose=camera_transform)
        else:
            self.set_camera_pose(pos=camera_pos, lookat=self._follow_lookat)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def res(self):
        return self._res

    @property
    def refresh_rate(self):
        return self._refresh_rate

    @property
    def max_FPS(self):
        return self._max_FPS

    @property
    def camera_pos(self):
        """
        Get the camera's current position.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])

    @property
    def camera_lookat(self):
        """
        Get the camera's current lookat point.
        """
        pos = np.array(self._pyrender_viewer._trackball._n_pose[:3, 3])
        z = self._pyrender_viewer._trackball._n_pose[:3, 2]
        return pos - z

    @property
    def camera_pose(self):
        """
        Get the camera's current pose represented by a 4x4 matrix.
        """
        return np.array(self._pyrender_viewer._trackball._n_pose)

    @property
    def camera_up(self):
        return self._camera_up

    @property
    def camera_fov(self):
        return self._camera_fov
