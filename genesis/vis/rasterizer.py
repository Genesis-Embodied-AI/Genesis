import os
import sys

import numpy as np

import OpenGL

import genesis as gs
from genesis.repr_base import RBC
from genesis.ext import pyrender


class Rasterizer(RBC):
    def __init__(self, viewer, context):
        self._viewer = viewer
        self._context = context
        self._camera_nodes = dict()
        self._camera_targets = dict()
        self._offscreen = self._viewer is None
        self._renderer = None
        self._buffer_updates = None

    def build(self):
        if self._context is None:
            return

        if self._offscreen:
            # Select PyOpenGL backend for `pyrender.OffscreenRenderer`.
            # If env variable is set, use specified platform if supported, otherwise some platform-specific default.
            platform = os.environ.get("PYOPENGL_PLATFORM", "egl" if gs.platform == "Linux" else "pyglet")
            if platform not in ("osmesa", "pyglet", "egl"):
                gs.logger.warning(f"PYOPENGL_PLATFORM='{platform}' not supported. Falling back to 'pyglet'.")
                platform = "pyglet"
            if sys.platform == "win32" and platform == "osmesa":
                gs.raise_exception("PYOPENGL_PLATFORM='osmesa' not supported on Windows OS. Falling back to 'pyglet'.")
                platform = "pyglet"

            # Start the viewer
            self._renderer = pyrender.OffscreenRenderer(
                pyopengl_platform=platform, seg_node_map=self._context.seg_node_map
            )

        self.visualizer = self._context.visualizer

    def add_camera(self, camera):
        self._camera_nodes[camera.uid] = self._context.add_node(
            pyrender.PerspectiveCamera(
                yfov=np.deg2rad(camera.fov),
                znear=camera.near,
                zfar=camera.far,
                aspectRatio=camera.aspect_ratio,
            ),
        )
        self._camera_targets[camera.uid] = pyrender.Renderer(camera.res[0], camera.res[1], self._context.jit)

    def update_camera(self, camera):
        self._camera_nodes[camera.uid].camera.yfov = np.deg2rad(camera.fov)
        self._context.set_node_pose(self._camera_nodes[camera.uid], camera.transform)
        self._context.update_camera_frustum(camera)

    def remove_camera(self, camera):
        self._context.removenode(self._camera_nodes[camera.uid])
        del self._camera_nodes[camera.uid]
        if self._offscreen:
            self._camera_targets[camera.uid].delete()
        else:
            self._viewer.close_offscreen(self._camera_targets[camera.uid])
        del self._camera_targets[camera.uid]

    def render_camera(self, camera, rgb=True, depth=False, segmentation=False, normal=False):
        rgb_arr, depth_arr, seg_idxc_arr, normal_arr = None, None, None, None
        if self._offscreen:
            # Set the context
            self._renderer.make_current()

            # Update the context if not already done before
            self._context.jit.update_buffer(self._context.buffer)
            self._context.buffer.clear()

            # Render
            try:
                if rgb or depth or normal:
                    retval = self._renderer.render(
                        self._context._scene,
                        self._camera_targets[camera.uid],
                        camera_node=self._camera_nodes[camera.uid],
                        env_separate_rigid=self._context.env_separate_rigid,
                        rgb=rgb,
                        normal=normal,
                        seg=False,
                        depth=depth,
                        plane_reflection=rgb and self._context.plane_reflection,
                        shadow=rgb and self._context.shadow,
                    )

                if segmentation:
                    seg_idxc_rgb_arr, *_ = self._renderer.render(
                        self._context._scene,
                        self._camera_targets[camera.uid],
                        camera_node=self._camera_nodes[camera.uid],
                        env_separate_rigid=self._context.env_separate_rigid,
                        rgb=False,
                        normal=False,
                        seg=True,
                        depth=False,
                        plane_reflection=False,
                        shadow=False,
                    )
            finally:
                # Unset the context
                self._renderer.make_uncurrent()
        else:
            # Render
            if rgb or depth or normal:
                retval = self._viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    rgb=rgb,
                    depth=depth,
                    normal=normal,
                    seg=False,
                )

            if segmentation:
                seg_idxc_rgb_arr, *_ = self._viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    rgb=False,
                    depth=False,
                    normal=False,
                    seg=True,
                )

        if segmentation:
            seg_idxc_arr = self._context.seg_idxc_rgb_arr_to_idxc_arr(seg_idxc_rgb_arr)

        if rgb:
            rgb_arr = retval[0]
        if depth:
            depth_arr = retval[int(rgb)]
        if normal:
            normal_arr = retval[int(rgb + depth)]
        return rgb_arr, depth_arr, seg_idxc_arr, normal_arr

    def update_scene(self, force_render: bool = False):
        self._context.update(force_render)

    def destroy(self):
        for node in self._camera_nodes.values():
            self._context.remove_node(node)
        self._camera_nodes.clear()
        for camera_target in self._camera_targets.values():
            try:
                if self._offscreen:
                    camera_target.delete()
                elif self._viewer is not None:
                    self._viewer.close_offscreen(camera_target)
            except OSError:
                pass
        self._camera_targets.clear()

        if self._offscreen and self._renderer is not None:
            try:
                self._renderer.make_current()
                self._renderer.delete()
            except (OpenGL.error.GLError, ImportError):
                pass
            del self._renderer
            self._renderer = None

    @property
    def viewer(self):
        return self._viewer

    @property
    def offscreen(self):
        return self._offscreen
