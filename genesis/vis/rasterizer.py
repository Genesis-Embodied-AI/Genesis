import os
import gc

import numpy as np

import genesis as gs
from genesis.repr_base import RBC


class Rasterizer(RBC):
    def __init__(self, viewer, context):
        self._viewer = viewer
        self._context = context
        self._camera_nodes = dict()
        self._camera_targets = dict()
        self._offscreen = self._viewer is None

    def build(self):
        if self._context is None:
            return

        if self._offscreen:
            from genesis.ext import pyrender

            # if environment variable is set, use the platform specified, otherwise some platform-specific default
            platform = os.environ.get("PYOPENGL_PLATFORM", "egl" if gs.platform == "Linux" else "pyglet")
            self._renderer = pyrender.OffscreenRenderer(
                pyopengl_platform=platform, seg_node_map=self._context.seg_node_map
            )

        self.visualizer = self._context.visualizer

    def add_camera(self, camera):
        from genesis.ext import pyrender

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

    def render_camera(self, camera, rgb=True, depth=False, segmentation=False, normal=False):

        if not self._offscreen:
            if rgb or depth:
                rgb_arr, depth_arr = self._viewer._pyrender_viewer.render_offscreen(
                    self._camera_nodes[camera.uid], self._camera_targets[camera.uid], depth=depth
                )

            if segmentation:
                seg_idxc_rgb_arr, _ = self._viewer._pyrender_viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    seg=True,
                )
                seg_idxc_arr = self._context.seg_idxc_rgb_arr_to_idxc_arr(seg_idxc_rgb_arr)

            if normal:
                normal_arr, _ = self._viewer._pyrender_viewer.render_offscreen(
                    self._camera_nodes[camera.uid],
                    self._camera_targets[camera.uid],
                    normal=True,
                )

        else:
            if rgb or depth:  # depth is always rendered together with rgb
                rgb_arr, depth_arr = self._renderer.render(
                    self._context._scene,
                    self._camera_targets[camera.uid],
                    camera_node=self._camera_nodes[camera.uid],
                    shadow=self._context.shadow,
                    plane_reflection=self._context.plane_reflection,
                    env_separate_rigid=self._context.env_separate_rigid,
                    ret_depth=depth,
                )

            if segmentation:
                seg_idxc_rgb_arr, _ = self._renderer.render(
                    self._context._scene,
                    self._camera_targets[camera.uid],
                    camera_node=self._camera_nodes[camera.uid],
                    env_separate_rigid=self._context.env_separate_rigid,
                    ret_depth=False,
                    seg=True,
                )
                seg_idxc_arr = self._context.seg_idxc_rgb_arr_to_idxc_arr(seg_idxc_rgb_arr)

            if normal:
                normal_arr = self._renderer.render(
                    self._context._scene,
                    self._camera_targets[camera.uid],
                    camera_node=self._camera_nodes[camera.uid],
                    env_separate_rigid=self._context.env_separate_rigid,
                    ret_depth=False,
                    normal=True,
                )[-1]

        if not rgb:
            rgb_arr = None

        if not depth:
            depth_arr = None

        if not segmentation:
            seg_idxc_arr = None

        if not normal:
            normal_arr = None

        return rgb_arr, depth_arr, seg_idxc_arr, normal_arr

    def update_scene(self):
        buffer_updates = self._context.update()
        self._context.jit.update_buffer(buffer_updates)

    def destroy(self):
        if self._offscreen:
            self._renderer._platform.make_current()
            for target in self._camera_targets:
                self._camera_targets[target].delete()
            self._renderer.delete()
            del self._renderer
            gc.collect()

    @property
    def viewer(self):
        return self._viewer

    @property
    def offscreen(self):
        return self._offscreen
