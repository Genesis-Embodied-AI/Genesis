import enum
import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC

try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter
except ImportError as e:
    gs.raise_exception_from("Madrona batch renderer is only supported on Linux x86-64.", e)


class IMAGE_TYPE(enum.IntEnum):
    RGB = 0
    DEPTH = 3
    NORMAL = 1
    SEGMENTATION = 2


class Light:
    def __init__(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = dir
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff


class BatchRenderer(RBC):
    """
    This class is used to manage batch rendering
    """

    def __init__(self, visualizer, renderer_options):
        self._visualizer = visualizer
        self._lights = gs.List()
        self._renderer_options = renderer_options
        self._renderer = None

        self._data_cache = {}
        self._t = -1

    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if len(self._visualizer._cameras) == 0:
            raise ValueError("No cameras to render")

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")

        self._cameras = gs.List([camera for camera in self._visualizer._cameras if not camera.debug])
        lights = self._lights
        rigid = self._visualizer.scene.rigid_solver
        n_envs = max(self._visualizer.scene.n_envs, 1)
        res = self._cameras[0].res
        gpu_id = gs.device.index
        use_rasterizer = self._renderer_options.use_rasterizer

        # Cameras
        n_cameras = len(self._cameras)
        cameras_pos = torch.stack([camera.get_pos() for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in self._cameras], dim=1)
        cameras_fov = torch.tensor([camera.fov for camera in self._cameras], dtype=gs.tc_float, device=gs.device)

        # Build taichi arrays to store light properties once. If later we need to support dynamic lights, we should
        # consider storing light properties as taichi fields in Genesis.
        n_lights = len(lights)
        light_pos = torch.tensor([light.pos for light in self._lights], dtype=gs.tc_float)
        light_dir = torch.tensor([light.dir for light in self._lights], dtype=gs.tc_float)
        light_intensity = torch.tensor([light.intensity for light in self._lights], dtype=gs.tc_float)
        light_directional = torch.tensor([light.directional for light in self._lights], dtype=gs.tc_int)
        light_castshadow = torch.tensor([light.castshadow for light in self._lights], dtype=gs.tc_int)
        light_cutoff = torch.tensor([light.cutoffRad for light in self._lights], dtype=gs.tc_float)

        self._renderer = MadronaBatchRendererAdapter(
            rigid, gpu_id, n_envs, n_cameras, n_lights, cameras_fov, *res, False, use_rasterizer
        )
        self._renderer.init(
            rigid,
            cameras_pos,
            cameras_quat,
            light_pos,
            light_dir,
            light_intensity,
            light_directional,
            light_castshadow,
            light_cutoff,
        )

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, force_render=False, aliasing=False):
        """
        Render all cameras in the batch.

        Parameters
        ----------
        rgb : bool, optional
            Whether to render the rgb image.
        depth : bool, optional
            Whether to render the depth image.
        segmentation : bool, optional
            Whether to render the segmentation image.
        normal : bool, optional
            Whether to render the normal image.
        force_render : bool, optional
            Whether to force render the scene.
        aliasing : bool, optional
            Whether to apply anti-aliasing.

        Returns
        -------
        rgb_arr : tuple of tensors
            The sequence of rgb images associated with each camera.
        depth_arr : tuple of tensors
            The sequence of depth images associated with each camera.
        """
        if normal:
            raise NotImplementedError("Normal rendering not supported from now")
        if segmentation:
            raise NotImplementedError("Segmentation rendering not supported from now")

        # Clear cache if requested or necessary
        if force_render or self._t < self._visualizer.scene.t:
            self._data_cache.clear()

        # Fetch available cached data
        cache_key = (aliasing,)
        rgb_arr = self._data_cache.get((IMAGE_TYPE.RGB, cache_key), None)
        depth_arr = self._data_cache.get((IMAGE_TYPE.DEPTH, cache_key), None)

        # Force disabling rendering whenever cached data is already available
        rgb_ = rgb and rgb_arr is None
        depth_ = depth and depth_arr is None

        # Early return if there is nothing to do
        if not (rgb_ or depth_):
            return rgb_arr if rgb else None, depth_arr if depth else None, None, None

        # Update scene
        self.update_scene()

        # Render frame
        cameras_pos = torch.stack([camera.get_pos() for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in self._cameras], dim=1)
        render_options = np.array((rgb_, depth_, False, False, aliasing), dtype=np.uint32)
        rgba_arr_all, depth_arr_all = self._renderer.render(
            self._visualizer.scene.rigid_solver, cameras_pos, cameras_quat, render_options
        )

        # Post-processing: Remove alpha channel from RGBA, squeeze env dim if necessary, and split along camera dim
        buffers = [rgba_arr_all[..., :3], depth_arr_all]
        for i, data in enumerate(buffers):
            if data is not None:
                data = data.swapaxes(0, 1)
                if self._visualizer.scene.n_envs == 0:
                    data = data.squeeze(1)
                buffers[i] = tuple(data)

        # Update cache
        self._t = self._visualizer.scene.t
        if rgb_:
            rgb_arr = self._data_cache[(IMAGE_TYPE.RGB, cache_key)] = buffers[0]
        if depth_:
            depth_arr = self._data_cache[(IMAGE_TYPE.DEPTH, cache_key)] = buffers[1]

        return rgb_arr, depth_arr, None, None

    def destroy(self):
        self._lights.clear()
        self._data_cache.clear()
        if self._renderer is not None:
            del self._renderer.madrona
            self._renderer = None

    def reset(self):
        self._t = -1

    @property
    def lights(self):
        return self._lights

    @property
    def cameras(self):
        return self._cameras
