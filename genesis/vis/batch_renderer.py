import numpy as np
import torch
import taichi as ti

import genesis as gs
from genesis.repr_base import RBC

try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter
except ImportError as e:
    gs.raise_exception(f"Failed to import Madrona batch renderer. {e.__class__.__name__}: {e}")


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
        return np.deg2rad(self._cutoff)

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
        self._rgb_torch = None
        self._depth_torch = None
        self._last_t = -1

    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if len(self._visualizer._cameras) == 0:
            raise ValueError("No cameras to render")
        cameras = self._visualizer._cameras
        lights = self._lights
        rigid = self._visualizer.scene.rigid_solver
        device = torch.cuda.current_device()
        n_envs = self._visualizer.scene.n_envs if self._visualizer.scene.n_envs > 0 else 1
        res = cameras[0].res
        use_rasterizer = self._renderer_options.use_rasterizer

        # Cameras
        n_cameras = len(cameras)
        camera_pos = self._visualizer.camera_pos
        camera_quat = self._visualizer.camera_quat
        camera_fov = self._visualizer.camera_fov

        # Build taichi arrays to store light properties once.
        # If later we need to support dynamic lights, we should consider storing light properties as taichi fields in Genesis.
        n_lights = len(lights)
        light_pos = self.light_pos_tensor
        light_dir = self.light_dir_tensor
        light_intensity = self.light_intensity_tensor
        light_directional = self.light_directional_tensor
        light_castshadow = self.light_castshadow_tensor
        light_cutoff = self.light_cutoff_tensor

        self._renderer = MadronaBatchRendererAdapter(
            rigid, device, n_envs, n_cameras, n_lights, camera_fov, *res, False, use_rasterizer
        )
        self._renderer.init(
            rigid,
            camera_pos,
            camera_quat,
            light_pos,
            light_dir,
            light_intensity,
            light_directional,
            light_castshadow,
            light_cutoff,
        )

    def update_scene(self):
        self._visualizer._context.update()

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, force_render=False):
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

        Returns
        -------
        rgb_torch : tuple of tensors
            The rgb image.
        depth_torch : tuple of tensors
            The depth image.
        """
        if normal:
            raise NotImplementedError("Normal rendering is not implemented")
        if segmentation:
            raise NotImplementedError("Segmentation rendering is not implemented")

        if not force_render and self._last_t == self._visualizer.scene.t:
            return self._rgb_torch, self._depth_torch, None, None

        # Update last_t to current time to avoid re-rendering if the scene is not updated
        self._last_t = self._visualizer.scene.t
        self.update_scene()

        rigid = self._visualizer.scene.rigid_solver
        camera_pos = self._visualizer.camera_pos
        camera_quat = self._visualizer.camera_quat
        # TODO: Control whether to render rgb, depth, segmentation, normal separately
        rgb_torch, depth_torch = self._renderer.render(rigid, camera_pos, camera_quat)

        if rgb_torch is not None:
            if rgb_torch.ndim == 4:
                rgb_torch = rgb_torch.squeeze(0)
            self._rgb_torch = tuple(rgb_torch.swapaxes(0, 1))
        else:
            self._rgb_torch = None

        if depth_torch is not None:
            if depth_torch.ndim == 4:
                depth_torch = depth_torch.squeeze(0)
            self._depth_torch = tuple(depth_torch.swapaxes(0, 1))
        else:
            self._depth_torch = None

        return self._rgb_torch, self._depth_torch, None, None

    def destroy(self):
        self._lights.clear()
        self._rgb_torch = None
        self._depth_torch = None

    @property
    def lights(self):
        return self._lights

    def has_lights(self):
        return len(self._lights) > 0

    @property
    def light_pos_tensor(self):
        return torch.tensor([light.pos for light in self._lights], dtype=gs.tc_float)

    @property
    def light_dir_tensor(self):
        return torch.tensor([light.dir for light in self._lights], dtype=gs.tc_float)

    @property
    def light_intensity_tensor(self):
        return torch.tensor([light.intensity for light in self._lights], dtype=gs.tc_float)

    @property
    def light_directional_tensor(self):
        return torch.tensor([light.directional for light in self._lights], dtype=gs.tc_int)

    @property
    def light_castshadow_tensor(self):
        return torch.tensor([light.castshadow for light in self._lights], dtype=gs.tc_int)

    @property
    def light_cutoff_tensor(self):
        return torch.tensor([light.cutoffRad for light in self._lights], dtype=gs.tc_float)
