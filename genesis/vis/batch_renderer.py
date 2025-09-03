import math

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.constants import IMAGE_TYPE

try:
    from gs_madrona.renderer_gs import MadronaBatchRendererAdapter
except ImportError as e:
    gs.raise_exception_from("Madrona batch renderer is only supported on Linux x86-64.", e)


def _transform_camera_quat(quat):
    # quat for Madrona needs to be transformed to y-forward
    w, x, y, z = torch.unbind(quat, dim=-1)
    return torch.stack([x + w, x - w, y - z, y + z], dim=-1) / math.sqrt(2.0)


def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


class Light:
    def __init__(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._pos = pos
        self._dir = tuple(dir / np.linalg.norm(dir))
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
        self._use_rasterizer = renderer_options.use_rasterizer
        self._renderer = None
        self._data_cache = {}
        self._t = -1

    def add_light(self, pos, dir, intensity, directional, castshadow, cutoff):
        self._lights.append(Light(pos, dir, intensity, directional, castshadow, cutoff))

    def build(self):
        """
        Build all cameras in the batch and initialize Moderona renderer
        """
        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRenderer requires CUDA backend.")
        gpu_id = gs.device.index if gs.device.index is not None else 0

        # Extract the complete list of non-debug cameras
        self._cameras = gs.List([camera for camera in self._visualizer._cameras if not camera.debug])
        if not self._cameras:
            gs.raise_exception("Please add at least one camera when using BatchRender.")

        # Make sure that all cameras have identical resolution
        try:
            ((camera_width, camera_height),) = set(camera.res for camera in self._cameras)
        except ValueError as e:
            gs.raise_exception_from("All cameras must have the exact same resolution when using BatchRender.", e)

        self._renderer = MadronaBatchRendererAdapter(
            rigid=self._visualizer.scene.rigid_solver,
            gpu_id=gs.device.index if gs.device.index is not None else 0,
            num_worlds=max(self._visualizer.scene.n_envs, 1),
            num_cameras=len(self._cameras),
            num_lights=len(self._lights),
            cam_fovs_tensor=_make_tensor([camera.fov for camera in self._cameras]),
            batch_render_view_width=camera_width,
            batch_render_view_height=camera_height,
            add_cam_debug_geo=False,
            use_rasterizer=self._use_rasterizer,
        )
        self._renderer.init(
            rigid=self._visualizer.scene.rigid_solver,
            cam_pos_tensor=torch.stack([camera.get_pos() for camera in self._cameras], dim=1),
            cam_rot_tensor=_transform_camera_quat(torch.stack([camera.get_quat() for camera in self._cameras], dim=1)),
            lights_pos_tensor=_make_tensor([light.pos for light in self._lights]).reshape((-1, 3)),
            lights_dir_tensor=_make_tensor([light.dir for light in self._lights]).reshape((-1, 3)),
            lights_intensity_tensor=_make_tensor([light.intensity for light in self._lights]),
            lights_directional_tensor=_make_tensor([light.directional for light in self._lights], dtype=torch.bool),
            lights_castshadow_tensor=_make_tensor([light.castshadow for light in self._lights], dtype=torch.bool),
            lights_cutoff_tensor=_make_tensor([light.cutoffRad for light in self._lights]),
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
        request = (rgb, depth, segmentation, normal)
        cache_key = (aliasing,)
        cached = [self._data_cache.get((img_type, cache_key), None) for img_type in IMAGE_TYPE]

        # Force disabling rendering whenever cached data is already available
        needed = tuple(req and arr is None for req, arr in zip(request, cached))

        # Early return if everything requested is already cached
        if not any(needed):
            return tuple(arr if req else None for req, arr in zip(request, cached))

        # Update scene
        self.update_scene()

        # Render only what is needed (flags still passed to renderer)
        cameras_pos = torch.stack([camera.get_pos() for camera in self._cameras], dim=1)
        cameras_quat = torch.stack([camera.get_quat() for camera in self._cameras], dim=1)
        cameras_quat = _transform_camera_quat(cameras_quat)
        render_flags = np.array(
            (
                *(
                    needed[img_type]
                    for img_type in (IMAGE_TYPE.RGB, IMAGE_TYPE.DEPTH, IMAGE_TYPE.NORMAL, IMAGE_TYPE.SEGMENTATION)
                ),
                aliasing,
            ),
            dtype=np.uint32,
        )
        rendered = list(
            self._renderer.render(self._visualizer.scene.rigid_solver, cameras_pos, cameras_quat, render_flags)
        )

        # Post-processing:
        # * Remove alpha channel from RGBA
        # * Squeeze env and channel dims if necessary
        # * Split along camera dim
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                data = data.swapaxes(0, 1)
                if self._visualizer.scene.n_envs == 0:
                    data = data.squeeze(1)
                rendered[img_type] = tuple(data[..., :3].squeeze(-1))

        # Update cache
        self._t = self._visualizer.scene.t
        for img_type, data in enumerate(rendered):
            if needed[img_type]:
                self._data_cache[(img_type, cache_key)] = rendered[img_type]

        # Return in the required order, or None if not requested
        return tuple(self._data_cache[(img_type, cache_key)] if needed[img_type] else None for img_type in IMAGE_TYPE)

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
