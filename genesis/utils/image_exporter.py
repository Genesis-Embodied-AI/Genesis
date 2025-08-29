import os
import cv2
import numpy as np
import torch
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import genesis as gs
from genesis.utils.misc import tensor_to_array


def _export_frame_rgb_camera(i_env, export_dir, i_cam, i_step, rgb):
    # Take the rgb channel in case the rgb tensor has RGBA channel.
    rgb = np.flip(tensor_to_array(rgb[i_env, ..., :3]), axis=-1)
    cv2.imwrite(f"{export_dir}/rgb_cam{i_cam}_env{i_env}_{i_step:03d}.png", rgb)


def _export_frame_depth_camera(i_env, export_dir, i_cam, i_step, depth):
    depth = tensor_to_array(depth[i_env])
    cv2.imwrite(f"{export_dir}/depth_cam{i_cam}_env{i_env}_{i_step:03d}.png", depth)


class FrameImageExporter:
    """
    This class enables exporting images from all cameras and all environments in batch and in parallel, unlike
    `Camera.(start|stop)_recording` API, which only allows for exporting images from a single camera and environment.
    """

    def __init__(self, export_dir, depth_clip_max=100, depth_scale="log"):
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self.depth_clip_max = depth_clip_max
        self.depth_scale = depth_scale

    def _normalize_depth(self, depth):
        """Normalize depth values for visualization.

        Args:
            depth: Tensor of depth values

        Returns:
            Normalized depth tensor as uint8
        """
        # Clip depth values
        depth = depth.clamp(0.0, self.depth_clip_max)

        # Apply scaling if specified
        if self.depth_scale == "log":
            depth = torch.log(depth + 1)

        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-2, -1), keepdim=True)
        depth_max = depth.amax(dim=(-2, -1), keepdim=True)

        # Normalize to 0-255 range
        return torch.where(
            depth_max - depth_min > gs.EPS, ((depth_max - depth) / (depth_max - depth_min) * 255).to(torch.uint8), 0
        )

    def export_frame_all_cameras(self, i_step, camera_idx=None, rgb=None, depth=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: rgb image is a sequence of tensors of shape (n_envs, H, W, 3).
            depth: Depth image is a sequence of tensors of shape (n_envs, H, W).
        """
        if rgb is None and depth is None:
            gs.logger.info("No rgb or depth images to export")
            return
        if rgb is not None and (not isinstance(rgb, (tuple, list)) or not rgb):
            gs.raise_exception("'rgb' must be a non-empty sequence of tensors.")
        if depth is not None and (not isinstance(depth, (tuple, list)) or not depth):
            gs.raise_exception("'depth' must be a non-empty sequence of tensors.")
        if camera_idx is None:
            camera_idx = range(len(depth or rgb))
        for i_cam in camera_idx:
            rgb_cam, depth_cam = None, None
            if rgb is not None:
                rgb_cam = rgb[i_cam]
            if depth is not None:
                depth_cam = depth[i_cam]
            self.export_frame_single_camera(i_step, i_cam, rgb_cam, depth_cam)

    def export_frame_single_camera(self, i_step, i_cam, rgb=None, depth=None):
        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            i_cam: The index of the camera.
            rgb: rgb image tensor of shape (n_envs, H, W, 3).
            depth: Depth tensor of shape (n_envs, H, W).
        """
        if rgb is not None:
            if isinstance(rgb, np.ndarray) and any(e < 0 for e in rgb.strides):
                # Torch does not support negative strides for now
                rgb = rgb.copy()
            rgb = torch.as_tensor(rgb, dtype=torch.uint8, device=gs.device)

            # Unsqueeze rgb to (n_envs, H, W, 3)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            if rgb.ndim != 4 or rgb.shape[-1] != 3:
                gs.raise_exception("'rgb' must be a tensor of shape (n_envs, H, W, 3)")

            rgb_job = partial(
                _export_frame_rgb_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                rgb=rgb,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(rgb_job, np.arange(len(rgb)))

        if depth is not None:
            if isinstance(depth, np.ndarray) and any(e < 0 for e in depth.strides):
                depth = depth.copy()
            depth = torch.as_tensor(depth, dtype=torch.float32, device=gs.device)

            # Unsqueeze depth to (n_envs, H, W)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            depth = self._normalize_depth(depth)
            if depth.ndim != 3:
                gs.raise_exception("'depth' must be a tensor of shape (n_envs, H, W)")

            depth_job = partial(
                _export_frame_depth_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                depth=depth,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(depth_job, np.arange(len(depth)))
