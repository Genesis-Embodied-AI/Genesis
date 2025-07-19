import os
import cv2
import numpy as np
import torch
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import genesis as gs
from genesis.utils.misc import tensor_to_array


class FrameImageExporter:
    @staticmethod
    def _export_frame_rgb_camera(i_env, export_dir, i_cam, i_step, rgb):
        # Take the rgb channel in case the rgb tensor has RGBA channel.
        rgb = np.flip(tensor_to_array(rgb[i_env, ..., :3]), axis=-1)
        cv2.imwrite(f"{export_dir}/rgb_cam{i_cam}_env{i_env}_{i_step:03d}.png", rgb)

    @staticmethod
    def _export_frame_depth_camera(i_env, export_dir, i_cam, i_step, depth):
        depth = tensor_to_array(depth[i_env])
        cv2.imwrite(f"{export_dir}/depth_cam{i_cam}_env{i_env}_{i_step:03d}.png", depth)

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
        depth = depth.clamp(0, self.depth_clip_max)

        # Apply scaling if specified
        if self.depth_scale == "log":
            depth = torch.log(depth + 1)

        # Calculate min/max for each image in the batch
        depth_min = depth.amin(dim=(-3, -2), keepdim=True)
        depth_max = depth.amax(dim=(-3, -2), keepdim=True)

        # Normalize to 0-255 range
        return ((depth - depth_min) / (depth_max - depth_min) * 255).to(torch.uint8)

    def export_frame_all_cameras(self, i_step, camera_idx=None, rgb=None, depth=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: rgb image is a tuple of tensors of shape (n_envs, H, W, 3).
            depth: Depth image is a tuple of tensors of shape (n_envs, H, W).
        """
        if rgb is None and depth is None:
            print("No rgb or depth to export")
            return
        if rgb is not None:
            assert isinstance(rgb, tuple) and len(rgb) > 0, "rgb must be a tuple of tensors with length > 0"
        if depth is not None:
            assert isinstance(depth, tuple) and len(depth) > 0, "depth must be a tuple of tensors with length > 0"
        if camera_idx is None:
            camera_idx = range(len(depth if rgb is None else rgb))
        for i_cam in camera_idx:
            rgb_cam = rgb[i_cam] if rgb is not None and i_cam < len(rgb) else None
            depth_cam = depth[i_cam] if depth is not None and i_cam < len(depth) else None
            if rgb_cam is not None or depth_cam is not None:
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
            rgb = torch.as_tensor(rgb, dtype=torch.uint8, device=gs.device)

            # Unsqueeze rgb to (n_envs, H, W, 3)
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            assert rgb.ndim == 4, "rgb must be of shape (n_envs, H, W, 3)"

            rgb_job = partial(
                FrameImageExporter._export_frame_rgb_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                rgb=rgb,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(rgb_job, np.arange(len(rgb)))

        if depth is not None:
            depth = torch.as_tensor(depth, dtype=torch.float32, device=gs.device)

            # Unsqueeze depth to (n_envs, H, W, 1)
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 2:
                depth = depth.reshape(1, depth.shape[0], depth.shape[1], 1)
            depth = self._normalize_depth(depth)
            assert depth.ndim == 4, "depth must be of shape (n_envs, H, W, 1)"

            depth_job = partial(
                FrameImageExporter._export_frame_depth_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                depth=depth,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(depth_job, np.arange(len(depth)))
