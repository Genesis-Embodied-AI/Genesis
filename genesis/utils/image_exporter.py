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

    @staticmethod
    def _export_frame_normal_camera(i_env, export_dir, i_cam, i_step, normal):
        # Take the normal channel in case the rgb tensor has RGBA channel.
        normal = np.flip(tensor_to_array(normal[i_env, ..., :3]), axis=-1)
        cv2.imwrite(f"{export_dir}/normal_cam{i_cam}_env{i_env}_{i_step:03d}.png", normal)

    @staticmethod
    def _export_frame_segmentation_camera(i_env, export_dir, i_cam, i_step, segmentation):
        segmentation = tensor_to_array(segmentation[i_env])
        cv2.imwrite(f"{export_dir}/segmentation_cam{i_cam}_env{i_env}_{i_step:03d}.png", segmentation)

    def __init__(self, export_dir, depth_clip_max=100, depth_scale="linear"):
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
        # Filter corrupted depth pixels. inf/-inf should be replaced with max/min values (not clip values)
        pinf_mask = torch.isposinf(depth)
        ninf_mask = torch.isneginf(depth)
        depth_min = torch.where(ninf_mask, float("inf"), depth).amin(dim=(-3, -2), keepdim=True)
        depth_max = torch.where(pinf_mask, -float("inf"), depth).amax(dim=(-3, -2), keepdim=True)
        depth_min = torch.maximum(depth_min, 0.0)
        depth_max = torch.minimum(depth_max, self.depth_clip_max)
        depth = depth.clamp(0.0, self.depth_clip_max)       # clip depth values
        depth = torch.where(ninf_mask, depth_min, depth)
        depth = torch.where(pinf_mask, depth_max, depth)

        # Apply scaling if specified
        if self.depth_scale == "log":
            depth = torch.log(depth + 1)

        # Normalize to 0-255 range
        return torch.where(
            depth_max - depth_min > gs.EPS, ((depth_max - depth) / (depth_max - depth_min) * 255).to(torch.uint8), 0
        )

    def export_frame_all_cameras(self, i_step, camera_idx=None, rgb=None, depth=None, normal=None, segmentation=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: rgb image is a sequence of tensors of shape (n_envs, H, W, 3).
            depth: Depth image is a sequence of tensors of shape (n_envs, H, W).
        """
        if rgb is None and depth is None and normal is None and segmentation is None:
            gs.logger.info("No images to export")
            return
        if rgb is not None and (not isinstance(rgb, (tuple, list)) or not rgb):
            gs.raise_exception("'rgb' must be a non-empty sequence of tensors.")
        if depth is not None and (not isinstance(depth, (tuple, list)) or not depth):
            gs.raise_exception("'depth' must be a non-empty sequence of tensors.")
        if normal is not None and (not isinstance(normal, (tuple, list)) or not normal):
            gs.raise_exception("'normal' must be a non-empty sequence of tensors.")
        if segmentation is not None and (not isinstance(segmentation, (tuple, list)) or not segmentation):
            gs.raise_exception("'segmentation' must be a non-empty sequence of tensors.")
        if camera_idx is None:
            camera_idx = range(len(depth or rgb))
        for i_cam in camera_idx:
            rgb_cam, depth_cam = None, None
            if rgb is not None:
                rgb_cam = rgb[i_cam]
            if depth is not None:
                depth_cam = depth[i_cam]
            if normal is not None:
                normal_cam = normal[i_cam]
            if segmentation is not None:
                segmentation_cam = segmentation[i_cam]
            self.export_frame_single_camera(i_step, i_cam, rgb_cam, depth_cam, normal_cam, segmentation_cam)

    def export_frame_single_camera(self, i_step, i_cam, rgb=None, depth=None, normal=None, segmentation=None):
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
            depth = torch.as_tensor(depth, dtype=torch.float32, device=gs.device)

            # Unsqueeze depth to (n_envs, H, W, 1)
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 2:
                depth = depth.reshape((1, *depth.shape, 1))
            depth = self._normalize_depth(depth)
            if depth.ndim != 4 or depth.shape[-1] != 1:
                gs.raise_exception("'rgb' must be a tensor of shape (n_envs, H, W, 1)")

            depth_job = partial(
                _export_frame_depth_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                depth=depth,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(depth_job, np.arange(len(depth)))

        if normal is not None:
            normal = torch.as_tensor(normal, dtype=gs.tc_float, device=gs.device)

            # Unsqueeze normal to (n_envs, H, W, 3)
            if normal.ndim == 3:
                normal = normal.unsqueeze(0)
            assert normal.ndim == 4, "normal must be of shape (n_envs, H, W, 3)"

            normal_job = partial(
                FrameImageExporter._export_frame_normal_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                normal=normal,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(normal_job, np.arange(len(normal)))

        if segmentation is not None:
            segmentation = torch.as_tensor(segmentation, dtype=gs.tc_float, device=gs.device)

            # Unsqueeze segmentation to (n_envs, H, W, 1)
            if segmentation.ndim == 3:
                segmentation = segmentation.unsqueeze(0)
            elif segmentation.ndim == 2:
                segmentation = segmentation.reshape(1, segmentation.shape[0], segmentation.shape[1], 1)
            segmentation = self._normalize_depth(segmentation)
            assert segmentation.ndim == 4, "segmentation must be of shape (n_envs, H, W, 1)"

            segmentation_job = partial(
                FrameImageExporter._export_frame_segmentation_camera,
                export_dir=self.export_dir,
                i_cam=i_cam,
                i_step=i_step,
                segmentation=segmentation,
            )

            with ThreadPoolExecutor() as executor:
                executor.map(segmentation_job, np.arange(len(segmentation)))
