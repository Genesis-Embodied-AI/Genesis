import os
import cv2
import numpy as np

from functools import partial
from concurrent.futures import ThreadPoolExecutor

import genesis as gs
from genesis.vis.visualizer import IMAGE_TYPE


class ImageComponent:
    def __init__(self, name, channel, normalize_func):
        self.name = name
        self.channel = channel
        self.normalize_func = normalize_func

    def export_frame_camera(self, i_env, export_dir, i_step, i_cam, frame):
        frame = frame[i_env]
        frame_path = os.path.join(export_dir, f"{self.name}_cam{i_cam}_env{i_env}_{i_step:03d}.png")
        cv2.imwrite(frame_path, frame)

    def check_frame_shape(self, frame):
        if frame.ndim == 3:
            if frame.shape[-1] == self.channel:
                frame = frame[None, ...]
            else:
                frame = frame[..., None]
        elif frame.ndim == 2:
            frame = frame.reshape((1, *frame.shape, 1))
        if frame.ndim != 4 or frame.shape[-1] != self.channel:
            gs.raise_exception(f"'{self.name}' must be an array of shape (n_envs, H, W, {self.channel})")
        return frame


def normalize_depth(depth, depth_clip_max, depth_scale):
    """Normalize depth values for visualization.

    Args:
        depth: Float ndarray of depth values.
        depth_clip_max: Maximum valid depth value (float).
        depth_scale: Storage of depth image, "linear" or "log".

    Returns:
        Normalized depth ndarray as uint8.
    """
    # Masks for infinities
    pinf_mask = np.isposinf(depth)
    ninf_mask = np.isneginf(depth)
    depth_min = np.where(ninf_mask, np.inf, depth).min(axis=(-3, -2), keepdims=True)
    depth_max = np.where(pinf_mask, -np.inf, depth).max(axis=(-3, -2), keepdims=True)
    depth_min = np.maximum(depth_min, 0.0)
    depth_max = np.minimum(depth_max, depth_clip_max)
    depth = np.clip(depth, 0.0, depth_clip_max)
    depth = np.where(ninf_mask, depth_min, depth)
    depth = np.where(pinf_mask, depth_max, depth)

    # Optional log scaling
    if depth_scale == "log":
        depth = np.log(depth + 1.0)

    # Normalize to 0–255
    denom = depth_max - depth_min
    out = np.zeros_like(depth, dtype=np.float32)
    np.divide(depth_max - depth, denom, out=out, where=denom > gs.EPS)  # safe masked divide
    return (out * 255.0).astype(np.uint8)


def normalize_segmentation(segmentation):
    """Normalize segmentation values for visualization.

    Args:
        segmentation: Int ndarray of labels.

    Returns:
        Normalized segmentation ndarray as uint8.
    """
    seg_min = segmentation.min(axis=(-3, -2), keepdims=True)
    seg_max = segmentation.max(axis=(-3, -2), keepdims=True)
    denom = seg_max - seg_min
    out = np.zeros_like(segmentation, dtype=np.float32)
    # using np.where will evaluate values in advance.
    np.divide(segmentation - seg_min, denom, out=out, where=denom > 0)
    return (out * 255.0).astype(np.uint8)


class FrameImageExporter:
    """
    This class enables exporting images from all cameras and all environments in batch and in parallel, unlike
    `Camera.(start|stop)_recording` API, which only allows for exporting images from a single camera and environment.
    """

    def __init__(self, export_dir, depth_clip_max=100, depth_scale="linear"):
        self.image_components = [
            ImageComponent(str(IMAGE_TYPE.RGB), 3, None),
            ImageComponent(
                str(IMAGE_TYPE.DEPTH),
                1,
                partial(normalize_depth, depth_clip_max=depth_clip_max, depth_scale=depth_scale),
            ),
            ImageComponent(str(IMAGE_TYPE.SEGMENTATION), 1, partial(normalize_segmentation)),
            ImageComponent(str(IMAGE_TYPE.NORMAL), 3, None),
        ]
        self.export_dir = export_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

    def export_frame_all_cameras(self, i_step, camera_idx=None, rgb=None, depth=None, segmentation=None, normal=None):
        """
        Export frames for all cameras.

        Args:
            i_step: The current step index.
            camera_idx: array of indices of cameras to export. If None, all cameras are exported.
            rgb: RGB image is a sequence of arrays of shape (n_envs, H, W, 3).
            depth: Depth image is a sequence of arrays of shape (n_envs, H, W).
            segmentation: Segmentation image is a sequence of arrays of shape (n_envs, H, W).
            normal: Normal image is a sequence of arrays of shape (n_envs, H, W, 3).
        """
        component_frames = [rgb, depth, segmentation, normal]
        ref_component = next((c for c in component_frames if c is not None), None)
        if ref_component is None:
            gs.raise_exception("No images to export")

        # Choose reference sequence for default camera indices
        if camera_idx is None:
            camera_idx = range(len(ref_component))

        for t in range(IMAGE_TYPE.NUM_TYPES):
            frames = component_frames[t]
            if frames is not None and (not isinstance(frames, (tuple, list)) or len(frames) == 0):
                gs.raise_exception(f"'{str(IMAGE_TYPE(t))}' must be a non-empty sequence of arrays.")

        for i_cam in camera_idx:
            frame_args = {}
            for t in range(IMAGE_TYPE.NUM_TYPES):
                frame_args[str(IMAGE_TYPE(t))] = None if frames is None else frames[i_cam]
            self.export_frame_single_camera(i_step, i_cam, **frame_args)

    def export_frame_single_camera(self, i_step, i_cam, rgb=None, depth=None, segmentation=None, normal=None):
        """
        Export frames for a single camera.

        Args:
            i_step: The current step index.
            i_cam: The index of the camera.
            rgb: RGB image array of shape (n_envs, H, W, 3).
            depth: Depth image array of shape (n_envs, H, W).
            segmentation: Segmentation image array of shape (n_envs, H, W).
            normal: Normal image array of shape (n_envs, H, W, 3).
        """
        component_frames = [rgb, depth, segmentation, normal]

        for t in range(IMAGE_TYPE.NUM_TYPES):
            frames = component_frames[t]
            if frames is None:
                continue
            component = self.image_components[t]
            frames = component.check_frame_shape(frames)
            if component.normalize_func is not None:
                frames = component.normalize_func(frames)
            frame_job = partial(
                component.export_frame_camera,
                export_dir=self.export_dir,
                i_step=i_step,
                i_cam=i_cam,
                frame=frames,
            )
            with ThreadPoolExecutor() as executor:
                executor.map(frame_job, np.arange(len(frames)))
