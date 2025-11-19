"""
Camera sensors for rendering: Rasterizer, Raytracer, and Batch Renderer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Type

import numpy as np
import torch

import genesis as gs
from genesis.options.sensors import (
    RasterizerCameraOptions,
    RaytracerCameraOptions,
    BatchRendererCameraOptions,
)
from genesis.utils.geom import pos_lookat_up_to_T

from .base_sensor import Sensor, SharedSensorMetadata
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


# ========================== Data Class ==========================


class CameraData(NamedTuple):
    """Camera sensor return data."""

    rgb: np.ndarray  # Shape: (H, W, 3) for single env, (B, H, W, 3) for batched


# ========================== Shared Metadata ==========================


@dataclass
class RasterizerCameraSharedMetadata(SharedSensorMetadata):
    """Shared metadata for all Rasterizer cameras."""

    renderer: Any = None  # Rasterizer instance
    context: Any = None  # RasterizerContext instance
    lights: Any = None  # gs.List of lights
    sensors: list = None  # List of RasterizerCameraSensor instances
    camera_nodes: dict = None  # {sensor_idx: camera_node}
    camera_targets: dict = None  # {sensor_idx: camera_target}
    image_cache: dict = None  # {sensor_idx: np.ndarray with shape (B, H, W, 3)}


@dataclass
class RaytracerCameraSharedMetadata(SharedSensorMetadata):
    """Shared metadata for all Raytracer cameras."""

    renderer: Any = None  # Raytracer instance
    lights: Any = None  # List of light objects
    sensors: list = None  # List of RaytracerCameraSensor instances
    image_cache: dict = None  # {sensor_idx: np.ndarray with shape (B, H, W, 3)}


@dataclass
class BatchRendererCameraSharedMetadata(SharedSensorMetadata):
    """Shared metadata for all Batch Renderer cameras."""

    renderer: Any = None  # BatchRenderer instance
    lights: Any = None  # gs.List of lights
    sensors: list = None  # List of BatchRendererCameraSensor instances
    image_cache: dict = None  # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    last_render_timestep: int = -1  # Track when batch was last rendered


# ========================== Rasterizer Camera Sensor ==========================


@register_sensor(RasterizerCameraOptions, RasterizerCameraSharedMetadata, CameraData)
class RasterizerCameraSensor(Sensor[RasterizerCameraSharedMetadata]):
    """
    Rasterizer camera sensor using OpenGL-based rendering.

    This sensor renders RGB images using the existing Rasterizer backend,
    but operates independently from the scene visualizer.
    """

    def __init__(
        self,
        options: RasterizerCameraOptions,
        idx: int,
        data_cls: Type[CameraData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, idx, data_cls, manager)
        self._options: RasterizerCameraOptions
        self._camera_node = None
        self._camera_target = None
        self._attached_link = None
        self._attached_offset_T = None

    # ========================== Sensor Lifecycle ==========================

    def build(self):
        """Initialize the rasterizer and register this camera."""
        super().build()

        scene = self._manager._sim.scene

        # Initialize shared metadata on first camera
        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.camera_nodes = {}
            self._shared_metadata.camera_targets = {}
            self._shared_metadata.image_cache = {}

            # Create standalone rasterizer context
            self._shared_metadata.context = self._create_standalone_context(scene)

            # Create offscreen rasterizer
            from genesis.vis.rasterizer import Rasterizer

            self._shared_metadata.renderer = Rasterizer(viewer=None, context=self._shared_metadata.context)
            self._shared_metadata.renderer.build()

        # Register this camera
        self._shared_metadata.sensors.append(self)

        # Add camera to rasterizer
        self._add_camera_to_rasterizer()

        # Initialize image cache for this camera
        n_envs = max(self._manager._sim._B, 1)
        h, w = self._options.res[1], self._options.res[0]
        self._shared_metadata.image_cache[self._idx] = np.zeros((n_envs, h, w, 3), dtype=np.uint8)

    def _create_standalone_context(self, scene):
        """Create a simplified RasterizerContext for camera sensors."""
        from genesis.vis.rasterizer_context import RasterizerContext
        from genesis.options.vis import VisOptions

        # Create minimal visualizer options
        vis_options = VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            rendered_envs_idx=list(range(max(self._manager._sim._B, 1))),
        )

        context = RasterizerContext(vis_options)
        context.build(scene)
        context.reset()
        return context

    def _add_camera_to_rasterizer(self):
        """Add this camera to the rasterizer."""
        # Use the rasterizer's add_camera method
        camera_wrapper = self._get_camera_wrapper()
        self._shared_metadata.renderer.add_camera(camera_wrapper)

        # Set initial camera pose
        self._update_camera_pose()

    def _update_camera_pose(self):
        """Update camera pose based on options."""
        pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(self._options.up, dtype=gs.tc_float, device=gs.device)

        transform = pos_lookat_up_to_T(pos, lookat, up)
        camera_wrapper = self._get_camera_wrapper()
        camera_wrapper.transform = transform.cpu().numpy()
        self._shared_metadata.renderer.update_camera(camera_wrapper)

    @classmethod
    def reset(cls, shared_metadata: RasterizerCameraSharedMetadata, envs_idx):
        """Reset camera sensor (no state to reset)."""
        pass

    # ========================== Cache Integration ==========================

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        """Return minimal cache format (1 float as timestamp)."""
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RasterizerCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """Update cache with current timestamp."""
        if shared_metadata.renderer is not None:
            current_time = shared_metadata.context.scene.t
            shared_ground_truth_cache.fill_(current_time)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RasterizerCameraSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """Copy ground truth to cache (no noise for images)."""
        shared_cache.copy_(shared_ground_truth_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """No debug drawing for cameras."""
        pass

    # ========================== Public API ==========================

    @gs.assert_built
    def render(self):
        """Render this camera and update cached image."""
        # Update camera pose if attached to a rigid link
        if self._attached_link is not None:
            self.move_to_attach()

        # Update scene state
        self._shared_metadata.context.update(force_render=False)

        # Render this camera
        rgb_arr, _, _, _ = self._shared_metadata.renderer.render_camera(
            self._get_camera_wrapper(),
            rgb=True,
            depth=False,
            segmentation=False,
            normal=False,
        )

        # Store in cache
        # rgb_arr shape: (H, W, 3) for single env or needs batching
        n_envs = self._manager._sim._B
        if n_envs == 0:
            # Single environment case - add batch dimension
            self._shared_metadata.image_cache[self._idx][0] = rgb_arr
        else:
            # TODO: Handle multi-env rendering properly
            # For now, just replicate the single render
            for i in range(n_envs):
                self._shared_metadata.image_cache[self._idx][i] = rgb_arr

    def _get_camera_wrapper(self):
        """Create a camera wrapper object for the renderer."""

        class CameraWrapper:
            def __init__(self, sensor):
                self.sensor = sensor
                self.uid = id(sensor)  # Use Python id as unique identifier
                self.res = sensor._options.res
                self.fov = sensor._options.fov
                self.near = sensor._options.near
                self.far = sensor._options.far
                self.aspect_ratio = self.res[0] / self.res[1]

        wrapper = CameraWrapper(self)
        # Make sure the uid matches what was stored
        wrapper.uid = self._idx
        return wrapper

    @gs.assert_built
    def read(self, envs_idx=None) -> CameraData:
        """Read the cached rendered image."""
        cached_image = self._shared_metadata.image_cache[self._idx]

        # Handle envs_idx
        if envs_idx is None:
            if self._manager._sim.n_envs == 0:
                # Return without batch dimension for single env
                return self._return_data_class(rgb=cached_image[0])
            else:
                # Return with batch dimension
                return self._return_data_class(rgb=cached_image)
        else:
            # Return specific environment(s)
            envs_idx = self._sanitize_envs_idx(envs_idx)
            if isinstance(envs_idx, (int, np.integer)):
                return self._return_data_class(rgb=cached_image[envs_idx])
            else:
                return self._return_data_class(rgb=cached_image[envs_idx])

    def _sanitize_envs_idx(self, envs_idx):
        """Sanitize envs_idx to valid indices."""
        if envs_idx is None:
            return None
        if isinstance(envs_idx, (int, np.integer)):
            return envs_idx
        return np.asarray(envs_idx)

    def add_light(
        self,
        pos=(0.0, 0.0, 5.0),
        dir=(0.0, 0.0, -1.0),
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        type="directional",
    ):
        """
        Add a light to the scene for all rasterizer cameras.

        Parameters
        ----------
        pos : tuple[float, float, float]
            Light position.
        dir : tuple[float, float, float]
            Light direction (for directional lights).
        color : tuple[float, float, float]
            Light color RGB.
        intensity : float
            Light intensity.
        type : str
            Light type: "directional" or "point".
        """
        if self._shared_metadata.lights is not None:
            light_dict = {
                "type": type,
                "pos": pos,
                "dir": dir,
                "color": tuple(np.array(color) * intensity),
                "intensity": intensity,
            }
            self._shared_metadata.lights.append(light_dict)

            # Add to context
            self._shared_metadata.context.add_light(light_dict)

    @gs.assert_built
    def attach(self, rigid_link, offset_T):
        """
        Attach the camera to a rigid link in the scene.

        Once attached, the camera will automatically update its pose relative to the
        attached link during rendering. This is useful for mounting cameras on robots
        or other dynamic objects.

        Parameters
        ----------
        rigid_link : genesis.RigidLink
            The rigid link to which the camera should be attached.
        offset_T : np.ndarray or torch.Tensor, shape (4, 4)
            The transformation matrix specifying the camera's pose relative to the rigid link.
        """
        self._attached_link = rigid_link
        self._attached_offset_T = torch.as_tensor(offset_T, dtype=gs.tc_float, device=gs.device)

    @gs.assert_built
    def detach(self):
        """
        Detach the camera from the currently attached rigid link.

        After detachment, the camera will stop following the motion of the rigid link
        and maintain its current world pose. Calling this method has no effect if the
        camera is not currently attached.
        """
        self._attached_link = None
        self._attached_offset_T = None

    @gs.assert_built
    def move_to_attach(self):
        """
        Move the camera to follow the currently attached rigid link.

        This method updates the camera's pose using the transform of the attached
        rigid link combined with the specified offset. It is automatically called
        during render() if the camera is attached.

        Raises
        ------
        Exception
            If the camera has not been attached to a rigid link.
        """
        if self._attached_link is None:
            gs.raise_exception("Camera not attached to any rigid link.")

        # Get link pose (for single env or first env)
        link_pos = self._attached_link.get_pos()
        link_quat = self._attached_link.get_quat()

        # Handle batched case - use first environment
        if link_pos.ndim > 1:
            link_pos = link_pos[0]
            link_quat = link_quat[0]

        # Compute camera transform
        from genesis.utils.geom import trans_quat_to_T

        link_T = trans_quat_to_T(link_pos, link_quat)
        camera_T = torch.matmul(link_T, self._attached_offset_T)

        # Update camera pose
        camera_wrapper = self._get_camera_wrapper()
        camera_wrapper.transform = camera_T.cpu().numpy()
        self._shared_metadata.renderer.update_camera(camera_wrapper)


# ========================== Raytracer Camera Sensor ==========================


@register_sensor(RaytracerCameraOptions, RaytracerCameraSharedMetadata, CameraData)
class RaytracerCameraSensor(Sensor[RaytracerCameraSharedMetadata]):
    """
    Raytracer camera sensor using LuisaRender path tracing.
    """

    def __init__(
        self,
        options: RaytracerCameraOptions,
        idx: int,
        data_cls: Type[CameraData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, idx, data_cls, manager)
        self._options: RaytracerCameraOptions

    def build(self):
        """Initialize the raytracer and register this camera."""
        super().build()
        gs.raise_exception("RaytracerCameraSensor not yet implemented. Coming soon!")

    @classmethod
    def reset(cls, shared_metadata: RaytracerCameraSharedMetadata, envs_idx):
        pass

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RaytracerCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        pass

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RaytracerCameraSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        pass

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        pass


# ========================== Batch Renderer Camera Sensor ==========================


@register_sensor(BatchRendererCameraOptions, BatchRendererCameraSharedMetadata, CameraData)
class BatchRendererCameraSensor(Sensor[BatchRendererCameraSharedMetadata]):
    """
    Batch renderer camera sensor using Madrona GPU batch rendering.
    """

    def __init__(
        self,
        options: BatchRendererCameraOptions,
        idx: int,
        data_cls: Type[CameraData],
        manager: "gs.SensorManager",
    ):
        super().__init__(options, idx, data_cls, manager)
        self._options: BatchRendererCameraOptions

    def build(self):
        """Initialize the batch renderer and register this camera."""
        super().build()
        gs.raise_exception("BatchRendererCameraSensor not yet implemented. Coming soon!")

    @classmethod
    def reset(cls, shared_metadata: BatchRendererCameraSharedMetadata, envs_idx):
        pass

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        return (1,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: BatchRendererCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        pass

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: BatchRendererCameraSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        pass

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        pass
