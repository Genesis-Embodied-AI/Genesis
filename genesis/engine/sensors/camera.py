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
    visualizer_wrapper: Any = None  # MinimalVisualizerWrapper instance


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

    Note: Raytracer only renders environment 0 (expensive path tracing).
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
        self._camera_obj = None
        self._attached_link = None
        self._attached_offset_T = None

    def build(self):
        """Register a raytracer camera that reuses the visualizer pipeline."""
        super().build()

        scene = self._manager._sim.scene
        visualizer = scene.visualizer

        renderer = getattr(visualizer, "raytracer", None)
        if renderer is None:
            gs.raise_exception(
                "RaytracerCameraSensor requires the scene to be created with "
                "`renderer=gs.renderers.RayTracer(...)` so that it can share "
                "the same LuisaRender backend as the main visualizer (see examples/rendering/demo.py)."
            )

        # Initialize shared metadata on first camera
        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = []
            self._shared_metadata.image_cache = {}
            self._shared_metadata.renderer = renderer

        # Register this sensor
        self._shared_metadata.sensors.append(self)

        # Create the underlying visualizer camera
        self._create_camera(visualizer)

        # Initialize image cache for this camera (only env 0)
        h, w = self._options.res[1], self._options.res[0]
        n_envs = max(self._manager._sim._B, 1)
        self._shared_metadata.image_cache[self._idx] = np.zeros((n_envs, h, w, 3), dtype=np.uint8)

    def _create_camera(self, visualizer):
        """Create the underlying visualizer camera that drives the raytracer."""
        opts = self._options
        self._camera_obj = visualizer.add_camera(
            res=opts.res,
            pos=opts.pos,
            lookat=opts.lookat,
            up=opts.up,
            model=opts.model,
            fov=opts.fov,
            aperture=opts.aperture,
            focus_dist=opts.focus_dist,
            GUI=False,
            spp=opts.spp,
            denoise=opts.denoise,
            near=0.05,
            far=100.0,
            env_idx=0,
            debug=False,
        )

    @classmethod
    def reset(cls, shared_metadata: RaytracerCameraSharedMetadata, envs_idx):
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
        cls, shared_metadata: RaytracerCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """Update cache with current timestamp."""
        # For raytracer cameras we only need a monotonically increasing timestamp;
        # just reuse the scene time from any registered sensor.
        if not shared_metadata.sensors:
            return
        sensor0 = shared_metadata.sensors[0]
        scene_t = sensor0._manager._sim.scene.t
        shared_ground_truth_cache.fill_(scene_t)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RaytracerCameraSharedMetadata,
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
        """Render this camera and update cached image (env 0 only)."""
        if self._attached_link is not None:
            self._camera_obj.move_to_attach()

        rgb_arr, _, _, _ = self._camera_obj.render(
            rgb=True,
            depth=False,
            segmentation=False,
            colorize_seg=False,
            normal=False,
            antialiasing=False,
            force_render=False,
        )

        # Store in cache (rgb_arr is already (H, W, 3) for env 0)
        n_envs = self._manager._sim._B
        if n_envs == 0:
            # Single environment case
            self._shared_metadata.image_cache[self._idx][0] = rgb_arr
        else:
            # Multi-env case: only render env 0, replicate for others
            for i in range(n_envs):
                self._shared_metadata.image_cache[self._idx][i] = rgb_arr

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
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        radius=0.5,
        pos=(0.0, 0.0, 5.0),
        revert_dir=False,
        double_sided=False,
        cutoff=180.0,
    ):
        """
        Add a mesh light to the scene for all raytracer cameras.

        Parameters
        ----------
        color : tuple[float, float, float]
            Light color RGB.
        intensity : float
            Light intensity.
        radius : float
            Radius of the (spherical) light source.
        pos : tuple[float, float, float]
            Light position.
        revert_dir : bool
            Whether to revert normal direction for mesh light.
        double_sided : bool
            Whether the mesh light is double-sided.
        cutoff : float
            Light cutoff angle.
        """
        # Use the same mesh-light path as the high-level Scene API by creating
        # a temporary sphere morph and letting the visualizer convert it.
        scene = self._manager._sim.scene
        if scene.is_built:
            gs.logger.warning("RaytracerCameraSensor.add_light() should be called before scene.build(). Ignoring.")
            return

        morph = gs.morphs.Sphere(pos=pos, radius=radius)
        scene.add_mesh_light(
            morph=morph,
            color=(*color, 1.0),
            intensity=intensity,
            revert_dir=revert_dir,
            double_sided=double_sided,
            cutoff=cutoff,
        )

    @gs.assert_built
    def attach(self, rigid_link, offset_T):
        """Attach the camera to a rigid link in the scene."""
        self._attached_link = rigid_link
        self._attached_offset_T = torch.as_tensor(offset_T, dtype=gs.tc_float, device=gs.device)
        # Keep the underlying visualizer camera in sync so its own attach logic works.
        self._camera_obj.attach(rigid_link, offset_T)

    @gs.assert_built
    def detach(self):
        """Detach the camera from the currently attached rigid link."""
        self._attached_link = None
        self._attached_offset_T = None
        self._camera_obj.detach()

    @gs.assert_built
    def move_to_attach(self):
        """Move the camera to follow the currently attached rigid link."""
        # Delegate to the underlying visualizer camera implementation
        self._camera_obj.move_to_attach()


# ========================== Batch Renderer Camera Sensor ==========================


@register_sensor(BatchRendererCameraOptions, BatchRendererCameraSharedMetadata, CameraData)
class BatchRendererCameraSensor(Sensor[BatchRendererCameraSharedMetadata]):
    """
    Batch renderer camera sensor using Madrona GPU batch rendering.

    Note: All batch renderer cameras must have the same resolution.
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
        self._camera_obj = None
        self._attached_link = None
        self._attached_offset_T = None

    def build(self):
        """Initialize the batch renderer and register this camera."""
        super().build()

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRendererCameraSensor requires CUDA backend.")

        scene = self._manager._sim.scene

        # Initialize shared metadata on first camera
        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.image_cache = {}
            self._shared_metadata.last_render_timestep = -1

            # Validate that all batch renderer cameras have the same resolution
            all_sensors = self._manager._sensors_by_type[type(self)]
            resolutions = [s._options.res for s in all_sensors]
            if len(set(resolutions)) > 1:
                gs.raise_exception(
                    f"All BatchRendererCameraSensor instances must have the same resolution. "
                    f"Found: {set(resolutions)}"
                )

            # Create batch renderer (will be initialized after all cameras are added)
            from genesis.vis.batch_renderer import BatchRenderer
            from genesis.options.renderers import BatchRenderer as BatchRendererOptions
            from genesis.options.vis import VisOptions

            br_options = BatchRendererOptions(
                use_rasterizer=self._options.use_rasterizer,
            )

            vis_options = VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                rendered_envs_idx=list(range(max(self._manager._sim._B, 1))),
            )

            # Create a minimal visualizer-like object for BatchRenderer
            class MinimalVisualizerWrapper:
                def __init__(self, scene, sensors, vis_options):
                    self.scene = scene
                    self._cameras = []  # Will be populated with camera wrappers
                    self._sensors = sensors  # Keep reference to sensors

                    # Create a minimal context for batch renderer
                    from genesis.vis.rasterizer_context import RasterizerContext

                    self._context = RasterizerContext(vis_options)
                    self._context.build(scene)
                    self._context.reset()

            self._shared_metadata.visualizer_wrapper = MinimalVisualizerWrapper(scene, all_sensors, vis_options)
            self._shared_metadata.renderer = BatchRenderer(
                self._shared_metadata.visualizer_wrapper, br_options, vis_options
            )

        # Register this camera
        self._shared_metadata.sensors.append(self)

        # Create camera object
        self._create_camera()

        # Build renderer once all cameras are registered (do it on last sensor)
        if len(self._shared_metadata.sensors) == len(self._manager._sensors_by_type[type(self)]):
            # Add all camera objects to visualizer wrapper
            self._shared_metadata.visualizer_wrapper._cameras = [s._camera_obj for s in self._shared_metadata.sensors]
            # Build the batch renderer
            self._shared_metadata.renderer.build()

        # Initialize image cache for this camera
        n_envs = max(self._manager._sim._B, 1)
        h, w = self._options.res[1], self._options.res[0]
        self._shared_metadata.image_cache[self._idx] = np.zeros((n_envs, h, w, 3), dtype=np.uint8)

    def _create_camera(self):
        """Create batch renderer camera object."""

        # Create a minimal camera wrapper
        class BatchRendererCameraWrapper:
            def __init__(self, sensor):
                self.sensor = sensor
                self.idx = len(sensor._shared_metadata.sensors)  # Camera index in batch
                self.uid = sensor._idx
                self.res = sensor._options.res
                self.fov = sensor._options.fov
                self.near = sensor._options.near
                self.far = sensor._options.far
                self.debug = False

                # Initial pose
                pos = torch.tensor(sensor._options.pos, dtype=gs.tc_float, device=gs.device)
                lookat = torch.tensor(sensor._options.lookat, dtype=gs.tc_float, device=gs.device)
                up = torch.tensor(sensor._options.up, dtype=gs.tc_float, device=gs.device)

                # Store pos/lookat/up for later updates
                self._pos = pos
                self._lookat = lookat
                self._up = up
                self.transform = pos_lookat_up_to_T(pos, lookat, up)

            def get_pos(self):
                """Get camera position (for batch renderer)."""
                n_envs = self.sensor._manager._sim._B
                if n_envs == 0:
                    return self._pos.unsqueeze(0)
                else:
                    return self._pos.unsqueeze(0).expand(n_envs, -1)

            def get_quat(self):
                """Get camera quaternion (for batch renderer)."""
                from genesis.utils.geom import T_to_trans_quat

                _, quat = T_to_trans_quat(self.transform)
                n_envs = self.sensor._manager._sim._B
                if n_envs == 0:
                    return quat.unsqueeze(0)
                else:
                    return quat.unsqueeze(0).expand(n_envs, -1)

        self._camera_obj = BatchRendererCameraWrapper(self)

    @classmethod
    def reset(cls, shared_metadata: BatchRendererCameraSharedMetadata, envs_idx):
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
        cls, shared_metadata: BatchRendererCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """Update cache with current timestamp."""
        if shared_metadata.renderer is not None:
            current_time = shared_metadata.renderer._visualizer.scene.t
            shared_ground_truth_cache.fill_(current_time)

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: BatchRendererCameraSharedMetadata,
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
        """
        Render all batch cameras and update cached images.

        Note: BatchRenderer renders all cameras at once for efficiency.
        Calling render() on any camera will update all camera caches.
        """
        scene = self._manager._sim.scene

        # Update camera pose if attached to a rigid link
        if self._attached_link is not None:
            self.move_to_attach()

        # Only render once per timestep (shared across all batch cameras)
        if self._shared_metadata.last_render_timestep == scene.t:
            return

        self._shared_metadata.last_render_timestep = scene.t

        # Update scene state
        self._shared_metadata.renderer.update_scene(force_render=False)

        # Render all cameras at once
        rgb_arr, _, _, _ = self._shared_metadata.renderer.render(
            rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=False
        )

        # rgb_arr shape: (n_cameras, n_envs, H, W, 3) or (n_cameras, H, W, 3) for single env
        # Convert to numpy and store each camera's output in its cache
        if isinstance(rgb_arr, torch.Tensor):
            rgb_arr = rgb_arr.cpu().numpy()

        for cam_idx, sensor in enumerate(self._shared_metadata.sensors):
            sensor._shared_metadata.image_cache[sensor._idx] = rgb_arr[cam_idx]

    @gs.assert_built
    def read(self, envs_idx=None) -> CameraData:
        """Read the cached rendered image."""
        cached_image = self._shared_metadata.image_cache[self._idx]

        # Convert to numpy if it's a torch tensor
        if isinstance(cached_image, torch.Tensor):
            cached_image = cached_image.cpu().numpy()

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
        directional=True,
        castshadow=True,
        cutoff=45.0,
        attenuation=(1.0, 0.0, 0.0),
    ):
        """
        Add a light to the scene for all batch renderer cameras.

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
        directional : bool
            Whether the light is directional.
        castshadow : bool
            Whether the light casts shadows.
        cutoff : float
            Light cutoff angle in degrees.
        attenuation : tuple[float, float, float]
            Light attenuation coefficients (constant, linear, quadratic).
        """
        if self._shared_metadata.renderer is not None:
            self._shared_metadata.renderer.add_light(
                pos=pos,
                dir=dir,
                color=color,
                intensity=intensity,
                directional=directional,
                castshadow=castshadow,
                cutoff=cutoff,
                attenuation=attenuation,
            )
            self._shared_metadata.lights.append(
                {
                    "pos": pos,
                    "dir": dir,
                    "color": color,
                    "intensity": intensity,
                    "directional": directional,
                }
            )

    @gs.assert_built
    def attach(self, rigid_link, offset_T):
        """Attach the camera to a rigid link in the scene."""
        self._attached_link = rigid_link
        self._attached_offset_T = torch.as_tensor(offset_T, dtype=gs.tc_float, device=gs.device)

    @gs.assert_built
    def detach(self):
        """Detach the camera from the currently attached rigid link."""
        self._attached_link = None
        self._attached_offset_T = None

    @gs.assert_built
    def move_to_attach(self):
        """Move the camera to follow the currently attached rigid link."""
        if self._attached_link is None:
            gs.raise_exception("Camera not attached to any rigid link.")

        # Get link pose (for all envs)
        link_pos = self._attached_link.get_pos()
        link_quat = self._attached_link.get_quat()

        # Compute camera transform for env 0 (batch renderer will handle multi-env)
        if link_pos.ndim > 1:
            link_pos = link_pos[0]
            link_quat = link_quat[0]

        # Compute camera transform
        from genesis.utils.geom import trans_quat_to_T, T_to_trans_quat

        link_T = trans_quat_to_T(link_pos, link_quat)
        camera_T = torch.matmul(link_T, self._attached_offset_T)

        # Update camera pose
        self._camera_obj.transform = camera_T
        camera_pos, camera_quat = T_to_trans_quat(camera_T)
        self._camera_obj._pos = camera_pos
        # Note: BatchRenderer will pick up the updated transform on next render
