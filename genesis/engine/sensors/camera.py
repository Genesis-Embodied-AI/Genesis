"""
Camera sensors for rendering: Rasterizer, Raytracer, and Batch Renderer.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Type

import numpy as np
import torch

import genesis as gs
from genesis.options.sensors import (
    RasterizerCameraOptions,
    RaytracerCameraOptions,
    BatchRendererCameraOptions,
    SensorOptions,
)

from genesis.utils.geom import pos_lookat_up_to_T, trans_quat_to_T
from genesis.utils.misc import tensor_to_array
from genesis.vis.batch_renderer import BatchRenderer
from genesis.options.renderers import BatchRenderer as BatchRendererOptions
from genesis.options.vis import VisOptions
from genesis.vis.rasterizer_context import RasterizerContext
from .base_sensor import Sensor, SharedSensorMetadata, RigidSensorMixin, RigidSensorMetadataMixin
from .sensor_manager import register_sensor


if TYPE_CHECKING:
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer import Rasterizer
    from genesis.vis.rasterizer_context import RasterizerContext
    from genesis.vis.batch_renderer import BatchRenderer
    from genesis.vis.raytracer import Raytracer
    from genesis.morphs.rigid_link import RigidLink


# ========================== Data Class ==========================


class CameraData(NamedTuple):
    """Camera sensor return data."""

    rgb: torch.Tensor


class MinimalVisualizerWrapper:
    """
    Minimal visualizer wrapper for BatchRenderer camera sensors.

    BatchRenderer requires a visualizer-like object to provide camera information and context,
    but camera sensors don't need the full visualizer functionality (viewer, UI, etc.).
    This wrapper provides just the minimal interface expected by BatchRenderer while
    avoiding the overhead of creating a full visualizer instance.
    """

    def __init__(self, scene, sensors, vis_options):
        self.scene = scene
        self._cameras = []  # Will be populated with camera wrappers
        self._sensors = sensors  # Keep reference to sensors

        # Create a minimal rasterizer context for camera frustum visualization
        # (required by BatchRenderer even though cameras don't render frustums)
        self._context = RasterizerContext(vis_options)
        self._context.build(scene)
        self._context.reset()


class BaseCameraWrapper:
    """Base class for camera wrappers to reduce code duplication."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.uid = sensor._idx
        self.res = sensor._options.res
        self.fov = sensor._options.fov
        self.near = sensor._options.near
        self.far = sensor._options.far


class RasterizerCameraWrapper(BaseCameraWrapper):
    """Lightweight wrapper object used by the rasterizer backend."""

    def __init__(self, sensor: "RasterizerCameraSensor"):
        super().__init__(sensor)
        self.aspect_ratio = self.res[0] / self.res[1]


class BatchRendererCameraWrapper(BaseCameraWrapper):
    """Wrapper object used by the batch renderer backend."""

    def __init__(self, sensor: "BatchRendererCameraSensor"):
        super().__init__(sensor)
        self.idx = len(sensor._shared_metadata.sensors)  # Camera index in batch
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

        _, quat = T_to_trans_quat(self.transform.unsqueeze(0))
        n_envs = self.sensor._manager._sim._B
        if n_envs > 0:
            return quat.expand(n_envs, -1)
        return quat


# ========================== Shared Metadata ==========================


@dataclass
class RasterizerCameraSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Rasterizer cameras."""

    # Rasterizer instance
    renderer: Optional["Rasterizer"] = None
    # RasterizerContext instance
    context: Optional["RasterizerContext"] = None
    # List of light dictionaries
    lights: Optional[List[Dict[str, Any]]] = None
    # List of RasterizerCameraSensor instances
    sensors: Optional[List["RasterizerCameraSensor"]] = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: Optional[Dict[int, np.ndarray]] = None
    # Track when rasterizer cameras were last updated
    last_render_timestep: int = -1


@dataclass
class RaytracerCameraSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Raytracer cameras."""

    # Raytracer instance
    renderer: Optional["Raytracer"] = None
    # List of light objects
    lights: Optional[List[Any]] = None
    # List of RaytracerCameraSensor instances
    sensors: Optional[List["RaytracerCameraSensor"]] = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: Optional[Dict[int, np.ndarray]] = None
    # Track when raytracer cameras were last updated
    last_render_timestep: int = -1


@dataclass
class BatchRendererCameraSharedMetadata(RigidSensorMetadataMixin, SharedSensorMetadata):
    """Shared metadata for all Batch Renderer cameras."""

    # BatchRenderer instance
    renderer: Optional["BatchRenderer"] = None
    # gs.List of lights
    lights: Optional[Any] = None
    # List of BatchRendererCameraSensor instances
    sensors: Optional[List["BatchRendererCameraSensor"]] = None
    # {sensor_idx: np.ndarray with shape (B, H, W, 3)}
    image_cache: Optional[Dict[int, np.ndarray]] = None
    # Track when batch was last rendered
    last_render_timestep: int = -1
    # MinimalVisualizerWrapper instance
    visualizer_wrapper: Optional["MinimalVisualizerWrapper"] = None


# ========================== Base Camera Sensor ==========================


class BaseCameraSensor(RigidSensorMixin, Sensor[SharedSensorMetadata]):
    """
    Base class for camera sensors that render RGB images into an internal image_cache.

    This class centralizes:
    - Attachment handling via RigidSensorMixin
    - The _stale flag used for auto-render-on-read
    - Common Sensor cache integration (shape/dtype)
    - Shared read() method returning torch tensors
    """

    def __init__(self, options: "SensorOptions", idx: int, data_cls: Type[CameraData], manager: "gs.SensorManager"):
        super().__init__(options, idx, data_cls, manager)
        self._stale: bool = True

    # ========================== Cache Integration (shared) ==========================

    def _get_return_format(self) -> tuple[tuple[int, ...], ...]:
        w, h = self._options.res
        return ((h, w, 3),)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return torch.uint8

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: SharedSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        pass

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: SharedSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        # No per-step measured-cache update for cameras (handled lazily on read()).
        pass

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """No debug drawing for cameras."""
        pass

    # ========================== Attachment handling ==========================

    @gs.assert_built
    def move_to_attach(self):
        """
        Move the camera to follow the currently attached rigid link.

        Uses a shared transform computation and delegates to _apply_camera_transform().
        """
        if self._link is None:
            gs.raise_exception("Camera not attached to any rigid link.")

        # Use pos directly as offset from link
        pos_offset = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        offset_T = trans_quat_to_T(pos_offset, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device))

        camera_T = _camera_compute_T_from_link(self._link, offset_T)
        self._apply_camera_transform(camera_T)

    # ========================== Hooks for subclasses ==========================

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Apply the computed camera transform to the backend-specific camera representation."""
        pass

    def _render_current_state(self):
        """Perform the actual render for the current state; subclasses must implement."""
        pass

    # ========================== Shared read() ==========================

    def _get_image_cache_entry(self):
        """Return this sensor's entry in the shared image cache."""
        return self._shared_metadata.image_cache[self._idx]

    def _ensure_rendered_for_current_state(self):
        """Ensure this camera has an up-to-date render before reading.

        Base handles staleness and timestamps; subclasses implement _render_current_state().
        """
        scene = self._manager._sim.scene

        # If the scene time advanced, mark all cameras as stale
        if self._shared_metadata.last_render_timestep != scene.t:
            if hasattr(self._shared_metadata, "sensors") and self._shared_metadata.sensors is not None:
                for sensor in self._shared_metadata.sensors:
                    sensor._stale = True
            self._shared_metadata.last_render_timestep = scene.t

        # If this camera is not stale, cache is considered fresh
        if not self._stale:
            return

        # Update camera pose only when attached; detached cameras keep their last world pose
        if self._link is not None:
            self.move_to_attach()

        # Call subclass-specific render
        self._render_current_state()

        # Mark as fresh
        self._stale = False

    def _sanitize_envs_idx(self, envs_idx):
        """Sanitize envs_idx to valid indices."""
        if envs_idx is None:
            return None
        if isinstance(envs_idx, (int, np.integer)):
            return envs_idx
        return np.asarray(envs_idx)

    @gs.assert_built
    def read(self, envs_idx=None) -> CameraData:
        """Render if needed, then read the cached image from the backend-specific cache."""
        self._ensure_rendered_for_current_state()
        cached_image = self._get_image_cache_entry()
        return _camera_read_from_image_cache(self, cached_image, envs_idx, to_numpy=False)

    @classmethod
    def reset(cls, shared_metadata, envs_idx):
        """Reset camera sensor (no state to reset)."""
        pass


# ========================== Camera Sensor Helpers ==========================
def _camera_read_from_image_cache(sensor, cached_image, envs_idx, *, to_numpy: bool) -> CameraData:
    """
    Shared helper to convert a cached RGB image array into CameraData with correct env handling.

    Parameters
    ----------
    sensor : any camera sensor with _manager and _return_data_class
    cached_image : np.ndarray | torch.Tensor
        Image cache for this camera, shaped (B, H, W, 3) or (H, W, 3) depending on n_envs.
    envs_idx : None | int | sequence
        Environment index/indices to select.
    to_numpy : bool
        If True and cached_image is a torch Tensor, convert to numpy first.
    """
    if to_numpy and isinstance(cached_image, torch.Tensor):
        cached_image = tensor_to_array(cached_image)

    n_envs = sensor._manager._sim.n_envs

    if envs_idx is None:
        if n_envs == 0:
            return sensor._return_data_class(rgb=cached_image[0])
        return sensor._return_data_class(rgb=cached_image)
    if isinstance(envs_idx, (int, np.integer)):
        return sensor._return_data_class(rgb=cached_image[envs_idx])
    return sensor._return_data_class(rgb=cached_image[envs_idx])


def _camera_compute_T_from_link(attached_link, attached_offset_T: torch.Tensor) -> torch.Tensor:
    """
    Compute camera transform from an attached link pose and offset.

    Uses env 0 if the link pose is batched.
    """
    link_pos = attached_link.get_pos()
    link_quat = attached_link.get_quat()

    # Handle batched case - use first environment
    if link_pos.ndim > 1:
        link_pos = link_pos[0]
        link_quat = link_quat[0]

    link_T = trans_quat_to_T(link_pos, link_quat)
    return torch.matmul(link_T, attached_offset_T)


# ========================== Rasterizer Camera Sensor ==========================


@register_sensor(RasterizerCameraOptions, RasterizerCameraSharedMetadata, CameraData)
class RasterizerCameraSensor(BaseCameraSensor):
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
        self._camera_wrapper = None

    # ========================== Sensor Lifecycle ==========================

    def build(self):
        """Initialize the rasterizer and register this camera."""
        super().build()

        scene = self._manager._sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.image_cache = {}

            self._shared_metadata.context = self._create_standalone_context(scene)

            from genesis.vis.rasterizer import Rasterizer

            self._shared_metadata.renderer = Rasterizer(viewer=None, context=self._shared_metadata.context)
            self._shared_metadata.renderer.build()

        self._shared_metadata.sensors.append(self)

        # Add lights from options to the shared metadata
        for light_config in self._options.lights:
            if self._shared_metadata.lights is not None:
                # Convert light config to rasterizer format
                light_dict = self._convert_light_config_to_rasterizer(light_config)
                # self._shared_metadata.lights.append(light_dict)
                self._shared_metadata.context.add_light(light_dict)

        self._add_camera_to_rasterizer()

        n_envs = max(self._manager._sim._B, 1)
        h, w = self._options.res[1], self._options.res[0]
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (n_envs, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    def _create_standalone_context(self, scene):
        """Create a simplified RasterizerContext for camera sensors."""
        vis_options = VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            rendered_envs_idx=range(max(self._manager._sim._B, 1)),
        )

        context = RasterizerContext(vis_options)
        context.build(scene)
        context.reset()
        return context

    def _convert_light_config_to_rasterizer(self, light_config):
        """Convert a light config dict to rasterizer format."""
        # Default values for rasterizer
        light_type = light_config.get("type", "directional")
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        dir = light_config.get("dir", (0.0, 0.0, -1.0))
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)

        return {
            "type": light_type,
            "pos": pos,
            "dir": dir,
            "color": tuple(np.array(color) * intensity),
            "intensity": intensity,
        }

    def _add_camera_to_rasterizer(self):
        """Add this camera to the rasterizer."""
        camera_wrapper = self._get_camera_wrapper()
        self._shared_metadata.renderer.add_camera(camera_wrapper)

        self._update_camera_pose()

    def _update_camera_pose(self):
        """Update camera pose based on options."""
        pos = torch.tensor(self._options.pos, dtype=gs.tc_float, device=gs.device)
        lookat = torch.tensor(self._options.lookat, dtype=gs.tc_float, device=gs.device)
        up = torch.tensor(self._options.up, dtype=gs.tc_float, device=gs.device)

        # If attached to a link and the link is built, pos is relative to link frame
        if self._link is not None and self._link.is_built:
            # Convert pos from link-relative to world coordinates
            link_pos = self._link.get_pos()
            link_quat = self._link.get_quat()

            # Handle batched case - use first environment
            if link_pos.ndim > 1:
                link_pos = link_pos[0]
                link_quat = link_quat[0]

            # Apply pos directly as offset from link
            from genesis.utils.geom import transform_by_quat

            pos_world = transform_by_quat(pos, link_quat) + link_pos
            pos = pos_world
        elif self._link is not None:
            # Link exists but not built yet - use configured pose as-is (treat as world coordinates for now)
            # This will be corrected when move_to_attach is called
            pass

        transform = pos_lookat_up_to_T(pos, lookat, up)
        camera_wrapper = self._get_camera_wrapper()
        camera_wrapper.transform = tensor_to_array(transform)
        self._shared_metadata.renderer.update_camera(camera_wrapper)

    def _get_camera_wrapper(self):
        """Get (and lazily create) the persistent camera wrapper for the renderer."""
        if self._camera_wrapper is None:
            self._camera_wrapper = RasterizerCameraWrapper(self)
        return self._camera_wrapper

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Update rasterizer camera wrapper from a world transform."""
        camera_wrapper = self._get_camera_wrapper()
        camera_wrapper.transform = tensor_to_array(camera_T)
        self._shared_metadata.renderer.update_camera(camera_wrapper)

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        self._shared_metadata.context.update(force_render=False)

        rgb_arr, _, _, _ = self._shared_metadata.renderer.render_camera(
            self._get_camera_wrapper(), rgb=True, depth=False, segmentation=False, normal=False
        )

        rgb_tensor = torch.from_numpy(rgb_arr.copy()).to(dtype=torch.uint8, device=gs.device)

        # Store in cache
        n_envs = self._manager._sim._B
        if n_envs <= 1:
            # Single environment case - add batch dimension
            self._shared_metadata.image_cache[self._idx][0] = rgb_tensor
        else:
            # Multi-environment rendering is not yet supported for Rasterizer cameras
            gs.raise_exception(
                f"Rasterizer camera sensors do not support multi-environment rendering (n_envs={n_envs}). "
                "Use BatchRenderer camera sensors for batched rendering."
            )


# ========================== Raytracer Camera Sensor ==========================
@register_sensor(RaytracerCameraOptions, RaytracerCameraSharedMetadata, CameraData)
class RaytracerCameraSensor(BaseCameraSensor):
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
        self._camera_obj = None

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

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = []
            self._shared_metadata.image_cache = {}
            self._shared_metadata.renderer = renderer

        self._shared_metadata.sensors.append(self)

        # Add lights from options as mesh lights to the scene
        scene = self._manager._sim.scene
        for light_config in self._options.lights:
            if not scene.is_built:
                self._add_light_as_mesh_light(scene, light_config)

        opts = self._options

        # Compute world pose for the camera
        pos = opts.pos
        lookat = opts.lookat
        up = opts.up

        # If attached to a link and the link is built, transform pos to world coordinates
        if self._link is not None and self._link.is_built:
            link_pos = self._link.get_pos()
            link_quat = self._link.get_quat()

            # Handle batched case - use first environment
            if link_pos.ndim > 1:
                link_pos = link_pos[0]
                link_quat = link_quat[0]

            # Apply pos directly as offset from link
            from genesis.utils.geom import transform_by_quat

            pos_world = transform_by_quat(torch.tensor(pos, dtype=gs.tc_float, device=gs.device), link_quat) + link_pos
            pos = pos_world.tolist()

            # Transform lookat and up (no rotation offset since rotation is defined by lookat/up)
            lookat_world = (
                transform_by_quat(torch.tensor(lookat, dtype=gs.tc_float, device=gs.device), link_quat) + link_pos
            )
            lookat = lookat_world.tolist()

            up_world = transform_by_quat(torch.tensor(up, dtype=gs.tc_float, device=gs.device), link_quat)
            up = up_world.tolist()
        elif self._link is not None:
            # Link exists but not built yet - use configured pose as-is (treat as world coordinates for now)
            # This will be corrected when move_to_attach is called
            pass

        self._camera_obj = visualizer.add_camera(
            res=opts.res,
            pos=pos,
            lookat=lookat,
            up=up,
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

        # Attach the visualizer camera to the link if this sensor is attached
        if self._link is not None:
            from genesis.utils.geom import trans_quat_to_T

            pos_offset = torch.tensor(opts.pos, dtype=gs.tc_float, device=gs.device)
            offset_T = trans_quat_to_T(
                pos_offset, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
            )
            self._camera_obj.attach(self._link, offset_T)

        h, w = self._options.res[1], self._options.res[0]
        n_envs = max(self._manager._sim._B, 1)
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (n_envs, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    def _add_light_as_mesh_light(self, scene, light_config):
        """Add a light as a mesh light to the scene."""
        # Default values for raytracer mesh lights
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        radius = light_config.get("radius", 0.5)
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        revert_dir = light_config.get("revert_dir", False)
        double_sided = light_config.get("double_sided", False)
        cutoff = light_config.get("cutoff", 180.0)

        morph = gs.morphs.Sphere(pos=pos, radius=radius)
        scene.add_mesh_light(
            morph=morph,
            color=(*color, 1.0),
            intensity=intensity,
            revert_dir=revert_dir,
            double_sided=double_sided,
            cutoff=cutoff,
        )

    def _on_attach_backend(self, rigid_link, offset_T):
        """Keep the underlying visualizer camera in sync when attaching."""
        if self._camera_obj is not None:
            self._camera_obj.attach(rigid_link, offset_T)

    def _on_detach_backend(self):
        """Keep the underlying visualizer camera in sync when detaching."""
        if self._camera_obj is not None:
            self._camera_obj.detach()

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        if self._link is not None:
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

        rgb_tensor = torch.from_numpy(rgb_arr.copy()).to(dtype=torch.uint8, device=gs.device)

        n_envs = self._manager._sim._B
        if n_envs <= 1:
            self._shared_metadata.image_cache[self._idx][0] = rgb_tensor
        else:
            # Multi-environment rendering is not yet supported for Raytracer cameras
            gs.raise_exception(
                f"Raytracer camera sensors do not support multi-environment rendering (n_envs={n_envs}). "
                "Use BatchRenderer camera sensors for batched rendering."
            )


# ========================== Batch Renderer Camera Sensor ==========================


@register_sensor(BatchRendererCameraOptions, BatchRendererCameraSharedMetadata, CameraData)
class BatchRendererCameraSensor(BaseCameraSensor):
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

    def build(self):
        """Initialize the batch renderer and register this camera."""
        super().build()

        if gs.backend != gs.cuda:
            gs.raise_exception("BatchRendererCameraSensor requires CUDA backend.")

        scene = self._manager._sim.scene

        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
            self._shared_metadata.lights = gs.List()
            self._shared_metadata.image_cache = {}
            self._shared_metadata.last_render_timestep = -1

            all_sensors = self._manager._sensors_by_type[type(self)]
            resolutions = [s._options.res for s in all_sensors]
            if len(set(resolutions)) > 1:
                gs.raise_exception(
                    f"All BatchRendererCameraSensor instances must have the same resolution. "
                    f"Found: {set(resolutions)}"
                )

            br_options = BatchRendererOptions(
                use_rasterizer=self._options.use_rasterizer,
            )

            vis_options = VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                rendered_envs_idx=range(max(self._manager._sim._B, 1)),
            )

            self._shared_metadata.visualizer_wrapper = MinimalVisualizerWrapper(scene, all_sensors, vis_options)
            self._shared_metadata.renderer = BatchRenderer(
                self._shared_metadata.visualizer_wrapper, br_options, vis_options
            )

        self._shared_metadata.sensors.append(self)

        # Add lights from options to the renderer
        for light_config in self._options.lights:
            if self._shared_metadata.renderer is not None:
                self._add_light_to_batch_renderer(light_config)

        self._camera_obj = BatchRendererCameraWrapper(self)

        if len(self._shared_metadata.sensors) == len(self._manager._sensors_by_type[type(self)]):
            self._shared_metadata.visualizer_wrapper._cameras = [s._camera_obj for s in self._shared_metadata.sensors]
            self._shared_metadata.renderer.build()

        n_envs = max(self._manager._sim._B, 1)
        h, w = self._options.res[1], self._options.res[0]
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (n_envs, h, w, 3), dtype=torch.uint8, device=gs.device
        )

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        sensors = self._shared_metadata.sensors or [self]

        for sensor in sensors:
            if sensor._link is not None:
                sensor.move_to_attach()

        self._shared_metadata.renderer.update_scene(force_render=False)

        rgb_arr, *_ = self._shared_metadata.renderer.render(
            rgb=True, depth=False, segmentation=False, normal=False, antialiasing=False, force_render=False
        )

        # rgb_arr might be a tuple of arrays (one per camera) or a single array
        # Handle both cases
        if isinstance(rgb_arr, (tuple, list)):
            rgb_arr = torch.stack([torch.as_tensor(arr).to(dtype=torch.uint8, device=gs.device) for arr in rgb_arr])
        else:
            rgb_arr = torch.as_tensor(rgb_arr).to(dtype=torch.uint8, device=gs.device)

        for cam_idx, sensor in enumerate(sensors):
            sensor._shared_metadata.image_cache[sensor._idx] = rgb_arr[cam_idx]
            sensor._stale = False

        self._shared_metadata.last_render_timestep = self._manager._sim.scene.t

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Update batch renderer camera from a world transform."""
        from genesis.utils.geom import T_to_trans_quat

        self._camera_obj.transform = camera_T
        camera_pos, camera_quat = T_to_trans_quat(camera_T)
        self._camera_obj._pos = camera_pos
        # Note: BatchRenderer will pick up the updated transform on next render

    def _add_light_to_batch_renderer(self, light_config):
        """Add a light to the batch renderer."""
        # Default values for batch renderer
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        dir = light_config.get("dir", (0.0, 0.0, -1.0))
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        directional = light_config.get("directional", True)
        castshadow = light_config.get("castshadow", True)
        cutoff = light_config.get("cutoff", 45.0)
        attenuation = light_config.get("attenuation", (1.0, 0.0, 0.0))

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
