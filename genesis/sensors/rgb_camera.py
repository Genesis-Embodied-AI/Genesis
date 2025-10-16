from typing import TYPE_CHECKING, Literal, Type

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.sensors.base_sensor import (
    RigidSensorOptionsMixin,
    Sensor,
    SensorOptions,
    SharedSensorMetadata,
    Tuple3FType,
)
from genesis.sensors.sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.sensors.sensor_manager import SensorManager
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext


class RGBCameraOptions(RigidSensorOptionsMixin, SensorOptions):
    """
    A camera which can be used to render RGB, depth, and segmentation images.

    Warning
    -------
    When 'pinhole' is used, the `aperture` and `focal_len` parameters are ignored.

    Parameters
    ----------
    pos : tuple[float, float, float], optional
        The position of the camera in the scene.
    euler : tuple[float, float, float], optional
        The orientation in degree angles of the camera in the scene.
    entity_idx : int, optional
        The global entity index of the RigidEntity to which the camera is attached. If provided, this will override
        pos and euler (use pos_ffset and euler_offset instead), and the camera's position and orientation will be
        updated relative to this entity. Defaults to -1 (not attached to any entity).
    model : Literal["pinhole", "thinlens"]
        Specifies the camera model. Options are 'pinhole' or 'thinlens'.
    res : tuple of int, shape (2,)
        The resolution of the camera, specified as a tuple (width, height).
    fov : float
        The vertical field of view of the camera in degrees.
    aperture : float
        The aperture size of the camera, controlling depth of field.
    focus_dist : float | None
        The focus distance of the camera. If None, it will be auto-computed using `pos` and `lookat`.
    spp : int, optional
        Samples per pixel. Only available when using the RayTracer renderer. Defaults to 256.
    denoise : bool
        Whether to denoise the camera's rendered image. Only available when using the RayTracer renderer.
        Defaults to True.  If OptiX denoiser is not available on your platform, consider enabling the OIDN denoiser
        option when building RayTracer.
    near : float
        Distance from camera center to near plane in meters.
        Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 0.05.
    far : float
        Distance from camera center to far plane in meters.
        Only available when using rasterizer in Rasterizer and BatchRender renderer. Defaults to 100.0.
    env_idx : int, optional
        The index of the environment to track the camera.
    debug : bool, optional
        Whether to use the debug camera. It enables to create cameras that can used to monitor / debug the
        simulation without being part of the "sensors". Their output is rendered by the usual simple Rasterizer
        systematically, no matter if BatchRayTracer is enabled. This way, it is possible to record the
        simulation with arbitrary resolution and camera pose, without interfering with what robots can perceive
        from their environment. Defaults to False.
    debug_color : float, optional
        The rgba color of the debug pyramid. Defaults to (1.0, 1.0, 1.0, 0.5).
    debug_size : float, optional
        The size of the debug pyramid. Defaults to 0.2.
    """

    pos: Tuple3FType = (0.5, 2.5, 3.5)
    euler: Tuple3FType = (0.0, 0.0, 0.0)
    # override from RigidSensorOptionsMixin to provide default value
    entity_idx: int = -1

    model: Literal["pinhole", "thinlens"] = "pinhole"
    res: tuple[int, int] = (320, 320)
    pos: Tuple3FType = (0.5, 2.5, 3.5)
    fov: float = 30.0
    aperture: float = 2.8
    focus_dist: float | None = None
    spp: int = 256
    denoise: bool = True
    near: float = 0.05
    far: float = 100.0
    env_idx: int | None = None

    debug: bool = False
    debug_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5)
    debug_size: float = 0.2

    def validate_rigid_link(self, scene: "Scene"):
        if self.entity_idx >= 0:
            super().validate_rigid_link(scene)


class RGBCameraSharedMetadata(SharedSensorMetadata):
    cameras: tuple["gs.RGBCamera"] = ()


@register_sensor(RGBCameraOptions, RGBCameraSharedMetadata, tuple)
class RGBCameraSensor(Sensor):

    def __init__(
        self,
        sensor_options: RGBCameraOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

        self._visualizer = self._manager._sim._scene._visualizer

        self._camera = self._visualizer.add_camera(
            res=self._options.res,
            pos=self._options.pos,
            transform=gu.trans_R_to_T(np.array(self._options.pos), gu.euler_to_R(self._options.euler)),
            lookat=(0.0, 0.0, 0.0),
            model=self._options.model,
            fov=self._options.fov,
            up=(0.0, 0.0, 1.0),
            aperture=self._options.aperture,
            focus_dist=self._options.focus_dist,
            GUI=False,
            spp=self._options.spp,
            denoise=self._options.denoise,
            near=self._options.near,
            far=self._options.far,
            env_idx=self._options.env_idx,
            debug=self._options.debug,
        )

        self.debug_object: "Mesh" | None = None
        self.offset_T = gu.trans_quat_to_T(
            np.array(self._options.pos_offset), gu.euler_to_quat(self._options.euler_offset)
        )

    def build(self):
        super().build()
        # _camera will be built by visualizer

        if self._visualizer.batch_renderer is None and self._manager._sim.n_envs > 1:
            gs.raise_exception(
                "BatchRenderer must be enabled in the Scene to use RGBCameraSensor with n_envs > 1. "
                "Add `renderer=gs.renderers.BatchRenderer()` when creating the scene."
            )

        if self._options.entity_idx >= 0:
            entity = self._manager._sim.rigid_solver.entities[self._options.entity_idx]
            link = entity.links[self._options.link_idx_local]
            self._camera.attach(link, self.offset_T)
        self._shared_metadata.cameras += (self._camera,)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.res[1], self._options.res[0], 3)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return torch.uint8

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: RGBCameraSharedMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        tensor_idx = 0
        for cam in shared_metadata.cameras:
            rgb, *_ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
            rgb_flattened = rgb.flatten()
            next_idx = tensor_idx + len(rgb_flattened)
            shared_ground_truth_cache[:, tensor_idx:next_idx] = torch.from_numpy(rgb_flattened).to(
                device=gs.device, dtype=gs.tc_float
            )
            tensor_idx = next_idx

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: RGBCameraSharedMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug pyramid where the camera is.
        """
        if self.debug_object is None:
            base_height = 2.0 * self._options.debug_size * np.tan(np.radians(self._options.fov) / 2.0)
            base_width = base_height * self._options.res[0] / self._options.res[1]

            self.debug_object = context.draw_debug_pyramid(
                T=self._camera.transform,
                base_width=base_width,
                base_height=base_height,
                height=self._options.debug_size,
                color=self._options.debug_color,
            )
        else:
            buffer_updates.update(context.get_buffer_debug_objects([self.debug_object], [self._camera.transform]))
