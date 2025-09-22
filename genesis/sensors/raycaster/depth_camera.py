import torch

import genesis as gs
from genesis.sensors.sensor_manager import register_sensor

from .patterns import DepthCameraPattern
from .raycaster import RaycasterOptions, RaycasterSensor, RaycasterSharedMetadata


class DepthCameraOptions(RaycasterOptions):
    """
    Depth camera that uses ray casting to obtain depth images.

    Parameters
    ----------
    entity_idx : int
        The global entity index of the RigidEntity to which this sensor is attached.
    link_idx_local : int, optional
        The local index of the RigidLink of the RigidEntity to which this sensor is attached.
    pos_offset : tuple[float, float, float], optional
        The mounting offset position of the sensor in the world frame. Defaults to (0.0, 0.0, 0.0).
    euler_offset : tuple[float, float, float], optional
        The mounting offset quaternion of the sensor in the world frame. Defaults to (0.0, 0.0, 0.0).
    pattern: DepthCameraPattern
        The raycasting pattern configuration for the sensor.
    min_range : float, optional
        The minimum sensing range in meters. Defaults to 0.0.
    max_range : float, optional
        The maximum sensing range in meters. Defaults to 20.0.
    return_world_frame : bool, optional
        Whether to return points in the world frame. Defaults to False (local frame).
    only_cast_fixed : bool, optional
        Whether to only cast rays on fixed geoms. Defaults to False. This is a shared option, so the value of this
        option for the **first** lidar sensor will be the behavior for **all** Lidar sensors.
    delay : float, optional
        The delay in seconds before the sensor data is read.
    update_ground_truth_only : bool, optional
        If True, the sensor will only update the ground truth cache, and not the measured cache.
    draw_debug : bool, optional
        If True and the rasterizer visualization is active, spheres will be drawn at every hit point.
    debug_sphere_radius: float, optional
        The radius of each debug hit point sphere drawn in the scene. Defaults to 0.02.
    """

    def validate(self, scene):
        super().validate(scene)
        if not isinstance(self.pattern, DepthCameraPattern):
            gs.raise_exception("DepthCamera pattern must be an instance of DepthCameraPattern")


@register_sensor(DepthCameraOptions, RaycasterSharedMetadata)
class DepthCameraSensor(RaycasterSensor):
    def read_image(self) -> torch.Tensor:
        """
        Read the depth image from the sensor.

        This method uses the hit distances from the underlying RaycasterSensor.read() method and reshapes into image.

        Returns
        -------
        torch.Tensor
            The depth image with shape (height, width).
        """
        return self.read()["hit_ranges"].reshape(self._options.pattern.height, self._options.pattern.width)
